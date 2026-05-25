"""Human-arm pose sources for webcam teleoperation.

A *pose source* yields the 3D positions of one arm's shoulder, elbow, and wrist
(``ArmLandmarks``) in a **robot-aligned world frame**:

    +x forward (away from the person, into the robot's workspace)
    +y to the person's/robot's left
    +z up

Two implementations:

* ``ScriptedPoseSource`` — synthetic, deterministic arm motion. No camera, no
  dependencies; used by the headless tests and the offline demo so the whole
  retargeting/teleop stack is verifiable without hardware.
* ``WebcamPoseSource`` — live webcam via MediaPipe Pose. Imported lazily so the
  package never hard-depends on ``mediapipe``/``opencv``; the live demo plugs in
  here unchanged.

The two share the same ``get()`` -> ``ArmLandmarks | None`` contract, so the
retargeter and controller don't care which one is feeding them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ArmLandmarks:
    """One arm's shoulder/elbow/wrist as 3D points in the robot-aligned frame.

    ``grasp`` is an optional hand-closure signal in [0, 1] (0 = open hand,
    1 = closed fist) used to drive the gripper; ``None`` means "not tracked".
    """
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    side: str = "right"          # "right" or "left" human arm
    grasp: float = None          # hand closure in [0,1], or None

    def __post_init__(self):
        self.shoulder = np.asarray(self.shoulder, dtype=float).reshape(3)
        self.elbow = np.asarray(self.elbow, dtype=float).reshape(3)
        self.wrist = np.asarray(self.wrist, dtype=float).reshape(3)

    @property
    def upper_arm(self) -> np.ndarray:
        """Shoulder -> elbow vector."""
        return self.elbow - self.shoulder

    @property
    def forearm(self) -> np.ndarray:
        """Elbow -> wrist vector (the direction the hand extends)."""
        return self.wrist - self.elbow

    @property
    def arm_length(self) -> float:
        """Total upper-arm + forearm length (the human's reach)."""
        return float(np.linalg.norm(self.upper_arm) + np.linalg.norm(self.forearm))

    def is_valid(self) -> bool:
        pts = np.stack([self.shoulder, self.elbow, self.wrist])
        return bool(np.all(np.isfinite(pts)) and self.arm_length > 1e-3)


class PoseSource:
    """Interface: ``get()`` returns the latest ``ArmLandmarks`` or ``None``."""

    def get(self) -> "ArmLandmarks | None":
        raise NotImplementedError

    def close(self):
        pass


class ScriptedPoseSource(PoseSource):
    """Deterministic synthetic arm motion in the robot-aligned frame.

    The shoulder is fixed; the upper arm and forearm swing through a smooth,
    repeatable reaching/waving pattern that stays in front of the body. Lets the
    retargeting + teleop stack be exercised end-to-end with no camera.
    """

    def __init__(self, side="right", upper=0.30, fore=0.27, rate=0.6,
                 shoulder=(0.0, 0.0, 0.0)):
        self.side = side
        self.upper = float(upper)          # human upper-arm length (m)
        self.fore = float(fore)            # human forearm length (m)
        self.rate = float(rate)            # motion speed (rad/s of the phase)
        self.shoulder = np.asarray(shoulder, dtype=float)
        self.t = 0.0
        self._sign = 1.0 if side == "right" else -1.0   # right arm reaches -y? see below

    def step(self, dt: float):
        self.t += dt

    def get(self) -> ArmLandmarks:
        t = self.t * self.rate
        # Upper-arm direction: mostly forward (+x), gently swinging in y and
        # up/down. For the right arm we bias toward -y (the robot's right
        # workspace); the left arm mirrors to +y. Amplitudes are modest so the
        # mapped wrist stays inside the arm's dexterous workspace.
        s = -1.0 if self.side == "right" else 1.0
        ua = np.array([
            0.85 + 0.07 * np.sin(t * 0.7),
            s * (0.30 + 0.09 * np.sin(t)),
            -0.02 + 0.15 * np.sin(t * 0.9 + 0.5),
        ])
        ua = ua / np.linalg.norm(ua)
        elbow = self.shoulder + self.upper * ua

        # Forearm bends gently relative to the upper arm (elbow flexion).
        bend = 0.9 + 0.22 * np.sin(t * 1.3)
        fa = ua + np.array([0.12 * np.cos(bend), 0.0, 0.20 * np.sin(bend)])
        fa = fa / np.linalg.norm(fa)
        wrist = elbow + self.fore * fa

        # Synthetic grasp signal: a slow open/close cycle so the gripper path is
        # exercised headlessly (0 = open, 1 = closed fist).
        grasp = 0.5 + 0.5 * np.sin(t * 0.8)
        return ArmLandmarks(self.shoulder, elbow, wrist, side=self.side, grasp=grasp)


# Pretrained MediaPipe Tasks model bundles (auto-downloaded + cached on first use).
_POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                   "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
_HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/latest/hand_landmarker.task")

# MediaPipe hand-landmark indices used for the grasp (fist-closure) signal.
_HAND_WRIST, _HAND_MID_MCP = 0, 9
_HAND_TIPS = (8, 12, 16, 20)            # index, middle, ring, pinky fingertips


def _cache_path(filename):
    return os.path.join(os.path.expanduser("~"), ".cache", "openarm", filename)


def _ensure_model(url, path):
    """Return a local model path, downloading it once (cached under ~/.cache)."""
    import urllib.request
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[teleop] downloading model -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def hand_closure(hand_landmarks):
    """Map 21 hand landmarks to a closure signal in [0,1] (0 open, 1 closed fist).

    Uses the mean fingertip-to-wrist distance normalised by hand size, so it is
    scale- and distance-invariant.
    """
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks])
    ref = np.linalg.norm(pts[_HAND_MID_MCP] - pts[_HAND_WRIST]) + 1e-6
    ext = np.mean(np.linalg.norm(pts[list(_HAND_TIPS)] - pts[_HAND_WRIST], axis=1)) / ref
    # ext ~ 1.0 for a closed fist, ~2.1 for an open hand.
    return float(np.clip((2.0 - ext) / (2.0 - 1.1), 0.0, 1.0))


class WebcamPoseSource(PoseSource):
    """Live webcam arm tracking via MediaPipe.

    Lazily imports ``cv2`` and ``mediapipe`` so the platform never hard-depends
    on them. Prefers the modern **Tasks API** (``PoseLandmarker``, the only one
    shipped by recent wheels) and auto-downloads its model bundle once; falls
    back to the legacy ``mp.solutions.pose`` API on older installs. Either way it
    converts the metric ``pose_world_landmarks`` for the chosen arm into the
    robot-aligned frame.

    MediaPipe world landmarks are metric and roughly hip-centered with axes
    ``x`` right, ``y`` down, ``z`` toward the camera. We map them to the robot
    frame (+x forward, +y left, +z up). With ``mirror=True`` the robot follows
    you like a mirror (your right hand -> the robot's right arm reaching toward
    you), which is the natural feel for face-on teleop.
    """

    # MediaPipe Pose landmark indices (same in both APIs).
    _IDX = {
        "right": (12, 14, 16),   # shoulder, elbow, wrist (subject's right)
        "left": (11, 13, 15),
    }

    def __init__(self, side="right", camera=0, mirror=True, min_conf=0.5,
                 scale=1.0, model_path=None, track_grasp=True):
        import cv2                              # noqa: F401  (lazy, optional dep)
        import mediapipe as mp                  # noqa: F401

        self.side = side
        self.mirror = bool(mirror)
        self.scale = float(scale)
        self.track_grasp = bool(track_grasp)
        self._cv2 = cv2
        self._mp = mp
        self._cap = cv2.VideoCapture(camera)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam {camera}")
        self.last_frame = None                  # BGR frame for optional preview
        self.last_image_lms = None              # [(x,y),...] image coords for drawing
        self.last_arm = None                    # last ArmLandmarks (robot frame)
        self._ts = 0                            # monotonic timestamp (ms) for VIDEO mode
        self._hands = None

        # Prefer the Tasks API (PoseLandmarker); fall back to legacy solutions.
        self._mode = None
        if hasattr(mp, "tasks") and hasattr(mp, "Image"):
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            model = _ensure_model(_POSE_MODEL_URL,
                                  model_path or _cache_path("pose_landmarker_lite.task"))
            self._landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model),
                    running_mode=vision.RunningMode.VIDEO, num_poses=1,
                    min_pose_detection_confidence=min_conf,
                    min_tracking_confidence=min_conf))
            self._mode = "tasks"
            if self.track_grasp:                # second model: hand landmarks -> grasp
                hmodel = _ensure_model(_HAND_MODEL_URL,
                                       _cache_path("hand_landmarker.task"))
                self._hands = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=hmodel),
                        running_mode=vision.RunningMode.VIDEO, num_hands=2,
                        min_hand_detection_confidence=min_conf,
                        min_tracking_confidence=min_conf))
        elif hasattr(mp, "solutions"):
            self._pose = mp.solutions.pose.Pose(
                model_complexity=1, min_detection_confidence=min_conf,
                min_tracking_confidence=min_conf)
            self._mode = "solutions"
        else:
            raise RuntimeError(
                "Installed mediapipe exposes neither the Tasks API nor "
                "mp.solutions.pose — try `pip install -U mediapipe`.")

    def _to_robot_frame(self, lm) -> np.ndarray:
        """MediaPipe world landmark (x right, y down, z toward cam) -> robot frame."""
        x, y, z = lm.x, lm.y, lm.z
        my = -1.0 if self.mirror else 1.0
        # forward = away from camera = -z(mediapipe); left = -x; up = -y.
        return self.scale * np.array([-z, my * (-x), -y])

    def _grasp(self, image, wrist_xy):
        """Closure of the hand nearest the tracked wrist (image coords), or None."""
        if self._hands is None:
            return None
        res = self._hands.detect_for_video(image, self._ts)
        if not res.hand_landmarks:
            return None
        # Associate the hand whose wrist is closest to the arm's wrist in-image.
        best, best_d = None, 1e9
        for hand in res.hand_landmarks:
            d = (hand[0].x - wrist_xy[0]) ** 2 + (hand[0].y - wrist_xy[1]) ** 2
            if d < best_d:
                best, best_d = hand, d
        return hand_closure(best)

    def get(self) -> "ArmLandmarks | None":
        ok, frame = self._cap.read()
        if not ok:
            return None
        self.last_frame = frame
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        si, ei, wi = self._IDX[self.side]

        grasp, img_lms = None, None
        if self._mode == "tasks":
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            self._ts += 33
            res = self._landmarker.detect_for_video(image, self._ts)
            if not res.pose_world_landmarks:
                return None
            world = res.pose_world_landmarks[0]
            if res.pose_landmarks:
                pl = res.pose_landmarks[0]
                img_lms = [(pl[i].x, pl[i].y) for i in (si, ei, wi)]
                if self.track_grasp:
                    grasp = self._grasp(image, (pl[wi].x, pl[wi].y))
        else:                                           # legacy solutions API
            res = self._pose.process(rgb)
            if not res.pose_world_landmarks:
                return None
            world = res.pose_world_landmarks.landmark

        la = ArmLandmarks(self._to_robot_frame(world[si]),
                          self._to_robot_frame(world[ei]),
                          self._to_robot_frame(world[wi]), side=self.side, grasp=grasp)
        if not la.is_valid():
            return None
        self.last_image_lms, self.last_arm = img_lms, la
        return la

    def close(self):
        try:
            self._cap.release()
            if self._mode == "tasks":
                self._landmarker.close()
                if self._hands is not None:
                    self._hands.close()
            elif self._mode == "solutions":
                self._pose.close()
        except Exception:
            pass
