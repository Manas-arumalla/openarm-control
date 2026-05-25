"""Vision-based 3D ball perception for catching.

Replaces ground-truth ball state with what cameras actually see: each RGB-D
camera detects the ball in its image and deprojects to a 3D point; the points
from all cameras are fused into one noisy estimate that feeds the Kalman filter.
The catcher never reads the ball's true pose — only this estimate.

Detection sits behind a small ``BallDetector`` interface so the classical
color-blob detector used now can be swapped for a learned detector later.
"""

import numpy as np
import mujoco

from .camera import Camera
from .detection import detect_color, detect_color_blobs


class BallDetector:
    """Interface: map an RGB image to a ball pixel (u, v) or None."""

    def detect(self, rgb):
        raise NotImplementedError


class ColorBlobDetector(BallDetector):
    """Classical: largest blob of a color -> centroid pixel. No training."""

    def __init__(self, color="orange", min_area=6):
        self.color = color
        self.min_area = min_area

    def detect(self, rgb):
        d = detect_color(rgb, self.color, min_area=self.min_area)
        return None if d is None else d["centroid"]


class BallPerception:
    """Fuse one or more RGB-D cameras into a single 3D ball estimate.

    ``observe()`` renders each camera, detects the ball, deprojects the detected
    pixel (correcting the near-surface bias by half a diameter along the camera
    ray so the estimate is the ball *centre*), and returns the mean over the
    cameras that saw it (or None if none did). Optional Gaussian noise models
    residual sensor error on top of the rendering/discretisation error.
    """

    def __init__(self, model, data, cam_names, detector=None, ball_radius=0.035,
                 width=320, height=240, noise_std=0.0, seed=0):
        self.cams = [Camera(model, data, n, width=width, height=height) for n in cam_names]
        self.detector = detector if detector is not None else ColorBlobDetector()
        self.ball_radius = ball_radius
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.last_per_cam = []          # diagnostics: per-camera 3D points last frame

    def _camera_estimate(self, cam):
        rgb = cam.rgb()
        uv = self.detector.detect(rgb)
        if uv is None:
            return None
        u, v = uv
        depth = cam.depth()
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= vi < cam.height and 0 <= ui < cam.width):
            return None
        z = float(depth[vi, ui])
        if z <= 0 or z > 50:            # no hit / sky
            return None
        surf = cam.deproject(u, v, z)   # point on the ball's near surface
        cam_pos, _ = cam.pose()
        ray = surf - cam_pos
        ray /= (np.linalg.norm(ray) + 1e-9)
        return surf + ray * self.ball_radius   # advance to the ball centre

    def observe(self):
        """Return a fused 3D ball-centre estimate (world frame) or None."""
        pts = [p for p in (self._camera_estimate(c) for c in self.cams) if p is not None]
        self.last_per_cam = pts
        if not pts:
            return None
        est = np.mean(pts, axis=0)
        if self.noise_std > 0:
            est = est + self.rng.normal(0, self.noise_std, 3)
        return est


class MultiBallPerception:
    """Detect MULTIPLE same-coloured balls and return anonymous 3D estimates.

    Each camera detects up to ``n_balls`` blobs and deprojects them (centre,
    radius-corrected); points from all cameras are then clustered by proximity so
    the two views of the same physical ball merge. Returns a list of <= n_balls
    3D points (unordered) — association into tracks is the tracker's job.
    """

    def __init__(self, model, data, cam_names, n_balls=2, color="orange",
                 ball_radius=0.035, width=320, height=240, noise_std=0.0,
                 cluster_tol=0.18, seed=0):
        self.cams = [Camera(model, data, n, width=width, height=height) for n in cam_names]
        self.color = color
        self.n_balls = n_balls
        self.ball_radius = ball_radius
        self.noise_std = noise_std
        self.cluster_tol = cluster_tol
        self.rng = np.random.default_rng(seed)
        self.last = []

    def _camera_points(self, cam):
        rgb = cam.rgb()
        uvs = detect_color_blobs(rgb, self.color, k=self.n_balls)
        if not uvs:
            return []
        depth = cam.depth()
        cam_pos, _ = cam.pose()
        pts = []
        for u, v in uvs:
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= vi < cam.height and 0 <= ui < cam.width):
                continue
            z = float(depth[vi, ui])
            if z <= 0 or z > 50:
                continue
            surf = cam.deproject(u, v, z)
            ray = surf - cam_pos
            ray /= (np.linalg.norm(ray) + 1e-9)
            pts.append(surf + ray * self.ball_radius)
        return pts

    def observe(self):
        allpts = [p for c in self.cams for p in self._camera_points(c)]
        # greedy proximity clustering: merge the two cameras' views of each ball
        clusters = []
        for p in allpts:
            for cl in clusters:
                if np.linalg.norm(p - np.mean(cl, axis=0)) < self.cluster_tol:
                    cl.append(p)
                    break
            else:
                clusters.append([p])
        clusters.sort(key=len, reverse=True)
        ests = [np.mean(cl, axis=0) for cl in clusters[:self.n_balls]]
        if self.noise_std > 0:
            ests = [e + self.rng.normal(0, self.noise_std, 3) for e in ests]
        self.last = ests
        return ests
