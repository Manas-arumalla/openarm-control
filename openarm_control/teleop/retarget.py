"""Retarget a human arm pose onto OpenArm joint angles (anatomical mimicry).

The OpenArm is anatomically arm-like (3-DOF shoulder, elbow, 3-DOF wrist), so we
mimic at the level of **limb directions**, anchored at the shoulder:

* upper-arm direction  ``û = elbow - shoulder``  -> place the robot *elbow* along
  ``û`` (at the robot's own upper-arm length), so the robot's upper arm points
  where yours does;
* forearm direction  ``f̂ = wrist - elbow``  -> place the robot *wrist* along
  ``f̂`` from that elbow, so the robot's forearm points where yours does.

A weighted IK then matches **both** the elbow and the wrist positions, so the
*whole arm* moves naturally (the shoulder-to-elbow segment included), not just the
hand. Because everything is built from directions measured *relative to your
shoulder*, the result is invariant to where your body is and to your other arm —
moving the other arm no longer drags this one (the previous absolute,
torso-anchored mapping coupled them).

Why direction-based reconstruction (vs joint-by-joint angle mapping): it is
independent of the exact joint-axis conventions, reuses the project's tested IK,
and the reconstructed targets are reachable by construction (they sit at the
robot's own link lengths). Warm-starting each solve from the previous solution (a
single descent, not the public multi-seed IK) keeps the joint trajectory
temporally coherent — no configuration flips — which teleop needs.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..kinematics import OpenArmKinematics, orientation_error
from ..catching import _unit
from ..config import IK_DAMPING


def _rot_between(a, b):
    """Rotation matrix taking unit-ish vector ``a`` onto ``b`` (Rodrigues)."""
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if s < 1e-8:                       # parallel (c~+1) or anti-parallel (c~-1)
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _best_rotation(src, dst):
    """Rotation best mapping the columns of ``src`` onto ``dst`` (Kabsch/SVD)."""
    H = np.asarray(src, float) @ np.asarray(dst, float).T
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ D @ U.T


class ArmRetargeter:
    """Map ``ArmLandmarks`` -> a 7-DOF joint target by anatomical reconstruction.

    Parameters
    ----------
    kinematics : OpenArmKinematics
        IK/FK for the arm being driven.
    shoulder_body, elbow_body : str
        Bodies whose origins are the robot's shoulder pivot and elbow
        (``openarm_<arm>_link1`` and ``...link4``).
    R_align : (3,3) or None
        Rotation from the landmark frame to the robot base frame (default
        identity: pose sources already deliver robot-aligned coordinates).
    track_orientation : bool
        Point the gripper along the forearm (low-weight, fills the remaining roll
        DOF after the elbow + wrist positions are matched).
    w_wrist, w_elbow, w_orient : float
        IK objective weights (wrist position, elbow/posture position, orientation).
    dir_smooth : float
        EMA factor for the limb directions (anti-jitter), in (0, 1].
    """

    def __init__(self, kinematics: OpenArmKinematics, shoulder_body: str,
                 elbow_body: str, R_align=None, track_orientation=True,
                 w_upper=1.0, w_fore=1.2, w_orient=0.1, dir_smooth=0.5,
                 reach_max=0.62, reach_min=0.16, fill=0.95):
        self.kin = kinematics
        self.R_align = np.eye(3) if R_align is None else np.asarray(R_align, float)
        self.track_orientation = bool(track_orientation)
        # Posture-matching weights: upper-arm direction, forearm direction, gripper.
        self.w_upper, self.w_fore, self.w_orient = w_upper, w_fore, w_orient
        self.dir_smooth = float(dir_smooth)
        self.reach_max, self.reach_min, self.fill = reach_max, reach_min, fill

        m, d = kinematics.model, kinematics.data
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, shoulder_body)
        eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, elbow_body)
        if sid == -1 or eid == -1:
            raise ValueError("shoulder/elbow body not found")
        self._elbow_bid = eid
        mujoco.mj_kinematics(m, d)
        self.p_shoulder = d.xpos[sid].copy()         # fixed in the base frame

        # Set on calibrate():
        self.scale = None                             # human->robot displacement gain
        self.wrist_rel_home = None                    # human wrist - shoulder at home
        self.ee_home = None                           # robot tool point at home
        self.R_cal = np.eye(3)                        # human->robot direction align
        self.R_home = None                            # robot gripper orientation home
        self.f_home = None                            # robot forearm dir home
        self._u_s = self._f_s = None                  # smoothed limb directions
        self.last_target = None                       # (wrist_pos, R) for logging
        self.last_elbow_target = None                 # elbow direction (for logging)
        self._q_prev = None

    # -------------------------------------------------------------- calibrate
    def calibrate(self, landmarks):
        """Anchor at a neutral pose: the human wrist (relative to the shoulder)
        maps to the robot tool point; the rotation aligning your limb directions
        to the robot's is fixed; the gain spans your reach over the robot's."""
        kin = self.kin
        p_wrist, R = kin.forward_kinematics()
        p_elbow = kin.data.xpos[self._elbow_bid].copy()
        u_r = _unit(p_elbow - self.p_shoulder)        # robot home upper-arm dir
        f_r = _unit(p_wrist - p_elbow)                # robot home forearm dir
        self.ee_home, self.R_home, self.f_home = p_wrist.copy(), R.copy(), f_r

        self.scale = self.fill * self.reach_max / max(landmarks.arm_length, 1e-3)
        # Direct, axis-correct mapping (the landmark frame is already robot-aligned
        # via the pose source). No neutral-pose alignment rotation: aligning the
        # human neutral to the robot's tilted home pose rotated all subsequent
        # motions (up<->left/right scrambling), so we map directions absolutely.
        self.R_cal = np.eye(3)
        u_h = _unit(self.R_align @ landmarks.upper_arm)
        f_h = _unit(self.R_align @ landmarks.forearm)
        self._u_s, self._f_s = u_h.copy(), f_h.copy()
        self._q_prev = kin.data.qpos[kin.qpos_indices].copy()

    def _ensure_calibrated(self, landmarks):
        if self.scale is None:
            self.calibrate(landmarks)

    # ----------------------------------------------------------- reconstruct
    def _arm_dirs(self, landmarks):
        """Smoothed robot-frame upper-arm and forearm unit directions."""
        u = self.R_cal @ _unit(self.R_align @ landmarks.upper_arm)
        f = self.R_cal @ _unit(self.R_align @ landmarks.forearm)
        a = self.dir_smooth
        self._u_s = _unit((1 - a) * self._u_s + a * u)
        self._f_s = _unit((1 - a) * self._f_s + a * f)
        return self._u_s, self._f_s

    def reconstruct(self, landmarks):
        """Smoothed robot-frame upper-arm and forearm directions (u, f)."""
        self._ensure_calibrated(landmarks)
        return self._arm_dirs(landmarks)

    def orientation_target(self, f_dir):
        """Gripper orientation: home orientation rotated by the change in forearm
        direction (continuous; fills the roll DOF left after positions match)."""
        return _rot_between(self.f_home, f_dir) @ self.R_home

    # --------------------------------------------------------------- IK solve
    def _solve(self, u_dir, f_dir, R_wrist, q_seed, max_iters=90):
        """Warm-started IK that matches the whole-arm **posture by direction**:
        rotate the upper arm (shoulder->elbow) onto ``u_dir`` and the forearm
        (elbow->wrist) onto ``f_dir``, plus a low-weight gripper orientation.

        Matching the two limb *directions* (not the wrist position) is what lets
        the *whole* arm mimic — the shoulder-to-elbow segment moves too — while
        staying self-consistent and reachable (no length/scale conflict). The
        hand position is emergent: it follows your reach at the robot's scale.
        Returns ``(q, posture_error_deg)``.
        """
        kin = self.kin
        m, d = kin.model, kin.data
        qpos_orig = d.qpos.copy()
        qadr, dadr = kin.qpos_indices, kin.dof_indices
        ebid = self._elbow_bid

        q = np.clip(np.asarray(q_seed, float).copy(), kin.jnt_low, kin.jnt_high)
        prev, post_err = np.inf, np.inf
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        ep = np.zeros((3, m.nv)); er = np.zeros((3, m.nv))

        for _ in range(max_iters):
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p_tool, R_tool = kin._tool_pose()
            p_elb = d.xpos[ebid]
            mujoco.mj_jac(m, d, jp, jr, p_tool, kin.ee_body_id)
            mujoco.mj_jac(m, d, ep, er, p_elb, ebid)

            # Tangential (direction-only) errors: rotate each segment onto target.
            r_up = np.linalg.norm(p_elb - self.p_shoulder)
            e_up = (self.p_shoulder + r_up * u_dir) - p_elb
            cur_f = p_tool - p_elb
            r_fo = np.linalg.norm(cur_f)
            e_fo = (p_elb + r_fo * f_dir) - p_tool
            post_err = np.degrees(np.arccos(np.clip(
                np.dot(_unit(p_elb - self.p_shoulder), u_dir), -1, 1)))

            rows = [self.w_upper * e_up, self.w_fore * e_fo]
            Js = [self.w_upper * ep[:, dadr], self.w_fore * jp[:, dadr]]
            if R_wrist is not None:
                rows.append(self.w_orient * orientation_error(R_tool, R_wrist))
                Js.append(self.w_orient * jr[:, dadr])

            e = np.concatenate(rows)
            ne = np.linalg.norm(e)
            lam = max(IK_DAMPING * 0.7, 1e-4) if ne < prev else min(IK_DAMPING * 2, 10.0)
            prev = ne
            J = np.vstack(Js)
            dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(J.shape[0]), e)
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, kin.jnt_low, kin.jnt_high)
            if step < 5e-4:
                break

        d.qpos[:] = qpos_orig
        mujoco.mj_kinematics(m, d)
        return q, post_err

    # --------------------------------------------------------------- retarget
    def retarget(self, landmarks, q_seed):
        """Return a 7-vector joint target reproducing the human arm pose."""
        if self._q_prev is None:
            self._q_prev = np.asarray(q_seed, dtype=float).copy()
        self._ensure_calibrated(landmarks)

        u, f = self.reconstruct(landmarks)
        R_target = self.orientation_target(f) if self.track_orientation else None
        q, _ = self._solve(u, f, R_target, self._q_prev)

        self._q_prev = q
        # FK of the solution for logging (emergent wrist position).
        p_wrist, _ = self.kin.forward_kinematics(q)
        self.last_target = (p_wrist, R_target)
        self.last_elbow_target = self.p_shoulder + 0.27 * u
        return q
