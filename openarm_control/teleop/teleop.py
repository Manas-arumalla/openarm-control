"""Real-time teleoperation: stream a pose source onto one arm, safely.

``TeleopController`` ties a ``PoseSource`` (webcam or scripted) to an
``ArmRetargeter`` and drives the arm's position actuators every control tick.
Between the raw retargeted target and the motors it applies three safety layers:

1. **EMA smoothing** — low-pass the joint target so jitter/dropouts don't snap
   the arm.
2. **Velocity limiting** — cap per-joint motion per tick (rad/s), so a big pose
   jump becomes a bounded, safe slew instead of a lunge.
3. **Joint-limit clamping** — never command outside the model's joint ranges.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..config import RIGHT_ARM, GRASP_LOCAL_OFFSET
from ..kinematics import OpenArmKinematics
from .retarget import ArmRetargeter
from .pose import ScriptedPoseSource


# Per-joint slew limits (rad/s) — generous enough for natural motion, low enough
# to stay safe; shoulder joints slower than wrist.
QD_LIMIT = np.array([2.5, 2.5, 3.0, 3.0, 4.0, 4.0, 5.0])


class TeleopController:
    def __init__(self, model, data, arm=RIGHT_ARM, source=None, retargeter=None,
                 control_dt=0.02, smooth=0.35, qd_limit=None):
        self.model = model
        self.data = data
        self.arm = arm
        self.dt = float(control_dt)
        self.smooth = float(smooth)                 # EMA factor (0..1]
        self.qd_limit = QD_LIMIT if qd_limit is None else np.asarray(qd_limit, float)

        self.kin = OpenArmKinematics(
            model, data, joint_names=arm.joints, site_name=arm.ee_site)
        self.source = source if source is not None else ScriptedPoseSource(arm.name)
        self.retargeter = retargeter if retargeter is not None else ArmRetargeter(
            self.kin, shoulder_body=f"openarm_{arm.name}_link1",
            elbow_body=f"openarm_{arm.name}_link4")

        self.qpos_idx = self.kin.qpos_indices
        self.jnt_low, self.jnt_high = self.kin.jnt_low, self.kin.jnt_high
        self._act_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            for a in arm.actuators])
        # Gripper actuator (driven by the hand-closure signal, if tracked).
        self._grip_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, arm.gripper_actuator)
        self._grip_cmd = float(arm.gripper_open)            # start open

        # Graspable objects: closing the hand near one welds it to the gripper
        # (a real pick), opening the hand releases it. Set by enable_grasping().
        self.graspables = []
        self._held = None
        self._ee_body = int(model.site_bodyid[self.kin.site_id])
        # Forgiving grab: close hand within ~13 cm of a block welds it (the hand
        # position is emergent under posture mapping, so precise aiming is hard).
        self.grasp_close_thr, self.grasp_open_thr, self.grasp_radius = 0.55, 0.35, 0.13

        self.q_cmd = self.data.qpos[self.qpos_idx].copy()   # commanded target
        self.calibrated = False

    def enable_grasping(self, blocks):
        """Let the hand pick up the given block bodies (weld-on-close)."""
        self.graspables = list(blocks)

    # -------------------------------------------------------------- helpers
    def q_now(self):
        return self.data.qpos[self.qpos_idx].copy()

    def calibrate(self):
        """Sample one pose and fix the human->robot scale (arm should be extended)."""
        la = self.source.get()
        if la is not None and la.is_valid():
            self.retargeter.calibrate(la)
            self.calibrated = True
        return self.calibrated

    # ----------------------------------------------------------------- step
    def step(self):
        """Advance one control tick: read pose, retarget, smooth/limit, command."""
        if isinstance(self.source, ScriptedPoseSource):
            self.source.step(self.dt)
        la = self.source.get()
        if la is None or not la.is_valid():
            self._apply(self.q_cmd)                  # hold last command
            return None

        if not self.calibrated:
            self.retargeter.calibrate(la)
            self.calibrated = True

        q_target = self.retargeter.retarget(la, self.q_cmd)

        # 1) EMA smoothing toward the new target.
        q_des = (1.0 - self.smooth) * self.q_cmd + self.smooth * q_target
        # 2) Velocity limiting (bounded slew per tick).
        dq = np.clip(q_des - self.q_cmd, -self.qd_limit * self.dt,
                     self.qd_limit * self.dt)
        self.q_cmd = self.q_cmd + dq
        # 3) Joint-limit clamping.
        self.q_cmd = np.clip(self.q_cmd, self.jnt_low, self.jnt_high)

        self._apply(self.q_cmd)
        self._apply_grasp(la.grasp)
        return q_target

    def _apply_grasp(self, grasp):
        """Map a hand-closure signal in [0,1] to the gripper actuator (smoothed),
        and weld/release a nearby block on close/open (a real pick)."""
        if grasp is None or self._grip_id < 0:
            return
        a_open, a_closed = self.arm.gripper_open, 0.0   # ctrl: open .. closed
        target = (1.0 - grasp) * a_open + grasp * a_closed
        self._grip_cmd += 0.3 * (target - self._grip_cmd)   # EMA, anti-jitter
        self.data.ctrl[self._grip_id] = self._grip_cmd
        self._update_grasp_attachment(grasp)

    def grasp_point(self):
        """World point between the gripper pads (where a grasp closes)."""
        p = self.data.site_xpos[self.kin.site_id]
        R = self.data.site_xmat[self.kin.site_id].reshape(3, 3)
        return p + R @ np.asarray(GRASP_LOCAL_OFFSET, float)

    def _weld_id(self, block):
        name = self.arm.weld(block.split("_")[-1])      # e.g. grasp_right_red
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def _attach(self, block):
        """Weld a block to the gripper at the current relative pose."""
        eid = self._weld_id(block)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, block)
        if eid < 0 or bid < 0:
            return
        p1, q1 = self.data.xpos[self._ee_body], self.data.xquat[self._ee_body]
        p2, q2 = self.data.xpos[bid], self.data.xquat[bid]
        nq1 = np.zeros(4); mujoco.mju_negQuat(nq1, q1)
        relpos = np.zeros(3); mujoco.mju_rotVecQuat(relpos, p2 - p1, nq1)
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, nq1, q2)
        self.model.eq_data[eid, 0:3] = 0.0
        self.model.eq_data[eid, 3:6] = relpos
        self.model.eq_data[eid, 6:10] = relquat
        self.model.eq_data[eid, 10] = 1.0
        self.data.eq_active[eid] = 1
        self._held = block

    def _detach(self):
        if self._held is not None:
            eid = self._weld_id(self._held)
            if eid >= 0:
                self.data.eq_active[eid] = 0
            self._held = None

    def _update_grasp_attachment(self, grasp):
        """Close the hand near a block -> weld it; open the hand -> release."""
        if not self.graspables:
            return
        if self._held is None and grasp > self.grasp_close_thr:
            gp = self.grasp_point()
            near = [(np.linalg.norm(self.data.xpos[
                        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)] - gp), b)
                    for b in self.graspables]
            d, b = min(near)
            if d < self.grasp_radius:
                self._attach(b)
        elif self._held is not None and grasp < self.grasp_open_thr:
            self._detach()

    def _apply(self, q):
        self.data.ctrl[self._act_ids] = q

    def ee_pos(self):
        """Current tool-point world position (for evaluation/logging)."""
        p, _ = self.kin.forward_kinematics()
        return p

    def close(self):
        self.source.close()
