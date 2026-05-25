"""Real-time resolved-rate Cartesian controller for one OpenArm arm.

Drives the end-effector (or grasp point) to a Cartesian target by integrating
joint-velocity commands from a damped least-squares inverse of the Jacobian.

Key design point: the controller integrates an *internal* desired joint state
``q_des`` rather than re-reading the (gravity-sagged) actual joints every step.
Error feedback is taken from the actual end-effector pose, so ``q_des``
accumulates until the actual pose reaches the target -- this gives integral-like
behaviour and converges to the goal instead of stalling one step behind it.
"""

import mujoco
import numpy as np

from .kinematics import OpenArmKinematics, orientation_error
from .config import RIGHT_ARM, CARTESIAN_POS_GAIN, CARTESIAN_ORI_GAIN


class CartesianController:
    """Resolved-rate Cartesian controller (damped least squares)."""

    def __init__(self, model, data, arm=RIGHT_ARM, kinematics=None,
                 pos_gain=CARTESIAN_POS_GAIN, ori_gain=CARTESIAN_ORI_GAIN,
                 damping=0.05, max_joint_vel=2.0):
        self.model = model
        self.data = data
        self.arm = arm
        self.kin = kinematics if kinematics is not None else OpenArmKinematics(
            model, data, joint_names=arm.joints, site_name=arm.ee_site)

        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in arm.actuators]
        if -1 in self.actuator_ids:
            raise ValueError(f"Actuators not found: {arm.actuators}")
        self.gripper_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, arm.gripper_actuator)

        self.pos_gain = pos_gain
        self.ori_gain = ori_gain
        self.damping = damping
        self.max_joint_vel = max_joint_vel

        self.target_pos = None
        self.target_mat = None
        self.target_gripper = 0.0      # 0 = open, 1 = closed
        self.q_des = None              # internal integrated desired joint state

    def reset(self):
        """Seed the internal desired joint state from the actual configuration."""
        self.q_des = self.data.qpos[self.kin.qpos_indices].copy()

    def set_target(self, pos, mat=None, gripper=None):
        self.target_pos = None if pos is None else np.asarray(pos, dtype=float)
        self.target_mat = None if mat is None else np.asarray(mat, dtype=float).reshape(3, 3)
        if gripper is not None:
            self.target_gripper = float(np.clip(gripper, 0.0, 1.0))

    def _command_gripper(self):
        if self.gripper_act_id != -1:
            self.data.ctrl[self.gripper_act_id] = (
                self.arm.gripper_open
                + self.target_gripper * (self.arm.gripper_closed - self.arm.gripper_open))

    def step(self):
        """One control step. Returns the position error norm (meters).

        Assumes mj_forward / mj_step has run for the current state (so the
        Jacobian's cdof is current) -- true inside a normal sim loop.
        """
        if self.q_des is None:
            self.reset()
        self._command_gripper()

        if self.target_pos is None:
            return 0.0

        cur_pos, cur_R = self.kin.forward_kinematics()
        err_pos = self.target_pos - cur_pos

        if self.target_mat is not None:
            err_rot = orientation_error(cur_R, self.target_mat)
            err = np.concatenate([self.pos_gain * err_pos, self.ori_gain * err_rot])
            J = self.kin._jacobian_current()
        else:
            err = self.pos_gain * err_pos
            J = self.kin._jacobian_current()[:3]

        # Damped least-squares joint velocity.
        n = J.shape[0]
        q_dot = J.T @ np.linalg.solve(J @ J.T + (self.damping ** 2) * np.eye(n), err)

        # Clamp joint speed for smoothness/safety.
        speed = np.linalg.norm(q_dot)
        if speed > self.max_joint_vel:
            q_dot *= self.max_joint_vel / speed

        # Integrate the internal desired state and command it.
        dt = self.model.opt.timestep
        self.q_des = np.clip(self.q_des + q_dot * dt, self.kin.jnt_low, self.kin.jnt_high)
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = self.q_des[i]

        return float(np.linalg.norm(err_pos))

    def is_converged(self, pos_tol=2e-3, ori_tol=np.deg2rad(2.0)):
        """True if the actual pose is within tolerance of the target."""
        if self.target_pos is None:
            return True
        cur_pos, cur_R = self.kin.forward_kinematics()
        if np.linalg.norm(self.target_pos - cur_pos) > pos_tol:
            return False
        if self.target_mat is not None:
            if np.linalg.norm(orientation_error(cur_R, self.target_mat)) > ori_tol:
                return False
        return True
