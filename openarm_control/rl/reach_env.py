"""Gymnasium environment: the OpenArm right arm learns to reach a target.

Task: drive the end-effector to a randomly placed target in the reachable
workspace. Joint-delta position control with gravity compensation (so the policy
learns *where* to move rather than how to fight gravity). Dense distance reward.

Observation (23):  qpos(7), qvel(7), ee_pos(3), target(3), ee->target(3)
Action (7):        joint-position deltas in [-1, 1], scaled by ACTION_SCALE
Reward:            -distance  - 0.01*|action|  (+ success bonus when within TOL)
Episode ends:      success (within TOL) or after `max_steps`.
"""

import os
import sys

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import REACH_SCENE, RIGHT_ARM
from openarm_control.kinematics import OpenArmKinematics

# Target sampling box (inside the right arm's reachable workspace, free space).
TARGET_LOW = np.array([0.15, -0.45, 0.40])
TARGET_HIGH = np.array([0.38, -0.08, 0.72])
ACTION_SCALE = 0.04       # rad per control step per joint
SUCCESS_TOL = 0.03        # 3 cm (tightened from 5 cm)
CONTROL_SUBSTEPS = 10     # sim steps per env step (timestep 0.002 -> 50 Hz control)
PRECISION_SIGMA = 0.03    # width of the near-target precision bonus (m)


class OpenArmReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=120, seed=None, success_tol=SUCCESS_TOL):
        super().__init__()
        self.success_tol = success_tol
        self.model = mujoco.MjModel.from_xml_path(REACH_SCENE)
        self.data = mujoco.MjData(self.model)
        self.kin = OpenArmKinematics(self.model, self.data,
                                     joint_names=RIGHT_ARM.joints, site_name=RIGHT_ARM.ee_site)
        self.act_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                                 for n in RIGHT_ARM.actuators])
        self.mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "reach_target")]
        self.key_ready = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")

        self.max_steps = max_steps
        self.render_mode = render_mode
        self._viewer = None
        self._step = 0
        self.target = np.zeros(3)
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        high = np.full(23, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # ------------------------------------------------------------------ core
    def _ee(self):
        return self.kin.forward_kinematics()[0]

    def _obs(self):
        q = self.data.qpos[self.kin.qpos_indices]
        qd = self.data.qvel[self.kin.dof_indices]
        ee = self._ee()
        return np.concatenate([q, qd, ee, self.target, self.target - ee]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_ready)
        # small joint noise for robustness
        self.data.qpos[self.kin.qpos_indices] += self._rng.uniform(-0.05, 0.05, 7)
        self.data.qvel[:] = 0.0
        for i, a in enumerate(self.act_ids):
            self.data.ctrl[a] = self.data.qpos[self.kin.qpos_indices[i]]
        self.target = self._rng.uniform(TARGET_LOW, TARGET_HIGH)
        self.data.mocap_pos[self.mocap_id] = self.target
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        q = self.data.qpos[self.kin.qpos_indices].copy()
        q_des = np.clip(q + action * ACTION_SCALE, self.kin.jnt_low, self.kin.jnt_high)
        for i, a in enumerate(self.act_ids):
            self.data.ctrl[a] = q_des[i]
        for _ in range(CONTROL_SUBSTEPS):
            # gravity compensation on the arm DOFs so position control tracks
            self.data.qfrc_applied[self.kin.dof_indices] = self.data.qfrc_bias[self.kin.dof_indices]
            mujoco.mj_step(self.model, self.data)

        ee = self._ee()
        dist = float(np.linalg.norm(self.target - ee))
        # Dense distance + a sharp near-target precision bonus (drives sub-cm
        # accuracy where the bare -dist gradient is weak); tiny action penalty.
        reward = (-dist
                  + 0.5 * np.exp(-(dist / PRECISION_SIGMA) ** 2)
                  - 0.001 * float(np.linalg.norm(action)))
        success = dist < self.success_tol
        if success:
            reward += 2.0
        self._step += 1
        terminated = success
        truncated = self._step >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, {"distance": dist, "is_success": success}

    # ---------------------------------------------------------------- render
    def render(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
