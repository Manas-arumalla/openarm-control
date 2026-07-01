"""Gymnasium environment: SAC learns to balance a rolling ball on a plate.

Reference for OpenArm-Bench: a learned SAC policy pitted against the classical
PD / LQR / MPC controllers on the same physics + hold pose. The environment
wraps the exact ``BallBalancer`` hold, tilt, and manual-pin machinery used by
those classical controllers -- the only thing that changes is who chooses the
commanded (roll, pitch) each step.

Observation (6):  [x, y, vx, vy, x - tx, y - ty]  -- ball state + error to target
Action (2):       [roll, pitch] in [-1, 1], scaled by ``MAX_TILT`` (~20 deg)
Reward:           -distance - 0.05*speed - 0.001*|action| + precision bonus
                  minus a one-shot -5 penalty if the ball rolls off the plate,
                  plus a +2 success bonus at end-of-episode if within tolerance.
Episode ends:     ball off plate (terminated), or after ``max_steps`` (truncated).
"""

import os
import sys

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import BALANCE_SCENE
from openarm_control.balance import BallBalancer

# 5 sim substeps at 500 Hz sim -> 100 Hz policy control rate (matches the
# tempo the classical controllers effectively hit after arm-servo lag).
CONTROL_SUBSTEPS = 5
# Plate half-width is 7.5 cm; treat ball outside a 7 cm disk as "rolled off"
# (leaves a small margin before the ball actually leaves the plate surface).
PLATE_RADIUS_OFF = 0.070
# Success band: at end of episode, within 2 cm of target and moving < 5 cm/s.
SUCCESS_TOL_M = 0.020
SUCCESS_TOL_V = 0.05
PRECISION_SIGMA = 0.010  # 1 cm precision bonus width


class OpenArmBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode=None, max_steps=300, seed=None,
                 control_substeps=CONTROL_SUBSTEPS, random_target=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(BALANCE_SCENE)
        self.data = mujoco.MjData(self.model)
        self.key_ready = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_ready)
        mujoco.mj_forward(self.model, self.data)

        # Shared hold + tilt + pin scaffolding (identical to what PD/LQR/MPC use).
        self._bal = BallBalancer(self.model, self.data)
        self._bal.setup_hold()

        self.max_steps = int(max_steps)
        self.control_substeps = int(control_substeps)
        self.random_target = bool(random_target)
        self.render_mode = render_mode
        self._viewer = None
        self._step = 0
        self.target = np.zeros(2)
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        high = np.full(6, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # ------------------------------------------------------------------ core
    def _obs(self):
        (x, y), (vx, vy) = self._bal.ball_state()
        tx, ty = float(self.target[0]), float(self.target[1])
        return np.array([x, y, vx, vy, x - tx, y - ty], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Bring the arm back to the ready keyframe and re-run the hold setup so
        # the plate is horizontal at the achieved gripper pose. Cheap: no IK,
        # just forward kinematics + a manual pin.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_ready)
        mujoco.mj_forward(self.model, self.data)
        self._bal.setup_hold()
        # Random starting ball offset in a disk (5 mm .. 30 mm).
        r  = float(self._rng.uniform(0.005, 0.030))
        th = float(self._rng.uniform(0.0, 2 * np.pi))
        offset = (r * np.cos(th), r * np.sin(th))
        self._bal.reset(ball_offset_xy=offset, settle_steps=200)
        # Static origin target unless random_target is on (curriculum hook).
        if self.random_target:
            r  = float(self._rng.uniform(0.0, 0.020))
            th = float(self._rng.uniform(0.0, 2 * np.pi))
            self.target = np.array([r * np.cos(th), r * np.sin(th)], dtype=float)
        else:
            self.target = np.zeros(2)
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        # Scale [-1, 1] action to actual tilt angles (~20 deg cap).
        roll_cmd  = float(action[0]) * self._bal.MAX_TILT
        pitch_cmd = float(action[1]) * self._bal.MAX_TILT
        # Hold the same commanded tilt across ``control_substeps`` sim steps
        # (zero-order hold, matches classical controllers' effective bandwidth).
        for _ in range(self.control_substeps):
            self._bal._apply_tilt_and_step(roll_cmd, pitch_cmd)

        (x, y), (vx, vy) = self._bal.ball_state()
        tx, ty = float(self.target[0]), float(self.target[1])
        err   = float(np.hypot(x - tx, y - ty))
        speed = float(np.hypot(vx, vy))
        # "Rolled off" test uses distance from the PLATE centre (0,0), not the
        # target -- the plate itself doesn't move with the target.
        off = float(np.hypot(x, y)) > PLATE_RADIUS_OFF

        reward = (- err
                  - 0.05 * speed
                  - 0.001 * float(np.linalg.norm(action))
                  + 0.5 * float(np.exp(-(err / PRECISION_SIGMA) ** 2)))
        if off:
            reward -= 5.0

        self._step += 1
        terminated = off
        truncated  = self._step >= self.max_steps
        success = (not off) and (err < SUCCESS_TOL_M) and (speed < SUCCESS_TOL_V)
        if (terminated or truncated) and success:
            reward += 2.0

        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, \
            {"distance": err, "is_success": success}

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
