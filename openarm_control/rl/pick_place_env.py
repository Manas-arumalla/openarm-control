"""Gymnasium environment: the OpenArm right arm learns pick-and-place.

Task: approach a block, grasp it, and carry it to a target spot on the table.
The grasp uses the scene's weld, auto-activated when the policy closes the
gripper with the block between the fingers (a reliable "magnetic" grasp common
in RL manipulation; the unreliable pure-friction grasp would make the reward
signal hopeless). The policy still must position, close at the right moment, and
keep holding until the target.

Observation (31): qpos(7), qvel(7), grasp_pt(3), block(3), target(3),
                  block->grasp(3), target->block(3), gripper_closed(1), grasped(1)
Action (8): 7 joint-position deltas + 1 gripper (>0 close, <0 open), all [-1,1]
Reward: staged — approach the block, +grasp bonus, carry to target, +success.
"""

import os
import sys

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import RL_PICK_SCENE, RIGHT_ARM
from openarm_control.pick_and_place import PickPlaceController

ACTION_SCALE = 0.04
CONTROL_SUBSTEPS = 10
GRASP_DIST = 0.045        # grasp point within this of block center -> can grasp
SUCCESS_TOL = 0.05        # block within 5 cm of target -> success
# Reset randomization ranges (on the table, reachable).
BLOCK_LOW, BLOCK_HIGH = np.array([0.18, -0.32]), np.array([0.26, -0.18])
TARGET_LOW, TARGET_HIGH = np.array([0.22, -0.34]), np.array([0.34, -0.16])
TABLE_Z = 0.43


class OpenArmPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=200, seed=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(RL_PICK_SCENE)
        self.data = mujoco.MjData(self.model)
        self.ppc = PickPlaceController(self.model, self.data, arm=RIGHT_ARM)
        self.king = self.ppc.king                       # grasp-point kinematics
        self.act_ids = np.array(self.ppc.arm_acts)
        self.grip_act = self.ppc.grip_act
        self.block_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        # block free-joint qpos address (resolved via the body's joints)
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == self.block_bid and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.block_qadr = int(self.model.jnt_qposadr[j])
        self.target_mocap = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "place_target")]
        self.key_ready = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")

        self.max_steps = max_steps
        self.render_mode = render_mode
        self._viewer = None
        self._step = 0
        self.grasped = False
        self.target = np.zeros(3)
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, (8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (31,), dtype=np.float32)

    # ------------------------------------------------------------------ core
    def _grasp_pt(self):
        return self.king.forward_kinematics()[0]

    def _block(self):
        return self.data.xpos[self.block_bid].copy()

    def _obs(self):
        q = self.data.qpos[self.king.qpos_indices]
        qd = self.data.qvel[self.king.dof_indices]
        gp = self._grasp_pt()
        blk = self._block()
        grip_closed = 1.0 if self.data.ctrl[self.grip_act] == self.ppc._grip_ctrl(True) else 0.0
        return np.concatenate([q, qd, gp, blk, self.target, blk - gp, self.target - blk,
                               [grip_closed, float(self.grasped)]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_ready)
        self.ppc.detach("block")
        self.grasped = False
        bxy = self._rng.uniform(BLOCK_LOW, BLOCK_HIGH)
        self.data.qpos[self.block_qadr:self.block_qadr + 3] = [bxy[0], bxy[1], TABLE_Z]
        self.data.qpos[self.block_qadr + 3:self.block_qadr + 7] = [1, 0, 0, 0]
        self.data.qpos[self.king.qpos_indices] += self._rng.uniform(-0.03, 0.03, 7)
        self.data.qvel[:] = 0.0
        for i, a in enumerate(self.act_ids):
            self.data.ctrl[a] = self.data.qpos[self.king.qpos_indices[i]]
        self.data.ctrl[self.grip_act] = self.ppc._grip_ctrl(False)   # open
        txy = self._rng.uniform(TARGET_LOW, TARGET_HIGH)
        self.target = np.array([txy[0], txy[1], TABLE_Z])
        self.data.mocap_pos[self.target_mocap] = self.target
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        q = self.data.qpos[self.king.qpos_indices].copy()
        q_des = np.clip(q + action[:7] * ACTION_SCALE, self.king.jnt_low, self.king.jnt_high)
        for i, a in enumerate(self.act_ids):
            self.data.ctrl[a] = q_des[i]
        close = action[7] > 0.0
        self.data.ctrl[self.grip_act] = self.ppc._grip_ctrl(close)

        # Auto-weld grasp: close + block between fingers -> attach; open -> release.
        d_gb = float(np.linalg.norm(self._grasp_pt() - self._block()))
        just_grasped = False
        if close and not self.grasped and d_gb < GRASP_DIST:
            self.ppc.attach("block"); self.grasped = True; just_grasped = True
        elif not close and self.grasped:
            self.ppc.detach("block"); self.grasped = False

        for _ in range(CONTROL_SUBSTEPS):
            self.data.qfrc_applied[self.king.dof_indices] = self.data.qfrc_bias[self.king.dof_indices]
            mujoco.mj_step(self.model, self.data)

        blk = self._block()
        d_gb = float(np.linalg.norm(self._grasp_pt() - blk))
        d_bt = float(np.linalg.norm(self.target - blk))
        if self.grasped:
            reward = 1.0 - d_bt + 0.5 * np.exp(-(d_bt / 0.05) ** 2)
        else:
            reward = -d_gb + 0.3 * np.exp(-(d_gb / 0.05) ** 2)
        if just_grasped:
            reward += 2.0
        reward -= 0.001 * float(np.linalg.norm(action))
        success = self.grasped and d_bt < SUCCESS_TOL
        if success:
            reward += 5.0

        self._step += 1
        if self.render_mode == "human":
            self.render()
        return (self._obs(), reward, bool(success), self._step >= self.max_steps,
                {"d_block_target": d_bt, "grasped": self.grasped, "is_success": success})

    def render(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
