"""Gymnasium environment: the OpenArm inserts a held peg into a socket.

The peg is held in the gripper (welded); each episode the **socket is randomly
repositioned** and the peg starts with a **random lateral offset**, with randomized
**friction** -- a single robot facing many hole positions/offsets, the domain
randomization that makes an insertion policy precise and robust (the user's design:
one robot, many holes, rather than many robots). Round peg -> position insertion (no
yaw alignment). Joint-delta position control with gravity compensation.

Observation (23):  qpos(7), qvel(7), peg_tip(3), socket(3), peg_tip->socket(3)
Action (7):        joint-position deltas in [-1, 1], scaled by ACTION_SCALE
Reward:            -dist(peg_tip, socket) + precision bonus + insertion success
Episode ends:      inserted (within tol, at depth) or after `max_steps`.
"""
import os
import sys

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import PEG_SOCKET_SCENE, RIGHT_ARM, GRASP_LOCAL_OFFSET
from openarm_control.kinematics import OpenArmKinematics, orientation_error
from openarm_control.grasp import topdown_orientation

TABLE_TOP_Z = 0.40
PEG_HALF = 0.045                 # peg half-length (tip is this far below the grasp)
SOCKET_LOW = np.array([0.28, -0.30])     # socket xy sampling box (reachable, right arm)
SOCKET_HIGH = np.array([0.35, -0.17])
ACTION_SCALE = 0.03
HOVER = 0.12                     # peg tip starts this far above the socket
SUCCESS_RADIAL = 0.012           # horizontal tol for "inserted" (within the socket clearance)
SUCCESS_DEPTH = 0.448            # peg-tip z below this = seated inside the socket (walls 0.40-0.45)
CONTROL_SUBSTEPS = 10


class OpenArmInsertEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, max_steps=120, seed=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(PEG_SOCKET_SCENE)
        self.data = mujoco.MjData(self.model)
        self.kin = OpenArmKinematics(self.model, self.data, joint_names=RIGHT_ARM.joints,
                                     site_name=RIGHT_ARM.ee_site, tool_offset=GRASP_LOCAL_OFFSET)
        self.act_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                                 for n in RIGHT_ARM.actuators])
        self.peg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.socket = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "socket")
        self.weld = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right_peg")
        self.key_ready = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        self.peg_qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.peg]]
        self._peg_friction0 = self.model.geom_friction[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")].copy()

        self.max_steps = max_steps
        self.render_mode = render_mode
        self._viewer = None
        self._step = 0
        self.socket_xy = np.zeros(2)
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        high = np.full(23, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _peg_tip(self):
        """World position of the peg's bottom tip."""
        R = self.data.xmat[self.peg].reshape(3, 3)
        return self.data.xpos[self.peg] + R @ np.array([0.0, 0.0, -PEG_HALF])

    def _socket_target(self):
        """Insertion target: the socket centre at seating depth."""
        return np.array([self.socket_xy[0], self.socket_xy[1], TABLE_TOP_Z + 0.005])

    def reachable_ik(self, pos, yaw_hint=None, ori_tol_deg=6.0):
        """IK to ``pos`` with an ACTUALLY top-down gripper at a reachable yaw. Searches
        yaw and -- crucially -- keeps only solutions whose achieved orientation is
        within ``ori_tol_deg`` of top-down (a position-only check accepts tilted
        grippers, which then tilt the welded peg). Returns (q, yaw) or (None, 0)."""
        yaws = [yaw_hint] if yaw_hint is not None else np.linspace(-np.pi, np.pi, 13)
        ori_tol = np.deg2rad(ori_tol_deg)
        best_q, best_err, best_yaw = None, np.inf, 0.0
        for yaw in yaws:
            R = topdown_orientation(yaw)
            q, info = self.kin.inverse_kinematics(pos, target_mat=R, return_info=True,
                                                  seed=0, restarts=2)
            _, achm = self.kin.forward_kinematics(q)
            oerr = float(np.linalg.norm(orientation_error(achm, R)))
            if info["success"] and oerr < ori_tol and info["error"] < best_err:
                best_q, best_err, best_yaw = q, info["error"], float(yaw)
        return best_q, best_yaw

    def _obs(self):
        q = self.data.qpos[self.kin.qpos_indices]
        qd = self.data.qvel[self.kin.dof_indices]
        tip = self._peg_tip()
        tgt = self._socket_target()
        return np.concatenate([q, qd, tip, tgt, tgt - tip]).astype(np.float32)

    # ------------------------------------------------------------------ core
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_ready)
        self.data.eq_active[self.weld] = 0

        # randomize the socket position (domain randomization) + peg friction
        self.socket_xy = self._rng.uniform(SOCKET_LOW, SOCKET_HIGH)
        self.model.body_pos[self.socket] = [self.socket_xy[0], self.socket_xy[1], TABLE_TOP_Z]
        peg_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
        fr = self._peg_friction0.copy()
        fr[0] *= self._rng.uniform(0.6, 1.4)
        self.model.geom_friction[peg_gid] = fr
        # randomize the peg radius (different objects -> different clearance): socket
        # inner apothem ~0.020, so r in [0.006, 0.009] gives ~11-14 mm radial clearance
        # (a forgiving fit a rigid descent can solve; tight-clearance precision is the
        # compliant-control / F2 variant -- a rigid descent jams on the rim).
        self.model.geom_size[peg_gid, 0] = self._rng.uniform(0.006, 0.009)

        # start the arm holding the peg above the socket, with a random lateral offset
        off = self._rng.uniform(-0.03, 0.03, 2)
        start = np.array([self.socket_xy[0] + off[0], self.socket_xy[1] + off[1],
                          TABLE_TOP_Z + PEG_HALF + HOVER])
        q_hold, self._yaw = self.reachable_ik(start)        # reachable top-down yaw
        if q_hold is None:
            q_hold = self.data.qpos[self.kin.qpos_indices].copy()
            self._yaw = 0.0
        self.data.qpos[self.kin.qpos_indices] = q_hold
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # place the peg in the gripper (tool point) and weld it there
        tool = self.kin.forward_kinematics()[0]
        self.data.qpos[self.peg_qadr:self.peg_qadr + 3] = tool + [0, 0, 0]
        self.data.qpos[self.peg_qadr + 3:self.peg_qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self._weld_peg()
        for i, a in enumerate(self.act_ids):
            self.data.ctrl[a] = self.data.qpos[self.kin.qpos_indices[i]]
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self._obs(), {}

    def _weld_peg(self):
        """Weld the peg to the gripper at its current relative pose."""
        ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_ARM.ee_body)
        p1, q1 = self.data.xpos[ee].copy(), self.data.xquat[ee].copy()
        p2, q2 = self.data.xpos[self.peg].copy(), self.data.xquat[self.peg].copy()
        nq1 = np.zeros(4); mujoco.mju_negQuat(nq1, q1)
        relpos = np.zeros(3); mujoco.mju_rotVecQuat(relpos, p2 - p1, nq1)
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, nq1, q2)
        self.model.eq_data[self.weld, 0:3] = 0.0
        self.model.eq_data[self.weld, 3:6] = relpos
        self.model.eq_data[self.weld, 6:10] = relquat
        self.model.eq_data[self.weld, 10] = 1.0
        self.model.eq_solref[self.weld] = [0.001, 1]      # very stiff: peg stays rigid in hand
        self.model.eq_solimp[self.weld] = [0.99, 0.999, 0.001, 0.5, 2]
        self.data.eq_active[self.weld] = 1

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        q = self.data.qpos[self.kin.qpos_indices].copy()
        q_des = np.clip(q + action * ACTION_SCALE, self.kin.jnt_low, self.kin.jnt_high)
        for i, a in enumerate(self.act_ids):
            self.data.ctrl[a] = q_des[i]
        for _ in range(CONTROL_SUBSTEPS):
            self.data.qfrc_applied[self.kin.dof_indices] = self.data.qfrc_bias[self.kin.dof_indices]
            mujoco.mj_step(self.model, self.data)

        tip = self._peg_tip()
        tgt = self._socket_target()
        dist = float(np.linalg.norm(tgt - tip))
        radial = float(np.linalg.norm(tgt[:2] - tip[:2]))
        reward = (-dist + 0.5 * np.exp(-(dist / 0.03) ** 2) - 0.001 * float(np.linalg.norm(action)))
        inserted = radial < SUCCESS_RADIAL and tip[2] < SUCCESS_DEPTH
        if inserted:
            reward += 3.0
        self._step += 1
        terminated = inserted
        truncated = self._step >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return (self._obs(), reward, terminated, truncated,
                {"distance": dist, "radial": radial, "is_success": inserted})

    def render(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
