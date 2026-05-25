"""Scripted experts that produce demonstrations in an RL env's action space.

The expert acts in the *same* observation/action space as the Gymnasium env, so
the behavior-cloning policy trained on its demos can be dropped straight into the
env and compared head-to-head with the RL policy.
"""

import numpy as np

from openarm_control.rl.reach_env import OpenArmReachEnv, ACTION_SCALE
from openarm_control.rl.insert_env import OpenArmInsertEnv, ACTION_SCALE as INSERT_SCALE, PEG_HALF


class ReachExpert:
    """Reach expert: solve IK to the target once, then each step command the
    joint-delta action that moves toward that solution (a proportional policy in
    the env's action space). Produces smooth, successful demonstrations."""

    def __init__(self, env):
        self.env = env
        self.q_target = None

    def reset(self):
        env = self.env
        q0 = env.data.qpos[env.kin.qpos_indices].copy()
        q, info = env.kin.inverse_kinematics(env.target, q_init=q0, return_info=True)
        self.q_target = q if (q is not None and info["success"]) else q0

    def act(self, obs):
        q = self.env.data.qpos[self.env.kin.qpos_indices]
        delta = (self.q_target - q) / ACTION_SCALE        # steps of one ACTION_SCALE
        return np.clip(delta, -1.0, 1.0).astype(np.float32)


class InsertExpert:
    """Insertion expert: align the peg over the socket, then descend along a
    **vertical Cartesian path** to seat it. (A joint-space proportional descent arcs
    sideways and drives the peg into the socket wall; a straight vertical descent is
    what threads it in -- the same point-down-descent insight as the classical M6
    insertion.) The classical baseline and the demo source for BC/ACT on insertion."""

    STEPS_PER_WP = 9

    def __init__(self, env):
        self.env = env
        self.path = None
        self._t = 0

    def reset(self):
        from openarm_control.grasp import topdown_orientation
        env = self.env
        tgt = env._socket_target()
        q_start = env.data.qpos[env.kin.qpos_indices].copy()
        start = env.kin.forward_kinematics()[0]              # current tool point (offset, hover)
        R = topdown_orientation(getattr(env, "_yaw", 0.0))   # the start's reachable top-down yaw
        hover_z = tgt[2] + PEG_HALF + 0.08
        # Cartesian waypoints at a FIXED top-down orientation: align (start xy -> socket
        # xy at hover) then descend (socket xy, hover -> seat). IK each, seeded from the
        # previous config -> orientation preserved throughout (the peg stays vertical;
        # a joint-space proportional move instead tilts the gripper and jams the peg).
        pts = [[start[0] + (tgt[0] - start[0]) * s, start[1] + (tgt[1] - start[1]) * s, hover_z]
               for s in np.linspace(0, 1, 5)[1:]]
        pts += [[tgt[0], tgt[1], z] for z in np.linspace(hover_z, tgt[2] + PEG_HALF, 6)[1:]]
        self.path, q_prev = [q_start], q_start
        for p in pts:
            q = env.kin.inverse_kinematics(p, target_mat=R, q_init=q_prev,
                                           restarts=0, rest_weight=0.0)
            q = q if q is not None else q_prev
            self.path.append(q); q_prev = q
        self._t = 0

    def act(self, obs):
        q = self.env.data.qpos[self.env.kin.qpos_indices]
        idx = min(self._t // self.STEPS_PER_WP, len(self.path) - 1)
        self._t += 1
        return np.clip((self.path[idx] - q) / INSERT_SCALE, -1.0, 1.0).astype(np.float32)


def make_env_and_expert(task="reach", **env_kwargs):
    """Return (env, expert) for a named task."""
    env, expert_cls = TASKS[task]
    e = env(**env_kwargs)
    return e, expert_cls(e)


# task name -> (env class, expert class)
TASKS = {
    "reach": (OpenArmReachEnv, ReachExpert),
    "insert": (OpenArmInsertEnv, InsertExpert),
}
