"""Collision checking for joint-space motion planning.

Classifies geoms into the arm and the environment-to-avoid (table, bins,
obstacle) once, then a config is "in collision" if any contact pairs an arm
geom with an avoid geom. State is saved/restored so checks don't disturb the
simulation. Adjacent-link self-contacts are ignored (MuJoCo already excludes
parent/child pairs); an explicit allow-list covers the gripper internals.
"""

import mujoco
import numpy as np

# Geom name prefixes that the arm must not hit while moving.
_AVOID_PREFIXES = ("table", "bin_", "obstacle")


class CollisionChecker:
    def __init__(self, model, data, kinematics, arm_name="right",
                 avoid_prefixes=_AVOID_PREFIXES, avoid_other_arm=False):
        self.model = model
        self.data = data
        self.kin = kinematics
        other = "left" if arm_name == "right" else "right"

        # Arm geoms = geoms on any body of this arm; environment geoms to avoid.
        self.arm_geoms = set()
        self.avoid_geoms = set()
        self.other_arm_geoms = set()
        for g in range(model.ngeom):
            body = model.geom_bodyid[g]
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if f"openarm_{arm_name}" in bname:
                self.arm_geoms.add(g)
            elif f"openarm_{other}" in bname:
                self.other_arm_geoms.add(g)
            elif gname.startswith(avoid_prefixes):
                self.avoid_geoms.add(g)
        # Optionally treat the other arm (at its current pose) as an obstacle.
        if avoid_other_arm:
            self.avoid_geoms |= self.other_arm_geoms

        # Carried-object state (set during transport): the held object's geoms are
        # treated as part of the arm (so they must avoid the environment too),
        # while its contact with the gripper -- the grasp -- is ignored. The object
        # is rigidly re-posed with the gripper at each checked config so its full
        # swept volume is tested.
        self.carried_geoms = set()
        self.extra_avoid_geoms = set()
        self._carried_qadr = None
        self._rel_pos = self._rel_quat = None
        self._ee_bid = int(kinematics.ee_body_id)

    def _geoms_of(self, body):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            return set()
        return {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid}

    def _free_qadr(self, body):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        for j in range(self.model.njnt):
            if (self.model.jnt_bodyid[j] == bid
                    and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                return int(self.model.jnt_qposadr[j])
        return None

    def set_carried(self, body=None, also_avoid=()):
        """Treat ``body`` as held: its geoms must avoid the environment (and the
        ``also_avoid`` bodies, e.g. the other objects), but not the gripper. The
        object is rigidly carried with the gripper at each checked config, so a
        transport that would knock it into something is reported as a collision --
        like real robotics. Call with ``body=None`` to clear."""
        self.extra_avoid_geoms = set()
        if not body:
            self.carried_geoms = set()
            self._carried_qadr = self._rel_pos = self._rel_quat = None
            return
        self.carried_geoms = self._geoms_of(body)
        self._carried_qadr = self._free_qadr(body)
        for b in also_avoid:
            self.extra_avoid_geoms |= self._geoms_of(b)
        # Object pose relative to the gripper at the current (grasped) config.
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        p_ee, q_ee = self.data.xpos[self._ee_bid].copy(), self.data.xquat[self._ee_bid].copy()
        p_o, q_o = self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
        nq = np.zeros(4); mujoco.mju_negQuat(nq, q_ee)
        self._rel_pos = np.zeros(3); mujoco.mju_rotVecQuat(self._rel_pos, p_o - p_ee, nq)
        self._rel_quat = np.zeros(4); mujoco.mju_mulQuat(self._rel_quat, nq, q_o)

    def in_collision(self, q, ignore_bodies=()):
        """True if config q collides the arm (or the carried object) with the
        environment. ``ignore_bodies`` names bodies to skip."""
        ignore_geoms = set()
        for name in ignore_bodies:
            ignore_geoms |= self._geoms_of(name)

        qpos_save = self.data.qpos.copy()
        self.data.qpos[self.kin.qpos_indices] = q
        mujoco.mj_kinematics(self.model, self.data)
        # Rigidly carry the held object with the gripper at this config.
        if self._carried_qadr is not None and self._rel_pos is not None:
            p_ee, q_ee = self.data.xpos[self._ee_bid], self.data.xquat[self._ee_bid]
            p_o = np.zeros(3); mujoco.mju_rotVecQuat(p_o, self._rel_pos, q_ee)
            p_o += p_ee
            q_o = np.zeros(4); mujoco.mju_mulQuat(q_o, q_ee, self._rel_quat)
            a = self._carried_qadr
            self.data.qpos[a:a + 3] = p_o
            self.data.qpos[a + 3:a + 7] = q_o
            mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

        avoid = self.avoid_geoms | self.extra_avoid_geoms
        hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in ignore_geoms or g2 in ignore_geoms:
                continue
            c1, c2 = g1 in self.carried_geoms, g2 in self.carried_geoms
            a1, a2 = g1 in self.arm_geoms, g2 in self.arm_geoms
            if (c1 and a2) or (c2 and a1) or (c1 and c2):
                continue                             # the grasp itself -- not a crash
            arm1, arm2 = a1 or c1, a2 or c2          # carried counts as part of the arm
            if (arm1 and g2 in avoid) or (arm2 and g1 in avoid):
                hit = True
                break

        self.data.qpos[:] = qpos_save
        mujoco.mj_kinematics(self.model, self.data)
        return hit

    def edge_clear(self, q1, q2, max_step=0.05, ignore_bodies=()):
        """Collision-free straight line between q1 and q2 (joint space)."""
        q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
        n = max(2, int(np.ceil(np.linalg.norm(q2 - q1) / max_step)))
        for i in range(n + 1):
            q = q1 + (q2 - q1) * (i / n)
            if self.in_collision(q, ignore_bodies):
                return False
        return True
