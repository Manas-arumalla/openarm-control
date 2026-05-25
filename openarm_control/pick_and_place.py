"""Autonomous pick-and-place for the OpenArm right arm.

Motion is expressed as a list of *segments*, each a polyline of joint configs
plus a gripper state and a duration. The executor plays each segment with a
quintic (zero-jerk) time scaling, interpolating along the polyline.

Vertical descend / lift / place moves are generated as **Cartesian** polylines
(many configs solved with continuous, on-branch IK) so the end-effector travels
in a straight line -- a joint-space interpolation between just the endpoints
would arc sideways and drag the grasped block. Gravity compensation keeps the
position actuators tracking accurately so grasps stay centered.
"""

import time

import mujoco
import numpy as np

from .config import RIGHT_ARM
from .grasp import GraspSolver, topdown_orientation
from .kinematics import orientation_error
from .trajectory import quintic_polynomial

TABLE_TOP_Z = 0.40
BLOCK_HALF = 0.03          # 60 mm tall blocks; center rests at table_top + 0.03
GRASP_DEPTH = 0.04         # grasp point above table top: grips upper block, tips clear table


class PickPlaceController:
    """Plans and executes a pick-and-place between two tabletop locations."""

    def __init__(self, model, data, arm=RIGHT_ARM, hover=0.12):
        self.model = model
        self.data = data
        self.arm = arm
        self.hover = hover
        self.gs = GraspSolver(model, data, arm=arm)
        self.king = self.gs.king                      # grasp-point kinematics
        self.arm_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                         for n in arm.actuators]
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                          arm.gripper_actuator)
        self.ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_body)

    # -- grasp weld --------------------------------------------------------
    def _weld_id(self, block):
        """Equality id of this arm's grasp weld for a block (e.g. grasp_right_red)."""
        name = self.arm.weld(block.split("_")[-1])
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def attach(self, block):
        """Weld a block rigidly to the gripper at the current relative pose.

        The lateral (in-gripper x,y) offset is zeroed so the object is held
        **centred on the gripper's approach axis** -- a top-down grasp targets the
        object's xy, so it should hang directly below the tool point. This corrects
        any small sideways shove from the closing fingers (or 1e-5 plan differences
        from BLAS/thread state), making the grasp pose exact and the carried
        placement robust; the grasp depth (along the approach axis) and the object's
        orientation are preserved, so clean grasps (already centred) are unchanged."""
        eid = self._weld_id(block)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, block)
        if eid < 0 or bid < 0:
            return
        p1, q1 = self.data.xpos[self.ee_body].copy(), self.data.xquat[self.ee_body].copy()
        p2, q2 = self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
        nq1 = np.zeros(4); mujoco.mju_negQuat(nq1, q1)
        relpos = np.zeros(3); mujoco.mju_rotVecQuat(relpos, p2 - p1, nq1)
        relpos[0] = 0.0                 # centre laterally on the approach axis
        relpos[1] = 0.0                 # (keep relpos[2]: the grasp depth)
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, nq1, q2)
        self.model.eq_data[eid, 0:3] = 0.0
        self.model.eq_data[eid, 3:6] = relpos
        self.model.eq_data[eid, 6:10] = relquat
        self.model.eq_data[eid, 10] = 1.0
        self.data.eq_active[eid] = 1

    def detach(self, block):
        eid = self._weld_id(block)
        if eid >= 0:
            self.data.eq_active[eid] = 0

    # -- IK helpers --------------------------------------------------------
    def _ik_line(self, q0, p0, p1, R=None, steps=20):
        """Polyline of configs tracking the straight line p0->p1 from q0.

        Each small step seeds from the previous solution with no nullspace
        drift / restarts, so the configs stay on one IK branch and the EE
        follows the Cartesian line. If ``R`` is None the orientation is left
        free (position-only) -- used while carrying, where the gripped object's
        orientation is irrelevant and an orientation constraint can fold/jump.
        Returns an (steps+1, 7) array [q0..q1].
        """
        q = np.asarray(q0, dtype=float).copy()
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        out = [q.copy()]
        for s in range(1, steps + 1):
            p = p0 + (p1 - p0) * (s / steps)
            qn = self.king.inverse_kinematics(p, target_mat=R, q_init=q,
                                              restarts=0, rest_weight=0.0)
            if qn is not None:
                q = qn
            out.append(q.copy())
        return np.array(out)

    def _ik_line_down(self, q0, p0, p1, steps=24, iters=80):
        """Track p0->p1 keeping the gripper pointing straight down, yaw FREE.

        A 5-DOF task (3 position + 2 tilt; yaw unconstrained) so the gripper
        approach axis stays vertical -- the grasped block hangs directly below
        the grasp point and lands on target -- while the free yaw keeps the
        whole line reachable and the configs continuous (seeded from previous).
        """
        king, model, data = self.king, self.model, self.data
        q = np.clip(np.asarray(q0, float).copy(), king.jnt_low, king.jnt_high)
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        z_up = np.array([0.0, 0.0, 1.0])
        qpos_save = data.qpos.copy()
        out = [q.copy()]
        for s in range(1, steps + 1):
            target = p0 + (p1 - p0) * (s / steps)
            for _ in range(iters):
                data.qpos[king.qpos_indices] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                pt, Rt = king._tool_pose()
                perr = target - pt
                zerr = np.cross(Rt[:, 2], z_up)        # align local z with world up
                if np.linalg.norm(perr) < 1e-4 and np.linalg.norm(zerr) < 1e-3:
                    break
                J = king._jacobian_current()           # 6x7
                err = np.concatenate([perr, zerr])
                dq = J.T @ np.linalg.solve(J @ J.T + 2.5e-3 * np.eye(6), err)
                n = np.linalg.norm(dq)
                if n > 0.2:
                    dq *= 0.2 / n
                q = np.clip(q + dq, king.jnt_low, king.jnt_high)
            out.append(q.copy())
        data.qpos[:] = qpos_save
        mujoco.mj_kinematics(model, data)
        return np.array(out)

    def _ik_line_oriented(self, q0, p0, p1, R, steps=28, iters=140):
        """Track p0->p1 holding the **full** orientation ``R`` (6-DOF), seeded
        continuously from the previous config so the path can't branch-jump.

        Used for a fixed-yaw, point-down insertion descent (a rectangular peg must
        keep the hole's orientation as it goes in) -- the position-only / free-yaw
        lines twist or jump under a full orientation constraint.
        """
        king, model, data = self.king, self.model, self.data
        q = np.clip(np.asarray(q0, float).copy(), king.jnt_low, king.jnt_high)
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        qpos_save = data.qpos.copy()
        out = [q.copy()]
        for s in range(1, steps + 1):
            target = p0 + (p1 - p0) * (s / steps)
            for _ in range(iters):
                data.qpos[king.qpos_indices] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                pt, Rt = king._tool_pose()
                perr = target - pt
                oerr = orientation_error(Rt, R)
                if np.linalg.norm(perr) < 1e-4 and np.linalg.norm(oerr) < 2e-3:
                    break
                J = king._jacobian_current()           # 6x7
                err = np.concatenate([perr, oerr])
                dq = J.T @ np.linalg.solve(J @ J.T + 2.5e-3 * np.eye(6), err)
                n = np.linalg.norm(dq)
                if n > 0.2:
                    dq *= 0.2 / n
                q = np.clip(q + dq, king.jnt_low, king.jnt_high)
            out.append(q.copy())
        data.qpos[:] = qpos_save
        mujoco.mj_kinematics(model, data)
        return np.array(out)

    @staticmethod
    def _max_jump(path):
        """Largest per-joint change between adjacent configs (radians)."""
        return float(np.max(np.abs(np.diff(path, axis=0)))) if len(path) > 1 else 0.0

    # -- planning ----------------------------------------------------------
    def plan(self, pick_xy, place_xy, grasp_z=None, place_z=None):
        """Build segments: list of (knots[K,7], gripper_closed, duration).

        The whole carry (lift -> transport -> place descend) is one continuous IK
        chain at a SINGLE fixed wrist orientation (the pick yaw), each segment
        seeded from the previous. This avoids both IK branch jumps and the
        yaw-interpolation folds that otherwise fling the grasped block.
        """
        grasp_z = TABLE_TOP_Z + GRASP_DEPTH if grasp_z is None else grasp_z
        place_z = TABLE_TOP_Z + GRASP_DEPTH + 0.04 if place_z is None else place_z
        pick = np.array([pick_xy[0], pick_xy[1], grasp_z])
        place = np.array([place_xy[0], place_xy[1], place_z])
        pick_hover = pick + np.array([0, 0, self.hover])
        place_hover = place + np.array([0, 0, self.hover])

        gq, info = self.gs.solve(pick, return_info=True)
        if not info["success"]:
            raise ValueError("Pick pose not reachable as a top-down grasp.")
        R = topdown_orientation(info["yaw"])

        # Lift: fixed yaw (vertical, no wrist twist that would shear the block).
        # Transport + descend: point-down with FREE yaw so the gripper stays
        # vertical (block hangs below, lands on target) while yaw adapts for
        # reachability. Each segment is seeded from the previous for continuity.
        # Lift fixed-yaw (clean vertical); transport + descend point-down with
        # free yaw (gripper stays vertical so the welded block hangs straight
        # below and lands on target). All seeded continuously.
        lift = self._ik_line(gq, pick, pick_hover, R)
        transport = self._ik_line_down(lift[-1], pick_hover, place_hover)
        descend = self._ik_line_down(transport[-1], place_hover, place)

        carry = np.vstack([lift, transport, descend])
        if self._max_jump(carry) > np.deg2rad(15.0):
            raise ValueError("Carry path discontinuous (IK branch jump).")

        OPEN, CLOSE = False, True
        return [
            (np.array([lift[-1]]),        OPEN,  1.5),   # 0 move above pick (free space)
            (lift[::-1],                  OPEN,  1.5),   # 1 descend hover->grasp
            (np.array([gq] * 2),          CLOSE, 2.0),   # 2 close + grip
            (lift,                        CLOSE, 2.5),   # 3 lift grasp->hover
            (transport,                   CLOSE, 3.0),   # 4 transport hover->hover
            (descend,                     CLOSE, 1.5),   # 5 descend into bin
            (np.array([descend[-1]] * 2), OPEN,  1.0),   # 6 release
            (descend[::-1],               OPEN,  1.5),   # 7 retreat to hover
        ]

    def plan_pick(self, pick_xy, grasp_z=None):
        """Pick + lift segments only: grasp the object and hold it above the table
        (segments 0-3 of ``plan``). Used for a 'grab and hold' command where the place
        comes in a later turn. Raises ValueError if the grasp/lift is unreachable."""
        grasp_z = TABLE_TOP_Z + GRASP_DEPTH if grasp_z is None else grasp_z
        pick = np.array([pick_xy[0], pick_xy[1], grasp_z])
        pick_hover = pick + np.array([0, 0, self.hover])
        gq, info = self.gs.solve(pick, return_info=True)
        if not info["success"]:
            raise ValueError("Pick pose not reachable as a top-down grasp.")
        R = topdown_orientation(info["yaw"])
        lift = self._ik_line(gq, pick, pick_hover, R)
        if self._max_jump(lift) > np.deg2rad(15.0):
            raise ValueError("Lift discontinuous (IK branch jump).")
        OPEN, CLOSE = False, True
        return [
            (np.array([lift[-1]]), OPEN,  1.5),    # move above the object
            (lift[::-1],           OPEN,  1.5),    # descend onto it
            (np.array([gq] * 2),   CLOSE, 2.0),    # close + grip (weld)
            (lift,                 CLOSE, 2.0),    # lift and hold
        ]

    def plan_place_held(self, place_xy, place_z=None):
        """Place an already-held object: transport from the CURRENT (held) config to
        above ``place_xy``, descend, release, retreat. Used after ``plan_pick``.
        Raises ValueError if the carry to the destination is discontinuous."""
        place_z = TABLE_TOP_Z + GRASP_DEPTH + 0.04 if place_z is None else place_z
        q_now = self.data.qpos[self.king.qpos_indices].copy()
        p_now, _ = self.king.forward_kinematics()            # current tool point
        place = np.array([place_xy[0], place_xy[1], place_z])
        place_hover = place + np.array([0, 0, self.hover])
        transport = self._ik_line_down(q_now, p_now, place_hover)
        descend = self._ik_line_down(transport[-1], place_hover, place)
        carry = np.vstack([transport, descend])
        if self._max_jump(carry) > np.deg2rad(15.0):
            raise ValueError("Held-carry path discontinuous (IK branch jump).")
        OPEN, CLOSE = False, True
        return [
            (transport,                   CLOSE, 3.0),    # carry held object above target
            (descend,                     CLOSE, 1.5),    # lower
            (np.array([descend[-1]] * 2), OPEN,  1.0),    # release
            (descend[::-1],               OPEN,  1.5),    # retreat
        ]

    # -- execution ---------------------------------------------------------
    def _apply_gravity_comp(self):
        self.data.qfrc_applied[self.king.dof_indices] = \
            self.data.qfrc_bias[self.king.dof_indices]

    def _grip_ctrl(self, closed):
        return self.arm.gripper_closed if closed else self.arm.gripper_open

    @staticmethod
    def _sample(knots, s):
        """Config at normalized arc position s in [0,1] along the polyline."""
        K = len(knots)
        if K == 1:
            return knots[0]
        x = np.clip(s, 0.0, 1.0) * (K - 1)
        lo = int(np.floor(x))
        if lo >= K - 1:
            return knots[-1]
        frac = x - lo
        return knots[lo] * (1.0 - frac) + knots[lo + 1] * frac

    def execute(self, segments, block=None, viewer=None, dt_realtime=False,
                gravity_comp=True):
        """Run the segments in the live simulation. Returns True on completion.

        If ``block`` is given, its grasp weld is activated at the START of the
        first close segment -- the gripper is already positioned at the grasp
        config, so the object is captured at its centred resting pose and the
        fingers then shut around it. (Welding only *after* the close lets the
        closing fingers shove a free object off-centre first -- the captured pose
        is then skewed, so a carried object can miss its target; capturing at the
        start makes the grasp exact and robust to contact/FP perturbations.) The
        weld is deactivated when the gripper next opens (release).
        """
        q_start = self.data.qpos[self.king.qpos_indices].copy()
        dt = self.model.opt.timestep
        prev_last = q_start
        welded = False
        for knots, closed, duration in segments:
            # Release: drop the weld as the gripper opens.
            if not closed and welded and block is not None:
                self.detach(block)
                welded = False
            # Grasp: weld at the START of the first close (gripper positioned at the
            # grasp, object still centred -- before the fingers can shove it).
            if closed and not welded and block is not None:
                self.attach(block)
                welded = True
            knots = np.vstack([prev_last, knots]) if not np.allclose(knots[0], prev_last) else knots
            n = max(1, int(duration / dt))
            for k in range(n):
                t0 = time.time()
                s, _, _ = quintic_polynomial(k * dt, 0.0, duration, 0.0, 1.0)
                q_cmd = self._sample(knots, s)
                for i, a in enumerate(self.arm_acts):
                    self.data.ctrl[a] = q_cmd[i]
                if self.grip_act != -1:
                    self.data.ctrl[self.grip_act] = self._grip_ctrl(closed)
                if gravity_comp:
                    self._apply_gravity_comp()
                mujoco.mj_step(self.model, self.data)
                if viewer is not None:
                    if not viewer.is_running():
                        return False
                    viewer.sync()
                    if dt_realtime:
                        time.sleep(max(0, dt - (time.time() - t0)))
            prev_last = knots[-1]
        return True
