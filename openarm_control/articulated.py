"""Articulated-object manipulation for the OpenArm (extension phase S3).

Operate the authored articulated fixtures on ``articulated_scene.xml``:
  - open a DRAWER (prismatic slide) -- a straight pull,
  - open a DOOR   (revolute hinge)  -- an arc with a matching wrist rotation,
  - turn a VALVE  (revolute hinge)  -- an arc with a matching wrist rotation.

Single-arm, weld-assisted: the arm grasps the handle, the handle is welded to the
gripper, then the arm moves the handle along the joint's allowed motion (the welded
fixture follows). For a revolute joint the handle traces an arc about the joint axis
*and* the gripper's wrist rotates by the joint angle, so the gripper genuinely
swings/turns the part. The non-working arm is held parked. (F2 admittance can be
layered on for force-guarded operation; the geometric motion is the foundation.)
"""
import time

import numpy as np
import mujoco

from .config import RIGHT_ARM, LEFT_ARM
from .grasp import topdown_orientation, front_orientation
from .pick_and_place import PickPlaceController
from .trajectory import quintic_polynomial

# Fixed seed for IK random restarts inside the skills: restart sampling is
# otherwise unseeded, so a skill could land on a different arm branch every
# run (measured: the frontal drawer pull varied 4-92 mm run to run).
IK_BRANCH_SEED = 1


class ArticulatedController:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.right = PickPlaceController(model, data, arm=RIGHT_ARM)
        self.left = PickPlaceController(model, data, arm=LEFT_ARM)

    # -- helpers ----------------------------------------------------------
    def _geom_pos(self, name):
        return self.data.geom_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)].copy()

    def _body_xy(self, name):
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)][:2].copy()

    def _park(self, arm):
        """Move the non-working arm to the neutral (zeros) pose, well clear of the
        working arm (a raised park sits in the working arm's swing path)."""
        self.data.qpos[arm.king.qpos_indices] = 0.0
        self.data.qvel[arm.king.dof_indices] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _weld(self, ppc, fixture):
        """Non-centred weld of the fixture body to the gripper at the actual pose
        (the handle is grasped off the body's origin, so centring would be wrong)."""
        eid = ppc._weld_id(fixture)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, fixture)
        if eid < 0 or bid < 0:
            return
        p1, q1 = self.data.xpos[ppc.ee_body].copy(), self.data.xquat[ppc.ee_body].copy()
        p2, q2 = self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
        nq1 = np.zeros(4); mujoco.mju_negQuat(nq1, q1)
        relpos = np.zeros(3); mujoco.mju_rotVecQuat(relpos, p2 - p1, nq1)
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, nq1, q2)
        self.model.eq_data[eid, 0:3] = 0.0
        self.model.eq_data[eid, 3:6] = relpos
        self.model.eq_data[eid, 6:10] = relquat
        self.model.eq_data[eid, 10] = 1.0
        self.data.eq_active[eid] = 1

    def _drive(self, active, other, path, dur, grip=None, viewer=None, dt_realtime=False):
        """Quintic-time the active arm along the config polyline ``path`` while the
        other arm holds its pose; both arms are gravity-compensated."""
        model, data = self.model, self.data
        dt = model.opt.timestep
        path = np.atleast_2d(np.asarray(path, float))
        prev = data.qpos[active.king.qpos_indices].copy()
        kk = np.vstack([prev, path]) if not np.allclose(path[0], prev) else path
        ohold = data.qpos[other.king.qpos_indices].copy()
        for k in range(max(1, int(dur / dt))):
            t0 = time.time()
            s, _, _ = quintic_polynomial(k * dt, 0.0, dur, 0.0, 1.0)
            q = active._sample(kk, s)
            for i, a in enumerate(active.arm_acts):
                data.ctrl[a] = q[i]
            for i, a in enumerate(other.arm_acts):
                data.ctrl[a] = ohold[i]
            if grip is not None and active.grip_act != -1:
                data.ctrl[active.grip_act] = grip
            data.qfrc_applied[active.king.dof_indices] = data.qfrc_bias[active.king.dof_indices]
            data.qfrc_applied[other.king.dof_indices] = data.qfrc_bias[other.king.dof_indices]
            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    return False
                viewer.sync()
                if dt_realtime:
                    time.sleep(max(0, dt - (time.time() - t0)))
        return True

    def _lift_vertical(self, active, other, z_to, viewer=None, dt_realtime=False):
        """Raise the gripper straight up on one IK branch before any lateral
        transit -- a direct joint-space swing at fixture height can graze the
        taller fixtures on the way to the hover point."""
        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, active.arm.ee_site)
        ee0 = self.data.site_xpos[site].copy()
        if ee0[2] >= z_to - 0.02:
            return
        q = self.data.qpos[active.king.qpos_indices].copy()
        path = []
        # retreat first, THEN rise: rising in place pitches the fingertips
        # forward and they can brush the taller fixtures' front corners
        for x in np.linspace(ee0[0], ee0[0] - 0.08, 3)[1:]:
            qn = active.king.inverse_kinematics(np.array([x, ee0[1], ee0[2]]),
                                                q_init=q, restarts=0, rest_weight=0.0)
            if qn is not None:
                q = qn
                path.append(q.copy())
        for z in np.linspace(ee0[2], z_to, 4)[1:]:
            qn = active.king.inverse_kinematics(np.array([ee0[0] - 0.08, ee0[1], z]),
                                                q_init=q, restarts=0, rest_weight=0.0)
            if qn is not None:
                q = qn
                path.append(q.copy())
        if path:
            self._drive(active, other, path, 1.6, viewer=viewer, dt_realtime=dt_realtime)

    def _approach_grasp(self, active, other, handle, fixture, viewer, dt_realtime):
        """Reorient overhead, translate at height, descend onto, and weld a
        handle. Cartesian-chained segments -- a single joint-space hop to the
        hover point takes a curved detour that can sweep the fingers through
        neighbouring fixtures. Returns (yaw, q_grasp) or None."""
        qg, info = active.gs.solve(handle, return_info=True)
        if qg is None:
            return None
        yaw = info["yaw"]
        R = topdown_orientation(yaw)
        hz = handle[2] + 0.10
        p0, _ = active.king.forward_kinematics()
        # 1) reorient to top-down in place, at hover height above own xy
        q = active.king.inverse_kinematics(np.array([p0[0], p0[1], hz]), target_mat=R,
                                           q_init=qg, restarts=2, seed=IK_BRANCH_SEED)
        if q is None:
            q = qg
        self._drive(active, other, [q], 1.2, grip=active.arm.gripper_open, viewer=viewer, dt_realtime=dt_realtime)
        # 2) straight lateral transit at hover height (chained, one branch)
        lat = []
        for f in np.linspace(0, 1, 5)[1:]:
            p = np.array([p0[0] + f * (handle[0] - p0[0]),
                          p0[1] + f * (handle[1] - p0[1]), hz])
            qn = active.king.inverse_kinematics(p, target_mat=R,
                                                q_init=q, restarts=0, rest_weight=0.0)
            if qn is not None:
                q = qn
                lat.append(q.copy())
        self._drive(active, other, lat, 1.5, grip=active.arm.gripper_open, viewer=viewer, dt_realtime=dt_realtime)
        # 3) vertical descent onto the handle (chained)
        desc = []
        for z in np.linspace(hz, handle[2], 5)[1:]:
            qn = active.king.inverse_kinematics(np.array([handle[0], handle[1], z]),
                                                target_mat=R, q_init=q, restarts=0, rest_weight=0.0)
            if qn is not None:
                q = qn
                desc.append(q.copy())
        self._drive(active, other, desc, 1.2, grip=active.arm.gripper_open, viewer=viewer, dt_realtime=dt_realtime)
        qg = q
        # Settle so the servo converges ON the handle (welding early freezes a
        # visible offset), then close the fingers slowly before the weld takes
        # over -- the grasp must read as a grasp.
        self._drive(active, other, qg, 0.6, grip=active.arm.gripper_open, viewer=viewer, dt_realtime=dt_realtime)
        self._drive(active, other, qg, 1.2, grip=active.arm.gripper_closed, viewer=viewer, dt_realtime=dt_realtime)
        self._weld(active, fixture)
        return yaw, qg

    # -- skills -----------------------------------------------------------
    def open_drawer(self, distance=0.095, viewer=None, dt_realtime=False):
        """Right arm: approach the drawer handle frontally (15-deg downward
        diagonal, like a human hand on a drawer knob), close the fingers on
        the bar, pull the drawer straight out toward the robot, release and
        withdraw."""
        a, other = self.right, self.left
        self._park(other)
        h = self._geom_pos("drawer_handle")
        grasp = np.array([h[0] - 0.010, h[1], h[2]])   # slightly toward the bar tip
        th = np.radians(15.0)
        R = front_orientation(th)
        u = np.array([np.cos(th), 0.0, -np.sin(th)])   # finger direction
        half_open = 0.5 * a.arm.gripper_open           # full open can graze the cabinet top
        pre = grasp - 0.10 * u
        q = a.king.inverse_kinematics(pre, target_mat=R,
                                      q_init=self.data.qpos[a.king.qpos_indices], restarts=3,
                                      seed=IK_BRANCH_SEED)
        if q is None:
            return False
        self._drive(a, other, [q], 1.8, grip=half_open, viewer=viewer, dt_realtime=dt_realtime)
        self._drive(a, other, [q], 0.6, grip=half_open, viewer=viewer, dt_realtime=dt_realtime)
        # advance the open cage onto the handle bar (chained IK, one branch)
        adv = []
        for t in np.linspace(0, 1, 7)[1:]:
            qn = a.king.inverse_kinematics(pre + t * 0.10 * u, target_mat=R,
                                           q_init=q, restarts=0, rest_weight=0.0)
            if qn is not None:
                q = qn
                adv.append(q.copy())
        self._drive(a, other, adv, 1.6, grip=half_open, viewer=viewer, dt_realtime=dt_realtime)
        self._drive(a, other, [q], 0.6, grip=half_open, viewer=viewer, dt_realtime=dt_realtime)
        # one-step bias correction: measure the tool error, re-command shifted
        site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, a.arm.ee_site)
        tool = self.data.site_xpos[site] + 0.135 * u
        bias = grasp - tool
        if np.linalg.norm(bias) > 0.002:
            qn = a.king.inverse_kinematics(grasp + bias, target_mat=R,
                                           q_init=q, restarts=0, rest_weight=0.0)
            if qn is not None:
                q = qn
                self._drive(a, other, [q], 0.8, grip=half_open, viewer=viewer, dt_realtime=dt_realtime)
                self._drive(a, other, [q], 0.4, grip=half_open, viewer=viewer, dt_realtime=dt_realtime)
        # close the fingers on the bar, visibly, then weld
        self._drive(a, other, [q], 1.2, grip=a.arm.gripper_closed, viewer=viewer, dt_realtime=dt_realtime)
        self._weld(a, "drawer")
        # pull straight back toward the robot, settle, release, withdraw
        pull = []
        for t in np.linspace(0, 1, 12)[1:]:
            qn = a.king.inverse_kinematics(grasp + [-distance * t, 0, 0] + bias, target_mat=R,
                                           q_init=q, restarts=0, rest_weight=0.0)
            if qn is not None:
                q = qn
                pull.append(q.copy())
        self._drive(a, other, pull, 3.0, grip=a.arm.gripper_closed, viewer=viewer, dt_realtime=dt_realtime)
        self._drive(a, other, [q], 0.5, grip=a.arm.gripper_closed, viewer=viewer, dt_realtime=dt_realtime)
        eid = a._weld_id("drawer")
        if eid >= 0:
            self.data.eq_active[eid] = 0
        back = a.king.inverse_kinematics(grasp + [-distance - 0.08, 0, 0], target_mat=R,
                                         q_init=q, restarts=0, rest_weight=0.0)
        if back is not None:
            self._drive(a, other, [back], 1.0, grip=a.arm.gripper_open, viewer=viewer, dt_realtime=dt_realtime)
        return True

    def _operate_revolute(self, active, other, fixture, handle_geom, axis_body, target,
                          viewer, dt_realtime):
        """Grasp a handle and sweep it on an arc about the joint axis, rotating the
        wrist by the joint angle so the part swings/turns with the gripper."""
        self._park(other)
        h = self._geom_pos(handle_geom)
        axis = self._body_xy(axis_body)
        hz = h[2]
        r = float(np.linalg.norm(h[:2] - axis))
        a0 = float(np.arctan2(h[1] - axis[1], h[0] - axis[0]))
        res = self._approach_grasp(active, other, np.array([h[0], h[1], hz + 0.005]),
                                   fixture, viewer, dt_realtime)
        if res is None:
            return False
        yaw, qg = res
        path, q = [], qg
        for th in np.linspace(0, target, 12)[1:]:
            pos = np.array([axis[0] + r * np.cos(a0 + th), axis[1] + r * np.sin(a0 + th), hz + 0.005])
            qn = active.king.inverse_kinematics(pos, target_mat=topdown_orientation(yaw + th),
                                                q_init=q, restarts=2, rest_weight=0.0)
            if qn is not None:
                q = qn
            path.append(q.copy())
        self._drive(active, other, path, 2.5, grip=active.arm.gripper_closed, viewer=viewer, dt_realtime=dt_realtime)
        # Settle so the constraint ring damps while still held, then release
        # and retreat -- leaving the weld active would drag the fixture during
        # the next skill of a multi-step command.
        self._drive(active, other, [q], 0.6, grip=active.arm.gripper_closed, viewer=viewer, dt_realtime=dt_realtime)
        eid = active._weld_id(fixture)
        if eid >= 0:
            self.data.eq_active[eid] = 0
        h2 = self._geom_pos(handle_geom)
        qr = active.king.inverse_kinematics(np.array([h2[0], h2[1], h2[2] + 0.12]),
                                            target_mat=topdown_orientation(yaw + target),
                                            q_init=q, restarts=0, rest_weight=0.0)
        if qr is not None:
            self._drive(active, other, [qr], 1.0, grip=active.arm.gripper_open, viewer=viewer, dt_realtime=dt_realtime)
        return True

    def open_door(self, target=0.8, viewer=None, dt_realtime=False):
        """Left arm: grasp the door handle and swing the door open about its hinge."""
        return self._operate_revolute(self.left, self.right, "door", "door_handle", "door",
                                      target, viewer, dt_realtime)

    def turn_valve(self, target=1.3, viewer=None, dt_realtime=False):
        """Right arm: grasp the valve grip and sweep it about the valve axis."""
        return self._operate_revolute(self.right, self.left, "valve", "valve_grip", "valve",
                                      target, viewer, dt_realtime)
