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
from .grasp import topdown_orientation
from .pick_and_place import PickPlaceController
from .trajectory import quintic_polynomial


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

    def _approach_grasp(self, active, other, handle, fixture, viewer, dt_realtime):
        """Hover over, descend onto, and weld a handle. Returns (yaw, q_grasp) or None."""
        qg, info = active.gs.solve(handle, return_info=True)
        if qg is None:
            return None
        yaw = info["yaw"]
        qh = active.king.inverse_kinematics(handle + [0, 0, 0.10], target_mat=topdown_orientation(yaw),
                                            q_init=qg, restarts=2)
        self._drive(active, other, qh, 1.5, grip=active.arm.gripper_open, viewer=viewer, dt_realtime=dt_realtime)
        self._drive(active, other, qg, 1.2, grip=active.arm.gripper_open, viewer=viewer, dt_realtime=dt_realtime)
        self._weld(active, fixture)
        return yaw, qg

    # -- skills -----------------------------------------------------------
    def open_drawer(self, distance=0.095, viewer=None, dt_realtime=False):
        """Right arm: grasp the drawer handle and pull it straight out (-x)."""
        a, other = self.right, self.left
        self._park(other)
        h = self._geom_pos("drawer_handle")
        grasp = np.array([h[0], h[1], 0.45])
        res = self._approach_grasp(a, other, grasp, "drawer", viewer, dt_realtime)
        if res is None:
            return False
        yaw, qg = res
        R = topdown_orientation(yaw)
        path, q = [], qg
        for s in np.linspace(0, 1, 10)[1:]:
            qn = a.king.inverse_kinematics(grasp + [-distance * s, 0, 0], target_mat=R,
                                           q_init=q, restarts=0, rest_weight=0.0)
            if qn is not None:
                q = qn
            path.append(q.copy())
        self._drive(a, other, path, 2.5, grip=a.arm.gripper_closed, viewer=viewer, dt_realtime=dt_realtime)
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
        return True

    def open_door(self, target=0.8, viewer=None, dt_realtime=False):
        """Left arm: grasp the door handle and swing the door open about its hinge."""
        return self._operate_revolute(self.left, self.right, "door", "door_handle", "door",
                                      target, viewer, dt_realtime)

    def turn_valve(self, target=1.3, viewer=None, dt_realtime=False):
        """Right arm: grasp the valve grip and sweep it about the valve axis."""
        return self._operate_revolute(self.right, self.left, "valve", "valve_grip", "valve",
                                      target, viewer, dt_realtime)
