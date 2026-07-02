"""Non-prehensile pushing: move an object to a target *without grasping it*.

The robot pushes an object across the table with the **closed gripper as a pusher**.
Each stroke: approach just behind the object (on the far side from the target),
descend, and push along the object→target line at a fixed point-down orientation;
then re-perceive the object and re-aim. The closed-loop re-planning corrects the
drift/slip inherent to pushing, so the object converges to the target. Classical
control, contact-rich, no weld and no grasp.

    pc = PushController(model, data)
    pc.push("puck", (0.33, -0.30))            # nudge the puck onto the goal
"""
from __future__ import annotations

import numpy as np
import mujoco

from .config import RIGHT_ARM
from .pick_and_place import PickPlaceController, TABLE_TOP_Z


class PushController:
    def __init__(self, model, data, arm=RIGHT_ARM, push_z=None, hover=0.12):
        self.model, self.data = model, data
        self.ppc = PickPlaceController(model, data, arm=arm, hover=hover)
        self.king = self.ppc.king
        self.hover = hover
        # Pusher height: tool point a little above the object centre so the closed
        # fingers contact the object's side (not its top).
        self.push_z = (TABLE_TOP_Z + 0.06) if push_z is None else push_z

    def _obj_xy(self, body):
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)][:2].copy()

    def _obj_half(self, body):
        """Horizontal half-extent of the object's first collision geom (for stand-off)."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == bid:
                return float(max(self.model.geom_size[g][:2]))
        return 0.035

    def _settle(self, viewer, n=120):
        if viewer is None:
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)

    def push(self, obj_body, target_xy, tol=0.05, max_strokes=6, stroke=0.06, viewer=None):
        """Push ``obj_body`` to ``target_xy`` (xy) with the closed gripper, re-aiming
        after each stroke. Returns (ok, message). ``tol`` is the success radius."""
        from .grasp import topdown_orientation
        target = np.asarray(target_xy, dtype=float)
        half = self._obj_half(obj_body)
        standoff = half + 0.05
        dist = np.linalg.norm(target - self._obj_xy(obj_body))
        for _ in range(max_strokes):
            O = self._obj_xy(obj_body)
            dist = float(np.linalg.norm(target - O))
            if dist < tol:
                return True, f"pushed '{obj_body}' onto the target ({dist*1000:.0f} mm)"
            d = (target - O) / dist
            push_len = min(stroke, dist)                 # don't drive past the target
            S = O - d * standoff                         # behind the object
            E = O + d * push_len                         # push the object forward by ~push_len
            # Find a *reachable* point-down config behind the object (the closed
            # gripper is a blunt pusher, so its yaw need not equal the push
            # direction -- only be reachable). Hold THAT orientation for the whole
            # stroke so the wrist never rotates mid-push (which would fling the
            # object); seeding the line tracker with this config keeps it smooth.
            gq, info = self.ppc.gs.solve(np.array([S[0], S[1], self.push_z]), return_info=True)
            if not info["success"]:
                return False, f"can't reach behind '{obj_body}' to push it"
            R = topdown_orientation(info["yaw"])
            up = self.ppc._ik_line_oriented(
                gq, np.array([S[0], S[1], self.push_z]),
                np.array([S[0], S[1], self.push_z + self.hover]), R)
            line = self.ppc._ik_line_oriented(
                gq, np.array([S[0], S[1], self.push_z]),
                np.array([E[0], E[1], self.push_z]), R)
            if self.ppc._max_jump(line) > np.deg2rad(120.0):
                return False, f"no smooth push path for '{obj_body}' from here"
            self.ppc.execute([(np.array([up[-1]]), True, 1.5),   # over the approach point
                              (up[::-1],            True, 1.0),   # descend behind the object
                              (line,                True, 2.5),   # push toward the target
                              (up,                  True, 1.0)],  # lift clear
                             block=None, viewer=viewer)
            self._settle(viewer)
        O = self._obj_xy(obj_body)
        dist = float(np.linalg.norm(target - O))
        ok = dist < tol * 1.6
        return ok, f"pushed '{obj_body}' to {dist*1000:.0f} mm from the target"


class ToolController:
    """Tool use: grasp a stick and use its tip to move an object the bare gripper
    can't reach. The stick is welded to the gripper; holding the grasp orientation
    keeps the stick pointing in a fixed world direction, so the tip is a constant
    offset from the gripper -- pushing the gripper along an offset line sweeps the
    tip across the object. Re-aims each stroke, like `PushController`."""

    def __init__(self, model, data, arm=RIGHT_ARM, hover=0.12, push_z=None):
        self.model, self.data = model, data
        self.ppc = PickPlaceController(model, data, arm=arm, hover=hover)
        self.king = self.ppc.king
        self.hover = hover
        self.push_z = (TABLE_TOP_Z + 0.025) if push_z is None else push_z   # tip at object mid-height
        self.offset = None       # tool-tip position relative to the tool point (world, at self.R)
        self.R = None            # held gripper orientation
        self.tool = None

    def _xy(self, body):
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)][:2].copy()

    def _obj_half(self, body):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == bid:
                return float(max(self.model.geom_size[g][:2]))
        return 0.03

    def _tool_tip(self, tool_body, toward_xy):
        """World position of the tool end nearest ``toward_xy`` (the pushing tip)."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, tool_body)
        # the long axis is the geom's largest half-dim; use the geom's local +/-x.
        g = next(gg for gg in range(self.model.ngeom) if self.model.geom_bodyid[gg] == bid)
        L = float(self.model.geom_size[g][0])
        R = self.data.xmat[bid].reshape(3, 3)
        p = self.data.xpos[bid]
        ends = [p + R @ np.array([L, 0, 0]), p + R @ np.array([-L, 0, 0])]
        return min(ends, key=lambda e: np.linalg.norm(e[:2] - np.asarray(toward_xy)))

    def grasp_tool(self, tool_body, viewer=None):
        """Grasp the tool at its centre and record the tip offset + held orientation."""
        from .pick_and_place import TABLE_TOP_Z as _T
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, tool_body)
        sp = self.data.xpos[bid].copy()
        gz = float(np.clip(sp[2] + 0.005, _T + 0.035, _T + 0.12))
        try:
            segs = self.ppc.plan_pick(sp[:2], grasp_z=gz)
        except ValueError:
            return False, f"can't grasp the {tool_body}"
        self.ppc.execute(segs, block=tool_body, viewer=viewer)
        self.tool = tool_body
        return True, f"grasped the {tool_body}"

    def push_with_tool(self, obj_body, target_xy, tol=0.05, max_strokes=7,
                       stroke=0.06, viewer=None):
        """Push ``obj_body`` to ``target_xy`` with the grasped tool's tip, re-aiming
        each stroke. Returns (ok, message)."""
        if self.tool is None:
            return False, "no tool grasped"
        target = np.asarray(target_xy, dtype=float)
        standoff = self._obj_half(obj_body) + 0.045
        for _ in range(max_strokes):
            O = self._xy(obj_body)
            dist = float(np.linalg.norm(target - O))
            if dist < tol:
                return True, f"used the {self.tool} to push '{obj_body}' onto the target ({dist*1000:.0f} mm)"
            d = (target - O) / dist
            # Refresh the held orientation + tip offset from the current pose (the
            # tool is rigid, so these are stable while we hold the grasp).
            P0, self.R = self.king.forward_kinematics()
            self.offset = self._tool_tip(self.tool, target) - P0
            push_len = min(stroke, dist)
            s_tip = np.array([O[0] - d[0] * standoff, O[1] - d[1] * standoff, self.push_z])
            e_tip = np.array([O[0] + d[0] * push_len, O[1] + d[1] * push_len, self.push_z])
            Sg, Eg = s_tip - self.offset, e_tip - self.offset
            q0, info = self.king.inverse_kinematics(Sg, target_mat=self.R, restarts=16, return_info=True)
            if not info["success"]:
                return False, f"can't position the {self.tool} behind '{obj_body}'"
            up = self.ppc._ik_line_oriented(q0, Sg, Sg + np.array([0, 0, self.hover]), self.R)
            line = self.ppc._ik_line_oriented(q0, Sg, Eg, self.R)
            if self.ppc._max_jump(line) > np.deg2rad(120.0):
                return False, f"no smooth tool path for '{obj_body}' from here"
            self.ppc.execute([(np.array([up[-1]]), True, 1.5),
                              (up[::-1],            True, 1.2),
                              (line,                True, 2.5),
                              (up,                  True, 1.2)],
                             block=None, viewer=viewer)   # block=None: the tool weld stays active
            if viewer is None:
                for _ in range(120):
                    mujoco.mj_step(self.model, self.data)
        O = self._xy(obj_body)
        dist = float(np.linalg.norm(target - O))
        return dist < tol * 1.6, f"pushed '{obj_body}' to {dist*1000:.0f} mm from the target with the {self.tool}"

    def use(self, tool_body, obj_body, target_xy, viewer=None):
        """Grasp ``tool_body`` and use it to push ``obj_body`` to ``target_xy``."""
        ok, msg = self.grasp_tool(tool_body, viewer=viewer)
        if not ok:
            return ok, msg
        return self.push_with_tool(obj_body, target_xy, viewer=viewer)
