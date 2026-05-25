"""Deformable cloth folding: grasp a corner of a cloth and fold it over.

The cloth is a MuJoCo **flex** grid (9x9, with self-collision so a folded layer rests
on the layer below it); each grid vertex is a body, so the gripper can grasp a corner
via the scene's weld and carry it across to fold the sheet. The carry uses the
continuous, branch-safe ``_ik_line_oriented`` (a free-yaw point-down line branch-jumps
near the workspace edges where the cloth corners sit). Classical scripted fold -- which
also serves as a demonstration source for an imitation-learned policy.

Single-arm: two close-mounted 7-DOF arms working over one centred cloth collide in
motion regardless of grasp separation (the upper arms cross) -- the same hardware
limit as the bimanual bottle. A two-arm half-fold IS more accurate (the edge lays onto
the opposite edge to ~mm) but is not reliably collision-free without a dedicated
collision-checked dual-arm planner; the single-arm corner fold here is collision-free.

    cf = ClothFoldController(model, data)
    cf.fold("cloth_0", target_xy)        # grasp corner 0, fold it onto target_xy
"""
from __future__ import annotations

import numpy as np
import mujoco

from .config import RIGHT_ARM
from .pick_and_place import PickPlaceController, TABLE_TOP_Z

OPEN, CLOSE = False, True

_READY_RIGHT = np.array([-0.720369, 2.27095, 0.290977, 1.90389, 1.23759, 0.785149, 0.558776])


def set_ready(model, data, settle=300):
    """Pose both arms ready (right + mirrored left) and let the cloth settle on the
    table (the cloth scene has no keyframe; the cloth starts at its flex rest)."""
    from .config import MIRROR_R2L
    for i in range(7):
        for arm, mir in (("right", 1.0), ("left", MIRROR_R2L[i])):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{arm}_joint{i+1}")
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = mir * _READY_RIGHT[i]
    mujoco.mj_forward(model, data)
    for _ in range(settle):
        mujoco.mj_step(model, data)


class ClothFoldController:
    def __init__(self, model, data, arm=RIGHT_ARM, hover=0.14):
        self.model, self.data = model, data
        self.ppc = PickPlaceController(model, data, arm=arm, hover=hover)
        self.king = self.ppc.king
        self.arm = arm
        self.hover = hover

    def corner_xy(self, body):
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)][:2].copy()

    def cloth_vertices(self, prefix="cloth_", n=None):
        """World positions of the cloth vertices (auto-detects the grid size)."""
        out = []
        i = 0
        while n is None or i < n:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{i}")
            if bid < 0:
                break
            out.append(self.data.xpos[bid].copy())
            i += 1
        return np.array(out)

    def settle(self, n=300, viewer=None):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None and viewer.is_running():
                viewer.sync()

    def fold(self, corner_body, target_xy, grasp_z=None, place_z=None, viewer=None):
        """Grasp ``corner_body`` (a cloth corner), lift it, carry it to ``target_xy``
        and lay it down -- folding the sheet. Returns (ok, message)."""
        from .grasp import topdown_orientation
        target = np.asarray(target_xy, dtype=float)
        cxy = self.corner_xy(corner_body)
        grasp_z = (TABLE_TOP_Z + 0.02) if grasp_z is None else grasp_z
        place_z = (TABLE_TOP_Z + 0.012) if place_z is None else place_z

        gq, info = self.ppc.gs.solve(np.array([cxy[0], cxy[1], grasp_z]), return_info=True)
        if not info["success"]:
            return False, f"can't reach cloth corner '{corner_body}'"
        R = topdown_orientation(info["yaw"])
        gz_hi = grasp_z + self.hover
        up = self.ppc._ik_line_oriented(gq, np.array([cxy[0], cxy[1], grasp_z]),
                                        np.array([cxy[0], cxy[1], gz_hi]), R)
        # carry the corner (at hover height) to above the target, then lay it down.
        carry = self.ppc._ik_line_oriented(up[-1], np.array([cxy[0], cxy[1], gz_hi]),
                                           np.array([target[0], target[1], place_z + self.hover]), R)
        down = self.ppc._ik_line_oriented(carry[-1], np.array([target[0], target[1], place_z + self.hover]),
                                          np.array([target[0], target[1], place_z]), R)
        if max(self.ppc._max_jump(carry), self.ppc._max_jump(down)) > np.deg2rad(20.0):
            return False, "fold path discontinuous (IK branch jump)"

        # Approach + descend onto the corner (gripper open, no weld yet).
        self.ppc.execute([(np.array([up[-1]]), OPEN, 1.5),
                          (up[::-1],            OPEN, 1.5)], block=None, viewer=viewer)
        self.ppc.attach(corner_body)                       # weld the corner to the gripper
        # Lift -> carry -> lay down (corner welded, gripper closed cosmetically).
        self.ppc.execute([(up,    CLOSE, 1.5),
                          (carry, CLOSE, 3.0),
                          (down,  CLOSE, 1.5)], block=None, viewer=viewer)
        self.ppc.detach(corner_body)                       # release the corner
        self.ppc.execute([(down[::-1], OPEN, 1.5)], block=None, viewer=viewer)
        self.settle(200, viewer=viewer)
        cf = self.corner_xy(corner_body)
        err = float(np.linalg.norm(cf - target))
        return err < 0.06, f"folded '{corner_body}' to {err*1000:.0f} mm from the target"
