"""Top-down grasp solving for the OpenArm gripper.

The gripper approaches a target from above (fingers pointing straight down) and
closes around it. The wrist *yaw* about the vertical is not free -- only some
yaw angles are kinematically reachable at a given point -- so the solver searches
yaw and returns a reachable, collision-aware top-down grasp configuration.
"""

import numpy as np

from .kinematics import OpenArmKinematics, orientation_error
from .config import RIGHT_ARM


def topdown_orientation(yaw):
    """Rotation matrix for a gripper pointing straight down at the given yaw.

    Local +z maps to world +z, so the fingers (local -z) point down; the grasp
    closing axis (local y) is rotated by ``yaw`` about the vertical.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def front_orientation(pitch=0.0):
    """Rotation matrix for a gripper pointing world +x, pitched down by
    ``pitch`` (rad), with the closing axis in the vertical plane -- the cage
    straddles a horizontal handle bar top/bottom. Used for frontal grasps
    (e.g. pulling a drawer toward the robot the way a human would)."""
    c, sn = np.cos(pitch), np.sin(pitch)
    return np.array([[0.0,  sn, -c],
                     [-1.0, 0.0, 0.0],
                     [0.0,  c,  sn]])


class GraspSolver:
    """Finds reachable top-down grasp configurations at the grasp point."""

    def __init__(self, model, data, arm=RIGHT_ARM, tool_offset=None):
        self.model = model
        self.data = data
        self.arm = arm
        offset = arm.grasp_offset if tool_offset is None else tool_offset
        self.king = OpenArmKinematics(model, data, joint_names=arm.joints,
                                      site_name=arm.ee_site, tool_offset=offset)

    def solve(self, grasp_pos, yaw_samples=13, ori_tol_deg=2.0, return_info=False, seed=0):
        """Return joint angles for a top-down grasp at ``grasp_pos``.

        Searches wrist yaw and keeps the most accurate reachable solution.
        ``seed`` fixes the IK restarts so grasps are reproducible run-to-run.
        Returns ``q`` (or ``(q, info)`` with success/error/yaw if requested).
        """
        grasp_pos = np.asarray(grasp_pos, dtype=float)
        ori_tol = np.deg2rad(ori_tol_deg)
        best_q, best_err, best_yaw = None, np.inf, 0.0
        for yaw in np.linspace(-np.pi, np.pi, yaw_samples):
            R = topdown_orientation(yaw)
            q, info = self.king.inverse_kinematics(grasp_pos, target_mat=R,
                                                   return_info=True, seed=seed)
            _, achm = self.king.forward_kinematics(q)
            oerr = np.linalg.norm(orientation_error(achm, R))
            if info["success"] and oerr < ori_tol and info["error"] < best_err:
                best_q, best_err, best_yaw = q, info["error"], yaw

        success = best_q is not None
        if return_info:
            return best_q, {"success": success,
                            "error": float(best_err) if success else np.inf,
                            "yaw": float(best_yaw)}
        return best_q

    def is_reachable(self, grasp_pos, **kw):
        _, info = self.solve(grasp_pos, return_info=True, **kw)
        return info["success"]
