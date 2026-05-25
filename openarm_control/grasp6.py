"""6-DOF grasp solving for the OpenArm gripper.

The top-down solver in :mod:`grasp` fixes the approach axis to straight-down and
only searches wrist yaw. That is ideal for flat, top-graspable objects, but some
objects (a mug by its body, a tall/irregular scanned mesh, or a target where a
vertical approach is kinematically unreachable) need the gripper to come in at an
**angle**. This solver generalises the top-down search to a full approach
**direction + roll**:

* the fingers (gripper local ``-z``) point along a chosen ``approach_dir``;
* the closing axis (local ``y``) is set by a ``roll`` about that approach axis.

``approach_dir = (0, 0, -1)`` reproduces the top-down grasp exactly, so this is a
strict superset of :class:`grasp.GraspSolver`. A small tilt penalty makes the
solver *prefer* a straight-down grasp and only tilt when that lowers the IK error
or when no vertical grasp is reachable — so adding it never degrades the
top-graspable cases the rest of the codebase already relies on.

This is purely geometric/analytic (no learning); a learned 6-DOF grasp proposer
can later be dropped in behind the same ``solve`` interface.
"""

import numpy as np

from .kinematics import OpenArmKinematics, orientation_error
from .config import RIGHT_ARM


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def approach_orientation(approach_dir, roll=0.0):
    """Gripper rotation whose fingers (local ``-z``) point along ``approach_dir``.

    ``roll`` rotates the closing axis about the approach axis. With
    ``approach_dir = (0, 0, -1)`` this matches ``grasp.topdown_orientation`` up to
    the roll offset, i.e. a straight-down grasp.
    """
    a = _normalize(approach_dir)
    zc = -a                                   # local +z in world (fingers are -z -> +a)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(zc @ ref)) > 0.95:           # zc parallel to world up -> pick another ref
        ref = np.array([1.0, 0.0, 0.0])
    xc = _normalize(np.cross(ref, zc))
    yc = np.cross(zc, xc)
    R0 = np.column_stack((xc, yc, zc))
    c, s = np.cos(roll), np.sin(roll)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R0 @ Rz


def approach_dir(tilt_deg, azimuth_rad):
    """Unit approach direction tilted ``tilt_deg`` from straight-down toward
    horizontal azimuth ``azimuth_rad``. ``tilt_deg = 0`` -> ``(0, 0, -1)``."""
    t = np.deg2rad(tilt_deg)
    return np.array([np.sin(t) * np.cos(azimuth_rad),
                     np.sin(t) * np.sin(azimuth_rad),
                     -np.cos(t)])


class Grasp6DOFSolver:
    """Finds a reachable grasp configuration over approach direction + roll.

    Mirrors :class:`grasp.GraspSolver` (same constructor and ``solve`` /
    ``is_reachable`` interface) but searches tilted approaches in addition to
    straight-down.
    """

    def __init__(self, model, data, arm=RIGHT_ARM, tool_offset=None):
        self.model = model
        self.data = data
        self.arm = arm
        offset = arm.grasp_offset if tool_offset is None else tool_offset
        self.king = OpenArmKinematics(model, data, joint_names=arm.joints,
                                      site_name=arm.ee_site, tool_offset=offset)

    def solve(self, grasp_pos, tilts_deg=(0.0, 20.0, 40.0), n_azimuth=6,
              roll_samples=9, ori_tol_deg=6.0, tilt_weight=1.0e-3,
              ik_restarts=2, return_info=False, seed=0):
        """Return joint angles for a reachable grasp at ``grasp_pos``.

        Searches approach directions (``tilts_deg`` from vertical, ``n_azimuth``
        headings each) and ``roll_samples`` rolls, keeping the reachable solution
        that minimises ``ik_error + tilt_weight * tilt`` — so a straight-down grasp
        wins ties and tilting only happens when it helps or is required.
        ``ik_restarts`` caps random IK restarts per candidate (the broad search
        scans many orientations, so a few seeds per candidate suffice and keep it
        fast). ``seed`` fixes the restarts for reproducibility. Returns ``q`` (or
        ``(q, info)`` with success/error/tilt/azimuth/roll/approach).
        """
        grasp_pos = np.asarray(grasp_pos, dtype=float)
        ori_tol = np.deg2rad(ori_tol_deg)
        rolls = np.linspace(-np.pi, np.pi, roll_samples)

        best = None  # (score, q, err, tilt, az, roll, approach)
        for tilt in tilts_deg:
            azimuths = [0.0] if tilt <= 1e-6 else np.linspace(0.0, 2 * np.pi, n_azimuth,
                                                              endpoint=False)
            for az in azimuths:
                a = approach_dir(tilt, az)
                for roll in rolls:
                    R = approach_orientation(a, roll)
                    q, info = self.king.inverse_kinematics(grasp_pos, target_mat=R,
                                                           restarts=ik_restarts,
                                                           return_info=True, seed=seed)
                    if not info["success"]:
                        continue
                    _, achm = self.king.forward_kinematics(q)
                    oerr = np.linalg.norm(orientation_error(achm, R))
                    if oerr >= ori_tol:
                        continue
                    score = info["error"] + tilt_weight * np.deg2rad(tilt)
                    if best is None or score < best[0]:
                        best = (score, q, info["error"], float(tilt), float(az),
                                float(roll), a.copy())

        success = best is not None
        if return_info:
            info = {"success": success,
                    "error": float(best[2]) if success else np.inf,
                    "tilt_deg": best[3] if success else None,
                    "azimuth": best[4] if success else None,
                    "roll": best[5] if success else None,
                    "approach": best[6] if success else None}
            return (best[1] if success else None), info
        return best[1] if success else None

    def is_reachable(self, grasp_pos, **kw):
        _, info = self.solve(grasp_pos, return_info=True, **kw)
        return info["success"]
