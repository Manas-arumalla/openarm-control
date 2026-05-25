"""F1b — 6-DOF grasp solver.

``Grasp6DOFSolver`` generalises the top-down grasp search to a full approach
direction + roll. It is a strict superset of the top-down solver: a straight-down
approach reproduces the top-down grasp, and a small tilt penalty makes it *prefer*
straight-down and only tilt when that helps or is required. These tests check the
superset property, the top-down preference, and that an angled grasp is found and
reachable when forced.
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import SCANNED_TABLE_SCENE
from openarm_control.grasp6 import Grasp6DOFSolver, approach_orientation

POINT = np.array([0.22, -0.18, 0.45])   # right-side, top-graspable point


def _load():
    model = mujoco.MjModel.from_xml_path(SCANNED_TABLE_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def test_down_approach_reduces_to_topdown():
    """A straight-down approach orients the fingers (local -z) straight down and
    is a valid rotation for any roll."""
    for roll in (-1.0, 0.0, 0.7):
        R = approach_orientation([0, 0, -1], roll)
        fingers = R @ np.array([0, 0, -1.0])
        assert np.allclose(fingers, [0, 0, -1], atol=1e-9), fingers
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_prefers_topdown_on_graspable_point():
    """When a straight-down grasp is reachable, the solver picks tilt = 0 and the
    end-effector reaches the target."""
    model, data = _load()
    gs6 = Grasp6DOFSolver(model, data)
    q, info = gs6.solve(POINT, tilts_deg=(0.0, 30.0), n_azimuth=4, roll_samples=5,
                        ik_restarts=1, return_info=True)
    assert info["success"], "no reachable grasp found"
    assert info["tilt_deg"] == 0.0, f"expected straight-down, got tilt {info['tilt_deg']}"
    pos, _ = gs6.king.forward_kinematics(q)
    assert np.linalg.norm(pos - POINT) < 0.01, f"EE off target: {np.round(pos, 3)}"


def test_finds_tilted_grasp_when_forced():
    """Restricting the search to a 40 deg tilt still yields a reachable grasp whose
    fingers point along the chosen approach direction."""
    model, data = _load()
    gs6 = Grasp6DOFSolver(model, data)
    q, info = gs6.solve(POINT, tilts_deg=(40.0,), n_azimuth=6, roll_samples=7,
                        ik_restarts=2, return_info=True)
    assert info["success"], "no reachable tilted grasp found"
    assert info["tilt_deg"] == 40.0
    pos, mat = gs6.king.forward_kinematics(q)
    assert np.linalg.norm(pos - POINT) < 0.01, f"EE off target: {np.round(pos, 3)}"
    fingers = mat @ np.array([0, 0, -1.0])
    assert float(fingers @ info["approach"]) > 0.9, "fingers not along approach dir"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
