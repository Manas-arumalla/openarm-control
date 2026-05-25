"""S3 — articulated-object manipulation (operate drawer / door / valve).

The arm grasps each fixture's handle and moves it along the joint's allowed motion:
a straight pull opens the drawer, an arc + matching wrist rotation swings the door
and turns the valve. Single-arm (the other arm is parked clear). These check the
controller actually drives each joint (the F1c test only checks the assets move when
pushed; this checks the *arm* operates them).
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import ARTICULATED_SCENE
from openarm_control.articulated import ArticulatedController


def _load():
    model = mujoco.MjModel.from_xml_path(ARTICULATED_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _run(method, joint):
    model, data = _load()
    qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
    q0 = data.qpos[qadr]
    ok = getattr(ArticulatedController(model, data), method)()
    assert ok, f"{method} did not complete"
    assert np.isfinite(data.qpos).all(), "simulation went unstable"
    return data.qpos[qadr] - q0


def test_open_drawer():
    moved = _run("open_drawer", "drawer_slide")
    assert moved < -0.05, f"drawer barely opened: {moved*1000:.0f} mm"


def test_open_door():
    moved = _run("open_door", "door_hinge")
    assert moved > 0.4, f"door barely opened: {np.degrees(moved):.0f} deg"


def test_turn_valve():
    moved = _run("turn_valve", "valve_turn")
    assert abs(moved) > 0.8, f"valve barely turned: {np.degrees(moved):.0f} deg"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
