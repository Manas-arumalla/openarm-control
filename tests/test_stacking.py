"""Phase I / M6 — stacking one object on another.

Grasp a cube and place it on top of another so it rests stably (block-on-block),
driven directly (the ``stack`` executor primitive) and by language ("stack the
red cube on the green cube"). Slowish (live grasp + carry + place).
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import STACK_SCENE
from openarm_control.vision.scene_perception import ScenePerception
from openarm_control.agent.executor import TaskExecutor
from openarm_control.agent.session import ManipulationSession
from openarm_control.agent.commands import parse_command

CUBES = ["cube_red", "cube_green", "cube_blue", "cube_orange"]


def _setup():
    model = mujoco.MjModel.from_xml_path(STACK_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    perc = ScenePerception(model, data, cam_name="tablecam")
    return model, data, perc


def _pos(model, data, name):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def _settle(model, data, n=400):
    for _ in range(n):
        mujoco.mj_step(model, data)


def _assert_stacked(model, data, top, base):
    """``top`` rests on ``base``: one cube height up, aligned, and stable."""
    t, b = _pos(model, data, top), _pos(model, data, base)
    dz, dxy = t[2] - b[2], np.linalg.norm(t[:2] - b[:2])
    assert 0.04 < dz < 0.06, f"{top} not one cube above {base}: dz={dz*1000:.0f} mm"
    assert dxy < 0.03, f"{top} not aligned over {base}: dxy={dxy*1000:.0f} mm"


def test_parse_stack_commands():
    """'stack X on Y' (and 'put X on top of Y') parse to a stack with both objects;
    'put X on the table' stays a place (the support is a location, not an object)."""
    i = parse_command("stack the red cube on the green cube")
    assert i.action == "stack" and i.target == "red cube" and i.destination == "green cube"
    i = parse_command("put the red cube on top of the blue cube")
    assert i.action == "stack" and i.target == "red cube" and i.destination == "blue cube"
    assert parse_command("put it on the table").action == "place"
    assert parse_command("put the ball in the bin").action == "place"


def test_stack_block_on_block():
    """The executor primitive stacks one cube on another, stably and aligned."""
    model, data, perc = _setup()
    ex = TaskExecutor(model, data, perception=perc, graspables=CUBES, bin_body="bin")
    ok, msg = ex.stack("red cube", "green cube")
    assert ok, f"stack failed: {msg}"
    assert ex.held_body is None, "gripper should be empty after a stack"
    _settle(model, data)
    _assert_stacked(model, data, "cube_red", "cube_green")


def test_stack_by_language():
    """A natural-language stack command runs end-to-end (parse -> ground -> stack)."""
    model, data, perc = _setup()
    sess = ManipulationSession(model, data, perception=perc, graspables=CUBES, bin_body="bin")
    ok, msg = sess.do("stack the orange cube on the blue cube")
    assert ok, f"stack command failed: {msg}"
    assert sess.held is None
    _settle(model, data)
    _assert_stacked(model, data, "cube_orange", "cube_blue")


def test_cannot_stack_on_itself():
    """Stacking an object on itself is refused, not attempted."""
    model, data, perc = _setup()
    ex = TaskExecutor(model, data, perception=perc, graspables=CUBES, bin_body="bin")
    ok, msg = ex.stack("red cube", "red cube")
    assert not ok and "itself" in msg


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
