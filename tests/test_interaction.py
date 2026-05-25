"""Phase I — conversational interaction polish.

Multi-step commands ("do A then B"), state queries ("what are you holding?"),
undo / "put it back", and clarification (list what's visible when a target isn't
found). The query/clarification/parse tests are instant; the multi-step + undo
test runs the live sim (slowish).
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import STACK_SCENE
from openarm_control.vision.scene_perception import ScenePerception
from openarm_control.agent.session import ManipulationSession
from openarm_control.agent.commands import parse_command, split_steps

CUBES = ["cube_red", "cube_green", "cube_blue", "cube_orange"]


def _session():
    model = mujoco.MjModel.from_xml_path(STACK_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    perc = ScenePerception(model, data, cam_name="tablecam")
    sess = ManipulationSession(model, data, perception=perc, graspables=CUBES, bin_body="bin")
    return model, data, sess


def _pos(model, data, name):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def test_parse_query_undo_and_split():
    assert parse_command("what are you holding?").action == "query"
    assert parse_command("what are you holding?").target == "held"
    assert parse_command("what is on the table?").target == "scene"
    assert parse_command("undo").action == "undo"
    assert parse_command("put it back").action == "undo"
    assert parse_command("put it back in the bin").action == "place"   # has a destination
    assert split_steps("pick the red cube then put it in the bin") == \
        ["pick the red cube", "put it in the bin"]
    assert split_steps("pick the red cube and put it in the bin") == \
        ["pick the red cube and put it in the bin"]                     # 'and' is not a split


def test_query_and_clarification():
    """The robot answers what it holds / sees, and lists visible objects when a
    requested object isn't there."""
    _, _, sess = _session()
    assert "empty" in sess.do("what are you holding?")[1].lower()
    ok, msg = sess.do("what is on the table?")
    assert ok and "red box" in msg and "green box" in msg
    ok, msg = sess.do("pick up the purple cube")          # not present
    assert not ok and "can see" in msg                    # clarification lists what's there


def test_multi_step_then_undo():
    """A two-step command runs in order; 'undo' returns the object to its origin."""
    model, data, sess = _session()
    red0 = _pos(model, data, "cube_red")
    binxy = _pos(model, data, "bin")[:2]
    ok, msg = sess.run("pick up the red cube then put it in the bin")
    assert ok, f"multi-step failed: {msg}"
    for _ in range(150):
        mujoco.mj_step(model, data)
    assert np.linalg.norm(_pos(model, data, "cube_red")[:2] - binxy) < 0.08, "red not in bin"
    assert "empty" in sess.do("what are you holding?")[1].lower()
    ok, msg = sess.do("undo")
    assert ok, f"undo failed: {msg}"
    for _ in range(150):
        mujoco.mj_step(model, data)
    assert np.linalg.norm(_pos(model, data, "cube_red")[:2] - red0[:2]) < 0.06, \
        "red not returned to its origin"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
