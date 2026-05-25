"""Phase I — stateful multi-turn manipulation session.

Follow-ups ("put it in the bin") and decoupled steps ("go to the bin" / "release")
resolve against the held object and carry over across turns. Slowish (live sim).
One shared ScenePerception (a single MuJoCo Renderer per process).
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import TABLETOP_SCENE
from openarm_control.vision import ScenePerception, ColorShapeDetector
from openarm_control.agent import ManipulationSession, parse_command

BODIES = ["ball_red", "box_green", "can_blue", "ball_orange"]


def _reset(model, data):
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)


def _pos(model, data, body):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)].copy()


def _in_bin(model, data, body, bin_xy):
    p = _pos(model, data, body)
    return np.linalg.norm(p[:2] - bin_xy) < 0.09 and p[2] < 0.50


def test_referent_resolution():
    """'it'/'that'/generic referents parse to a referent target."""
    assert parse_command("put it in the bin").target == "it"
    assert parse_command("take that object out").target == "object"
    assert parse_command("go to the bin").action == "goto"
    assert parse_command("release").action == "release"


def test_multi_turn_pick_then_put_it(tmp_path):
    """'pick up the blue cylinder' then 'put it in the bin' — 'it' = the held
    object — lands it in the bin; then a step-by-step go-to/release also works."""
    model = mujoco.MjModel.from_xml_path(TABLETOP_SCENE)
    data = mujoco.MjData(model)
    _reset(model, data)
    sp = ScenePerception(model, data, "tablecam", detector=ColorShapeDetector())

    # follow-up reference
    s = ManipulationSession(model, data, perception=sp, graspables=BODIES)
    ok, _ = s.do("pick up the blue cylinder")
    assert ok and s.held is not None
    ok, _ = s.do("put it in the bin")
    assert ok and _in_bin(model, data, "can_blue", s.ex.bin_xy), "follow-up place failed"

    # decoupled steps: go to the bin, then release (reuse the one renderer)
    _reset(model, data)
    s2 = ManipulationSession(model, data, perception=sp, graspables=BODIES)
    assert s2.do("pick the red ball")[0]
    assert s2.do("go to the bin")[0]
    assert s2.do("release")[0]
    assert _in_bin(model, data, "ball_red", s2.ex.bin_xy), "step-by-step place failed"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
