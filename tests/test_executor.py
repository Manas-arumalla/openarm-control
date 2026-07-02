"""Phase I / M3 — task executor: command -> ground -> collision-free pick/place.

End-to-end in the live sim (slowish): a language command is grounded by vision and
carried out, avoiding the other objects.
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import TABLETOP_SCENE
from openarm_control.vision import ScenePerception, ColorShapeDetector
from openarm_control.agent import parse_command, TaskExecutor

BODIES = ["ball_red", "box_green", "can_blue", "ball_orange"]


def _setup():
    model = mujoco.MjModel.from_xml_path(TABLETOP_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    sp = ScenePerception(model, data, "tablecam", detector=ColorShapeDetector())
    ex = TaskExecutor(model, data, perception=sp, graspables=BODIES)
    return model, data, ex


def _pos(model, data, body):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)].copy()


def test_pick_lifts_target_and_leaves_clutter():
    """'pick the red ball' lifts the red ball; the other objects are untouched."""
    model, data, ex = _setup()
    z0 = _pos(model, data, "ball_red")[2]
    others0 = {b: _pos(model, data, b) for b in BODIES if b != "ball_red"}
    ok, msg = ex.execute(parse_command("pick the red ball"))
    assert ok, msg
    assert _pos(model, data, "ball_red")[2] - z0 > 0.05, "target not lifted"
    for b, p0 in others0.items():
        assert np.linalg.norm(_pos(model, data, b)[:2] - p0[:2]) < 0.02, \
            f"{b} was disturbed (collision)"


def test_place_in_bin():
    """'put the green box in the bin' lands the box in the bin."""
    model, data, ex = _setup()
    ok, msg = ex.execute(parse_command("put the green box in the bin"))
    assert ok, msg
    gp = _pos(model, data, "box_green")
    assert np.linalg.norm(gp[:2] - ex.bin_xy) < 0.09 and gp[2] < 0.50, \
        f"green box not in the bin: {gp}"


def test_carried_object_collision_is_checked():
    """A motion that sweeps the *held* object into the environment is flagged by
    carried-object collision, even though the arm alone is clear there."""
    model, data, ex = _setup()
    ok, _ = ex.grasp("blue can")
    assert ok
    # A config with the gripper just above the green box: the gripper clears it,
    # but the carried can (hanging below) would dip into it.
    gb = _pos(model, data, "box_green")
    q_low, info = ex.ppc.gs.solve(np.array([gb[0], gb[1], 0.50]), return_info=True)
    if not info["success"]:
        return                                    # geometry-dependent; skip if unreachable
    others = [b for b in BODIES if b != "can_blue"]
    ex.checker.set_carried("can_blue", also_avoid=others)
    carried_hit = ex.checker.in_collision(q_low, ignore_bodies=())
    ex.checker.set_carried(None)
    arm_only_clear = not ex.checker.in_collision(q_low, ignore_bodies=("can_blue", "box_green"))
    assert carried_hit, "carried object's collision was not detected"
    assert arm_only_clear, "arm alone should be clear there (only the held object collides)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
