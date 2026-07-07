"""Phase 3a — non-prehensile pushing.

Push a puck to a goal *without grasping it*: the closed gripper nudges it along the
object→target line, re-aiming after each stroke. Contact-rich classical control.
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import MOVE_PUCK_SCENE
from openarm_control.pushing import PushController
from openarm_control.agent.commands import parse_command


def _load():
    model = mujoco.MjModel.from_xml_path(MOVE_PUCK_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _xy(model, data, body):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)][:2].copy()


def test_parse_push_command():
    """'push'/'nudge' parse as a (non-prehensile) push, distinct from grasp-and-move."""
    assert parse_command("push the puck to goal a").action == "push"
    assert parse_command("nudge the puck toward the goal").action == "push"
    # a plain "move" is still a grasp-and-carry, not a push
    assert parse_command("move the puck to the bin").action == "move"


@pytest.mark.parametrize("goal", ["goal_a", "goal_b"])
def test_push_puck_onto_goal(goal):
    """The controller pushes the puck onto each goal (within tolerance) and the puck
    stays on the table -- it is pushed, never grasped/lifted."""
    model, data = _load()
    pc = PushController(model, data)
    target = _xy(model, data, goal)
    start = _xy(model, data, "puck")
    assert np.linalg.norm(start - target) > 0.12, "puck should start away from the goal"
    ok, msg = pc.push("puck", target, tol=0.05)
    assert ok, msg
    for _ in range(150):
        mujoco.mj_step(model, data)
    p = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "puck")]
    assert np.linalg.norm(p[:2] - target) < 0.06, f"puck not on {goal}: {np.round(p, 3)}"
    assert 0.40 < p[2] < 0.46, f"puck left the table (z={p[2]:.3f}) -- it was not just pushed"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
