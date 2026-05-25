"""I1 — language for the articulated skills.

The parser maps "open the drawer", "open the cabinet door", "turn the valve",
"unscrew the cap" to the articulated actions (and keeps "open the gripper" =
release), and a multi-step command splits into ordered clauses. The
ArticulatedSession dispatches a parsed command to the controller end-to-end.
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.agent.commands import parse_command, split_steps
from openarm_control.config import ARTICULATED_SCENE
from openarm_control.agent.articulated_session import ArticulatedSession


def test_parse_articulated_commands():
    assert parse_command("open the drawer").action == "open_drawer"
    assert parse_command("pull the drawer open").action == "open_drawer"
    assert parse_command("open the cabinet door").action == "open_door"
    assert parse_command("turn the valve").action == "turn_valve"
    assert parse_command("rotate the valve").action == "turn_valve"
    assert parse_command("unscrew the cap").action == "unscrew"
    assert parse_command("open the bottle").action == "unscrew"


def test_open_hand_still_parses_as_release():
    # "open" without an articulated fixture must remain release-the-gripper
    assert parse_command("open the gripper").action == "release"
    assert parse_command("let go").action == "release"


def test_multi_step_splits():
    steps = split_steps("open the drawer then turn the valve")
    assert len(steps) == 2
    assert parse_command(steps[0]).action == "open_drawer"
    assert parse_command(steps[1]).action == "turn_valve"


def test_session_dispatches_command_to_controller():
    """An end-to-end language command opens the drawer (joint actually moves)."""
    model = mujoco.MjModel.from_xml_path(ARTICULATED_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")]
    q0 = data.qpos[qadr]
    results = ArticulatedSession(model, data).do("open the drawer")
    assert len(results) == 1 and results[0][1] is True, results
    assert abs(data.qpos[qadr] - q0) > 0.05, "drawer did not open via the command"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
