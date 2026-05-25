"""Phase 3b — tool use (reach extension).

The block and goal are past the gripper's bare top-down reach; the robot grasps a
stick and pushes the block onto the far goal with the stick's tip. Verifies the
tool is actually *needed* (bare-unreachable) and that it gets the job done.
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import TOOL_SCENE
from openarm_control.pushing import ToolController


def _load():
    model = mujoco.MjModel.from_xml_path(TOOL_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _xy(model, data, body):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)][:2].copy()


def test_block_and_goal_are_beyond_bare_reach():
    """The whole point of the tool: the bare gripper can't reach the block or goal."""
    model, data = _load()
    tc = ToolController(model, data)
    for body in ("block", "goal"):
        p = _xy(model, data, body)
        assert not tc.ppc.gs.is_reachable(np.array([p[0], p[1], 0.45])), \
            f"{body} is reachable bare-handed -- the tool isn't needed"


def test_tool_use_pushes_far_block_to_goal():
    """Grasp the stick and use it to push the out-of-reach block onto the goal."""
    model, data = _load()
    tc = ToolController(model, data)
    goal = _xy(model, data, "goal")
    start = _xy(model, data, "block")
    ok, msg = tc.use("stick", "block", goal)
    assert ok, msg
    for _ in range(200):
        mujoco.mj_step(model, data)
    p = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")]
    moved = np.linalg.norm(p[:2] - start)
    assert moved > 0.05, f"the block barely moved ({moved*1000:.0f} mm) -- not a real push"
    assert np.linalg.norm(p[:2] - goal) < 0.06, f"block not on the goal: {np.round(p, 3)}"
    assert 0.40 < p[2] < 0.46, f"block left the table (z={p[2]:.3f})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
