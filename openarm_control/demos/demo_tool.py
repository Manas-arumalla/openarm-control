"""Tool use (Phase 3b): grasp a stick and use it to reach BEYOND the arm's workspace.

The block and the goal both sit past the gripper's bare top-down reach. The robot
grasps the stick and pushes the block onto the far goal with the stick's tip -- a
task it cannot do bare-handed.

    python -m openarm_control.demos.demo_tool
    python -m openarm_control.demos.demo_tool --headless
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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


def run(headless=False):
    print("=" * 64)
    print("Tool use: the block + goal are past the bare-arm reach. Grasp the stick")
    print("and push the block onto the far goal with the stick's tip.")
    print("=" * 64)
    model, data = _load()
    tc = ToolController(model, data)
    # show the bare arm can't reach them
    bare = tc.ppc.gs.is_reachable(np.array([*_xy(model, data, "block"), 0.45]))
    print(f"  (bare gripper can reach the block directly: {bare})")
    goal = _xy(model, data, "goal")
    if headless:
        ok, msg = tc.use("stick", "block", goal)
        print(f"    {'OK' if ok else 'FAIL'}: {msg}")
        for _ in range(150):
            mujoco.mj_step(model, data)
        return
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        ok, msg = tc.use("stick", "block", goal, viewer=viewer)
        print(f"    {'OK' if ok else 'FAIL'}: {msg}")
        import time
        end = time.time() + 4.0
        while viewer.is_running() and time.time() < end:
            mujoco.mj_step(model, data)
            viewer.sync()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tool use: grasp a stick, push a far block to a far goal.")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)
    run(headless=args.headless)


if __name__ == "__main__":
    main()
