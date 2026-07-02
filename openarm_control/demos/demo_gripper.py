"""Gripper demo: reach the green block, grasp it, and lift it.

Shows the top-down grasp + gripper close + weld-assisted lift primitives that
the pick-and-place pipeline is built from. Run with the MuJoCo viewer:

    python control/demos/demo_gripper.py
"""
import os
import sys
import time

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import SINGLE_ARM_SCENE
from openarm_control.pick_and_place import PickPlaceController


def main():
    model = mujoco.MjModel.from_xml_path(SINGLE_ARM_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)

    ppc = PickPlaceController(model, data)
    for i, aid in enumerate(ppc.arm_acts):
        data.ctrl[aid] = data.qpos[ppc.king.qpos_indices[i]]

    # Build a pick that lifts the green block straight back up (pick == place xy).
    segs = ppc.plan(pick_xy=(0.22, -0.23), place_xy=(0.22, -0.23))

    print("=" * 50)
    print("Gripper Demo: grasp and lift the green block.")
    print("=" * 50)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        ppc.execute(segs, block="block_green", viewer=viewer, dt_realtime=True)
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
