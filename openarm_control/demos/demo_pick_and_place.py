"""Autonomous pick-and-place sorting demo.

The right arm picks each colored block and drops it into the matching bin,
using top-down grasps, smooth joint trajectories, gravity compensation, and a
grasp weld for a reliable hold. Run with the MuJoCo viewer:

    python control/demos/demo_pick_and_place.py
"""
import os
import sys

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import SINGLE_ARM_SCENE
from openarm_control.autonomy import SortingTask


def main():
    model = mujoco.MjModel.from_xml_path(SINGLE_ARM_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)

    # Sort each block into its matching bin (positions read live from the sim).
    task = SortingTask(model, data)
    for i, aid in enumerate(task.ppc.arm_acts):
        data.ctrl[aid] = data.qpos[task.ppc.king.qpos_indices[i]]

    print("=" * 56)
    print("Autonomous Pick-and-Place Sorting")
    print("Right arm sorts each block into its matching bin.")
    print("=" * 56)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        import time
        time.sleep(1.0)
        placed = task.run(viewer=viewer, dt_realtime=True)
        print(f"Sorting complete: {placed}/3 blocks placed.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
