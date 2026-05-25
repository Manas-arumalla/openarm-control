"""Motion-planning demo: collision-free joint path around an obstacle wall.

A wall sits between the pick lane and the bin lane. The planner (RRT-Connect by
default, or PRM) finds a collision-free joint-space path from above a block to
above a bin on the far side, and the arm executes it as a smooth trajectory.

    python control/demos/demo_motion_planning.py            # RRT-Connect
    python control/demos/demo_motion_planning.py --planner prm
"""
import os
import sys
import time
import argparse

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import OBSTACLE_SCENE
from openarm_control.grasp import GraspSolver
from openarm_control.trajectory import JointTrajectory
from openarm_control.planners.rrt import RRTPlanner
from openarm_control.planners.prm import PRMPlanner


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openarm plan")
    parser.add_argument("--planner", choices=["rrt", "prm"], default="rrt")
    args = parser.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(OBSTACLE_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)

    gs = GraspSolver(model, data)
    kin = gs.king
    arm_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_joint{i}_ctrl")
                for i in range(1, 8)]
    for i, aid in enumerate(arm_acts):
        data.ctrl[aid] = data.qpos[kin.qpos_indices[i]]

    q_start = gs.solve(np.array([0.18, -0.12, 0.55]))   # above red block
    q_goal = gs.solve(np.array([0.36, -0.38, 0.55]))    # above blue bin (across wall)

    print("=" * 56)
    print(f"Motion Planning Demo ({args.planner.upper()}) — avoid the wall")
    print("=" * 56)
    if args.planner == "rrt":
        planner = RRTPlanner(model, data, kin, seed=1)
    else:
        planner = PRMPlanner(model, data, kin, num_samples=400, seed=1)
    t0 = time.time()
    path = planner.plan(q_start, q_goal)
    if path is None:
        print("No path found."); return
    print(f"Found path: {len(path)} waypoints in {time.time()-t0:.2f}s")

    # Time-parameterize each segment by its joint distance.
    segs = [JointTrajectory(path[i], path[i + 1], max(0.8, np.linalg.norm(path[i+1]-path[i]) * 2.0))
            for i in range(len(path) - 1)]

    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        for traj in segs:
            t = 0.0
            while t < traj.duration and viewer.is_running():
                t0 = time.time()
                q = traj.evaluate(t)
                for i, aid in enumerate(arm_acts):
                    data.ctrl[aid] = q[i]
                data.qfrc_applied[kin.dof_indices] = data.qfrc_bias[kin.dof_indices]
                mujoco.mj_step(model, data)
                viewer.sync()
                t += dt
                time.sleep(max(0, dt - (time.time() - t0)))
        print("Execution complete.")
        while viewer.is_running():
            data.qfrc_applied[kin.dof_indices] = data.qfrc_bias[kin.dof_indices]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
