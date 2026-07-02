"""Trajectory demo: trace a square in the vertical plane with the end-effector.

Uses the resolved-rate Cartesian controller (position tracking) to follow
quintic-timed straight-line segments between corners. Run with the viewer:

    python control/demos/demo_trajectory.py
"""
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import SINGLE_ARM_SCENE
from openarm_control.kinematics import OpenArmKinematics
from openarm_control.controller import CartesianController
from openarm_control.trajectory import quintic_polynomial


def main():
    model = mujoco.MjModel.from_xml_path(SINGLE_ARM_SCENE)
    data = mujoco.MjData(model)
    # free space: no contact with the obstacle for this tracing demo
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_wall_geom")
    model.geom_contype[gid] = 0
    model.geom_conaffinity[gid] = 0
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)

    kin = OpenArmKinematics(model, data)
    ctrl = CartesianController(model, data, kinematics=kin)
    ctrl.reset()
    for i, aid in enumerate(ctrl.actuator_ids):
        data.ctrl[aid] = data.qpos[kin.qpos_indices[i]]

    mocap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap")
    center = np.array([0.26, -0.23, 0.55])
    s = 0.10
    corners = [center + np.array([0, dy, dz]) for dy, dz in
               [(-s, s), (s, s), (s, -s), (-s, -s), (-s, s)]]

    print("=" * 50)
    print("Trajectory Demo: end-effector traces a square (Y-Z plane).")
    print("=" * 50)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        for a, b in zip(corners[:-1], corners[1:]):
            dur = 2.5
            t = 0.0
            while t < dur and viewer.is_running():
                t0 = time.time()
                s_t, _, _ = quintic_polynomial(t, 0.0, dur, 0.0, 1.0)
                target = a + (b - a) * s_t
                data.mocap_pos[model.body_mocapid[mocap]] = target
                ctrl.set_target(target)
                ctrl.step()
                mujoco.mj_step(model, data)
                viewer.sync()
                t += model.opt.timestep
                time.sleep(max(0, model.opt.timestep - (time.time() - t0)))
        while viewer.is_running():
            ctrl.step()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
