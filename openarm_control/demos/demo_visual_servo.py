"""Vision-guided manipulation: the arm SEES a red cube and picks-and-places it.

Camera: a near-top-down camera (`topcam` in vision_scene.xml) renders RGB+depth.
The cube's 3D position is estimated purely from the image (color detection +
depth deprojection) -- the controller never reads the cube's true pose. The
estimate is then fed to the proven top-down pick-and-place pipeline, so the
gripper descends *vertically* onto the cube (it does not push it).

    python -m openarm_control.demos.demo_visual_servo
"""
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import VISION_SCENE, RIGHT_ARM
from openarm_control.pick_and_place import PickPlaceController
from openarm_control.vision import Camera, detect_color


def estimate_cube(cam, color="red"):
    """3D world position of the cube from the camera only (detect + deproject)."""
    det = detect_color(cam.rgb(), color)
    if det is None:
        return None
    u, v = det["centroid"]
    z = float(cam.depth()[int(round(v)), int(round(u))])
    if not np.isfinite(z) or z <= 0:
        return None
    return cam.deproject(u, v, z)


def main(argv=None):
    model = mujoco.MjModel.from_xml_path(VISION_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "park"))
    mujoco.mj_forward(model, data)

    cam = Camera(model, data, "topcam", width=320, height=240)
    ppc = PickPlaceController(model, data, arm=RIGHT_ARM)
    for i, aid in enumerate(ppc.arm_acts):
        data.ctrl[aid] = data.qpos[ppc.king.qpos_indices[i]]

    # Estimate the cube position from vision (arm parked aside -> no occlusion).
    ests = [e for e in (estimate_cube(cam) for _ in range(20)) if e is not None]
    cube = np.median(ests, axis=0)
    print("=" * 56)
    print("Vision-guided pick-and-place (camera: topcam, near-top-down).")
    print(f"  cube position estimated from the image: {np.round(cube, 3)}")
    print("=" * 56)

    segs = ppc.plan(pick_xy=(cube[0], cube[1]), place_xy=(0.30, -0.33))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        ppc.execute(segs, block="cube", viewer=viewer, dt_realtime=True, gravity_comp=True)
        print("  done: saw the cube, grasped it top-down (no pushing), placed it.")
        while viewer.is_running():
            data.qfrc_applied[ppc.king.dof_indices] = data.qfrc_bias[ppc.king.dof_indices]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
