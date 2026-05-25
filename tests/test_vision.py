"""Perception & visual-servoing tests (headless; offscreen render)."""
import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import VISION_SCENE, RIGHT_ARM, GRASP_LOCAL_OFFSET
from openarm_control.kinematics import OpenArmKinematics
from openarm_control.controller import CartesianController
from openarm_control.pick_and_place import PickPlaceController
from openarm_control.vision import Camera, detect_color, VisualServo


def _load():
    model = mujoco.MjModel.from_xml_path(VISION_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "park"))
    mujoco.mj_forward(model, data)
    return model, data


def _cube(model, data):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")].copy()


def test_camera_renders_rgb_and_depth():
    model, data = _load()
    cam = Camera(model, data, "topcam", width=160, height=120)
    rgb = cam.rgb()
    depth = cam.depth()
    assert rgb.shape == (120, 160, 3) and rgb.dtype == np.uint8
    assert depth.shape == (120, 160) and np.isfinite(depth).all()


def test_detection_finds_cube_and_rejects_absent_color():
    model, data = _load()
    cam = Camera(model, data, "topcam")
    rgb = cam.rgb()
    assert detect_color(rgb, "red") is not None          # the cube is red
    assert detect_color(rgb, "green") is None            # no green object present


def test_deprojection_matches_ground_truth():
    """Camera-only 3D estimate of the cube is within 2 cm (xy) of truth."""
    model, data = _load()
    cam = Camera(model, data, "topcam", width=320, height=240)
    det = detect_color(cam.rgb(), "red")
    u, v = det["centroid"]
    z = float(cam.depth()[int(round(v)), int(round(u))])
    est = cam.deproject(u, v, z)
    true = _cube(model, data)
    assert np.linalg.norm(est[:2] - true[:2]) < 0.02


def test_visual_servo_reaches_seen_cube():
    """Arm reaches the cube using only the camera estimate (no true pose)."""
    model, data = _load()
    king = OpenArmKinematics(model, data, joint_names=RIGHT_ARM.joints,
                             site_name=RIGHT_ARM.ee_site, tool_offset=GRASP_LOCAL_OFFSET)
    ctrl = CartesianController(model, data, arm=RIGHT_ARM, kinematics=king, pos_gain=6.0)
    ctrl.reset()
    for i, a in enumerate(ctrl.actuator_ids):
        data.ctrl[a] = data.qpos[king.qpos_indices[i]]
    cam = Camera(model, data, "topcam", width=320, height=240)
    servo = VisualServo(model, data, cam, ctrl, "red", approach_offset=(0, 0, -0.005))
    for t in range(4000):
        if t % 20 == 0:
            p = servo.estimate_target()
            if p is not None:
                ctrl.set_target(p + servo.approach_offset)
        ctrl.step()
        data.qfrc_applied[king.dof_indices] = data.qfrc_bias[king.dof_indices]
        mujoco.mj_step(model, data)
    gp, _ = king.forward_kinematics()
    assert np.linalg.norm(gp - _cube(model, data)) < 0.03


def test_vision_guided_pick_and_place():
    """See the cube (camera only) -> top-down pick -> place; cube reaches the
    target WITHOUT being pushed away during the (vertical) approach."""
    from openarm_control.demos.demo_visual_servo import estimate_cube
    model, data = _load()
    cam = Camera(model, data, "topcam", width=320, height=240)
    ppc = PickPlaceController(model, data, arm=RIGHT_ARM)
    for i, a in enumerate(ppc.arm_acts):
        data.ctrl[a] = data.qpos[ppc.king.qpos_indices[i]]
    ests = [e for e in (estimate_cube(cam) for _ in range(20)) if e is not None]
    cube_est = np.median(ests, axis=0)
    # the estimate (camera only) matches truth
    assert np.linalg.norm(cube_est[:2] - _cube(model, data)[:2]) < 0.02
    place = (0.30, -0.33)
    ppc.execute(ppc.plan(pick_xy=(cube_est[0], cube_est[1]), place_xy=place),
                block="cube", gravity_comp=True)
    data.qfrc_applied[:] = 0
    for _ in range(800):
        mujoco.mj_step(model, data)
    end = _cube(model, data)
    assert abs(end[0] - place[0]) < 0.06 and abs(end[1] - place[1]) < 0.06, "cube not placed on target"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
