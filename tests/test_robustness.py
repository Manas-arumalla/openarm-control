"""Regression tests for two reliability fixes (additive; no behaviour change).

1. The grasp weld captures the object **centred on the gripper's approach axis**
   (lateral in-gripper offset zeroed), so a sideways shove from the closing fingers
   -- or a tiny BLAS/thread-state plan difference -- can't make a carried object miss
   its target. (This is what made the full-suite `place_in_bin` deterministic.)
2. The offscreen `Camera`/perception free their MuJoCo `Renderer` (GL context), so
   creating many of them across a long process / suite doesn't leak contexts.

Both are deterministic (no physics rollouts), so they can't flake.
"""
import os
import sys

import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import STACK_SCENE, RIGHT_ARM
from openarm_control.pick_and_place import PickPlaceController
from openarm_control.vision.camera import Camera
from openarm_control.vision.scene_perception import ScenePerception


def _ready(model, data):
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)


def test_grasp_weld_centers_object_laterally():
    """Even with the object offset sideways, attach() stores a weld whose lateral
    (in-gripper x,y) offset is zero -- the object is held on the approach axis."""
    model = mujoco.MjModel.from_xml_path(STACK_SCENE)
    data = mujoco.MjData(model)
    _ready(model, data)
    ppc = PickPlaceController(model, data, arm=RIGHT_ARM)

    # Shove cube_red well off to the side, then weld it.
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_red")
    qadr = model.jnt_qposadr[model.body_jntadr[bid]]      # free joint of cube_red
    data.qpos[qadr:qadr + 3] = [0.30, -0.05, 0.45]        # offset from the gripper
    mujoco.mj_forward(model, data)

    ppc.attach("cube_red")
    eid = ppc._weld_id("cube_red")
    assert eid >= 0
    relpos = model.eq_data[eid, 3:6]
    assert abs(relpos[0]) < 1e-9 and abs(relpos[1]) < 1e-9, \
        f"weld not centred on the approach axis: lateral offset {relpos[:2]}"
    assert model.eq_data[eid, 10] == 1.0 and data.eq_active[eid] == 1


def test_camera_close_frees_renderer_and_is_idempotent():
    model = mujoco.MjModel.from_xml_path(STACK_SCENE)
    data = mujoco.MjData(model)
    _ready(model, data)
    cam = Camera(model, data, "tablecam", width=64, height=48)
    assert cam.renderer is not None
    cam.close()
    assert cam.renderer is None
    cam.close()                                           # second close must be safe


def test_scene_perception_close():
    model = mujoco.MjModel.from_xml_path(STACK_SCENE)
    data = mujoco.MjData(model)
    _ready(model, data)
    perc = ScenePerception(model, data, "tablecam")
    assert perc.camera.renderer is not None
    perc.close()
    assert perc.camera.renderer is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
