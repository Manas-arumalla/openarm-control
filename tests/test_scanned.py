"""Phase I — scanned-object testing harness.

Real Google Scanned Objects on the large bimanual table: depth-based localisation
(objects stick up from the table) finds them, and the dual-arm coordinator
delivers one into a bin. Detection *labels* (open-vocab) are exercised by the live
demo, not here -- this test uses ``detector=None`` (pure geometric localisation)
so it needs no YOLO weights and stays fast.
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import V2_MODEL_DIR
from openarm_control.vision.scene_perception import SegmentClassifyPerception
from openarm_control.bimanual import BimanualCoordinator
from openarm_control.pick_and_place import TABLE_TOP_Z

SCENE = os.path.join(V2_MODEL_DIR, "scanned_table_scene.xml")
TRUE = {"mug": (0.20, 0.20), "elephant": (0.32, 0.06),
        "bowl": (0.32, -0.06), "clock": (0.20, -0.20)}


def _setup():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _pos(model, data, name):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def test_depth_localisation_finds_objects():
    """Depth segmentation locates most of the scanned objects (bins excluded)."""
    model, data = _setup()
    for _ in range(150):
        mujoco.mj_step(model, data)
    binL, binR = _pos(model, data, "bin_left")[:2], _pos(model, data, "bin_right")[:2]
    perc = SegmentClassifyPerception(model, data, "tablecam", detector=None,
                                     exclude_xy=[binL, binR])
    found = perc.perceive()
    assert len(found) >= 3, f"depth localisation found only {len(found)} objects"
    # every located point lies on the table near a real object, not on a bin
    for o in found:
        assert min(np.linalg.norm(o.position[:2] - np.array(t)) for t in TRUE.values()) < 0.06
        assert np.linalg.norm(o.position[:2] - binL) > 0.07
        assert np.linalg.norm(o.position[:2] - binR) > 0.07


def test_bimanual_delivers_scanned_object():
    """The coordinator grasps a scanned object (box collision) and bins it."""
    model, data = _setup()
    co = BimanualCoordinator(model, data)
    binR = _pos(model, data, "bin_right")[:2]
    clock = _pos(model, data, "clock")
    grasp_z = float(np.clip(clock[2] + 0.025, TABLE_TOP_Z + 0.02, TABLE_TOP_Z + 0.12))
    ok, msg = co.pick_place(clock[:2], binR, "clock", grasp_z=grasp_z, place_z=0.52, verbose=False)
    assert ok, f"could not deliver the scanned clock: {msg}"
    data.qfrc_applied[:] = 0
    for _ in range(300):
        mujoco.mj_step(model, data)
    p = _pos(model, data, "clock")
    assert np.linalg.norm(p[:2] - binR) < 0.09 and p[2] < 0.55, f"clock not in bin: {np.round(p,3)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
