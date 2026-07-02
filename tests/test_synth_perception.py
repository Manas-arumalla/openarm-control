"""Phase 2 — perception hardening: segmentation-labeled synthetic data + multi-view.

These cover the *headless* machinery (no GPU training): segmentation-derived bounding
boxes, the YOLO-dataset writer under domain randomization, and multi-view detection
fusion. The actual detector fine-tuning is user-run (ultralytics), like RL/BC, so it
isn't tested here.
"""
import os
import sys

import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import BIMANUAL_TABLE_SCENE
from openarm_control.vision.synthgen import (SyntheticDataGenerator, TABLE_CLASSES,
                                             body_geom_ids, segmentation_bboxes)
from openarm_control.vision.multiview import MultiViewPerception
from openarm_control.vision import ColorShapeDetector

BLOCKS = ["block_red", "block_green", "block_blue", "block_orange"]


def test_segmentation_gives_per_object_bboxes():
    """The segmentation renderer yields an exact 2D box for each visible block."""
    model = mujoco.MjModel.from_xml_path(BIMANUAL_TABLE_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, height=240, width=320)
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "tablecam"))
    seg = r.render()
    geoms = {b: body_geom_ids(model, b) for b in BLOCKS}
    boxes = segmentation_bboxes(seg, geoms)
    r.close()
    assert set(boxes) == set(BLOCKS), f"missing boxes: {set(BLOCKS) - set(boxes)}"
    for b, (u0, v0, u1, v1) in boxes.items():
        assert 0 <= u0 < u1 <= 320 and 0 <= v0 < v1 <= 240, f"{b} bad bbox {(u0, v0, u1, v1)}"


def test_synthgen_writes_yolo_dataset(tmp_path):
    """Generating a small dataset writes images + valid YOLO labels + data.yaml."""
    gen = SyntheticDataGenerator(BIMANUAL_TABLE_SCENE, TABLE_CLASSES, width=160, height=120)
    out = str(tmp_path / "ds")
    yaml_path = gen.generate(out, n=6, seed=0)
    gen.close()
    assert os.path.exists(yaml_path)
    n_imgs = sum(len(os.listdir(os.path.join(out, "images", s))) for s in ("train", "val"))
    n_lbls = sum(len(os.listdir(os.path.join(out, "labels", s))) for s in ("train", "val"))
    assert n_imgs == 6 and n_lbls == 6
    # at least one label file has a valid YOLO line with an in-range class id
    nonempty = 0
    for s in ("train", "val"):
        for fn in os.listdir(os.path.join(out, "labels", s)):
            for line in open(os.path.join(out, "labels", s, fn)).read().splitlines():
                cls, cx, cy, bw, bh = line.split()
                assert 0 <= int(cls) < len(gen.class_names)
                assert all(0.0 <= float(x) <= 1.0 for x in (cx, cy, bw, bh))
                nonempty += 1
    assert nonempty > 0, "no labeled objects in any image"


def test_multiview_fuses_two_views():
    """Two camera views fuse to one detection per physical object (no duplicates),
    and a colour query grounds to the right object."""
    model = mujoco.MjModel.from_xml_path(BIMANUAL_TABLE_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    mv = MultiViewPerception(model, data, cam_names=("tablecam", "frontcam"),
                             detector=ColorShapeDetector())
    objs = mv.perceive()
    colors = {o.label.split()[0] for o in objs}
    assert len(objs) == 4, f"expected 4 fused objects, got {len(objs)}: {[o.label for o in objs]}"
    assert {"red", "green", "blue", "orange"} <= colors
    g = mv.ground("green block")
    mv.close()
    assert g is not None and "green" in g.label


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
