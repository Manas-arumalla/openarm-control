"""Phase 3c — deformable cloth folding.

A flex-grid cloth on the table; the robot grasps a corner (a weld to the corner
vertex body) and carries it across to fold the sheet. Verifies the cloth scene
simulates stably and that the fold actually reduces the cloth's extent (it folded).
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import CLOTH_SCENE
from openarm_control.cloth import ClothFoldController, set_ready


def _load():
    model = mujoco.MjModel.from_xml_path(CLOTH_SCENE)
    data = mujoco.MjData(model)
    return model, data


def test_cloth_scene_simulates_stably():
    """The finer 9x9 flex cloth settles flat on the table without blowing up, and its
    corners (0, 8, 72, 80) are distinct weldable bodies."""
    model, data = _load()
    set_ready(model, data, settle=400)
    assert np.all(np.isfinite(data.qpos)), "cloth simulation diverged"
    cf = ClothFoldController(model, data)
    verts = cf.cloth_vertices()
    assert len(verts) == 81, f"expected a 9x9 cloth, got {len(verts)} vertices"
    assert np.all(verts[:, 2] < 0.45) and np.all(verts[:, 2] > 0.38), "cloth not resting on the table"
    assert (verts[:, 2].max() - verts[:, 2].min()) < 0.02, "cloth not settled flat"
    for c in ("0", "8", "72", "80"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_right_{c}") >= 0


def test_fold_reduces_cloth_extent():
    """Grasping a corner and folding it onto the opposite corner of the near edge
    folds the cloth over -- its span across the fold must shrink.

    The gate is deliberately loose (10% reduction): deformable dynamics are
    chaotic, so the exact fold depth varies across MuJoCo/numpy builds (44%
    reduction on one platform, ~16% on another, from identical code and
    settings). A broken fold (grasp misses, carry no-ops) leaves the span at
    ~100% or more, which this still catches. The achieved fold strength is
    measured and reported by benchmarks/openarm_bench.py, not gated here."""
    model, data = _load()
    set_ready(model, data, settle=400)
    cf = ClothFoldController(model, data)
    before = cf.cloth_vertices()
    span0 = before[:, 1].max() - before[:, 1].min()
    cf.fold("cloth_0", cf.corner_xy("cloth_8"))
    after = cf.cloth_vertices()
    assert np.all(np.isfinite(data.qpos)), "cloth diverged during the fold"
    span1 = after[:, 1].max() - after[:, 1].min()
    assert span1 < span0 * 0.90, f"cloth not folded: y-span {span0*1000:.0f} -> {span1*1000:.0f} mm"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
