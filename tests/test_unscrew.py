"""S2 — single-arm bottle opening (unscrew a threaded cap).

The bottle is held fixed in a stand and the RIGHT arm unscrews the threaded cap
over several re-gripping bursts, then lifts it clear. (Two 7-DOF arms working at
one small bottle unavoidably collide, so the holding is done by the clamp; this
gives the working arm full room and a collision-free, proper multi-turn unscrew.)
The cap is jointed (turn hinge + lift slide), so it can't be knocked off and rises
as it turns. The test checks the cap is unscrewed (turned a lot + risen).
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import UNSCREW_SCENE
from openarm_control.bimanual import UnscrewTask


def test_scene_compiles_with_bottle_and_cap():
    model = mujoco.MjModel.from_xml_path(UNSCREW_SCENE)
    assert model.nq > 0 and model.nu > 0
    for b in ("bottle", "cap"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_right_cap") >= 0
    for j in ("cap_turn", "cap_lift"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cap_lift_drive") >= 0


def test_unscrews_and_lifts_cap():
    model = mujoco.MjModel.from_xml_path(UNSCREW_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    cap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cap")
    turn_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cap_turn")]
    capz0 = data.xpos[cap][2]

    ok = UnscrewTask(model, data).run()
    turned = abs(np.degrees(data.qpos[turn_qadr]))
    for _ in range(60):
        mujoco.mj_step(model, data)

    assert ok, "unscrew task did not complete"
    assert np.isfinite(data.qpos).all(), "simulation went unstable"
    assert turned > 100.0, f"cap barely turned: {turned:.0f} deg"
    assert data.xpos[cap][2] - capz0 > 0.008, "cap did not rise / lift off"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
