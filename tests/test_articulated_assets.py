"""F1c — articulated-object assets (drawer / door / valve).

These are physics-only checks on the authored ``articulated_scene.xml``: the
scene compiles, exposes the three articulated joints with the right types and
ranges and both-arm grasp welds, sits still at rest, and each joint actually
moves (within its limits) when pushed. The manipulation skills that *operate*
these fixtures (open drawer/door, turn valve) are built on top in phase S3.
"""
import os
import sys

import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import ARTICULATED_SCENE

JT = {mujoco.mjtJoint.mjJNT_SLIDE: "slide", mujoco.mjtJoint.mjJNT_HINGE: "hinge"}
JOINTS = {"drawer_slide": "slide", "door_hinge": "hinge", "valve_turn": "hinge"}
WELDS = ["grasp_right_drawer", "grasp_left_drawer", "grasp_right_door",
         "grasp_left_door", "grasp_right_valve", "grasp_left_valve"]


def _load():
    model = mujoco.MjModel.from_xml_path(ARTICULATED_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _qadr(model, name):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]


def test_articulated_scene_compiles():
    """Scene loads with the three articulated joints (right type + range) and the
    both-arm grasp welds."""
    model, data = _load()
    assert model.nq > 0 and model.nu > 0
    for nm, want in JOINTS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        assert jid >= 0, f"missing joint {nm}"
        assert JT[model.jnt_type[jid]] == want, f"{nm} wrong joint type"
        lo, hi = model.jnt_range[jid]
        assert hi > lo, f"{nm} has empty range"
    for w in WELDS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, w) >= 0, f"missing weld {w}"


def test_articulated_fixtures_stable_at_rest():
    """With no applied force the fixtures stay put (no instability/explosion)."""
    model, data = _load()
    q0 = {nm: data.qpos[_qadr(model, nm)] for nm in JOINTS}
    for _ in range(150):
        mujoco.mj_step(model, data)
    drift = max(abs(data.qpos[_qadr(model, nm)] - q0[nm]) for nm in JOINTS)
    assert drift < 0.01, f"fixtures drifted {drift:.4f} at rest"


def test_articulated_joints_move_within_range():
    """Each joint articulates when pushed and stays inside its limits."""
    model, data = _load()
    pushes = {"drawer_slide": -8.0, "door_hinge": 2.0, "valve_turn": 1.0}
    for nm, f in pushes.items():
        mujoco.mj_resetDataKeyframe(
            model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
        mujoco.mj_forward(model, data)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        qadr, dadr = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
        lo, hi = model.jnt_range[jid]
        q_start = data.qpos[qadr]
        for _ in range(600):
            data.qfrc_applied[dadr] = f
            mujoco.mj_step(model, data)
        q_end = data.qpos[qadr]
        assert abs(q_end - q_start) > 0.03, f"{nm} did not articulate"
        assert (lo - 0.02) <= q_end <= (hi + 0.02), f"{nm} left its range: {q_end}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

def test_ready_keyframe_clear_of_arms():
    """The ready keyframe must not start with the arms penetrating any fixture
    (a raised fixture once intersected the parked gripper's resting zone --
    every skill then began in contact)."""
    model, data = _load()
    fixtures = {"cabinet_drawer", "drawer", "cabinet_door", "door", "valve", "valve_base"}
    bad = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
        if c.dist < -0.0005 and ("openarm" in b1 or "openarm" in b2) and (b1 in fixtures or b2 in fixtures):
            bad.append((b1, b2, round(float(c.dist) * 1000, 2)))
    assert not bad, f"arms start in contact with fixtures: {bad}"
