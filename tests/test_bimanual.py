"""Bimanual coordination tests (headless)."""
import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import (BIMANUAL_SCENE, BIMANUAL_STACK_SCENE,
                                     BIMANUAL_HANDOVER_SCENE, RIGHT_ARM, LEFT_ARM)
from openarm_control.bimanual import (BimanualController, ParallelSort, RelayHandoff,
                              BimanualStack, BimanualCoordinator, synchronized_move, mirror_config)
from openarm_control.grasp import topdown_orientation


def _load():
    model = mujoco.MjModel.from_xml_path(BIMANUAL_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _pos(model, data, name):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def _in_bin(model, data, block, bin_name):
    b = _pos(model, data, block)
    c = _pos(model, data, bin_name)
    return abs(b[0] - c[0]) < 0.06 and abs(b[1] - c[1]) < 0.06 and b[2] > 0.41


def test_bimanual_scene_loads():
    model, data = _load()
    assert model.nq == 46           # 18 arm + 4 free-joint blocks
    # both grippers have welds to every block
    welds = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e) for e in range(model.neq)}
    for arm in ("right", "left"):
        for blk in ("r1", "r2", "l1", "l2"):
            assert f"grasp_{arm}_{blk}" in welds


def test_parallel_sort_both_arms():
    """Both arms sort their side's blocks into matching bins simultaneously."""
    model, data = _load()
    task = ParallelSort(model, data)
    assert task.run()
    data.qfrc_applied[:] = 0
    for _ in range(800):
        mujoco.mj_step(model, data)
    for blk, bn in [("block_r1", "bin_r1"), ("block_r2", "bin_r2"),
                    ("block_l1", "bin_l1"), ("block_l2", "bin_l2")]:
        assert _in_bin(model, data, blk, bn), f"{blk} not in {bn}"


def test_synchronized_motion_is_mirrored():
    """The left arm exactly mirrors the right during synchronized motion."""
    model, data = _load()
    bi = BimanualController(model, data)
    bi.sync_ctrl()
    rk = bi.right.king
    lk = bi.left.king
    rq = rk.inverse_kinematics(np.array([0.28, -0.20, 0.55]),
                               target_mat=topdown_orientation(1.5),
                               q_init=data.qpos[rk.qpos_indices], restarts=0, rest_weight=0.0)
    synchronized_move(bi, np.array([rq]), 2.0)
    pr, _ = rk.forward_kinematics()
    pl, _ = lk.forward_kinematics()
    assert np.linalg.norm(pl - np.array([pr[0], -pr[1], pr[2]])) < 2e-3


def test_relay_handoff_no_arm_collision():
    """Coordinated hand-off completes with no right<->left arm penetration."""
    model, data = _load()
    relay = RelayHandoff(model, data)
    rg, lg = relay.bi.right_checker.arm_geoms, relay.bi.left_checker.arm_geoms
    orig = mujoco.mj_step
    worst = {"pen": 0.0}

    def monitored(m, d, *a, **k):
        orig(m, d)
        for ci in range(d.ncon):
            c = d.contact[ci]
            cross = (c.geom1 in rg and c.geom2 in lg) or (c.geom2 in rg and c.geom1 in lg)
            if cross and c.dist < 0:
                worst["pen"] = max(worst["pen"], -c.dist)

    mujoco.mj_step = monitored
    try:
        assert relay.run()
    finally:
        mujoco.mj_step = orig
    data.qfrc_applied[:] = 0
    for _ in range(800):
        mujoco.mj_step(model, data)
    assert _in_bin(model, data, "block_r1", "bin_l1")
    assert worst["pen"] < 2e-3, f"arms collided ({worst['pen']*1000:.1f} mm)"


def test_bimanual_stack_two_towers():
    """Both arms stack their top cube on their base cube simultaneously (two
    towers), each cube landing ~one cube up and aligned over its base."""
    model = mujoco.MjModel.from_xml_path(BIMANUAL_STACK_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    assert BimanualStack(model, data).run(), "bimanual stack run failed"
    data.qfrc_applied[:] = 0
    for _ in range(400):
        mujoco.mj_step(model, data)
    for top, base in (("box_green", "box_red"), ("box_orange", "box_blue")):
        t, b = _pos(model, data, top), _pos(model, data, base)
        dz, dxy = t[2] - b[2], np.linalg.norm(t[:2] - b[:2])
        assert 0.04 < dz < 0.06, f"{top} not stacked on {base}: dz={dz*1000:.0f} mm"
        assert dxy < 0.03, f"{top} not aligned over {base}: dxy={dxy*1000:.0f} mm"


def test_interactive_table_detection_and_delivery():
    """On the large interactive table, the camera detects all four blocks, and the
    coordinator delivers a right-side block (single arm) and a left-side block
    (hand-over) into the right bin."""
    from openarm_control.config import BIMANUAL_TABLE_SCENE
    from openarm_control.vision.scene_perception import ScenePerception
    model = mujoco.MjModel.from_xml_path(BIMANUAL_TABLE_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    perc = ScenePerception(model, data, "tablecam")
    colors = {o.label.split()[0] for o in perc.perceive()}
    assert {"red", "green", "blue", "orange"} <= colors, f"missed objects: {colors}"

    co = BimanualCoordinator(model, data)
    binxy = _pos(model, data, "bin_right")[:2]
    ok, msg = co.pick_place(_pos(model, data, "block_green")[:2], binxy, "block_green", place_z=0.52)
    assert ok and "right arm picked" in msg, msg
    ok, msg = co.pick_place(_pos(model, data, "block_red")[:2], binxy, "block_red", place_z=0.52)
    assert ok and "handed" in msg, msg
    data.qfrc_applied[:] = 0
    for _ in range(400):
        mujoco.mj_step(model, data)
    for blk in ("block_green", "block_red"):
        p = _pos(model, data, blk)
        assert np.linalg.norm(p[:2] - binxy) < 0.08 and p[2] < 0.55, f"{blk} not in bin: {np.round(p,3)}"


def test_bimanual_coordination_single_and_handover():
    """The coordinator routes each task to the right arm: the right arm handles the
    green cube on its side alone, and hands the left-side red cube over to itself to
    reach the right bin. Both end up in the bin."""
    model = mujoco.MjModel.from_xml_path(BIMANUAL_HANDOVER_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    co = BimanualCoordinator(model, data)
    binxy = _pos(model, data, "bin")[:2]

    ok, msg = co.pick_place(_pos(model, data, "box_green")[:2], binxy, "box_green", place_z=0.52)
    assert ok and "right arm picked and placed" in msg, msg     # right side -> single arm
    ok, msg = co.pick_place(_pos(model, data, "box_red")[:2], binxy, "box_red", place_z=0.52)
    assert ok and "handed" in msg, msg                          # left side -> hand-over to right

    data.qfrc_applied[:] = 0
    for _ in range(400):
        mujoco.mj_step(model, data)
    for cube in ("box_green", "box_red"):
        p = _pos(model, data, cube)
        assert np.linalg.norm(p[:2] - binxy) < 0.07 and p[2] < 0.55, \
            f"{cube} not delivered to the bin: {np.round(p, 3)}"


def _language_session():
    from openarm_control.config import BIMANUAL_TABLE_SCENE
    from openarm_control.vision import ScenePerception, ColorShapeDetector
    from openarm_control.agent.bimanual_session import BimanualSession
    model = mujoco.MjModel.from_xml_path(BIMANUAL_TABLE_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    perc = ScenePerception(model, data, "tablecam", detector=ColorShapeDetector())
    sess = BimanualSession(model, data, perc,
                           ["block_red", "block_green", "block_blue", "block_orange"])
    return model, data, sess


def test_parse_transfer_is_a_move_with_destination():
    """'transfer'/'hand over' parse as a relocation with a target and destination."""
    from openarm_control.agent.commands import parse_command
    i = parse_command("transfer the red block to the left bin")
    assert i.action == "move" and "red" in i.target and i.destination == "left"
    assert parse_command("move the green block to the right bin").destination == "right"


def test_language_grab_query_and_place_held():
    """'grab X' holds it with the better-placed arm (answered by a query), and
    'move it to the <side> bin' on the holding arm's side puts it in that bin."""
    model, data, sess = _language_session()
    ok, msg = sess.do("grab the red block")             # red is on the +y side -> left arm
    assert ok, msg
    assert sess.co.held is not None and sess.co.held["name"] == "left"
    ok, ans = sess.do("which arm is holding it?")
    assert ok and "left arm is holding" in ans
    ok, msg = sess.do("move it to the left bin")        # left bin is on the holding arm's side
    assert ok, msg
    assert sess.co.held is None
    data.qfrc_applied[:] = 0
    for _ in range(400):
        mujoco.mj_step(model, data)
    p = _pos(model, data, "block_red")
    assert _in_bin(model, data, "block_red", "bin_left"), f"red not in left bin: {np.round(p, 3)}"


def test_language_transfer_triggers_handover():
    """A one-shot 'transfer X to the far bin' auto-selects the arm and hands over when
    only the other arm can reach the destination (green is on -y, left bin is on +y)."""
    model, data, sess = _language_session()
    ok, msg = sess.do("transfer the green block to the left bin")
    assert ok, msg
    assert "handed" in msg, f"expected a hand-over: {msg}"
    data.qfrc_applied[:] = 0
    for _ in range(400):
        mujoco.mj_step(model, data)
    assert _in_bin(model, data, "block_green", "bin_left"), \
        f"green not delivered to left bin: {np.round(_pos(model, data, 'block_green'), 3)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
