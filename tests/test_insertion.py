"""Phase I / M6 — peg-in-hole insertion.

Grasp a cylindrical peg and thread it into a socket with a precise vertical
descent (a few mm of clearance), driven directly (the ``insert`` primitive) and
by language ("insert the peg into the socket"). Slowish (live grasp + insert).
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import PEG_SOCKET_SCENE, PEG_SQUARE_SCENE, PEG_CUBOID_SCENE
from openarm_control.vision.scene_perception import ScenePerception
from openarm_control.agent.executor import TaskExecutor
from openarm_control.agent.session import ManipulationSession
from openarm_control.agent.commands import parse_command

PEG_HALF_LEN = 0.045          # matches peg_socket_scene.xml


def _setup():
    model = mujoco.MjModel.from_xml_path(PEG_SOCKET_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    perc = ScenePerception(model, data, cam_name="tablecam")
    return model, data, perc


def _pos(model, data, name):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def _assert_inserted(model, data):
    """The peg sits centred in the socket, upright, with its base near the table."""
    for _ in range(400):
        mujoco.mj_step(model, data)
    peg, sock = _pos(model, data, "peg"), _pos(model, data, "socket")
    dxy = np.linalg.norm(peg[:2] - sock[:2])
    R = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")].reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    base_z = peg[2] - PEG_HALF_LEN
    assert dxy < 0.015, f"peg not centred in socket: {dxy*1000:.0f} mm"
    assert tilt < 15, f"peg not upright: {tilt:.0f} deg"
    assert base_z < 0.42, f"peg not seated (base z={base_z:.3f}, table top 0.40)"


def test_parse_insert_commands():
    i = parse_command("insert the blue peg into the socket")
    assert i.action == "insert" and "blue" in i.target and i.destination == "socket"
    assert parse_command("insert the peg into the hole").action == "insert"
    assert parse_command("plug the peg in").action == "insert"


def test_insert_peg():
    """The executor primitive grasps the peg and threads it into the socket."""
    model, data, perc = _setup()
    ex = TaskExecutor(model, data, perception=perc, graspables=["peg"], bin_body="socket")
    ok, msg = ex.insert("blue peg", socket_body="socket")
    assert ok, f"insert failed: {msg}"
    assert ex.held_body is None
    _assert_inserted(model, data)


def test_insert_by_language():
    """A natural-language insert command runs end-to-end."""
    model, data, perc = _setup()
    sess = ManipulationSession(model, data, perception=perc, graspables=["peg"], bin_body="socket")
    ok, msg = sess.do("insert the blue peg into the socket")
    assert ok, f"insert command failed: {msg}"
    _assert_inserted(model, data)


def _assert_aligned_insert(scene, method, sym, half_h=0.035):
    """Grasp a prismatic peg, align it to a rotated hole of cross-section symmetry
    ``sym`` (deg), thread it in; assert centred, upright, orientation-matched, seated."""
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "socket")
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    Rs = data.xmat[sid].reshape(3, 3)
    syaw = np.degrees(np.arctan2(Rs[1, 0], Rs[0, 0])) % sym
    spos = data.xpos[sid].copy()

    ex = TaskExecutor(model, data, graspables=["peg"], bin_body="socket")
    ok, msg = getattr(ex, method)("peg", "socket")
    assert ok, f"{method} failed: {msg}"
    for _ in range(400):
        mujoco.mj_step(model, data)
    p = data.xpos[pid].copy()
    R = data.xmat[pid].reshape(3, 3)
    dxy = np.linalg.norm(p[:2] - spos[:2])
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    pyaw = np.degrees(np.arctan2(R[1, 0], R[0, 0])) % sym
    yawerr = min(abs(pyaw - syaw), sym - abs(pyaw - syaw))
    assert dxy < 0.013, f"block not centred in socket: {dxy*1000:.0f} mm"
    assert tilt < 15, f"block not upright: {tilt:.0f} deg"
    assert yawerr < 15, f"block not aligned to the hole: {yawerr:.0f} deg yaw error"
    assert p[2] - half_h < 0.43, f"block not seated (base z={p[2]-half_h:.3f})"


def test_insert_square_into_square_hole():
    """A square peg snaps to the nearest reachable 90-deg orientation of a rotated
    square hole and threads in (4-fold-symmetric alignment)."""
    _assert_aligned_insert(PEG_SQUARE_SCENE, "insert_square", sym=90)


def test_insert_cuboid_into_rotated_hole():
    """A rectangular block is grasped, yaw-aligned to a rotated rectangular slot,
    and threaded in — seated, upright, and orientation-matched (180-deg symmetry)."""
    _assert_aligned_insert(PEG_CUBOID_SCENE, "insert_cuboid", sym=180)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
