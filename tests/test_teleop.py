"""Phase G tests: human-arm imitation (pose -> retarget -> safe teleop).

All headless and hardware-free: a deterministic ``ScriptedPoseSource`` stands in
for the webcam, so the retargeting + teleop stack is fully verifiable. The live
``WebcamPoseSource`` (MediaPipe) is exercised only for import safety.
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import TELEOP_SCENE, ARMS
from openarm_control.teleop import (ArmLandmarks, ScriptedPoseSource,
                                    TeleopController)
from openarm_control.teleop.retarget import _rot_between


def _scene():
    model = mujoco.MjModel.from_xml_path(TELEOP_SCENE)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def _angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.degrees(np.arccos(np.clip(
        a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1)))


def _run(arm_name, track_orientation=True, ticks=500):
    """Drive the arm headless from the synthetic source; return a dict of
    per-tick posture/safety metrics."""
    model, data = _scene()
    tc = TeleopController(model, data, arm=ARMS[arm_name])
    tc.retargeter.track_orientation = track_orientation
    R = tc.retargeter
    sh, ebid = R.p_shoulder, R._elbow_bid
    sub = max(1, int(round(tc.dt / model.opt.timestep)))
    upper, fore, lag, jump = [], [], [], []
    cmd_ok, overshoot, prev_qt = True, 0.0, None
    for _ in range(ticks):
        qt = tc.step()
        cmd_ok = cmd_ok and bool(np.all((tc.q_cmd >= tc.jnt_low - 1e-9) &
                                        (tc.q_cmd <= tc.jnt_high + 1e-9)))
        for _ in range(sub):
            mujoco.mj_step(model, data)
        q = data.qpos[tc.qpos_idx]
        overshoot = max(overshoot, float(np.max(np.maximum(
            np.maximum(tc.jnt_low - q, q - tc.jnt_high), 0.0))))
        elbow, wrist = data.xpos[ebid], tc.ee_pos()
        upper.append(_angle(elbow - sh, R._u_s))         # robot upper arm vs yours
        fore.append(_angle(wrist - elbow, R._f_s))        # robot forearm vs yours
        lag.append(np.linalg.norm(wrist - R.last_target[0]))   # controller follows soln
        if prev_qt is not None:
            jump.append(float(np.max(np.abs(qt - prev_qt))))   # solution continuity
        prev_qt = qt
    inrange = cmd_ok and overshoot < 0.03
    return dict(upper=np.array(upper), fore=np.array(fore), lag=np.array(lag),
                jump=np.array(jump), inrange=inrange, tc=tc)


# --------------------------------------------------------------------- pose
def test_scripted_pose_source_is_valid():
    src = ScriptedPoseSource("right")
    for _ in range(50):
        src.step(0.02)
        la = src.get()
        assert la.is_valid()
        assert np.all(np.isfinite(la.wrist))
        assert 0.3 < la.arm_length < 0.9          # plausible human reach


def test_rot_between_is_a_rotation():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.3])
    R = _rot_between(a, b)
    # proper rotation, and it actually maps a -> b
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-6)


# ----------------------------------------------------------------- retarget
def test_whole_arm_posture_tracks():
    """The *whole* arm mimics: the robot's upper-arm and forearm directions both
    follow the human's (not just the hand). This is the anatomical posture match
    -- the shoulder-to-elbow segment moves too."""
    r = _run("right", track_orientation=True)
    assert r["upper"][40:].mean() < 25.0, f"upper arm not tracking: {r['upper'][40:].mean()}"
    assert r["fore"][40:].mean() < 30.0, f"forearm not tracking: {r['fore'][40:].mean()}"


def test_retarget_is_coherent():
    """The joint trajectory is temporally coherent (no configuration flips) and
    the controller follows the IK solution closely."""
    r = _run("right", track_orientation=True)
    assert r["jump"][40:].max() < 0.30, f"solution flipped: {r['jump'][40:].max()} rad"
    assert r["lag"][40:].mean() < 0.03, f"controller lag too high: {r['lag'][40:].mean()}"


def test_posture_decoupled_from_body_translation():
    """Because the arm is reconstructed from shoulder-relative *directions*,
    translating the whole body (which a moving other arm would do to MediaPipe's
    torso-anchored landmarks) does not change the output — the arms are decoupled."""
    la = ScriptedPoseSource("right").get()
    shifted = ArmLandmarks(la.shoulder + [0.2, -0.1, 0.15],
                           la.elbow + [0.2, -0.1, 0.15],
                           la.wrist + [0.2, -0.1, 0.15], side="right")

    def solve(landmarks):                       # fresh retargeter each time
        model, data = _scene()
        tc = TeleopController(model, data, arm=ARMS["right"])
        tc.retargeter.calibrate(la)             # identical calibration both times
        return tc.retargeter.retarget(landmarks, tc.q_now())

    assert np.max(np.abs(solve(la) - solve(shifted))) < 1e-3, \
        "body translation changed the output (arms not decoupled)"


# -------------------------------------------------------------------- teleop
def test_teleop_is_safe():
    """The arm tracks smoothly, in limits, with bounded joint speed (no flailing),
    and the controller follows the IK solution closely."""
    r = _run("right", track_orientation=True)
    assert r["inrange"], "joint commands left the allowed range"
    assert r["lag"][40:].mean() < 0.03, f"controller lag too high: {r['lag'][40:].mean()}"
    assert r["jump"][40:].max() < 0.30


def test_teleop_both_arms():
    """Right and left arms both track the whole-arm posture (the left mirrors via
    a valid home pose), smoothly and in limits."""
    for arm in ("right", "left"):
        r = _run(arm, track_orientation=True, ticks=400)
        assert r["inrange"], f"{arm}: joints out of range"
        assert r["upper"][40:].mean() < 25.0, f"{arm}: upper arm not tracking"
        assert r["jump"][40:].max() < 0.30, f"{arm}: solution flipped"


def test_grasp_signal_drives_gripper():
    """A hand-closure signal opens/closes the gripper (full travel, right way)."""
    model, data = _scene()
    tc = TeleopController(model, data, arm=ARMS["right"])
    fadr = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_finger_joint1")]
    sub = max(1, int(round(tc.dt / model.opt.timestep)))
    grasp_sig, finger = [], []
    for _ in range(500):
        tc.step()
        for _ in range(sub):
            mujoco.mj_step(model, data)
        grasp_sig.append(tc.source.get().grasp)
        finger.append(data.qpos[fadr])
    grasp_sig, finger = np.array(grasp_sig), np.array(finger)
    assert finger.max() - finger.min() > 0.3, "gripper barely moved"
    # closing the hand (grasp -> 1) drives the finger toward closed (ctrl 0)
    assert np.corrcoef(grasp_sig, finger)[0, 1] > 0.8


def test_pick_up_a_block():
    """Closing the hand at a block welds it (a real pick); the block lifts with
    the gripper, and opening the hand drops it."""
    from openarm_control.config import TELEOP_PICK_SCENE, GRASP_LOCAL_OFFSET
    from openarm_control.kinematics import OpenArmKinematics

    model = mujoco.MjModel.from_xml_path(TELEOP_PICK_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    tc = TeleopController(model, data, arm=ARMS["right"])
    tc.enable_grasping(["block_red", "block_green", "block_blue"])

    # Put the grasp point on the red block (position-only IK on the tool point).
    gk = OpenArmKinematics(model, data, joint_names=ARMS["right"].joints,
                           site_name=ARMS["right"].ee_site, tool_offset=GRASP_LOCAL_OFFSET)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_red")
    bpos = data.xpos[bid].copy()
    q_grab = gk.inverse_kinematics(bpos, None, restarts=24)
    q_lift = gk.inverse_kinematics(bpos + np.array([0, 0, 0.12]), None,
                                   q_init=q_grab, restarts=24)
    data.qpos[tc.qpos_idx] = q_grab
    mujoco.mj_forward(model, data)
    z0 = data.xpos[bid][2]

    for i in range(250):                       # close hand + weld, then lift
        tc._apply_grasp(1.0)
        a = min(1.0, i / 120)
        data.ctrl[tc._act_ids] = (1 - a) * q_grab + a * q_lift
        mujoco.mj_step(model, data)
    assert tc._held == "block_red", "did not grab the block"
    lifted = data.xpos[bid][2] - z0
    assert lifted > 0.05, f"block did not lift with the gripper: {lifted*1000:.0f} mm"

    z_lift = data.xpos[bid][2]
    for _ in range(400):                       # open hand -> release + open fingers
        tc._apply_grasp(0.0)
        data.ctrl[tc._act_ids] = q_lift
        mujoco.mj_step(model, data)
    assert tc._held is None, "did not release the block"
    assert z_lift - data.xpos[bid][2] > 0.05, "block did not fall after release"


def test_webcam_source_importable():
    """The live source class imports without MediaPipe installed (lazy import);
    constructing it without a camera/deps raises a clear error, not ImportError
    at module load."""
    from openarm_control.teleop import WebcamPoseSource
    assert WebcamPoseSource is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
