"""Phase I / M5 — dynamic throwing into a bin.

Grasp a ball, plan the release by simulation-in-the-loop (size the swing from the
ballistic velocity, search the release step/speed against the *true* simulated
landing), throw, and land it in the narrow bin; refuse targets outside the throw
envelope. Slowish (live sim with a grasp + swing + release search).
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import PROJECT_ROOT, THROW_MULTI_SCENE
from openarm_control.throwing import ThrowController

SCENE = os.path.join(PROJECT_ROOT, "v2", "openarm_mujoco_v2", "throw_scene.xml")


def _setup():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _bin(model, data):
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin")].copy()


def test_throw_lands_in_bin():
    model, data = _setup()
    tc = ThrowController(model, data, ball="ball_orange")
    assert tc.grasp_ball(), "failed to grasp the ball"
    binpos = _bin(model, data)
    ok, landing = tc.throw(np.array([binpos[0], binpos[1], 0.06]))
    assert ok, f"throw refused: {landing}"
    err = np.linalg.norm(landing[:2] - binpos[:2])
    # lands inside the narrow bin (half-width 0.08) and settles low (didn't fly out)
    assert err < 0.08 and landing[2] < 0.20, f"ball not in bin: landed {landing}, err {err*1000:.0f} mm"
    # the sim-in-the-loop search predicts the landing it then reproduces (deterministic)
    assert tc.pred_err < 0.06, f"planned error too large: {tc.pred_err*1000:.0f} mm"
    assert np.linalg.norm(tc.pred_landing[:2] - landing[:2]) < 0.02, "execution diverged from plan"


def test_throw_refuses_unreachable_pose():
    """A bin far beyond reach has no valid release pose / needs an impossible speed
    -> refused at plan time (not attempted blindly)."""
    model, data = _setup()
    tc = ThrowController(model, data, ball="ball_orange")
    tc.grasp_ball()
    assert tc.plan(np.array([2.0, -0.10, 0.06])) is None
    assert "fast" in tc.reason or "unreachable" in tc.reason


def test_throw_refuses_out_of_envelope():
    """A pose-reachable but too-far bin (x=0.80) falls in the achievable-landing
    gap, so no release lands within the gate -> plan_release refuses."""
    model, data = _setup()
    tc = ThrowController(model, data, ball="ball_orange")
    tc.grasp_ball()
    assert tc.plan_release(np.array([0.80, -0.25, 0.06])) is None
    assert "envelope" in tc.reason


def test_multi_bin_throws():
    """In the 5-bin scene, throwing to two different bins (each from a clean reset)
    lands both balls inside their bins -- the multi-target throwing show."""
    for ball, binname in [("ball_orange", "bin0"), ("ball_blue", "bin3")]:
        model = mujoco.MjModel.from_xml_path(THROW_MULTI_SCENE)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(
            model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
        mujoco.mj_forward(model, data)
        tc = ThrowController(model, data, ball=ball)
        binpos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, binname)].copy()
        assert tc.grasp_ball(disable_table=False), f"failed to grasp {ball}"
        assert tc.plan_release(np.array([binpos[0], binpos[1], 0.06])) is not None, \
            f"{binname} refused: {tc.reason}"
        tc.execute()
        for _ in range(300):                               # settle in the bin
            mujoco.mj_step(model, data)
        bp = data.xpos[tc.ball_bid]
        err = np.linalg.norm(bp[:2] - binpos[:2])
        assert err < 0.06 and bp[2] < 0.10, \
            f"{ball} not settled in {binname}: {err*1000:.0f} mm, z={bp[2]:.3f}"


def test_swing_speed_is_a_range_knob():
    """Scaling the swing speed changes the release velocity (continuous range
    control) -- and does not saturate the joint-limit clip at nominal speed."""
    model, data = _setup()
    tc = ThrowController(model, data, ball="ball_orange")
    binpos = _bin(model, data)
    target = np.array([binpos[0], binpos[1], 0.06])
    assert tc.plan(target, speed=1.0) is not None
    q_rel = tc.plan_["q_rel"].copy()
    amp_1 = np.linalg.norm(tc.plan_["q_wind"] - q_rel)
    assert tc.plan(target, speed=1.5) is not None
    amp_15 = np.linalg.norm(tc.plan_["q_wind"] - q_rel)
    # a faster swing winds up wider (further from the release config), i.e. the
    # amplitude scales with speed instead of being pinned at the joint limits
    assert amp_15 > amp_1 + 1e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
