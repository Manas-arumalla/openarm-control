"""Airborne ball-catching tests (research-grade MPC catcher) — headless.

Covers the pipeline pieces (ballistic Kalman estimator, interception solver,
orientation/trajectory maths) and an end-to-end check that the arm DYNAMICALLY
catches balls thrown on random ballistic arcs: it must move to meet the ball
(not be pre-positioned at the arrival point) and achieve a real, clean grasp.
"""
import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import CATCH_SCENE
from openarm_control.grasp import topdown_orientation
from openarm_control.kinematics import orientation_error
from openarm_control.config import CATCH_BIMANUAL_SCENE, CATCH_TWOBALL_SCENE
from openarm_control.catching import (
    BallisticKalmanFilter, CatchController,
    BimanualCatchController, TwoBallCatchController, MultiBallTracker,
    look_at_orientation, quintic, sample_throw, sample_throw_bimanual,
)
from openarm_control.vision import BallPerception, MultiBallPerception

G = np.array([0.0, 0.0, -9.81])


def _ball_addrs(model):
    j = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    return model.jnt_qposadr[j], model.jnt_dofadr[j]


# ----------------------------------------------------------------- unit maths
def test_look_at_reduces_to_topdown():
    R = look_at_orientation((0.0, 0.0, -1.0))
    assert np.allclose(R, topdown_orientation(0.0), atol=1e-9)


def test_look_at_points_approach_axis():
    # The gripper approach axis is local -z; it must align with approach_dir.
    for d in [(1, 0, 0), (0, -1, 0), (0.4, 0.0, 0.6)]:
        R = look_at_orientation(d)
        d = np.array(d, float); d /= np.linalg.norm(d)
        assert np.allclose(-R[:, 2], d, atol=1e-9)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)   # orthonormal


def test_quintic_boundary_conditions():
    q0 = np.zeros(3); qd0 = np.array([0.2, 0.0, -0.1])
    qf = np.array([1.0, -0.5, 0.3]); qdf = np.array([0.5, 0.0, -0.5])
    q, qd, _ = quintic(0.0, 1.0, q0, qd0, qf, qdf)
    assert np.allclose(q, q0) and np.allclose(qd, qd0)
    q, qd, _ = quintic(1.0, 1.0, q0, qd0, qf, qdf)
    assert np.allclose(q, qf) and np.allclose(qd, qdf)


# ------------------------------------------------------------ state estimation
def test_kalman_predicts_parabola():
    p0, v0 = np.array([1.1, -0.2, 1.05]), np.array([-2.0, 0.1, 1.4])
    kf = BallisticKalmanFilter(G, pos_noise=1e-4)
    for k in range(20):
        t = k * 0.002
        kf.observe(t, p0 + v0 * t + 0.5 * G * t**2)
    assert kf.ready
    # kf.vel is the velocity at the latest observation time (v0 + g*t), not v0.
    assert np.allclose(kf.vel, v0 + G * kf.t, atol=0.05)
    # The prediction that matters for catching: where the ball will be ahead.
    t_ahead = 0.30
    truth = p0 + v0 * t_ahead + 0.5 * G * t_ahead**2
    assert np.linalg.norm(kf.position_at(t_ahead) - truth) < 5e-3


# --------------------------------------------------------------- interception
def test_interception_is_reachable_and_future():
    model = mujoco.MjModel.from_xml_path(CATCH_SCENE)
    data = mujoco.MjData(model)
    catcher = CatchController(model, data)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    q_now = data.qpos[catcher.qpos_i].copy()

    # A ball whose arc passes through the reachable volume.
    A, L, Tf = np.array([0.40, -0.22, 0.95]), np.array([1.2, -0.2, 1.1]), 0.5
    v0 = (A - L - 0.5 * G * Tf**2) / Tf
    kf = BallisticKalmanFilter(G, pos_noise=1e-4)
    for k in range(12):
        t = k * 0.002
        kf.observe(t, L + v0 * t + 0.5 * G * t**2)

    plan = catcher.solver.solve(kf, t_now=kf.t, q_now=q_now)
    assert plan is not None, "no interception found for a catchable throw"
    assert plan.t_catch > kf.t
    r = np.linalg.norm(plan.p - catcher.shoulder)
    assert 0.20 < r < 0.62


def test_interception_returns_none_when_unreachable():
    model = mujoco.MjModel.from_xml_path(CATCH_SCENE)
    data = mujoco.MjData(model)
    catcher = CatchController(model, data)
    q_now = data.qpos[catcher.qpos_i].copy()
    # A ball flying AWAY from the arm (never enters the workspace).
    kf = BallisticKalmanFilter(G, pos_noise=1e-4)
    p0, v0 = np.array([0.4, -0.2, 1.0]), np.array([3.0, 0.0, 0.5])
    for k in range(10):
        t = k * 0.002
        kf.observe(t, p0 + v0 * t + 0.5 * G * t**2)
    assert catcher.solver.solve(kf, t_now=kf.t, q_now=q_now) is None


# --------------------------------------------------------- end-to-end catching
def test_dynamic_clean_catch():
    """Over random ballistic throws the arm DYNAMICALLY catches the ball:
    it moves to intercept (it is not parked at the arrival point) and ends with
    a real, clean grasp (fingers closed on the ball, held, no wild rotation)."""
    model = mujoco.MjModel.from_xml_path(CATCH_SCENE)
    data = mujoco.MjData(model)
    bq, bdof = _ball_addrs(model)
    fadr = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_finger_joint1")]
    catcher = CatchController(model, data)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    rng = np.random.default_rng(3)

    N, clean, moved = 6, 0, 0
    preposition_gap = []
    for _ in range(N):
        mujoco.mj_resetDataKeyframe(model, data, key)
        L, v0 = sample_throw(rng, model.opt.gravity)
        data.qpos[bq:bq + 3] = L
        data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        data.qvel[bdof:bdof + 3] = v0
        catcher.reset()
        gp0 = catcher.grasp_pos()                   # hand position before any motion
        q0 = data.qpos[catcher.qpos_i].copy()
        max_dev, ori_at_catch = 0.0, None
        for _ in range(800):
            catcher.step()
            mujoco.mj_step(model, data)
            if not catcher.caught:
                max_dev = max(max_dev, np.max(np.abs(data.qpos[catcher.qpos_i] - q0)))
            elif ori_at_catch is None:              # gripper faced the ball at the catch
                _, R = catcher.king.forward_kinematics()
                ori_at_catch = np.degrees(np.linalg.norm(orientation_error(R, catcher.plan.R)))
        if not catcher.caught:
            continue
        moved += int(max_dev > 0.05)                # the arm actively moved
        # the hand was NOT already sitting at the ball's arrival point
        preposition_gap.append(np.linalg.norm(catcher.plan.p - gp0))
        near = np.linalg.norm(catcher.ball_pos() - catcher.grasp_pos())
        held = catcher.ball_pos()[2] > 0.45 and near < 0.10
        gripped = data.qpos[fadr] > -0.45           # fingers closed off the open stop
        if held and gripped and ori_at_catch < 20:
            clean += 1

    assert clean >= int(0.8 * N), f"clean catches {clean}/{N}"
    assert moved >= clean, "arm did not move to intercept (pre-positioned?)"
    assert np.mean(preposition_gap) > 0.04, "hand was already at the arrival point"


# ----------------------------------------------------------- vision perception
def test_vision_estimate_is_accurate():
    """Both RGB-D cameras see the ball and the fused 3D estimate (centre,
    radius-corrected) is within 2 cm of ground truth — and it is NOT the raw
    MuJoCo pose (perception, not cheating)."""
    model = mujoco.MjModel.from_xml_path(CATCH_SCENE)
    data = mujoco.MjData(model)
    bbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    bq, _ = _ball_addrs(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[bq:bq + 3] = [0.70, -0.15, 1.00]      # in view of both cameras
    mujoco.mj_forward(model, data)

    perc = BallPerception(model, data, ["ballcam0", "ballcam1"])
    est = perc.observe()
    assert est is not None
    assert len(perc.last_per_cam) == 2              # both cameras detected it
    assert np.linalg.norm(est - data.xpos[bbid]) < 0.02


def test_vision_driven_catch():
    """Driven ONLY by the two cameras (no ground-truth ball state), the arm still
    catches thrown balls cleanly."""
    model = mujoco.MjModel.from_xml_path(CATCH_SCENE)
    data = mujoco.MjData(model)
    bq, bdof = _ball_addrs(model)
    fadr = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_finger_joint1")]
    perc = BallPerception(model, data, ["ballcam0", "ballcam1"])
    catcher = CatchController(model, data, perception=perc, cam_period=5)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    rng = np.random.default_rng(1)

    N, clean = 3, 0
    for _ in range(N):
        mujoco.mj_resetDataKeyframe(model, data, key)
        L, v0 = sample_throw(rng, model.opt.gravity)
        data.qpos[bq:bq + 3] = L
        data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        data.qvel[bdof:bdof + 3] = v0
        catcher.reset()
        for _ in range(800):
            catcher.step()
            mujoco.mj_step(model, data)
        if catcher.caught:
            near = np.linalg.norm(catcher.ball_pos() - catcher.grasp_pos())
            if catcher.ball_pos()[2] > 0.45 and near < 0.10 and data.qpos[fadr] > -0.45:
                clean += 1
    assert clean >= 2, f"vision-driven clean catches {clean}/{N}"


# ------------------------------------------------------------------ bimanual
def test_bimanual_picks_correct_arm_and_no_collision():
    """Throws toward each side are caught by the arm on that side, and the two
    arms never collide (min grasp-point separation stays well clear)."""
    model = mujoco.MjModel.from_xml_path(CATCH_BIMANUAL_SCENE)
    data = mujoco.MjData(model)
    bq, bdof = _ball_addrs(model)
    bi = BimanualCatchController(model, data)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    rng = np.random.default_rng(0)

    caught = 0
    sides = ["right", "left", "right", "left"]
    for side in sides:
        mujoco.mj_resetDataKeyframe(model, data, key)
        L, v0, _ = sample_throw_bimanual(rng, model.opt.gravity, side=side)
        data.qpos[bq:bq + 3] = L
        data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        data.qvel[bdof:bdof + 3] = v0
        bi.reset()
        min_sep = np.inf
        for _ in range(800):
            bi.step()
            mujoco.mj_step(model, data)
            min_sep = min(min_sep, bi.arm_separation())
        assert min_sep > 0.05, f"arms collided (min sep {min_sep:.3f} m)"
        if bi.caught:
            caught += 1
            assert bi.active == side, f"{side} throw caught by {bi.active} arm"
    assert caught >= len(sides) - 1     # allow at most one miss


# ----------------------------------------------------------- two balls (MOT)
def test_multiball_tracker_associates_two_balls():
    """Two ballistic trajectories, detections handed in SHUFFLED order each frame:
    the tracker's two filters must each lock onto a distinct ball."""
    pA0, vA0 = np.array([0.5, -0.2, 1.0]), np.array([-1.5, 0.0, 1.0])
    pB0, vB0 = np.array([0.5, 0.2, 1.0]), np.array([-1.6, 0.0, 1.3])
    tr = MultiBallTracker(G, n=2)
    tlast = 0.0
    for k in range(24):
        t = k * 0.002; tlast = t
        a = pA0 + vA0 * t + 0.5 * G * t**2
        b = pB0 + vB0 * t + 0.5 * G * t**2
        tr.update(t, [a, b] if k % 2 else [b, a])      # shuffled order
    assert tr.ready()
    tA = pA0 + vA0 * tlast + 0.5 * G * tlast**2
    tB = pB0 + vB0 * tlast + 0.5 * G * tlast**2
    e0, e1 = tr.kfs[0].pos, tr.kfs[1].pos
    err = min(max(np.linalg.norm(e0 - tA), np.linalg.norm(e1 - tB)),
              max(np.linalg.norm(e0 - tB), np.linalg.norm(e1 - tA)))
    assert err < 0.02


def test_multiball_perception_finds_both():
    model = mujoco.MjModel.from_xml_path(CATCH_TWOBALL_SCENE)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("ball", "ball2")]
    js = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[model.jnt_qposadr[js[0]]:model.jnt_qposadr[js[0]] + 3] = [0.5, -0.2, 1.0]
    data.qpos[model.jnt_qposadr[js[1]]:model.jnt_qposadr[js[1]] + 3] = [0.5, 0.2, 1.0]
    mujoco.mj_forward(model, data)
    perc = MultiBallPerception(model, data, ["ballcam0", "ballcam1"], n_balls=2)
    ests = perc.observe()
    assert len(ests) == 2
    for bid in bids:                          # each truth has a nearby estimate
        assert min(np.linalg.norm(data.xpos[bid] - e) for e in ests) < 0.02


def test_twoball_dual_catch_no_collision():
    """Two balls thrown at once (one per side): both caught by distinct arms,
    arms never collide."""
    model = mujoco.MjModel.from_xml_path(CATCH_TWOBALL_SCENE)
    data = mujoco.MjData(model)
    js = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    (q1, d1), (q2, d2) = [(model.jnt_qposadr[j], model.jnt_dofadr[j]) for j in js[:2]]
    bi = TwoBallCatchController(model, data)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    rng = np.random.default_rng(1)

    both = 0
    for _ in range(3):
        mujoco.mj_resetDataKeyframe(model, data, key)
        L1, v1, _ = sample_throw_bimanual(rng, model.opt.gravity, side="right")
        L2, v2, _ = sample_throw_bimanual(rng, model.opt.gravity, side="left")
        data.qpos[q1:q1 + 3] = L1; data.qpos[q1 + 3:q1 + 7] = [1, 0, 0, 0]
        data.qpos[q2:q2 + 3] = L2; data.qpos[q2 + 3:q2 + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        data.qvel[d1:d1 + 3] = v1
        data.qvel[d2:d2 + 3] = v2
        bi.reset()
        min_sep = np.inf
        for _ in range(820):
            bi.step()
            mujoco.mj_step(model, data)
            min_sep = min(min_sep, bi.arm_separation())
        assert min_sep > 0.04, f"arms collided (min sep {min_sep:.3f} m)"
        if bi.num_caught == 2:
            both += 1
            assert bi.cr.caught_ball != bi.cl.caught_ball   # each grabbed its own ball
    assert both >= 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
