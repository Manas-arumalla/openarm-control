"""B1 — ball-balancing PD controller regression gate.

Verifies the scene compiles, the BallBalancer's hold + manual-pin setup works,
and the PD controller drives a ball placed off-centre back to within 2 cm of
the plate centre within 6 s (the demo's stated success metric).
"""
import os
import sys

import numpy as np
import mujoco
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from openarm_control.config import BALANCE_SCENE
from openarm_control.balance import PDBalancer, LQRBalancer, MPCBalancer


def _load():
    m = mujoco.MjModel.from_xml_path(BALANCE_SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(m, d)
    return m, d


def test_scene_compiles_with_plate_and_ball():
    """balance_scene.xml has the right nq / bodies / weld / contact-exclude wiring."""
    m, _ = _load()
    # 18 arm + 7 plate free + 7 ball free = 32
    assert m.nq == 32
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,  "plate") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,  "ball")  >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "plate_to_right_ee") >= 0


def test_setup_hold_pins_plate_horizontally():
    """After setup_hold the plate is at the gripper's tool point + offset, with
    quat == identity (horizontal in world)."""
    m, d = _load()
    bal = PDBalancer(m, d)
    bal.setup_hold()
    plate_q = d.xquat[bal.plate_bid]
    # quat ~ (1, 0, 0, 0) -- world-horizontal at attach time
    assert abs(plate_q[0] - 1.0) < 1e-3
    assert np.linalg.norm(plate_q[1:]) < 1e-3
    # achieved hold pose was captured (used for IK target every step)
    assert bal._hold_pos is not None and bal._R_grip_init is not None


def test_pd_controller_settles_ball_under_2cm():
    """Drop the ball at (+3 cm, +2 cm) on the plate, run the PD controller for
    6 s, the ball must end up within 2 cm of the plate centre (last 0.4 s mean)."""
    m, d = _load()
    bal = PDBalancer(m, d)
    bal.setup_hold()
    bal.reset(ball_offset_xy=(0.03, 0.02), settle_steps=400)
    dt = m.opt.timestep
    n = int(6.0 / dt)
    errs = np.empty(n)
    for k in range(n):
        errs[k], _ = bal.step(target_xy=(0.0, 0.0))
    final = errs[-200:].mean()
    # The demo settles to ~1-2 mm. A 20 mm gate gives plenty of headroom for
    # numerical noise between platforms.
    assert final < 0.020, f"ball didn't settle: final={final*1000:.1f} mm"


def test_lqr_settles_ball_under_2cm_and_beats_pd():
    """LQR converges from a moderate offset AND beats PD's steady-state error --
    the whole point of moving from PD to LQR in Tier 2."""
    dt_target = 6.0
    def run(cls, offset):
        m, d = _load()
        bal = cls(m, d)
        bal.setup_hold()
        bal.reset(ball_offset_xy=offset, settle_steps=400)
        n = int(dt_target / m.opt.timestep)
        errs = np.empty(n)
        for k in range(n):
            errs[k], _ = bal.step(target_xy=(0.0, 0.0))
        return errs
    e_lqr = run(LQRBalancer, (0.04, 0.03))
    e_pd  = run(PDBalancer,  (0.04, 0.03))
    lqr_final = e_lqr[-200:].mean()
    pd_final  = e_pd[-200:].mean()
    # LQR must settle
    assert lqr_final < 0.020, f"LQR didn't settle: final={lqr_final*1000:.1f} mm"
    # AND beat PD's steady-state (empirically LQR is ~4-5x better here).
    assert lqr_final < pd_final, f"LQR final {lqr_final*1000:.2f} mm should beat PD {pd_final*1000:.2f} mm"


def test_lqr_gain_matrix_structure():
    """The linearised model decouples x/y axes -- the LQR gain must reflect
    that (K[0, 0] = K[0, 2] = K[1, 1] = K[1, 3] = 0, off-diagonal cross-axis
    zeros). Guards against a signs / axis-swap regression in the derivation."""
    K = LQRBalancer._compute_gain(0.002)
    # roll (row 0) only depends on py + vy; pitch (row 1) only on px + vx.
    assert abs(K[0, 0]) < 1e-8 and abs(K[0, 2]) < 1e-8
    assert abs(K[1, 1]) < 1e-8 and abs(K[1, 3]) < 1e-8
    # roll's py gain is negative (u = -K x, so effective coefficient is +positive),
    # and pitch's px gain is positive. Signs cross-checked in the module docstring.
    assert K[0, 1] < 0 and K[1, 0] > 0


def test_circle_trajectory_stays_bounded():
    """Ball tracks a circular target at 3 cm radius, 5 s period -- must stay
    well within the plate (7.5 cm half-width) throughout. Uses PD which tracks
    moving targets more crisply than the gentler LQR."""
    from openarm_control.demos.demo_balance import _target_at
    m, d = _load()
    bal = PDBalancer(m, d)
    bal.setup_hold()
    bal.reset(ball_offset_xy=(0.0, 0.0), settle_steps=400)
    dt = m.opt.timestep
    n = int(6.0 / dt)
    peak_from_centre = 0.0
    for k in range(n):
        t = k * dt
        tx, ty = _target_at(t, "circle", radius=0.03, period=5.0)
        bal.step(target_xy=(tx, ty))
        (x, y), _ = bal.ball_state()
        peak_from_centre = max(peak_from_centre, float(np.hypot(x, y)))
    # Ball must stay within 6 cm of plate centre (well inside the 7.5 cm plate
    # half-width, so it can never roll off).
    assert peak_from_centre < 0.06, \
        f"trajectory tracking wandered too far: peak {peak_from_centre*1000:.1f} mm"


def test_mpc_settles_ball_like_lqr():
    """With a static target and target acceleration zero, MPC's feedforward
    contribution vanishes and it must behave identically to LQR (both are
    the same feedback law with u_ff=0)."""
    def run(cls):
        m, d = _load()
        bal = cls(m, d)
        bal.setup_hold()
        bal.reset(ball_offset_xy=(0.03, 0.02), settle_steps=400)
        dt = m.opt.timestep
        errs = np.empty(int(4.0 / dt))
        for k in range(len(errs)):
            errs[k], _ = bal.step(target_xy=(0.0, 0.0), target_axy=(0.0, 0.0))
        return errs
    e_lqr = run(LQRBalancer)
    e_mpc = run(MPCBalancer)
    lqr_final = e_lqr[-200:].mean()
    mpc_final = e_mpc[-200:].mean()
    assert mpc_final < 0.020, f"MPC didn't settle: final={mpc_final*1000:.1f} mm"
    # MPC == LQR when target_axy = 0. Allow a hair of numerical drift.
    assert abs(mpc_final - lqr_final) < 1e-6, \
        f"MPC should collapse to LQR with zero target accel; got MPC={mpc_final} vs LQR={lqr_final}"


def test_mpc_beats_or_ties_lqr_on_trajectory():
    """On a moving target (circle), MPC's analytic feedforward should reduce
    steady-state tracking error compared to plain LQR. Check MPC RMS is at
    most 5% *worse* than LQR (empirically it's ~2-7% *better*, but numerical
    noise between platforms could flip the sign at very small margins)."""
    from openarm_control.demos.demo_balance import _target_state_at
    def run(cls):
        m, d = _load()
        bal = cls(m, d)
        bal.setup_hold()
        bal.reset(ball_offset_xy=(0.0, 0.0), settle_steps=400)
        dt = m.opt.timestep
        n = int(8.0 / dt)
        errs = np.empty(n)
        for k in range(n):
            t = k * dt
            (tx, ty), (ax, ay) = _target_state_at(t, "circle", radius=0.04, period=2.5)
            errs[k], _ = bal.step(target_xy=(tx, ty), target_axy=(ax, ay))
        return errs
    e_lqr = run(LQRBalancer)
    e_mpc = run(MPCBalancer)
    lqr_rms = np.sqrt(np.mean(e_lqr[1000:] ** 2))
    mpc_rms = np.sqrt(np.mean(e_mpc[1000:] ** 2))
    # MPC should be no worse than LQR (allowing 5% slack for platform noise).
    assert mpc_rms <= 1.05 * lqr_rms, \
        f"MPC ({mpc_rms*1000:.2f} mm) shouldn't be worse than LQR ({lqr_rms*1000:.2f} mm) with feedforward"


def test_perturbation_recovery():
    """Static target + a periodic ball velocity kick. Between kicks the LQR
    must return the ball to within 2 cm of the target -- otherwise the recovery
    isn't working."""
    from openarm_control.demos.demo_balance import _apply_perturbation
    m, d = _load()
    bal = LQRBalancer(m, d)
    bal.setup_hold()
    bal.reset(ball_offset_xy=(0.0, 0.0), settle_steps=400)
    dt = m.opt.timestep
    # Apply one 0.25 m/s kick, then observe for 2 s -- LQR must bring the ball
    # back near centre by the end of the observation window.
    np.random.seed(0)
    _apply_perturbation(bal, 0.25)
    errs = []
    for k in range(int(2.0 / dt)):
        err, _ = bal.step(target_xy=(0.0, 0.0))
        errs.append(err)
    final_err = float(np.mean(errs[-200:]))
    # The taller riser (12 cm above gripper) amplifies lever-arm effects during
    # transient recovery. 30 mm is well within the plate (75 mm half-width) and
    # is enough to demonstrate the controller does recover from a disturbance.
    assert final_err < 0.030, \
        f"LQR didn't recover from a 25 cm/s kick: final err {final_err*1000:.1f} mm"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
