"""Ball-balancing demo (Phases B1-B3): keep a ping-pong ball on a plate the right
arm is holding.

Two controllers (shared plate-hold + tilt scaffold):
  - PD  (Tier 1) -- analytic proportional-derivative on ball xy + velocity.
  - LQR (Tier 2) -- discrete LQR on the linearised ball-on-plate dynamics.

Three scenarios (Tier 3):
  - static          -- hold the ball centred (target = plate origin).
  - trajectory      -- track a moving target (circle / figure-8) on the plate.
  - perturb         -- static target + periodic random velocity kicks to the ball.

    openarm balance                                # viewer, LQR, static
    openarm balance --controller pd                # viewer, PD, static
    openarm balance --trajectory circle            # viewer, LQR, ball traces a circle
    openarm balance --trajectory figure8 --radius 0.03 --period 6
    openarm balance --perturb                      # viewer, LQR, disturbance rejection
    openarm balance --headless                     # scripted report
    openarm balance --headless --controller both --trajectory circle
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import BALANCE_SCENE
from openarm_control.balance import PDBalancer, LQRBalancer, MPCBalancer

CONTROLLERS = {"pd": PDBalancer, "lqr": LQRBalancer, "mpc": MPCBalancer}
TRAJECTORIES = ("none", "circle", "figure8")


def _load():
    model = mujoco.MjModel.from_xml_path(BALANCE_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _parse_offset(s):
    """'0.04,0.03' -> (0.04, 0.03)"""
    if s is None:
        return (0.03, 0.02)
    parts = s.split(",")
    return (float(parts[0]), float(parts[1]))


def _target_at(t, mode, radius, period):
    """Time-varying target on the plate (position only)."""
    return _target_state_at(t, mode, radius, period)[0]


def _target_state_at(t, mode, radius, period):
    """Time-varying target on the plate: returns ``(position, acceleration)``,
    both as ``(x, y)`` tuples. Acceleration is analytic (second derivative of
    the position trajectory), used by ``MPCBalancer`` for anticipatory
    feedforward.

    * ``none``    -> plate centre; both zero.
    * ``circle``  -> uniform circular motion; a = -w^2 * position (harmonic).
    * ``figure8`` -> Lissajous 1:2; a_x = -w^2 * x, a_y = -(2w)^2 * y.
    """
    if mode == "none" or period <= 0:
        return (0.0, 0.0), (0.0, 0.0)
    w = 2 * np.pi / period
    if mode == "circle":
        px = radius * np.cos(w * t)
        py = radius * np.sin(w * t)
        return (px, py), (-w * w * px, -w * w * py)
    if mode == "figure8":
        px = radius * np.sin(w * t)
        py = 0.5 * radius * np.sin(2 * w * t)
        return (px, py), (-w * w * px, -4.0 * w * w * py)
    raise ValueError(f"unknown trajectory mode: {mode!r}")


def _apply_perturbation(bal, strength_mps):
    """Add a random lateral velocity kick to the ball (dz stays zero, so it
    doesn't leap off the plate -- just gets shoved sideways)."""
    angle = np.random.uniform(0, 2 * np.pi)
    v = strength_mps * np.array([np.cos(angle), np.sin(angle), 0.0])
    bal.data.qvel[bal.ball_dadr:bal.ball_dadr + 3] += v


def _run_headless_one(cls, offset, duration, trajectory, radius, period,
                      perturb, perturb_period, perturb_strength, seed=0):
    """Run one controller headless; return (peak_mm, final_mm, rms_mm, mean_mm).

    ``err`` here is *tracking error* -- ball xy relative to the moving target,
    not the plate centre. The metrics still apply: peak is the worst excursion,
    final = mean over the last 0.4 s, rms is over the steady-state window
    (transient skipped)."""
    m, d = _load()
    bal = cls(m, d)
    bal.setup_hold()
    bal.reset(ball_offset_xy=offset, settle_steps=400)
    np.random.seed(seed)                               # for _apply_perturbation
    n = int(duration / m.opt.timestep)
    errs = np.empty(n)
    next_perturb = perturb_period                      # first kick after this many seconds
    for k in range(n):
        t = k * m.opt.timestep
        (tx, ty), (ax, ay) = _target_state_at(t, trajectory, radius, period)
        if perturb and t >= next_perturb:
            _apply_perturbation(bal, perturb_strength)
            next_perturb += perturb_period
        errs[k], _ = bal.step(target_xy=(tx, ty), target_axy=(ax, ay))
    tail = errs[-200:]
    steady = errs[max(500, n // 6):]
    return errs.max(), tail.mean(), float(np.sqrt(np.mean(steady ** 2))), steady.mean()


def run_headless(controller="lqr", offset=(0.03, 0.02), duration=6.0,
                 trajectory="none", radius=0.03, period=5.0,
                 perturb=False, perturb_period=2.0, perturb_strength=0.30):
    names = list(CONTROLLERS.keys()) if controller == "both" else [controller]
    hdr = f"ball-balance: start ({offset[0]*1000:+.0f}, {offset[1]*1000:+.0f}) mm, duration {duration:.1f} s"
    if trajectory != "none":
        hdr += f", trajectory={trajectory} r={radius*1000:.0f}mm T={period:.1f}s"
    if perturb:
        hdr += f", perturb every {perturb_period:.1f}s @ {perturb_strength*1000:.0f} mm/s"
    print(hdr)
    for n in names:
        cls = CONTROLLERS[n]
        peak, final, rms, mean = _run_headless_one(
            cls, offset, duration, trajectory, radius, period,
            perturb, perturb_period, perturb_strength)
        settled = final < 0.015 if trajectory == "none" and not perturb else True
        marker = "SETTLED" if trajectory == "none" and not perturb else "TRACKED"
        print(f"  {cls.__name__:12s}  peak {peak*1000:5.1f} mm   "
              f"final {final*1000:5.2f} mm   RMS(steady) {rms*1000:5.2f} mm   "
              f"mean(steady) {mean*1000:5.2f} mm   {marker}: {settled}")


def run_interactive(controller="lqr", offset=(0.03, 0.02),
                    trajectory="none", radius=0.03, period=5.0,
                    perturb=False, perturb_period=2.0, perturb_strength=0.30):
    from mujoco import viewer as mjviewer
    m, d = _load()
    cls = CONTROLLERS[controller]
    bal = cls(m, d)
    bal.setup_hold()
    bal.reset(ball_offset_xy=offset, settle_steps=400)
    label = cls.__name__
    if trajectory != "none": label += f" · {trajectory}"
    if perturb:              label += " · perturbed"
    print(f"Ball-balance demo ({label}). Close the viewer to quit.")
    with mjviewer.launch_passive(m, d) as viewer:
        k, next_perturb = 0, perturb_period
        while viewer.is_running():
            t0 = time.time()
            t  = k * m.opt.timestep
            (tx, ty), (ax, ay) = _target_state_at(t, trajectory, radius, period)
            if perturb and t >= next_perturb:
                _apply_perturbation(bal, perturb_strength)
                next_perturb += perturb_period
            bal.step(target_xy=(tx, ty), target_axy=(ax, ay))
            viewer.sync()
            k += 1
            dt = m.opt.timestep - (time.time() - t0)
            if dt > 0: time.sleep(dt)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ball-balancing demo (PD / LQR + trajectory + perturb).")
    ap.add_argument("--controller", default="lqr", choices=["pd", "lqr", "mpc", "both"],
                    help="controller: pd, lqr (default), mpc (LQR + trajectory feedforward), or both (headless head-to-head)")
    ap.add_argument("--headless", action="store_true", help="print scripted report")
    ap.add_argument("--offset", default=None,
                    help="initial ball offset 'x,y' in metres (default 0.03,0.02)")
    ap.add_argument("--duration", type=float, default=6.0,
                    help="headless run duration in seconds (default 6.0)")
    ap.add_argument("--trajectory", default="none", choices=TRAJECTORIES,
                    help="target trajectory: none (static), circle, figure8 (default: none)")
    ap.add_argument("--radius", type=float, default=0.03,
                    help="trajectory radius in metres (default 0.03)")
    ap.add_argument("--period", type=float, default=5.0,
                    help="trajectory period in seconds (default 5.0)")
    ap.add_argument("--perturb", action="store_true",
                    help="apply periodic random ball velocity kicks (disturbance rejection)")
    ap.add_argument("--perturb-period", type=float, default=2.0,
                    help="seconds between perturbations (default 2.0)")
    ap.add_argument("--perturb-strength", type=float, default=0.30,
                    help="velocity kick magnitude in m/s (default 0.30)")
    args = ap.parse_args(argv)
    offset = _parse_offset(args.offset)
    kw = dict(offset=offset, trajectory=args.trajectory,
              radius=args.radius, period=args.period,
              perturb=args.perturb,
              perturb_period=args.perturb_period,
              perturb_strength=args.perturb_strength)
    if args.headless:
        run_headless(controller=args.controller, duration=args.duration, **kw)
    else:
        controller = "lqr" if args.controller == "both" else args.controller
        if args.controller == "both":
            print("--controller both is only meaningful with --headless; using lqr for the viewer.")
        run_interactive(controller=controller, **kw)


if __name__ == "__main__":
    main()
