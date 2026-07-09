"""Reproducible evaluation of the catching system -> CSV tables + figures.

Produces four headless benchmarks under benchmarks/results/ (CSV) and
benchmarks/figures/ (PNG):

  1. catch-rate vs throw difficulty (flight time / reaction time)
  2. catch-rate vs vision sensor noise
  3. component ablations (velocity-matching, weld-at-closest, MPC replanning)
  4. ballistic estimator error vs prediction lookahead

    python benchmarks/catching_benchmark.py            # full run
    python benchmarks/catching_benchmark.py --quick     # fast smoke run
"""
import argparse
import csv
import os
import sys

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import CATCH_SCENE
from openarm_control.catching import CatchController, BallisticKalmanFilter, sample_throw
from openarm_control.vision import BallPerception

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
G = np.array([0.0, 0.0, -9.81])
EPISODE_STEPS = 720          # enough for flight (~0.5 s) + settle (~0.5 s); set by main()


def _scene():
    model = mujoco.MjModel.from_xml_path(CATCH_SCENE)
    data = mujoco.MjData(model)
    js = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    bq, bdof = model.jnt_qposadr[js], model.jnt_dofadr[js]
    fadr = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_finger_joint1")]
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    return model, data, bq, bdof, fadr, key


def _episode(model, data, bq, bdof, fadr, key, catcher, L, v0, steps=None):
    steps = EPISODE_STEPS if steps is None else steps
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[bq:bq + 3] = L
    data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    data.qvel[bdof:bdof + 3] = v0
    catcher.reset()
    for _ in range(steps):
        catcher.step()
        mujoco.mj_step(model, data)
    if not catcher.caught:
        return False
    near = np.linalg.norm(catcher.ball_pos() - catcher.grasp_pos())
    # fingers must be HELD APART by the ball: fully open is -0.7854, fully
    # closed (on nothing) settles at ~0 -- both are failures. A gripped 70 mm
    # ball leaves the finger joint in a mid band.
    return bool(catcher.ball_pos()[2] > 0.45 and near < 0.10
                and -0.45 < data.qpos[fadr] < -0.2)


def _rate(model, data, addrs, key, catcher, n, seed, tf_range=(0.38, 0.60)):
    bq, bdof, fadr = addrs
    rng = np.random.default_rng(seed)
    clean = 0
    for _ in range(n):
        L, v0 = sample_throw(rng, model.opt.gravity, tf_range=tf_range)
        clean += int(_episode(model, data, bq, bdof, fadr, key, catcher, L, v0))
    return clean / n


def _save_csv(name, header, rows):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------- benchmarks
def bench_difficulty(n, seed):
    model, data, bq, bdof, fadr, key = _scene()
    catcher = CatchController(model, data)
    tfs = [0.32, 0.40, 0.46, 0.52, 0.58]
    rates = [_rate(model, data, (bq, bdof, fadr), key, catcher, n, seed, (tf, tf)) for tf in tfs]
    _save_csv("difficulty.csv", ["flight_time_s", "catch_rate"], list(zip(tfs, rates)))
    plt.figure(figsize=(6, 4))
    plt.plot([t * 1000 for t in tfs], [r * 100 for r in rates], "o-", color="#1f77b4", lw=2)
    plt.xlabel("flight time = reaction time (ms)")
    plt.ylabel("clean-catch rate (%)")
    plt.title(f"Catch rate vs throw difficulty (n={n}/point)")
    plt.ylim(0, 105); plt.grid(alpha=0.3); plt.gca().invert_xaxis()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "catch_rate_vs_difficulty.png"), dpi=130)
    plt.close()
    print("difficulty:", dict(zip(tfs, [round(r, 2) for r in rates])))


def bench_vision_noise(n, seed):
    model, data, bq, bdof, fadr, key = _scene()
    noises_mm = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0]
    rates = []
    for nm in noises_mm:
        perc = BallPerception(model, data, ["ballcam0", "ballcam1"], noise_std=nm / 1000.0)
        catcher = CatchController(model, data, perception=perc)
        rates.append(_rate(model, data, (bq, bdof, fadr), key, catcher, n, seed))
    _save_csv("vision_noise.csv", ["noise_std_mm", "catch_rate"], list(zip(noises_mm, rates)))
    plt.figure(figsize=(6, 4))
    plt.plot(noises_mm, [r * 100 for r in rates], "s-", color="#d62728", lw=2)
    plt.xlabel("added camera noise std (mm)")
    plt.ylabel("clean-catch rate (%)")
    plt.title(f"Vision-driven catch rate vs sensor noise (n={n}/point)")
    plt.ylim(0, 105); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "catch_rate_vs_vision_noise.png"), dpi=130)
    plt.close()
    print("vision noise:", dict(zip(noises_mm, [round(r, 2) for r in rates])))


def bench_ablations(n, seed):
    model, data, bq, bdof, fadr, key = _scene()
    configs = {
        "full system": {},
        "no velocity\nmatching": {"velocity_match": False},
        "weld at jaw\nentry": {"weld_mode": "entry"},
        "no MPC\nreplanning": {"replan_every": 10 ** 9},
    }
    names, rates = [], []
    for name, kw in configs.items():
        catcher = CatchController(model, data, **kw)
        names.append(name)
        rates.append(_rate(model, data, (bq, bdof, fadr), key, catcher, n, seed))
    _save_csv("ablations.csv", ["config", "catch_rate"],
              [(nm.replace("\n", " "), r) for nm, r in zip(names, rates)])
    colors = ["#2ca02c"] + ["#7f7f7f"] * (len(names) - 1)
    plt.figure(figsize=(6.5, 4))
    bars = plt.bar(names, [r * 100 for r in rates], color=colors)
    for b, r in zip(bars, rates):
        plt.text(b.get_x() + b.get_width() / 2, r * 100 + 1, f"{r*100:.0f}%", ha="center", fontsize=9)
    plt.ylabel("clean-catch rate (%)")
    plt.title(f"Component ablations (n={n} each)")
    plt.ylim(0, 105); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "ablations.png"), dpi=130)
    plt.close()
    print("ablations:", dict(zip([nm.replace(chr(10), ' ') for nm in names],
                                  [round(r, 2) for r in rates])))


def bench_estimation(seed):
    """Estimator error vs prediction lookahead (clean vs noisy observations)."""
    rng = np.random.default_rng(seed)
    lookaheads = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    curves = {}
    for noise_mm, label in [(0.0, "noiseless obs"), (5.0, "+5 mm obs noise")]:
        errs = {la: [] for la in lookaheads}
        for _ in range(40):
            p0 = np.array([rng.uniform(1.0, 1.5), rng.uniform(-0.4, 0.4), rng.uniform(0.9, 1.3)])
            v0 = np.array([rng.uniform(-3, -2), rng.uniform(-0.4, 0.4), rng.uniform(1.5, 2.6)])
            kf = BallisticKalmanFilter(G)
            n_obs = 12
            for k in range(n_obs):
                t = k * 0.002
                p = p0 + v0 * t + 0.5 * G * t**2
                if noise_mm:
                    p = p + rng.normal(0, noise_mm / 1000.0, 3)
                kf.observe(t, p)
            t_now = (n_obs - 1) * 0.002
            for la in lookaheads:
                truth = p0 + v0 * (t_now + la) + 0.5 * G * (t_now + la)**2
                errs[la].append(np.linalg.norm(kf.position_at(t_now + la) - truth))
        curves[label] = [np.mean(errs[la]) * 1000 for la in lookaheads]
    _save_csv("estimation.csv", ["lookahead_s"] + list(curves),
              [[la] + [curves[k][i] for k in curves] for i, la in enumerate(lookaheads)])
    plt.figure(figsize=(6, 4))
    for label, ys in curves.items():
        plt.plot([la * 1000 for la in lookaheads], ys, "o-", lw=2, label=label)
    plt.xlabel("prediction lookahead (ms)")
    plt.ylabel("ball position error (mm)")
    plt.title("Ballistic Kalman estimator: prediction error")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "estimation_error.png"), dpi=130)
    plt.close()
    print("estimation (noiseless, mm):", [round(v, 1) for v in curves["noiseless obs"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Catching benchmark suite -> CSV + figures.")
    ap.add_argument("--quick", action="store_true", help="small N for a fast smoke run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["difficulty", "noise", "ablations", "estimation"])
    args = ap.parse_args(argv)
    os.makedirs(FIG, exist_ok=True)
    n_gt = 8 if args.quick else 20      # ground-truth episodes per point
    n_vis = 4 if args.quick else 10     # vision episodes per point (slower)

    runs = {
        "difficulty": lambda: bench_difficulty(n_gt, args.seed),
        "noise": lambda: bench_vision_noise(n_vis, args.seed),
        "ablations": lambda: bench_ablations(n_gt, args.seed),
        "estimation": lambda: bench_estimation(args.seed),
    }
    for name, fn in runs.items():
        if args.only is None or args.only == name:
            print(f"== {name} ==")
            fn()
    print(f"\nfigures -> {FIG}\nresults -> {RES}")


if __name__ == "__main__":
    main()
