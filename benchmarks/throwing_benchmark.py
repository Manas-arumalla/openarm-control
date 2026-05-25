"""Reproducible evaluation of the throwing system -> CSV tables + figures.

The arm throws a ball into a narrow bin (16 cm opening) repositioned across a
spread of locations inside the precise-throw envelope. Each trial perturbs the
ball's start pose, then runs the full pipeline (grasp -> sim-in-the-loop release
planning -> swing + release) and records where the ball lands.

Reports, under benchmarks/results/ (CSV) and benchmarks/figures/ (PNG):

  1. precision/consistency per bin (mean +/- std landing error, in-bin success)
  2. a top-down map of every bin and its landing scatter
  3. reachability vs forward distance (where the throw envelope ends)

    python benchmarks/throwing_benchmark.py             # full run (~few min)
    python benchmarks/throwing_benchmark.py --quick      # fast smoke run
    python benchmarks/throwing_benchmark.py --trials 12  # more trials/bin
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
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import THROW_SCENE
from openarm_control.throwing import ThrowController, BENCH_BINS

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")

BIN_HALF = 0.08          # bin opening half-width (matches throw_scene.xml)
BALL_R = 0.03            # ball radius
GRASP_JITTER = 0.005     # ball start xy perturbation (m, uniform +/-) -> trial variance


def _fresh():
    """A fresh scene every call: ``grasp_ball`` mutates the model (it disables the
    table collision as a swing fail-safe and writes the weld offset), so each trial
    must start from a clean model -- not a reused one."""
    model = mujoco.MjModel.from_xml_path(THROW_SCENE)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    bin_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball_orange")
    ball_qadr = model.jnt_qposadr[next(
        j for j in range(model.njnt)
        if model.jnt_bodyid[j] == ball_bid and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)]
    return model, data, key, bin_bid, int(ball_qadr)


def _trial(bx, by, jitter):
    """One throw to a bin at (bx,by) with the ball start perturbed by ``jitter``.
    Returns (planned, landing_xy, err_m, in_bin)."""
    model, data, key, bin_bid, ball_qadr = _fresh()
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[ball_qadr:ball_qadr + 2] += jitter            # perturb ball start
    model.body_pos[bin_bid][0] = bx
    model.body_pos[bin_bid][1] = by
    mujoco.mj_forward(model, data)

    tc = ThrowController(model, data, ball="ball_orange")
    target = np.array([bx, by, 0.06])
    if not tc.grasp_ball():
        return False, None, None, False
    if tc.plan_release(target) is None:
        return False, None, None, False
    res = tc.execute()
    land = res[:2].copy()
    err = float(np.linalg.norm(land - target[:2]))
    in_bin = bool(abs(land[0] - bx) < BIN_HALF and abs(land[1] - by) < BIN_HALF and res[2] < 0.13)
    return True, land, err, in_bin


def _save_csv(name, header, rows):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------- benchmarks
def bench_precision(trials, seed):
    """Throw to each bin ``trials`` times (perturbed); measure precision + success."""
    rng = np.random.default_rng(seed)
    rows, summary, scatter = [], [], {}
    for bx, by in BENCH_BINS:
        errs, n_plan, n_in = [], 0, 0
        pts = []
        for t in range(trials):
            jit = rng.uniform(-GRASP_JITTER, GRASP_JITTER, 2)
            planned, land, err, in_bin = _trial(bx, by, jit)
            n_plan += int(planned)
            if planned:
                n_in += int(in_bin)
                errs.append(err)
                pts.append(land)
                rows.append([bx, by, t, 1, round(land[0], 4), round(land[1], 4),
                             round(err * 1000, 1), int(in_bin)])
            else:
                rows.append([bx, by, t, 0, "", "", "", 0])
        scatter[(bx, by)] = np.array(pts) if pts else np.empty((0, 2))
        em = np.array(errs) * 1000 if errs else np.array([np.nan])
        summary.append([bx, by, trials, n_plan, n_in,
                        round(float(np.mean(em)), 1), round(float(np.std(em)), 1),
                        round(float(np.max(em)), 1)])
        print(f"bin ({bx:+.2f},{by:+.2f}): planned {n_plan}/{trials}  in-bin {n_in}/{n_plan}  "
              f"err {np.mean(em):.1f}+/-{np.std(em):.1f} mm  (max {np.max(em):.1f})")

    _save_csv("throwing_trials.csv",
              ["bin_x", "bin_y", "trial", "planned", "land_x", "land_y", "err_mm", "in_bin"], rows)
    _save_csv("throwing_summary.csv",
              ["bin_x", "bin_y", "trials", "planned", "in_bin", "mean_err_mm", "std_err_mm", "max_err_mm"],
              summary)

    all_err = [r[6] for r in rows if r[3] == 1]
    tot_in = sum(s[4] for s in summary)
    tot_plan = sum(s[3] for s in summary)
    print(f"\nOVERALL: {tot_plan}/{len(BENCH_BINS)*trials} planned, "
          f"{tot_in}/{tot_plan} in-bin ({100*tot_in/max(1,tot_plan):.0f}%), "
          f"mean error {np.mean(all_err):.1f} mm")

    _plot_map(scatter, summary)
    _plot_bars(summary)


def _plot_map(scatter, summary):
    """Top-down: each bin (square) + its landing scatter + centre."""
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    colors = plt.cm.tab10(np.linspace(0, 1, len(BENCH_BINS)))
    for (bx, by), c in zip(BENCH_BINS, colors):
        ax.add_patch(Rectangle((bx - BIN_HALF, by - BIN_HALF), 2 * BIN_HALF, 2 * BIN_HALF,
                               fill=False, ec=c, lw=1.5))
        ax.plot(bx, by, "+", color=c, ms=9, mew=2)
        pts = scatter[(bx, by)]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=18, color=c, alpha=0.8, edgecolors="none")
    ax.plot(0.0, 0.0, "k^", ms=10)                          # arm base (approx)
    ax.annotate("arm base", (0.0, 0.0), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("x forward (m)"); ax.set_ylabel("y lateral (m)")
    ax.set_title(f"Throw landing map: {len(BENCH_BINS)} bins (16 cm), squares=bins, dots=landings")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "throwing_landing_map.png"), dpi=130)
    plt.close()


def _plot_bars(summary):
    """Per-bin mean +/- std landing error, with in-bin success annotated."""
    labels = [f"({s[0]:.2f},\n{s[1]:+.2f})" for s in summary]
    means = [s[5] for s in summary]
    stds = [s[6] for s in summary]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, means, yerr=stds, capsize=4, color="#2ca02c", alpha=0.85)
    for b, s in zip(bars, summary):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + (s[6] or 0) + 0.4,
                 f"{s[4]}/{s[3]}", ha="center", fontsize=8)
    plt.axhline(BIN_HALF * 1000, ls="--", color="#777", lw=1)
    plt.text(len(summary) - 0.5, BIN_HALF * 1000 + 1, "bin edge (80 mm)", ha="right", fontsize=8, color="#555")
    plt.ylabel("landing error from bin centre (mm)")
    plt.xlabel("bin location (x, y)")
    plt.title("Throw precision per bin (bar = mean, whisker = std; label = in-bin/planned)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "throwing_precision.png"), dpi=130)
    plt.close()


def bench_reachability(seed):
    """Map best achievable landing error vs forward distance (the envelope edge)."""
    xs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    by = -0.10
    best, reach = [], []
    for bx in xs:
        model, data, key, bin_bid, ball_qadr = _fresh()
        mujoco.mj_resetDataKeyframe(model, data, key)
        model.body_pos[bin_bid][0] = bx; model.body_pos[bin_bid][1] = by
        mujoco.mj_forward(model, data)
        tc = ThrowController(model, data, ball="ball_orange")
        tc.grasp_ball()
        if tc.plan_release(np.array([bx, by, 0.06]), gate=0.5) is None:
            best.append(np.nan); reach.append(0)
        else:
            best.append(tc.pred_err * 1000)
            reach.append(int(tc.pred_err < 0.06))
    _save_csv("throwing_reachability.csv", ["bin_x", "best_err_mm", "within_gate"],
              list(zip(xs, [round(b, 1) for b in best], reach)))
    plt.figure(figsize=(6.5, 4))
    plt.plot(xs, best, "o-", color="#1f77b4", lw=2)
    plt.axhline(60, ls="--", color="#d62728", lw=1)
    plt.text(xs[0], 63, "throw gate (60 mm)", fontsize=8, color="#d62728")
    plt.xlabel("bin forward distance x (m)"); plt.ylabel("best achievable error (mm)")
    plt.title("Precise-throw envelope: best landing error vs distance")
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "throwing_envelope.png"), dpi=130)
    plt.close()
    print("reachability:", dict(zip(xs, [round(b, 0) for b in best])))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Throwing benchmark suite -> CSV + figures.")
    ap.add_argument("--quick", action="store_true", help="few trials for a fast smoke run")
    ap.add_argument("--trials", type=int, default=None, help="trials per bin (overrides default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["precision", "reachability"])
    args = ap.parse_args(argv)
    os.makedirs(FIG, exist_ok=True)
    trials = args.trials if args.trials is not None else (3 if args.quick else 8)

    if args.only is None or args.only == "precision":
        print(f"== precision ({trials} trials/bin, {len(BENCH_BINS)} bins) ==")
        bench_precision(trials, args.seed)
    if args.only is None or args.only == "reachability":
        print("== reachability ==")
        bench_reachability(args.seed)
    print(f"\nfigures -> {FIG}\nresults -> {RES}")


if __name__ == "__main__":
    main()
