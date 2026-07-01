"""Plot OpenArm-Bench results (reads results/openarm_bench.csv -> figures/).

Run the benchmark first to (re)generate the CSV, then plot:

    python benchmarks/openarm_bench.py          # writes results/openarm_bench.csv
    python benchmarks/plot_openarm_bench.py     # writes figures/openarm_bench_*.png

Three figures:
  * openarm_bench_methods.png  — classical vs BC vs ACT success rate (learned story)
  * openarm_bench_admittance.png — compliant vs rigid contact force (~8x softer)
  * openarm_bench_balance.png  — PD vs LQR vs MPC on the ball-balance task
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "results", "openarm_bench.csv")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

CLASSICAL = "#2ca02c"   # green  — classical/scripted
LEARNED = "#1f77b4"     # blue   — behaviour cloning
LEARNED2 = "#9467bd"    # purple — ACT
RIGID = "#d62728"       # red    — rigid (bad)


def load():
    rows = []
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["skill"], r["method"], r["metric"], float(r["result"])))
    return rows


def plot_methods(rows):
    """Grouped success-rate bars: classical vs learned, per skill."""
    succ = {(s, m): v for s, m, mt, v in rows if mt == "success"}
    groups = [
        ("Insertion\n(peg-in-hole)", [("classical", "classical"), ("BC", "BC (state)")]),
        ("Reach", [("BC", "BC (state)"), ("ACT", "ACT (vision)")]),
    ]
    color = {"classical": CLASSICAL, "BC": LEARNED, "ACT": LEARNED2}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x, ticks, labels = 0, [], []
    for gname, methods in groups:
        xs = []
        skill = "insertion" if "Insertion" in gname else "reach"
        for method, lbl in methods:
            v = succ.get((skill, method), 0.0) * 100
            b = ax.bar(x, v, width=0.8, color=color[method], edgecolor="white")
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
            ax.text(x, -7, lbl, ha="center", fontsize=8.5, color="#333")
            xs.append(x); x += 1
        ticks.append(sum(xs) / len(xs)); labels.append(gname); x += 0.8
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="x", length=0, pad=22)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("OpenArm-Bench — classical vs learned policies (n=20, fixed seeds)")
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CLASSICAL, LEARNED, LEARNED2)]
    ax.legend(handles, ["classical (scripted)", "behaviour cloning", "ACT (vision+state)"],
              loc="lower left", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(FIG, "openarm_bench_methods.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"  wrote {out}")


def plot_admittance(rows):
    """Compliant vs rigid contact force pressing the same depth."""
    f = {m: v for s, m, mt, v in rows if s == "admittance"}
    if not f:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    bars = ax.bar(["compliant\n(admittance)", "rigid\n(position)"],
                  [f.get("compliant", 0), f.get("rigid", 0)],
                  color=[CLASSICAL, RIGID], edgecolor="white", width=0.6)
    for b, v in zip(bars, [f.get("compliant", 0), f.get("rigid", 0)]):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f} N",
                ha="center", fontsize=11, fontweight="bold")
    ratio = f.get("rigid", 1) / max(f.get("compliant", 1), 1e-6)
    ax.set_ylabel("steady contact force (N)")
    ax.set_ylim(0, max(f.values()) * 1.18)
    ax.set_title(f"Compliant control yields on contact\n(~{ratio:.0f}x softer, same 3 cm press)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG, "openarm_bench_admittance.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"  wrote {out}")


def plot_balance(rows):
    """PD vs LQR vs MPC vs SAC on the balance task -- two side-by-side panels:
    static settle final error, moving-target (circle) tracking RMS. Lower is
    better; the LQR-outperforms-PD story, the MPC-feedforward win on
    trajectory tracking, and the SAC-vs-classical head-to-head are visible.
    SAC bars are only drawn if a trained model was included in the CSV."""
    b = {(m, mt): v for s, m, mt, v in rows if s == "balance"}
    if not b:
        return
    all_methods = ["PD", "LQR", "MPC", "SAC", "LQR+SAC"]
    methods = [m for m in all_methods if (m, "static final err (mm)") in b]
    static = [b.get((m, "static final err (mm)"), 0) for m in methods]
    track  = [b.get((m, "circle track RMS (mm)"), 0) for m in methods]
    # LQR+SAC = residual policy (classical baseline + learned correction), teal.
    color_of = {"PD": LEARNED, "LQR": "#ff7f0e", "MPC": CLASSICAL,
                "SAC": LEARNED2, "LQR+SAC": "#17becf"}
    colors = [color_of[m] for m in methods]
    # bench_balance caps SAC at 100 mm when the ball rolls off the plate.
    # Any bar exactly at 100 mm is a task-failure marker, not a measurement.
    fail_static = [(m == "SAC" and abs(v - 100.0) < 1e-6) for m, v in zip(methods, static)]
    fail_track  = [(m == "SAC" and abs(v - 100.0) < 1e-6) for m, v in zip(methods, track)]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, values, title, ylabel, fails in (
        (axL, static, "Static hold — settle final error", "final error (mm)", fail_static),
        (axR, track,  "Circle track (r=4 cm, T=2.5 s) RMS", "RMS tracking error (mm)", fail_track),
    ):
        bars = ax.bar(methods, values, color=colors, edgecolor="white", width=0.6)
        for bb, v, is_fail in zip(bars, values, fails):
            if is_fail:
                ax.text(bb.get_x() + bb.get_width() / 2, v * 1.02 + 0.02,
                        "FAILED\n(ball off plate)", ha="center", fontsize=9,
                        fontweight="bold", color="#a11")
                # Red X overlay to reinforce "did not complete the task".
                x0 = bb.get_x(); w = bb.get_width()
                ax.plot([x0, x0 + w], [0, v * 0.95], color="#a11", lw=2, alpha=0.7)
                ax.plot([x0 + w, x0], [0, v * 0.95], color="#a11", lw=2, alpha=0.7)
            else:
                ax.text(bb.get_x() + bb.get_width() / 2, v * 1.02 + 0.02,
                        f"{v:.2f}" if v < 1 else f"{v:.1f}",
                        ha="center", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", alpha=0.3)
    title = "Ball balance — classical vs learned controllers" if "SAC" in methods \
        else "Ball balance — classical controllers head-to-head"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out = os.path.join(FIG, "openarm_bench_balance.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"  wrote {out}")


def main():
    rows = load()
    plot_methods(rows)
    plot_admittance(rows)
    plot_balance(rows)


if __name__ == "__main__":
    main()
