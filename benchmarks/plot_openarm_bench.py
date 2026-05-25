"""Plot OpenArm-Bench results (reads results/openarm_bench.csv -> figures/).

Run the benchmark first to (re)generate the CSV, then plot:

    python benchmarks/openarm_bench.py          # writes results/openarm_bench.csv
    python benchmarks/plot_openarm_bench.py     # writes figures/openarm_bench_*.png

Two figures:
  * openarm_bench_methods.png  — classical vs BC vs ACT success rate (the learned story)
  * openarm_bench_admittance.png — compliant vs rigid contact force (~8x softer)
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


def main():
    rows = load()
    plot_methods(rows)
    plot_admittance(rows)


if __name__ == "__main__":
    main()
