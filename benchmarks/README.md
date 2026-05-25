# Benchmarks

Reproducible, headless evaluation suites. Each writes a CSV table to
[`results/`](results/) and figures to [`figures/`](figures/).

- **Catching** (`catching_benchmark.py`) — the dynamic flagship: rate-vs-difficulty,
  vision-noise robustness, component ablations, estimator error (documented below).
- **Throwing** (`throwing_benchmark.py`) — landing precision across the reachable envelope.
- **OpenArm-Bench** (`openarm_bench.py` + `plot_openarm_bench.py`) — a unified
  manipulation eval with the **classical vs BC vs ACT vs RL** comparison
  ([jump to it](#openarm-bench--unified-manipulation-eval)).

---

## Catching

Reproducible, headless evaluation of the catching system.

```bash
python benchmarks/catching_benchmark.py            # full run (writes CSVs + figures)
python benchmarks/catching_benchmark.py --quick     # fast smoke run (small N)
python benchmarks/catching_benchmark.py --only ablations
```

All runs are seeded (`--seed`, default 0). "Clean catch" = the ball is held
(z > 0.45 m, within 10 cm of the grasp point) with the fingers closed on it.

## 1. Catch rate vs throw difficulty
![catch rate vs difficulty](figures/catch_rate_vs_difficulty.png)

Difficulty is set by the **flight time = the reaction time the arm has**. Shorter
flights are faster balls with less time to perceive, plan, and move. The curve
shows the operating envelope and where it degrades.

## 2. Vision-driven catch rate vs sensor noise
![catch rate vs vision noise](figures/catch_rate_vs_vision_noise.png)

Catching driven *only* by the two RGB-D cameras, with extra Gaussian noise added
to the fused 3D estimate. Shows how robust the perception + Kalman filter are as
sensing degrades.

## 3. Component ablations
![ablations](figures/ablations.png)

Clean-catch rate with each key component removed, vs the full system:
- **no velocity matching** — the hand does not match the ball's velocity at the catch.
- **weld at jaw entry** — grab the instant the ball enters the jaws (vs at the closest approach, centred between the pads).
- **no MPC replanning** — commit to the first interception plan instead of re-planning every step.

## 4. Ballistic estimator error vs prediction lookahead
![estimation error](figures/estimation_error.png)

After 12 observations (~24 ms), how accurately the Kalman filter predicts the
ball's future position, for noiseless vs noisy observations. This is what the
interception solver relies on.

---

## OpenArm-Bench — unified manipulation eval

A single benchmark over the extension-arc manipulation skills with a standardized
protocol (fixed seeds, `n=20`) and the **classical vs BC vs ACT vs RL** method
comparison.

```bash
python benchmarks/openarm_bench.py            # full run -> results/openarm_bench.csv
python benchmarks/openarm_bench.py --quick    # fewer episodes
python benchmarks/openarm_bench.py --only insertion,reach
python benchmarks/plot_openarm_bench.py       # figures from the CSV
```

| | |
|---|---|
| ![classical vs learned](figures/openarm_bench_methods.png) | ![compliant vs rigid](figures/openarm_bench_admittance.png) |

- **Insertion (peg-in-hole):** classical (scripted) **100%** vs behaviour cloning **~70%**.
- **Reach:** behaviour cloning **~95%** vs ACT (vision+state) **~80%**.
- **Articulated:** drawer **~77 mm**, door **~40°**, valve **~75°** opened (classical).
- **Admittance:** compliant **27 N** vs rigid **217 N** pressing the same depth (~8× softer).
- **Cloth fold:** **~44%** span reduction (single-arm, self-colliding flex cloth).

All numbers are deterministic sim measurements written to `results/openarm_bench.csv`.
