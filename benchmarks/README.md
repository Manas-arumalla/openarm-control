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

### Protocol, per cell

Every cell in the leaderboard is reproducible with one command. Stochastic tasks
run `n=20` episodes on fixed seeds; deterministic tasks are seed-invariant single
measurements (running them twice produces identical numbers, so no seed spread is
reported — reporting a confidence interval over identical runs would be noise
theater).

| Cell | Episodes / seeds | Success criterion / metric | Reproduce |
|---|---|---|---|
| Insertion — classical | n=20, seeds 0–19 | peg tip within tolerance at socket depth | `python benchmarks/openarm_bench.py --only insertion` |
| Insertion — BC | n=20, seeds 1000–1019 | same | same command (loads `demos/insert_bc.pt`) |
| Reach — BC | n=20, seeds 700–719 | end-effector within 3 cm of target | `python benchmarks/openarm_bench.py --only reach` |
| Reach — ACT | n=20, seeds 700–719 | same | same command (loads `demos/reach_act.pt`) |
| Drawer / door / valve | deterministic | opened distance / swing angle / turn angle | `python benchmarks/openarm_bench.py --only articulated` |
| Admittance vs rigid | deterministic | steady contact force at the same 3 cm press | `python benchmarks/openarm_bench.py --only admittance` |
| Cloth fold | deterministic | y-span reduction after fold | `python benchmarks/openarm_bench.py --only cloth_fold` |
| Balance — PD / LQR / MPC | deterministic | static settle error; circle-tracking RMS (r=4 cm, T=2.5 s) | `python benchmarks/openarm_bench.py --only balance` |
| Balance — SAC, LQR+SAC | deterministic eval of a trained policy | same; ball-off-plate reported as failure | train first: `openarm rl-train --task balance --timesteps 200000` and `openarm rl-train --task balance_residual --timesteps 30000`, then the balance command above |

The learned-policy cells (BC, ACT) evaluate the small trained policies that are
versioned in `demos/` — they run from a fresh clone with no training step. The
SAC balance cells require local training first (the SAC weights follow the same
not-versioned convention as `reach_sac.zip`); the bench auto-detects the trained
models and appends their rows when present.

| | |
|---|---|
| ![classical vs learned](figures/openarm_bench_methods.png) | ![compliant vs rigid](figures/openarm_bench_admittance.png) |

- **Insertion (peg-in-hole):** classical (scripted) **100%** vs behaviour cloning **~70%**.
- **Reach:** behaviour cloning **~95%** vs ACT (vision+state) **~80%**.
- **Articulated:** drawer **~95 mm** (frontal grasp), door **~54°**, valve **~78°** opened (classical).
- **Admittance:** compliant **27 N** vs rigid **217 N** pressing the same depth (~8× softer).
- **Cloth fold:** **~44%** span reduction (single-arm, self-colliding flex cloth).
- **Ball balance:** PD / LQR / MPC / SAC / LQR+SAC on the same tilting-plate physics.
  Static hold: PD **0.44**, LQR **0.39**, MPC **0.39**, LQR+SAC **5.2 mm**; SAC
  from-scratch fails (ball off plate). Circle track (r=4 cm, T=2.5 s):
  PD **40.3**, LQR **39.2**, **MPC 37.7**, LQR+SAC **42.9 mm**. See
  [`../docs/IMPLEMENTATION_LOG.md`](../docs/IMPLEMENTATION_LOG.md)
  for the training curves.

![ball balance — five-way comparison](figures/openarm_bench_balance.png)

All numbers are deterministic sim measurements written to `results/openarm_bench.csv`.
