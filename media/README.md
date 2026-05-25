# Media gallery

Showcase screenshots and GIFs used in the top-level [README](../README.md). Everything
here is **rendered headless** from the project's own controllers — nothing is captured
by hand — so it is fully reproducible.

## Manipulation skills (extension arc)

| File | Skill | Reproduce |
|---|---|---|
| `insert.gif` / `insert_hero.png` | Peg-in-hole insertion | `python scripts/gen_showcase_media.py --only insert` |
| `drawer.gif` / `drawer_hero.png` | Open a sliding drawer | `… --only drawer` |
| `door.gif` / `door_hero.png` | Swing a cabinet door | `… --only door` |
| `valve.gif` / `valve_hero.png` | Turn a valve | `… --only valve` |
| `cloth.gif` / `cloth_hero.png` | Fold a deformable cloth | `… --only cloth` |
| `unscrew.gif` / `unscrew_hero.png` | Unscrew a bottle cap | `… --only unscrew` |
| `admittance.gif` / `admittance_hero.png` | Compliant (admittance) press | `… --only admittance` |

Regenerate all of them at once with `python scripts/gen_showcase_media.py`.

## Dynamic catching (flagship)

| File | What it shows |
|---|---|
| `catch_demo.gif` | Single-arm airborne ball catch (Kalman + MPC interception). |
| `catch_bimanual.gif` | Bimanual catch with best-arm selection, collision-free. |
| `catch_twoball.gif` | Two balls thrown at once; multi-object tracking, dual catch. |
| `catch_camera_view.png` | The robot's-eye RGB-D view with the detected ball crosshaired. |
| `catch_*_held.png`, `catch_1/2/3_*.png` | Key frames (incoming → catch moment → held). |

These are produced by the catching demos/benchmarks (`openarm catch …`,
`python benchmarks/catching_benchmark.py`).

## Other

| File | What it shows |
|---|---|
| `v2.png` | The OpenArm v2 bimanual model (hero image). |

> Benchmark **plots** (catch-rate curves, ablations, OpenArm-Bench bars) live separately
> under [`../benchmarks/figures/`](../benchmarks/figures/).
