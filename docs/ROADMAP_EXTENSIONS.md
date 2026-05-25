# OpenArm — Extension Roadmap (post-v1)

> This is an **additive** roadmap for the next development arc. The existing
> system (123 passing tests, all current apps/scenes/CLI) is **frozen and
> protected** — every phase below adds new packages/scenes/tests and **never
> edits working code paths**. The existing test suite is a **regression gate**:
> it must stay green after every phase.

Build order (2026-05-24):

```
F1 → F2 → S2 → S3 → F3 → I2 → S1 → I1 → I3 → E1
FOUNDATION ──── SKILLS ──── (harness) ─ LEARN ─ RL ─ LANG ─ VLA ─ BENCH
```

**Status:** ✅ **EXTENSION ROADMAP COMPLETE** — F1, F2, S2, S3, cloth, F3, I2, S1,
I1, E1 all done (**150 tests green**). I3/TinyVLA deferred (8.5 GB VRAM is tight for
a full VLA). Detail below.

✅ **F1**, ✅ **F2**, ✅ **S2**, ✅ **S3** done (136 tests green) — see
`docs/IMPLEMENTATION_LOG.md`. F2 admittance: ~8x lower contact force (27 N vs
217 N). S2 = **single-arm bottle opening** (clamped bottle; fingertip-pinch a knob,
twist ~356° in place on the neck, lift off; bimanual at one small object was
collision-prone, so the clamp holds it). S3 = **articulated manipulation**: open
drawer (~77 mm), swing cabinet door (~40°), turn valve (~75°) — single-arm,
weld-assisted, collision-free. **Cloth recheck done:** model upgraded to a finer
9×9 flex grid with self-collision (settles flat, folds into layers); single-arm
fold is collision-free; a bimanual half-fold is accurate (~4 mm) but can't be made
reliably collision-free on these close-mounted arms (same limit as the bottle).
**F3 done:** CUDA verified (RTX 5060, 8.5 GB, torch 2.11+cu128); GPU-aware BC
training; image-observation logging in the demo collector (foundation for vision
policies). **I2 done:** a self-contained **ACT** (action-chunking transformer,
vision+state, ~1.4 M params, no torchvision) trained on the GPU — reach ~75-85%
success (the first non-baseline *learned* policy). **S1 done:** a domain-randomized
peg-in-hole RL env (randomized socket/offset/friction/peg-radius) — classical
scripted insertion **100% @ 1.5 mm**, BC (state) **67%**, RL/SAC user-run. **I1
done:** the articulated skills are language-commandable ("open the drawer then turn
the valve"). Next: **I3** (TinyVLA — decide), **E1** (OpenArm-Bench).

## Guiding principles

1. **Same robot.** We never replace or modify the OpenArm v2 model. We only
   enrich the *world* it acts in and the *perception/control* around it.
2. **Additive only.** New code in new packages (`contact/`, `articulated/`,
   `rl2/`, `learned/`), new scene XMLs, new tests. Working controllers, scenes,
   and their tests are not touched.
3. **Regression gate.** `pytest tests` (currently 123 green) must stay green
   after every phase before moving on.
4. **Docs/CLI/memory updated each phase** (IMPLEMENTATION_LOG, CLI command list,
   this file, memory checkpoint).

## Phases

| Phase | Goal | Compute | Depends on |
|---|---|---|---|
| **F1** ✅ | Realistic objects (finish GSO) + **6-DOF grasp** + **authored** articulated assets (drawer/door/valve) | CPU | — |
| **F2** ✅ | **Admittance control** (reuse position stack: sum EE contact force → soften Cartesian reference). Impedance/torque variant later. | CPU | — |
| **S2** ✅ | Bimanual coordination — **unscrew** (hold-while-manipulate): left holds jar, right turns lid off. (Simultaneous rigid two-arm carry deferred — closed-chain.) | CPU | bimanual infra, F2 |
| **S3** ✅ | Articulated manipulation — open drawer (~77 mm), open cabinet door (~40°), turn valve (~75°); single-arm, weld-assisted, collision-free | CPU | F1, F2 |
| **F3** ✅ | Learning harness: self-contained npz + **image-observation logging**, **GPU-aware training**; CUDA verified (RTX 5060, 8.5 GB) | GPU (train) | scripted skills |
| **I2** ✅ | **ACT** (action-chunking transformer, vision+state) learned policy; trained on GPU, reach ~75-85% (first non-baseline learned result) | GPU | F3 + skills |
| **S1** ✅ | RL insertion env: **one robot + randomized socket/offset/friction/peg-radius**; classical (scripted) **100%** vs BC **67%** vs RL (SAC, user-run) | GPU | F2, F3 |
| **I1** ✅ | Language for the new skills: "open the drawer / door", "turn the valve", "unscrew" — parsed + multi-step dispatched to the controllers | CPU | S3, F1 |
| **I3** | TinyVLA fine-tune on our sim demos (stretch; decide at this phase, VRAM-dependent) | GPU | I2, F3 |
| **E1** ✅ | **OpenArm-Bench**: unified eval over the manipulation skills + classical/BC/ACT/RL comparison (table + CSV) | CPU/GPU | everything |

## Key decisions & constraints

- **GPU:** the project targets a CUDA GPU. PyTorch-CUDA (I2/I3 supervised training) is a big
  native-Windows win. RL *rollouts* stay CPU-bound with standard MuJoCo; the
  massively-parallel GPU path (MJX/JAX) needs WSL2 on Windows — decide at S1.
- **RL design:** generalization comes from **task diversity** (one robot, many
  randomized objects/holes/clearances), not extra robots. Parallel env *copies*
  are only for throughput.
- **Bimanual "different tasks at once":** great for **demos** and truly-bimanual
  tasks, but does **not** speed training (larger coupled action space hurts RL).
  The useful generalization form is a **task-conditioned multi-task policy**
  (one arm per env, sample a task per episode).
- **Datasets:** other robots' HF/LeRobot datasets mostly **don't transfer**
  (embodiment gap). Reuse pretrained VLA **weights** fine-tuned on **our** sim
  demos; generate demos with our scripted experts.
- **Demos for ACT/Diffusion:** use the **scripted controllers** (clean,
  deterministic), not the webcam mimic (too noisy to train on).

## F1 starting state (already on disk)

- `build_scanned.py` (curate GSO) + `gen_scanned_scene.py` (scene gen) exist.
- `assets/scanned/{mug,bowl,clock,elephant}` curated; `scanned_table_scene.xml`
  registered as `SCENES["scanned_table"]`.
- `grasp.py` is **top-down only** (`topdown_orientation` + `GraspSolver`,
  4-DOF x/y/z/yaw) → this is the 6-DOF grasp gap to fill.
