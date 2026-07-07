# OpenArm Control & Simulation — Project Roadmap

## Vision

A high-end, open-source robotics platform built on the Enactic **OpenArm v2**
(7-DOF × 2, MuJoCo). It spans the full stack — classical control & motion
planning, bimanual coordination, reinforcement & imitation learning, and
vision-driven autonomy — for single-arm **and** bimanual operation, culminating
in **real-time webcam-based human-arm imitation**.

The goal is not just demos but a *serious, reusable, well-tested, well-documented*
codebase that another engineer or researcher could pick up, install, and extend.

---

## Status snapshot

| Stage | Scope | Status |
|------|-------|--------|
| **Phase 0** | Foundation: FK, robust IK, Cartesian control, gravity comp, trajectories, verified scene | ✅ done |
| **Track 1** | Pick-and-place, color sorting, RRT-Connect obstacle avoidance | ✅ done |
| **Track 2** | Bimanual: parallel sort, mirrored sync, collision-aware hand-off | ✅ done |
| **Track 3** | RL: Gymnasium reach env + SAC train/eval | ✅ baseline (reach learned; to be strengthened in Phase E) |
| **Phase A** | Repository hardening: installable `openarm-control` pkg, README, extras | ✅ done |
| **Phase B** | Integrated system: `openarm` CLI + scene registry + showcase | ✅ done |
| **Phase C** | Perception & visual servoing (camera, color detection, PBVS) | ✅ done |
| **Phase D** | Dynamic manipulation: airborne catch (Kalman + interception + MPC) | ✅ done |
| **Phase D2a** | Vision-driven catching: two RGB-D cameras feed the estimator (no ground truth) | ✅ done |
| **Phase D2b** | Bimanual reactive catching: best-arm selection + collision-free catch | ✅ done |
| **Phase E** | Reinforcement learning: stronger reach + RL pick-place envs | ✅ envs done (user runs full training) |
| **Phase F** | Imitation learning (behavior cloning from scripted demos; BC vs RL) | ✅ done (BC 80–85% vs SAC 47% on reach) |
| **Phase G** | Webcam human-arm imitation (capstone) | ✅ done (whole-arm posture + hand grasping + pick-up; live webcam) |
| **Phase H** | Full integration & showcase | ✅ done (grand-tour showcase + full-system integration tests) |

Tests: 73 headless (`python -m pytest tests/`). Implementation history:
[`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md).

---

## Guiding principles (how we build)

1. **Verification-first** — every capability has a headless test with measured
   numbers (mm, deg, success-rate); we don't claim "works" without proof.
2. **Clean separation** — original model files stay untouched where possible;
   all logic lives in the `openarm_control/` package; scenes are declarative XML.
3. **Reproducible** — fixed seeds, saved configs/checkpoints, logged metrics.
4. **Documented** — docstrings, `IMPLEMENTATION_LOG.md`, and per-phase notes.
5. **Incremental & demoable** — each phase ends with a runnable demo + tests.

---

## Roadmap

### Phase A — Repository hardening ("high-end GitHub")
**Goal:** turn the working code into a professional, installable, CI-tested repo.

- Restructure into an installable package (`pyproject.toml`, `pip install -e .`),
  keep the `openarm_control/` import path; add `openarm` console entry point.
- Top-level **README** rewrite: overview, architecture diagram, capability GIFs,
  quickstart, demo index, results tables.
- **Docs site** (mkdocs-material or Sphinx) generated from docstrings +
  `IMPLEMENTATION_LOG`; hosted via GitHub Pages.
- **CI** (GitHub Actions): headless `pytest`, lint (`ruff`), format (`black`),
  type-check (optional `mypy`); badge in README.
- Dev tooling: `requirements.txt`/optional extras (`[rl]`, `[vision]`),
  `pre-commit`, `.gitignore` for `models/`, logs, caches.
- Demo-recording scripts (offscreen render → MP4/GIF) for the README.

**Acceptance:** fresh clone → `pip install -e .[all]` → `pytest` green in CI →
README shows real GIFs → `openarm --help` lists all demos.

---

### Phase B — Integrated system (unified orchestrator)
**Goal:** one coherent entry point and behavior layer over all capabilities.

- `openarm` **CLI** dispatching every demo/task (`sort`, `plan`, `bimanual`,
  `rl-reach`, later `servo`, `catch`, `mimic`).
- A **scene registry** + shared config so scenes/tasks are discoverable.
- A lightweight **behavior/task layer**: perceive → plan → act state machine that
  the higher phases (vision, RL policies) plug into.
- A polished **showcase scene** and a scripted "grand tour" demo.

**Acceptance:** `openarm <task>` runs each capability; one command runs an
end-to-end showcase; tasks share config/scene infrastructure.

---

### Phase C — Perception & visual servoing
**Goal:** close the loop with vision; foundation for imitation later.

- **Camera pipeline:** offscreen RGB/depth rendering from MuJoCo (wrist camera
  `camera_wrist_right` + an external cam); a `Camera` helper.
- **Object detection in sim:** start with color/blob segmentation; optional
  **YOLO** for richer detection *(optional dataset if needed)*.
- **Image-based visual servoing (IBVS):** drive the EE so the detected object
  centers/approaches in the image, using the image Jacobian + our Cartesian
  controller.
- Demo: arm visually tracks and reaches a colored object it *sees* (no ground-
  truth pose).

**Acceptance:** with object pose hidden from the controller, the arm reaches a
seen target to < 2 cm using only camera input; headless test on rendered frames.

---

### Phase D — Dynamic manipulation (catching a ball thrown through the air)  ✅
**Goal:** time-critical control — predict and intercept. **Done, research-grade.**

- Ball launched on a **ballistic arc** from a random airborne point (`catch_scene.xml`,
  `sample_throw`).
- **State estimation + trajectory prediction:** constant-acceleration **Kalman
  filter** with gravity as a known input; predicts the parabola to < 5 mm at 0.3 s.
- **Interception planning:** `InterceptionSolver` finds the earliest catch point
  reachable within the flight time, gripper facing the incoming velocity; returns
  `None` (honest miss) when unreachable.
- **MPC:** receding-horizon **minimum-jerk** joint replanning to arrive on time;
  **velocity-matched soft catch**, fingers close at the closest approach.
- Demo: `openarm catch` (viewer) / `openarm catch --benchmark` (headless rate).

**Acceptance met:** **160/160 clean mid-air catches across 4 seeds** over wide
randomized throws (descent up to ~60°), ~11 cm mean reach to the interception;
7 headless tests including an end-to-end dynamic-catch check.

---

### Phase D2 — Vision-driven bimanual reactive catching
**Goal:** catch a ball seen *through cameras* (no ground-truth state), with both
arms and no self-collision. Staged: D2a (vision) → D2b (bimanual).

**D2a — vision in the loop  ✅ done.**
- Two RGB-D cameras (`ballcam0/1`) observe the ball; `BallPerception` detects
  (color blob, learned-detector seam ready), deprojects, radius-corrects, and
  fuses to a 3D estimate that feeds the Kalman filter at the camera rate.
- The controller's only ball knowledge is this estimate + the filter's
  prediction — even the final grab uses the estimate, never `data.xpos`.
- **Verified:** ~7 mm fused estimate error, vision-driven catch 9–10/10 (robust to
  +5 mm sensor noise). `openarm catch --vision`.

**D2b — bimanual arbitration  ✅ done.**
- `BimanualCatchController`: one shared Kalman estimator → per-arm interception
  solver (reuses `ArmSpec` + `MIRROR_R2L`); `catch_bimanual_scene.xml` has both
  arms catch-ready (mirrored) + both grasp welds + centre-aimed cameras.
- **Arm selection:** the feasible arm with the largest time margin whose
  ready→catch path is collision-free (`CollisionChecker(avoid_other_arm=True)`,
  `edge_clear`); the other arm holds a safe ready pose.
- **Acceptance met:** ground truth 18/18, vision 12/12; arm choice matches the
  thrown side; **min inter-arm separation ~17–22 cm — zero collisions**.
  `openarm catch --bimanual [--vision]`.

**Refinement — two balls at once (multi-object tracking)  ✅ done.**
- `catch_twoball_scene.xml` (two identical balls), `MultiBallPerception`
  (connected-component blobs → deproject → cluster), `MultiBallTracker` (a Kalman
  filter per ball + nearest-neighbour data association), `TwoBallCatchController`
  (each arm assigned a track, dual simultaneous catch, welds the nearest ball).
- **Verified:** 8/8 both caught (ground truth & vision), distinct balls, no
  collision. `openarm catch --twoball [--vision]`.

---

### Phase E — Reinforcement learning (deepened)
**Goal:** robust, benchmarked RL beyond the baseline reach.

- **Strengthen reach:** reward shaping (precision bonus, reduce action penalty),
  tolerance **curriculum**, vectorized envs, TensorBoard logging → target 90 %+
  at a tighter (≤ 3 cm) tolerance.
- **RL pick-and-place / grasp:** obs includes block + gripper state; staged
  reward (approach → grasp → lift → place); HER for sparse variants.
- **Bimanual RL** (stretch): coordinated dual-arm policy.
- Optional **domain randomization** (masses, friction, target ranges) for
  robustness. Saved checkpoints + reproducible configs + results table.

**Acceptance:** reach ≥ 90 % @ ≤ 3 cm; a pick-and-place policy that places ≥ X %;
all training reproducible from a config + seed; metrics logged.

---

### Phase F — Imitation learning  ✅
**Goal:** learn policies from demonstrations. **Done.**

- **Scripted expert** (`imitation/expert.py`): `ReachExpert` solves IK to the
  target and acts in the env's own action space (clipped joint deltas), so demos
  are on-distribution for the learner.
- **Demo collection** (`imitation/collect.py`): rollouts → self-contained **`.npz`**
  dataset (`obs`, `act`, `ep_lens`); success-only filtering. `openarm bc-collect`.
- **Behavior Cloning** (`imitation/bc.py`): 2×256-MLP `BCPolicy` with **observation
  normalisation baked into the model** (`register_buffer` mean/std → critical:
  obs mixes radians/velocities/positions); Adam + MSE. `openarm bc-train`.
- **BC vs RL head-to-head** (`imitation/eval.py`): same seeds, same env;
  `--compare-rl` loads the trained SAC for a fair comparison. `openarm bc-eval`.

**Acceptance met:** BC trained purely on scripted demos reaches **80–85 %**
(MSE 0.24→0.0009 once obs are normalised) vs **SAC 47 %** on the same seeds;
dataset + training fully reproducible from a seed. 3 headless tests
(expert reliability, policy shapes/normalisation, end-to-end clone-and-reach).

---

### Phase G — Webcam human-arm imitation (capstone)  ✅
**Goal:** the robot mimics the user's arm from a webcam in real time. **Done.**

- **Pose source** (`teleop/pose.py`): a `PoseSource` interface yields the human's
  shoulder/elbow/wrist in a robot-aligned frame. `WebcamPoseSource` wraps
  **MediaPipe Pose** (lazy import — no hard dep on cv2/mediapipe); a deterministic
  `ScriptedPoseSource` stands in for the camera so the whole stack is verifiable
  headless.
- **Retargeting** (`teleop/retarget.py`): task-level, **relative-to-home** ("clutch")
  map — the human wrist displacement (scaled into the robot's reach) drives the
  tool point and the gripper is rotated by the *change* in forearm direction. A
  single **warm-started IK descent** keeps the joint trajectory temporally
  coherent (no solution flips), reusing the project's robust IK.
- **Real-time teleop** (`teleop/teleop.py`): EMA smoothing + per-joint velocity
  limiting + joint-limit clamping before driving the position actuators —
  single arm or both. `openarm mimic [--webcam] [--bimanual]`.

**Acceptance met:** with the synthetic source the tool point tracks the mapped
human wrist to **~9 mm mean (≤15 mm)**, gripper-along-forearm to **~0.1°**, with
**zero IK flips**, peak joint speed ~0.5 rad/s, commands always in limits — for
**both arms**. The live webcam path plugs into the same stack unchanged.

**Full arm + hand upgrade:** added (1) **hand tracking → gripper**
via MediaPipe Hands (fist-closure → open/close); (2) a **pick-up scene**
(`teleop_pick_scene.xml`, weld-on-grasp) so closing your hand near a block grabs
it and opening drops it (block lifted ~10 cm/released). Then, to fix arms that
felt coupled and an upper arm that stayed static, **redesigned the
retargeting to be anatomical and direction-based**: anchored at the shoulder, it
matches your **upper-arm direction and forearm direction** (whole-arm posture; the
hand position is emergent) — so the shoulder-to-elbow segment moves too, and since
it's built from shoulder-relative *directions* it's decoupled from body motion and
the other arm. Verified: upper-arm direction follows within ~14°, output invariant
to body translation, zero IK flips. `openarm mimic [--webcam] [--pick]`. 10 headless
tests.

---

### Phase H — Full integration & showcase  ✅
**Goal:** everything working together as one system. **Done.**

- **Grand-tour showcase** (`openarm showcase`): a curated viewer sequence walking
  the whole stack — classical sorting → RRT planning → bimanual → visual servoing
  → airborne catch → webcam imitation.
- **Full-system integration tests** (`tests/test_integration.py`): prove the
  platform is wired together — every CLI command resolves to a runnable module,
  every registered scene compiles to a valid model, the CLI dispatcher runs, and
  the end-to-end **headless** pipelines (catch: estimate→intercept→MPC; mimic:
  pose→retarget→safe teleop) run without error.
- Final docs/README polish; the `openarm` CLI exposes every capability.

**Acceptance met:** `openarm showcase` exercises perception, planning, learning,
dynamic control, and bimanual/teleop end-to-end; 6 integration tests lock the
wiring in CI. The catch pipeline already chains see→predict→decide→act in one
scenario (vision → Kalman → interception → MPC grasp).

---

## Phase I — Intelligent manipulation (vision-grounded, language-commanded, reactive)
**Goal (startup-grade):** *"Tell the robot what to do in plain language; it sees and
understands the table, plans a collision-free motion, executes — adapting in real
time if things move — and can compute and throw a ball into a bin."* Built
foundation-first; every milestone ends with headless tests + measured numbers.

- **M1 — Multi-object perception & grounding.** Detect several objects (color +
  shape + 3D position) from the cameras; resolve a query like *"the red cube"* to a
  specific object. Builds on `vision/`. *(Open-vocab YOLO/CLIP is a later drop-in;
  color+shape is the dependency-free baseline.)*
- **M2 — Lightweight NL command parser.** *"pick the red cube and put it in the
  bin," "move the red cube left," "take that object out"* → `Intent(action, target,
  destination)`. Compact rule/keyword parser over the detected-object vocabulary —
  no heavy LLM (optional LLM-API mode off by default).
- **M3 — Task executor with obstacle avoidance.** Intent → ground target → **RRT/PRM
  collision-free** path around the other objects/bin/table → grasp/place. Threads
  obstacle avoidance through every motion. Builds on `planners/`, `grasp.py`,
  `pick_and_place.py`.
- **M4 — Reactive / dynamic replanning.** ⏸ *Deferred.* Re-perceive while executing;
  if the target or an obstacle moves, replan in real time and continue (the catching
  MPC idea applied to reaching). A first closed-loop servo version was built but was
  unreliable (the arm got stuck reaching) and **removed**; to be redone properly at
  the end, likely on the planned/MPC machinery rather than a raw joint servo.
- **M5 — Dynamic throwing into a bin.** Detect the bin; invert the projectile
  equations for a release velocity; **optimize the joint swing** to hit that release
  point+velocity within limits; timed gripper release; **only throw if the bin is in
  the achievable envelope**. Builds on `catching.py` ballistic math.
- **M6 — Complex skill: stacking / insertion ✅.** Stacking — `stack(target,
  support)` grasps an object and places it on another's perceived top (a point-down
  descent so the held object lands aligned, ~2 mm), driven directly and by language
  ("stack the red cube on the green cube"); a `home()` step clears the arm from the
  camera between commands. Peg-in-hole — `insert(peg, socket)` grasps a cylindrical
  peg and threads it into a socket with a precise point-down descent (a few mm radial
  clearance → seats **1 mm off-centre, upright**); `openarm insert` / "insert the peg
  into the socket". A **selectable family of matched peg/hole shapes** of increasing
  alignment difficulty (`openarm insert --shape round|square|cuboid`): **round** — a
  cylinder into a *circular* (octagonal-ring) hole, no yaw alignment; **square** — a
  square peg into a *rotated square* hole, **4-fold** symmetric (snap to the nearest
  reachable 90° multiple); **cuboid** — a rectangular block into a *rotated rectangular*
  slot, **180°** symmetric (the tightest). The two aligned cases share
  `_insert_aligned(peg, socket, n_fold)`: grasp at a reachable yaw, rotate the block's
  cross-section to whichever symmetry-equivalent socket orientation is reachable, and a
  6-DOF fixed-orientation descent threads it in → seats **~1–4 mm off-centre, ~0° yaw
  error**. All classical control + optimization, not RL. (Deformable folding deferred as
  research-grade; a learned ACT/Diffusion policy from teleop demos is the startup-grade
  route to such skills.)

**Decisions (locked):** build order foundation-first (M1→M6); lightweight language
parser; **open-vocabulary** detection (YOLO-World, hardening deferred); complex
skill = stacking/insertion.

**Status:** M1 ✅ (perception + grounding), M2 ✅ (NL parser), M3 ✅
(obstacle-avoiding executor; `openarm manipulate`), M4 ⏸ **deferred** (the
closed-loop reactive object-following picker was built but was unreliable — the
servo got stuck reaching for the object — so it was **removed for now** and will be
re-implemented properly once the rest of the stack is stable), M5 ✅ (dynamic
throwing into a bin; `openarm throw`, incl. a **5-bin / 5-ball multi-target show**:
`openarm throw --multi`),
M6 ✅ (**stacking** — block-on-block and "stack the red cube on the green cube",
`openarm stack`; **peg-in-hole insertion** — `openarm insert`). All six milestones
done. Refinements done from live testing: **adaptive grasp height** (grasp the
object's perceived top — fixes pressing on tall objects), a **stateful multi-turn
session** (held-object memory, "it" resolution, "go to"/"release" primitives), and
**carried-object collision avoidance** (the held object is re-posed along the path
and must avoid the environment + other objects). (Detection hardening via synthetic
fine-tuning landed later — see "Next milestones → Phase 2" below.) 123 headless tests.

**Bimanual coordination (from live feedback — make both arms work).**
- **Bimanual simultaneous stacking** (`openarm bimanual --mode stack`): on a **wide
  table**, both arms build a tower on their own side at the same time. The left arm
  runs the exact joint-space **mirror** of the right arm's (collision-free) stack
  plan, so the motion is symmetric and sidesteps the left arm's IK branch jumps.
- **Intelligent dual-arm pick-and-place with hand-over** (`BimanualCoordinator`;
  `openarm bimanual --mode coordinate`): given an object and a destination, it picks
  the arm on the object's side (if feasible), and if **that arm can reach the
  destination it finishes alone**; if it **can't reach the destination but the other
  arm can**, it carries the object to a centre midpoint, **hands it to the other
  arm**, and that arm completes the place. Demo: the right arm bins the right-side
  cube alone, and the left arm hands the left-side cube over to the right arm to
  reach the right bin. Built on planned, collision-checked motions (reliable); the
  left arm is planned by **mirroring** a right-arm plan (its own IK branch-jumps).
- **Interactive bimanual playground** (`openarm interactive`): a **large table** with
  objects spread across both arms' reach and a bin on each side. The sim opens, the
  camera detects the objects, the **terminal lists them**, and you pick one + a
  destination bin; the coordinator carries it out (single-arm or hand-over). Two
  scenes: coloured blocks (`bimanual_table_scene`, colour/shape detection) and **real
  Google Scanned Objects** (`openarm interactive --scanned`, `scanned_table_scene`).
  For the scanned objects, detection is **depth-localisation + open-vocab labelling**
  (YOLO-World default, `--detector yoloe`): a top-down view can't reliably *name* small
  objects, so you **select by number** and the label is a best-effort hint; the meshes
  are the real scanned visuals with a box collision for a reliable grasp. *(This
  best-arm + hand-over intelligence is the pattern to extend to other tasks next.)*
- **Open-vocab backends.** `OpenVocabDetector` supports YOLO-World (default; lighter and
  scored higher on our top-down renders) and **YOLOE** (`backend="yoloe"`).
- **Bimanual start-pose fix.** All bimanual scenes now start **both arms raised** (left
  = joint-space mirror of the right); the left arm previously started parked low and
  could clip the table when moving (the bimanual coordinator runs trajectories
  open-loop, with no per-move collision check). Objects sit clear of the bins and of
  both arms' camera shadows.

**Conversational interaction polish.** The session now handles:
- **Multi-step commands** — `sess.run("pick up the red cube then put it in the bin")`
  splits on "then"/"after that"/";" and runs each clause in order ("pick X *and* put
  it in Y" stays one intent).
- **State queries** — "what are you holding?" → the held object; "what's on the
  table?" → the objects perception sees.
- **Undo / "put it back"** — remembers the last relocation and returns that object to
  its origin (ignoring the bin while extracting, so it can come back out).
- **Clarification** — when a requested object isn't found, the reply lists what the
  robot *can* see. `openarm manipulate` showcases all of these.

*(A dexterous 16-DOF Allegro hand replacing the parallel gripper was prototyped and
then **removed**: getting the mount/arm integration to look and behave right — and
load-bearing real finger contact — proved research-grade and not worth the complexity
for now. The parallel gripper remains the end-effector across all skills. A different
end-effector or in-hand-manipulation effort may revisit this later.)*

---

## Next milestones

In order. Each milestone keeps the project's discipline: new CLI + `--help`, README /
ROADMAP / implementation-log updates, and headless tests.

- **Phase 1 — Bimanual best-arm + hand-over, via language. ✅ DONE.** `BimanualSession`
  (`agent/bimanual_session.py`) grounds an object label + a destination through perception
  and drives `BimanualCoordinator`: "grab X" holds it with the better-placed arm; "move/
  transfer X to the left/right bin" auto-selects the arm and **hands the object over** when
  only the other arm can reach the destination. Adds held-state, "which arm is holding it?"
  queries, undo, and clarification. New coordinator methods `pick`/`place_held` (+
  `PickPlaceController.plan_pick`/`plan_place_held`) are additive — `pick_place` is unchanged.
  `openarm bimanual --mode language ["…"] [--interactive]`. Tests: parse, grab→query→place-held,
  one-shot transfer-with-hand-over.
- **Phase 2 — Perception hardening (synthetic fine-tuning). ✅ DONE.**
  `vision/synthgen.py` auto-labels synthetic data from MuJoCo's **segmentation renderer**
  (exact per-object masks → 2D boxes, zero manual labeling) under **domain randomisation**
  (camera pose, lighting, object placement/yaw, colour jitter) and writes a YOLO dataset
  (`openarm gen-data`). `vision/finetune.py` fine-tunes / evaluates a YOLO on it
  (`openarm detect train|eval`; ultralytics, **user-run**). `vision/multiview.py`
  `MultiViewPerception` fuses a top-down + an angled camera by 3D proximity (most-confident
  label wins) to resolve single-view ambiguity, using one renderer. A fine-tuned model drops
  into the existing `OpenVocabDetector(model=…)`. All additive; the colour/shape fallback and
  existing perception are unchanged. Tests: seg-bbox, dataset writer, multi-view fusion.
- **Phase 3 — Advanced manipulation.**
  - **3a Non-prehensile pushing ✅ DONE.** `pushing.py` `PushController.push(object,
    target_xy)`: approach just behind the object (far side from the target), push along
    the object→target line with the **closed gripper** at a reachable point-down
    orientation held continuously (`_ik_line_oriented`, so the wrist never rotates
    mid-push and flings it), and **re-aim after each stroke** until the object lands on
    the goal. No weld, no grasp. `move_puck_scene.xml` (puck + two goal regions),
    parser `push` action, `openarm push [--goal a|b]`. Lands the puck **~30–40 mm** from
    the goal. Tests: parse-push, push-onto-each-goal (stays on the table).
  - **3b Tool use ✅ DONE.** `pushing.py` `ToolController`: grasp a stick (weld), record
    the tool-tip offset + held orientation, then push a block that's **beyond the bare
    arm's reach** onto a far goal with the stick's tip — the gripper sweeps along an
    offset line so the tip tracks the object, re-aiming each stroke (the tip is angled by
    the grasp; the offset accounts for it). `tool_scene.xml` (stick + block + goal all
    past the ~0.36 m bare reach), `openarm tool`. Moves the block ~84 mm onto the goal
    (~39 mm). Tests assert the targets are bare-unreachable (tool is *needed*) and the job
    gets done. *(Reach-extension is kinematically tight for this 7-DOF arm — the forward
    reach + base proximity leave a narrow band; the demo is tuned within it.)*
  - **3c Cloth folding ✅ (scripted fold done; learned policy = next research step).**
    `cloth.py` `ClothFoldController` folds a **MuJoCo flex-grid cloth**: each grid vertex is
    a body, so the gripper grasps a **corner** via a weld, lifts it, and carries it across
    (continuous `_ik_line_oriented`, branch-safe near the edges) to fold the sheet onto the
    opposite edge. `cloth_scene.xml`, `openarm cloth [--corner …]`. Folds the corner to
    ~20 mm of the target and **shrinks the cloth's extent ~148→86 mm** (it folded), stably.
    Tests assert the flex cloth simulates without diverging and that the fold reduces the
    extent. *Honest scope:* a **learned ACT/Diffusion folding policy** (the original
    "standout learned skill") is a separate research effort; the scripted fold here is the
    deformable-manipulation foundation **and** a demonstration source for it, and the
    project's imitation infra (`imitation/`) is the training hook.

  *(Rope/cable manipulation was considered and **dropped** — hardest to get a clean,
  strong result; revisit only if a later need arises. Benchmarks remain parked until
  explicitly requested.)*

### Future plans (deferred) — learned cloth-folding policy

A *learned* (not scripted) folding policy is the natural research-grade extension of 3c.
Deferred by choice; captured here so it's actionable later. It reuses the existing
imitation stack (`imitation/`, modeled on the RL Gym envs); each stage is additive and
headlessly testable.

1. **Cloth-fold Gym env** (`rl/cloth_fold_env.py`, like `rl/reach_env.py`): obs = cloth
   state + arm state; action = `Box(8)` (7 joint deltas + grasp toggle / auto-weld a corner
   when closing near it); `reset` randomises the cloth pose; reward = the existing
   fold-quality metric (extent-shrink + corner-to-target) — only needed if RL is also wanted.
2. **Cloth-fold expert** (`imitation/expert.py` `TASKS["cloth_fold"]`): wrap the working
   `ClothFoldController` to emit actions in the env's action space; collect demos with
   `openarm bc-collect --task cloth_fold` (domain-randomised starts).
3. **Policy** (increasing power/effort): **BC** (existing `imitation/bc.py` MLP — baseline,
   long-horizon drift expected) → **ACT** (action-chunking transformer, new `imitation/
   act.py` — the realistic learned fold) → optional **Diffusion Policy**.
4. **Observation choice (biggest lever):** *privileged* cloth-vertex state first (low-dim,
   far easier); *vision* (depth/RGB → CNN encoder) is the realistic, research-grade version.

Realistic outcome of a focused effort: a privileged-state **ACT** policy that folds reliably
on in-distribution starts — the repo's first non-baseline learned result. Training is
GPU/user-run (the RL/BC pattern). Vision + Diffusion + sim-to-real are beyond a quick effort.

**Approach decision (algorithms vs. learning).** Stacking and insertion are built
with **robust classical control + optimization** (precise IK, a point-down
straight-line descent, collision-free RRT, and — for throwing — a
simulation-in-the-loop release optimizer), not RL. In sim these are deterministic,
verifiable to the millimetre, and need no training data; RL would add variance and
a training pipeline for no accuracy gain here. Learned policies (ACT/Diffusion from
teleop demos) remain the route reserved for genuinely contact-rich / deformable
skills (e.g. folding), per the locked plan.

**M5 throwing — accuracy hardening (advanced control).** The first throw had a
persistent ~93 mm overshoot. Root-caused to two compounding swing-shaping bugs:
the rest-to-rest quintic peaks at **1.875×** its amplitude/time ratio (so the EE
swept ~1.9× faster than the ballistic solution needed — the ball always overshot
and only the bin's far wall stopped it), and the fixed ±0.30 rad wind-up
**saturated against the joint limits** (so the speed/range knob did nothing).
Fixed by **sizing the swing amplitude from the desired peak velocity**
(`half = speed·qd_rel·T/3.75`, so the midpoint speed *equals* the ballistic
velocity and the swing stays off the limits) and replacing the biased analytic
release with a **simulation-in-the-loop release search**: roll the *true* swing
forward on a saved state, evaluate the actual landing per release step (coarse→
fine around the swing midpoint), and pick the step that genuinely lands closest;
escalate the continuous swing-**speed** knob for bins near the envelope edge.
**Result: 5–9 mm landing error** across a spread of 7 narrow-bin (16 cm) targets
(was ~93 mm), and out-of-envelope bins are correctly **refused**. Benchmark:
`benchmarks/throwing_benchmark.py` (precision/consistency map + reachability
envelope → CSV + figures); precise-throw envelope is forward x ∈ [0.56, 0.67].

**Multi-target show (`openarm throw --multi`).** A `throw_multi_scene.xml` with
five balls on a table and five narrow (12 cm) bins spread across the envelope; the
arm grasps each ball and throws it into a different bin, **landing all 5/5** (the
table collision stays on so the remaining balls keep resting on it). Each throw is
reset to a pristine start config (a clean arm pose, with the already-landed balls
kept in their bins) — throwing from the previous frozen follow-through degraded
accuracy, and resetting rather than sweeping the arm back avoids knocking the
balls still on the table.

---

## Sequencing & dependencies

```
A (repo) ──┐
           ├─> B (integration) ──┐
C (vision) ┘                     ├─> H (full integration / showcase)
D (catch) depends on C(pose/predict)+fast control ┘
E (RL deepen) ── independent, can run in parallel ─┘
F (imitation) depends on B + (G teleop for real demos)
G (webcam)   depends on C (camera/retarget infra) ── capstone before H
```

**Recommended order:** **A → B → C → E → D → F → G → H**
(A/B make it a real project; C unlocks vision for D/G; E proceeds in parallel;
G is the headline capstone; H ties it together.)

---

## External resources / datasets (to source when each phase starts)
- **Phase C/visual:** optional YOLO weights or a small in-sim detection dataset.
- **Phase F/imitation:** HuggingFace/LeRobot datasets, or demos we generate in-sim.
- **Phase G/webcam:** MediaPipe / MoveNet pose models (downloaded at runtime).

---

## Scope
- All phases demoable via the `openarm` CLI.
- Capstone: live webcam human-arm imitation, single and bimanual.
