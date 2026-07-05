# OpenArm Control & Simulation — Implementation Log

A running record of everything implemented and changed, for end-of-project
documentation. Newest milestones at the bottom.

---

## Phase 0 — Foundation audit & fixes (single right arm)

Started from the Enactic OpenArm v2 bimanual MuJoCo model plus a partially
working `openarm_control/` package. An audit found the "working foundation" had several
real bugs; all were fixed and locked behind an automated test suite.

### Bugs found & fixed
| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| 1 | IK unreliable (~148 mm avg error, oscillating) | `mj_jac` read a **stale Jacobian** — the IK loop called `mj_kinematics` but never `mj_comPos`, so `data.cdof` stayed frozen at the seed config | Call `mj_comPos` before every Jacobian; rewrote IK as adaptive Levenberg–Marquardt + nullspace + restarts |
| 2 | Table/blocks/bins **unreachable** | Objects placed outside the right arm's workspace (bins 140–330 mm out, blocks on the wrong side) | Rebuilt scene inside the measured top-down-graspable workspace |
| 3 | Blocks **teleported to origin** on keyframe load | A partial keyframe zeros the free-joint qpos | Full 39-value keyframes that place the blocks |
| 4 | Cartesian controller **stalled** (187 mm droop) | Commanded one timestep ahead of the *actual* (gravity-sagged) joints | Integrate an internal desired joint state `q_des` |
| 5 | Gripper polarity **inverted** | Code assumed `ctrl=0` open; actually `ctrl=0` closes (30 mm), `ctrl=-0.785` opens (134 mm) | Fixed `GRIPPER_OPEN_CTRL=-0.7854`, `CLOSED=0.0` |

### Modules (final state of Phase 0)
- `openarm_control/config.py` — names, gains, grasp offset, gripper polarity, IK params.
- `openarm_control/kinematics.py` — `OpenArmKinematics` (FK, tool-point Jacobian, robust
  IK; configurable joint set / site / tool offset). `orientation_error` helper.
- `openarm_control/controller.py` — `CartesianController` resolved-rate, integrates
  `q_des`, position & 6-DOF, gravity-aware.
- `openarm_control/grasp.py` — `GraspSolver` top-down grasp IK with wrist-yaw search.
- `openarm_control/gravity_compensation.py` — mass matrix / bias-force utilities.
- `openarm_control/trajectory.py` — quintic joint & Cartesian trajectories.
- `tests/test_foundation.py` — headless pytest suite (FK, IK round-trips,
  reachability, controller convergence, trajectory smoothness, …).

### Verified numbers
- IK round-trip: position 0.03 mm mean, 6-DOF 0.009 mm / 0.002°, 100% success.
- Controller position tracking: < 0.1 mm on reachable targets.
- Grasp point offset: 0.135 m below the wrist site (fingertips ~0.164 m).

---

## Track 1 — Classical autonomy & motion planning  ✅

### 1a. Pick-and-place (`openarm_control/pick_and_place.py`)
- Motion = list of segments (polyline of joint configs + gripper state +
  duration), executed with quintic time-scaling.
- Carry path: **lift** fixed-yaw vertical, **transport + place-descend**
  point-down with free yaw (`_ik_line_down`, a 5-DOF position+approach-axis IK)
  so the gripper stays vertical and the object lands on target; all seeded
  continuously to avoid IK branch jumps.
- **Gravity compensation** via `data.qfrc_applied = qfrc_bias` keeps the
  position actuators tracking accurately (else cm-level droop de-centers grasps).
- **Grasp:** pure-physics top-down grasping plateaued at ~40% reliability even
  after raising finger `kp` 30→150 and block friction to 2.5 (blocks slip on
  lift, branch-dependent). Switched to a **weld-assisted
  grasp** (scene welds `grasp_red/green/blue`; controller `attach()/detach()`
  set the weld relpose to the current grasp and toggle `data.eq_active` on
  close/open). Fingers still physically close. Result: ~95%+ reliable, exact
  placement. This is the standard MuJoCo manipulation/RL technique and is
  important for stable RL reward later.

### 1b. Scene redesign (`single_arm_scene.xml`)
- Bigger table (0.40×0.46 m, top z=0.40).
- **3 blocks** (40×40×60 mm pillars, x=0.18) and **3 bins** (red/green/blue,
  x=0.36) on lanes y = -0.12 / -0.25 / -0.38, 0.13 m apart for gripper clearance.
- Blocks are 60 mm tall so a top-down grasp clamps the sides while the
  fingertips clear the table.
- Obstacle wall **removed** from the sorting scene (it had spanned the full width
  in front of the blocks and collided with the open gripper during grasps).
- `openarm_control/autonomy.py` — `SortingTask` reads block/bin positions live and sorts
  each block into its matching bin. Verified 3/3 headless.

### 1c. Motion planning / obstacle avoidance
- `openarm_control/planners/collision.py` — `CollisionChecker` classifies arm vs.
  environment geoms once; `in_collision(q)` and `edge_clear(q1,q2)` with
  save/restore; supports ignoring a grasped body.
- `openarm_control/planners/rrt.py` — **RRT-Connect** (bidirectional) + path
  shortcutting. Finds a collision-free joint path around the obstacle in ~0.5 s.
- `openarm_control/planners/prm.py` — PRM roadmap using the same checker.
- `single_arm_scene_obstacle.xml` — separate scene with a wall between the pick
  and bin lanes; the wall has a collision `margin` (planning clearance) and a
  large `gap` (no contact force) so it acts as a virtual planning obstacle.
- `demo_motion_planning.py` — plans around the wall (RRT or `--planner prm`) and
  executes the path.
- Known follow-up: on some random seeds the executed path grazes a bin wall near
  the goal (endpoints sit ~11 mm from the wall); hardening pending.

### Demos (`openarm_control/demos/`)
`demo_fk`, `demo_ik`, `demo_cartesian_control`, `demo_trajectory`,
`demo_gripper`, `demo_pick_and_place` (sorting), `demo_motion_planning`.

### Tests: 21 passing (`python -m pytest tests/`), fully headless.

---

## Track 2 — Bimanual coordination  ✅

### 2a. Generalized the code to either arm (`ArmSpec`)
- `config.ArmSpec` captures everything arm-specific: joints, actuators, gripper
  actuator/site/body names, grasp offset, and **gripper polarity** (right opens
  at -0.7854, left at +0.7854; both close at 0.0). `RIGHT_ARM`, `LEFT_ARM`.
- `GraspSolver`, `CartesianController`, `PickPlaceController` now take an `arm=`
  argument (default `RIGHT_ARM`, so all single-arm code/tests are unchanged).
- Weld naming standardized to `grasp_{arm}_{obj}` (e.g. `grasp_right_red`).

### 2b. Bimanual scene (`bimanual_scene.xml`)
- Wide shared table (y ∈ [-0.47, 0.47], top z=0.40).
- 4 blocks (r1,r2 on the right; l1,l2 on the left) and 4 matching bins.
- 8 grasp welds (each arm ↔ each block) for parallel sort + hand-off.
- `ready` keyframe places both arms in symmetric hover poses (nq=46).

### 2c / 2d. `openarm_control/bimanual.py`
- `BimanualController` — one `PickPlaceController` per arm, shared model/data,
  gravity-comp for both.
- `_ArmRunner` — per-arm state machine advancing one timestep at a time through
  a flat list of motion segments (with grasp-weld attach/detach), so two arms
  can be stepped in **lockstep**.
- **`ParallelSort`** (headline) — both arms sort their side's blocks into the
  matching bins **simultaneously**. Verified 4/4 headless.
- **`synchronized_move`** — both arms move at once along mirrored joint paths.
- **`RelayHandoff`** — right arm relays a block to a shared midpoint
  (0.24, -0.05); left arm picks it up and places it in a left bin (object
  transfer across the workspace). Verified headless.
- Demo: `openarm_control/demos/demo_bimanual.py [--mode sort|sync|handoff]`.

Design note: the table was widened and given blocks/bins on both sides
specifically so the two arms do useful work *simultaneously*,
rather than one arm idling during a hand-off.

### Tests: 24 passing total (`tests/test_foundation.py` + `tests/test_bimanual.py`).

### 2e. Coordination refinements
- **Perfect mirrored synchronization.** The left arm now follows the exact
  joint-space mirror of the right: `q_left = MIRROR_R2L * q_right` with
  `MIRROR_R2L = [-1,-1,-1,+1,-1,-1,-1]` (derived from the mirrored joint axes;
  only joint 4 keeps its sign). Verified to reproduce the y-reflected EE pose to
  0.000 mm / 0.000 deg. `synchronized_move` now takes only the right-arm path and
  mirrors it, so the two arms are perfectly symmetric (previously each arm solved
  IK independently and drifted).
- **Collision-aware hand-off.** `CollisionChecker` gained `arm_name` +
  `avoid_other_arm` so an arm can treat the *other* arm (at its current pose) as
  an obstacle. `BimanualController.path_blocked()` checks a planned path against
  the other arm. `RelayHandoff` now: right carries the block to the midpoint and
  releases → **checks whether the right arm blocks the left arm's planned path**
  → if so, intelligently moves the right arm to a clear pose first → then the
  left arm picks up and places. Verified: hand-off completes with 0.00 mm
  right↔left arm penetration (previously the arms collided in the middle).
- Tests: 25 passing (added mirror-symmetry and inter-arm no-collision tests).

---

## Track 3 — Reinforcement Learning / Gymnasium  (in progress)

### Reach task environment (`openarm_control/rl/reach_env.py`)
- `OpenArmReachEnv(gymnasium.Env)` — the right arm learns to drive its
  end-effector to a randomly placed target in the reachable workspace.
- **Scene** `reach_scene.xml`: arm on pedestal + a floating target marker
  (mocap), no clutter; sim timestep 0.002 s.
- **Observation (23):** joint pos(7), joint vel(7), EE pos(3), target(3),
  EE→target(3).
- **Action (7):** joint-position deltas in [-1,1] × 0.04 rad, applied to the
  position actuators (gravity-compensated each substep so tracking is clean and
  the policy learns *where* to move). 10 substeps/step → 50 Hz control.
- **Reward:** −distance − 0.01·‖action‖ (+1 success bonus within 5 cm).
- **Episode:** terminates on success, truncates at `max_steps` (120 → ~2.4 s).
- Passes `stable_baselines3.common.env_checker.check_env`. Env dynamics verified
  sane: a Jacobian step toward the target reduces EE distance on every seed.

### Training & eval
- `openarm_control/rl/train.py` — trains **SAC** (SB3, MLP [256,256]); reports
  success-rate / mean-distance before & after; saves to `openarm_control/rl/models/`.
  `python openarm_control/rl/train.py --timesteps 200000`.
- `openarm_control/rl/eval.py` — loads a saved policy and runs it in the MuJoCo viewer
  (green sphere = target).
- Tests: `tests/test_rl.py` (env compliance, reset/step contract, distance
  decreases toward target, success terminates) — fast, no training.
- Result: SAC trained 200k steps learns to reach (≈50% strict / ~73 mm on hard
  target seeds, reliable on typical targets). To be strengthened in Phase E.

---

## Phase A — Repository hardening  ✅

Made the project a proper installable, tested package (per ROADMAP.md;
"eventually/unsure" publishing level → solid packaging + README now, CI/docs-site
/GIFs deferred).

- **`pyproject.toml`** (root): installable as `openarm-control` (import path
  `control`), `pip install -e .` with extras `[rl]`, `[vision]`, `[dev]`, `[all]`.
  Core deps: mujoco, numpy, scikit-learn, networkx. ruff + pytest config included.
- Added missing `__init__.py` (`openarm_control/planners`, `openarm_control/demos`); planners
  package now exports `CollisionChecker`, `RRTPlanner`, `PRMPlanner`.
- **README** rewritten around the platform (overview, capability table, install,
  quickstart, structure, credits to Enactic for the upstream model).
- **`.gitignore`** extended (models/, logs, tensorboard, caches, build/dist).
- **`requirements.txt`** convenience file. **`ROADMAP.md`** added at repo root.
- Verified: `pip install -e .` clean; planners/bimanual import; 29 tests pass.

### Package rename (`control` → `openarm_control`)
The import package was renamed from `control` to **`openarm_control`** because
`control` **collides with the popular `python-control` library** (PyPI `control`,
present on the dev machine) — running `openarm` from outside the repo imported
the wrong package. Renamed the directory and all absolute imports (relative
imports unchanged), updated `pyproject.toml` packages + entry point, README,
ROADMAP, and this log. Verified `openarm_control` now imports from the repo even
outside it, and the `openarm` console command works.

---

## Phase B — Integrated system (unified CLI)  ✅

- **`openarm_control/cli.py`** — the `openarm` command (also `python -m
  openarm_control.cli`). A `COMMANDS` registry maps each capability to its demo
  module (every demo module exposes `main(argv=None)`), so:
  `openarm sort | plan | bimanual [--mode] | gripper | trajectory | fk | ik |
  cartesian | rl-train | rl-eval`, plus `list`, `scenes`, `showcase`, `test`.
- **Scene registry** `config.SCENES` (single / obstacle / bimanual / reach),
  surfaced via `openarm scenes`.
- **`showcase`** runs a curated sequence (sort → plan → bimanual).
- Console entry point registered in `pyproject.toml` (`openarm = openarm_control.cli:main`).
- Demo `main()` signatures unified to `main(argv=None)` for clean dispatch.
- Tests: `tests/test_cli.py` (all command modules import + expose `main`,
  `list`/`scenes` run, unknown command → exit 2, registered scenes exist,
  showcase commands valid). **34 tests passing total.**

---

## Phase C — Perception & visual servoing  ✅

Closes the loop with vision (foundation for Phases D & G).

- **`openarm_control/vision/camera.py`** — `Camera`: offscreen RGB + depth from a
  named MuJoCo camera (`mujoco.Renderer`), pinhole intrinsics from the camera
  FOV, and `deproject(u,v,depth) → world point` (MuJoCo camera convention:
  x right, y up, looks down −z). Verified working headless (EGL/GL available).
- **`openarm_control/vision/detection.py`** — `detect_color`: dependency-free
  channel-dominance color blob detection → centroid / bbox / area / mask
  (red/green/blue). A YOLO detector can later slot behind the same interface.
- **`openarm_control/vision/servoing.py`** — `VisualServo` (PBVS): estimates the
  object's 3D world position from the camera *only* (detect → read depth at the
  centroid → deproject) and drives a Cartesian controller toward it; keeps the
  last estimate when the view is occluded by the moving arm.
- **`vision_scene.xml`** — a single red cube + a near-top-down `topcam`, and a
  `park` keyframe (arm parked aside so it doesn't occlude the first estimate).
  Also added an `external` camera to `single_arm_scene.xml`.
- Demo `demo_visual_servo.py` (`openarm servo`) — the arm **sees, reaches, and
  grabs** the cube using camera input only.
- Verified: vision estimate within **4.8 mm (xy) / 0 mm (z)** of ground truth;
  the servo reaches the seen cube (grasp point at the cube) using no true pose.
- Tests: `tests/test_vision.py` (render, detect, deproject < 2 cm, servo reaches
  < 3 cm, **grasp-and-lift**). **39 tests passing total.**

Notes / fixes:
- Color must be unique — the first attempt detected the block AND a same-colored
  bin and put the centroid between them; a dedicated single-object scene +
  near-top-down camera fixed accuracy.
- First grasp left the cube **floating below the gripper**: the servo targeted
  the cube's *top surface* (offset −0.005), so the weld captured the cube ~2 cm
  below the grasp point and lifting left it hanging. Fixed by (a) a taller cube
  pillar (50×50×60 mm) so a center grasp clears the table, (b) targeting the cube
  *center* (offset −0.03), and (c) the correct order: descend → close fingers →
  weld → lift. Cube now rises 171 mm enclosed (9.5 mm gap).

---

## Phase E — Reinforcement learning (deepened)  (in progress)

### E1. Strengthened reach
- Reward shaping in `reach_env.py`: dense `−distance` **+ a sharp near-target
  precision bonus** `0.5·exp(−(d/σ)²)` (σ=3 cm) to drive sub-cm accuracy where
  the bare `−dist` gradient is weak; action penalty cut 0.01→0.001; success
  bonus 1→2; **tolerance tightened 5 cm → 3 cm** (`success_tol` configurable).
- `train.py` now logs to **TensorBoard** and reports success/metric before vs.
  after. (Baseline was ~50 % @ 5 cm after 200 k; full retrain at the new reward
  is run separately — target ≥ 90 % @ ≤ 3 cm.)

### E2. RL pick-and-place (`pick_place_env.py`, `rl_pick_scene.xml`)
- `OpenArmPickPlaceEnv` (Gymnasium): approach a block, grasp it, carry it to a
  randomized target on the table. **Obs (31)**, **action (8)** = 7 joint deltas
  + 1 gripper. Grasp uses the scene weld **auto-activated** when the policy
  closes the gripper with the block between the fingers (reliable "magnetic"
  grasp; pure-friction grasping is too unreliable for a usable reward). Staged
  reward: approach → +grasp bonus → carry-to-target → +success (≤ 5 cm).
- Verified: passes `check_env`; a Jacobian **scripted oracle solves 5/5**
  (grasp + place), confirming the mechanics, auto-weld, and reward are sound.

### RL tooling
- `openarm_control/rl/__init__.py` exposes a `TASKS` registry; `train.py` and
  `eval.py` are task-agnostic: `openarm rl-train --task reach|pick`,
  `openarm rl-eval --task reach|pick` (saves `models/<task>_sac.zip`).
- Tests: `tests/test_rl.py` now covers both envs (compliance + the pick oracle).

---

## Phase D — Dynamic manipulation (catching a thrown ball)  ✅

- **`catch_scene.xml`** — a ball (free body) the env/demo launches toward the arm.
- **`openarm_control/catching.py`**:
  - `BallPredictor` — least-squares fit of a constant-acceleration (gravity)
    model to observed positions; predicts a future point to < 1 cm.
  - `CatchController` — finds a **stable interception point** where the ball
    crosses a fixed catch plane `x = X_CATCH` (so the arm commits and waits
    rather than chasing a moving target), pre-positions the gripper there with a
    fast Cartesian controller (gain 10, vel 5), and **welds the ball on contact**.
- Demo `demo_catch.py` (`openarm catch`); CLI `catch`.
- Verified: **12/12** catches in the tuned launch envelope; the headless test
  asserts ≥ 75 % over randomized throws. **44 tests passing total.**

### Catch reworked into a REAL grasp
The first version welded the ball wherever it touched the robot (near the wrist)
— a fake catch: the ball just stuck to the side of the hand. Reworked:
- A ball *falling from above* can't be caught in the finger gap (an upward "cup"
  pose is unreachable; the ball hits the back of the down-pointing hand). So the
  task is now a ball **sliding across a table** at constant height.
- The gripper waits **top-down with fingers open**, fingers closing in y, so the
  sliding ball passes BETWEEN the vertical fingers; the catch triggers only when
  the ball is in the **finger gap** (near the grasp point), then the fingers
  physically **close** around it (the weld only stabilizes a ball genuinely
  between the pads — like the pick-and-place grasp).
- Key bug: the catch-ready keyframe had the fingers at qpos 0 = **closed** for
  the right gripper (open = −0.7854), so the ball was blocked before reaching the
  gap. Opening the fingers in the keyframe fixed it.

Further fixes (symptoms — "gripper stays open / ball rotates
in the air"):
- **Gripper wasn't actually closing:** `CartesianController.step()` re-commands
  the gripper to *open* every step (its `target_gripper` defaults to 0), which
  overrode the catch's close. Fixed by closing via the controller
  (`ctrl.target_gripper = 1.0`) so it isn't reopened; the fingers now grip the ball.
- **Arm reoriented ~75° / ball spun during the lift:** the resolved-rate
  Cartesian lift reconfigured the redundant arm, and the ball (welded while still
  moving at ~1.3 m/s) swung. Fixed by (a) zeroing the ball's velocity on catch
  (the gripper absorbs the momentum) and (b) replacing the Cartesian lift with a
  **joint-space interpolation** to a pre-solved lifted config (seeded from the
  catch config, continuity-checked), so orientation is held to ~0.2°.
- **Longer runway table** (x ∈ [0.06, 0.70], lower friction) so the ball slides
  ~0.36 m → the predictor gets more frames and the arm more reaction time
  (catch at t≈0.25 s vs 0.10 s before).
- Verified: **8/8 clean catches** — fingers closed, ball lifted ~58 mm, arm
  orientation error 0.2°, no spin. Test asserts ≥ 75 % clean (lifted + gripped +
  no rotation).

## Phase C fix — visual servo no longer pushes the cube
The servo used **position-only** control, so the gripper descended at an
uncontrolled orientation and shoved the cube. Rewrote `demo_visual_servo` as
**vision-guided pick-and-place**: estimate the cube's 3D position from the
`topcam` image (color detect + depth deproject, camera-only), then feed it to the
proven top-down pick pipeline (vertical descent → no pushing) and place it.
Verified headless: estimate within 4.5 mm of truth; cube placed on target with no
push. Cameras documented: `topcam` (near-top-down, used) + the model's
`camera_wrist_right/left`.

## Phase D redesign — research-grade airborne catching (MPC + optimal interception)  ✅
The earlier sliding-ball catcher (above) was a placeholder: it fit a constant-
velocity line and **parked the hand at the single arrival point**, then waited.
No real-time tracking, no dynamics, ball on a table. Rebuilt from scratch into a
full robotic ball-catching pipeline that catches a ball **thrown through the air
from a random launch point**.

**Scene** (`catch_scene.xml`): table removed; the ball is a 70 mm sphere launched
on a ballistic arc (env/demo set its launch pose + velocity); caught mid-air, a
miss falls to the floor. New forward catch-ready keyframe (baked from the
feasibility probe) places the open gripper at the centre of the reachable volume.

**Pipeline** (`openarm_control/catching.py`):
1. **`BallisticKalmanFilter`** — constant-acceleration Kalman filter with gravity
   as a known control input. From noisy position observations it estimates the
   ball's full state and predicts the parabola forward analytically
   (`position_at`/`velocity_at`). Verified: predicts 0.30 s ahead to < 5 mm.
2. **`InterceptionSolver`** — searches the predicted arc for the *earliest* catch
   point that is (a) inside the arm's reachable shell and (b) reachable within the
   remaining flight time (a per-joint min-jerk speed budget bounds the move), with
   the gripper oriented to **face the incoming velocity** (`look_at_orientation`,
   the general case of top-down). IK at that point gives the catch config `q*`.
   Returns `None` (honest miss) when no point is reachable in time.
3. **Receding-horizon replanning (MPC)** — every 5 ms it re-estimates and
   re-fits a **minimum-jerk joint trajectory** (`quintic`, with velocity boundary
   conditions) from the current reference state to `q*`, arriving just before the
   catch instant; it commits (stops re-routing) in the last 120 ms.
4. **Velocity-matched soft catch** — the catch config's joint velocity is set
   (via the damped Jacobian inverse) so the hand moves *with* the ball at the
   catch. A two-phase grab starts closing the fingers at the jaw boundary but
   **welds at the closest approach** (ball centred between the pads, so the fingers
   actually clamp it, not grab at the edge), matches the ball's velocity to the
   hand, then a quintic settle/hold absorbs the follow-through.

**Feasibility probe** mapped the catchable workspace (42/46 grid points reachable,
0.0° orientation error) and produced the ready pose. Throw envelope
(`sample_throw`) forward-generates a sane airborne launch whose parabola passes
through the reachable volume, rejecting underground/near-vertical ("cup") throws.

**Verified headless** over wide random throws (descent up to ~60°, arrival speed
to ~3 m/s): **160/160 clean catches across 4 seeds** (40/40 each), caught mid-air
at ~0.95–1.0 m, ~11° gripper-facing error at the catch, **~11 cm mean reach to the
interception** (i.e. the arm actively moves to meet the ball — it is not parked at
the arrival point). `tests/test_catch.py` (7 tests) covers the Kalman estimator,
interception reachability/rejection, orientation/trajectory maths, and an
end-to-end dynamic-catch check that asserts the arm *moves to intercept* and ends
with a real clean grasp. Reproduce: `openarm catch --benchmark` (headless rate) or
`openarm catch` (viewer).

> **Honesty note (audited).** The catcher never receives the throw parameters —
> its only ball input is `ball_pos()` read once per step (`catching.py:370`); the
> launch `(pos, vel)` from `sample_throw` is written to the ball and *never passed
> to the controller*. Velocity and the parabola are *estimated* by the Kalman
> filter and the solver predicts from that estimate. The one idealisation was that
> `ball_pos()` was ground-truth state, not vision — removed in Phase D2a below.

## Phase D2a — Vision-driven catching (cameras in the loop, no ground truth)  ✅
Closed the perception gap: the controller now observes the ball **through two
RGB-D cameras** and never reads its true pose.

- **Scene:** added `ballcam0`/`ballcam1` (opposite sides, arc midpoint, 58° FOV,
  look-at the catch zone) to `catch_scene.xml`; both see the whole ballistic arc.
- **Perception** (`openarm_control/vision/ball_tracker.py`): a `BallDetector`
  interface (so a learned detector can drop in later) with a `ColorBlobDetector`
  (added an `orange` rule to `detection.py`); `BallPerception` renders each
  camera, detects the ball, deprojects pixel+depth to 3D, **corrects the
  near-surface bias by +1 radius along the camera ray** (depth hits the front of
  the sphere, not its centre), and **fuses** the cameras (mean; opposite-side
  cameras cancel residual bias). Optional Gaussian sensor noise.
- **Integration:** `CatchController(perception=…, cam_period=5)` observes only at
  the camera rate (~100 Hz); between frames the Kalman filter predicts. In vision
  mode the controller's *entire* ball knowledge is the vision estimate + filter —
  even the final grab uses `kf.position_at(t)`, never `data.xpos`.
- **Verified headless:** fused 3D estimate **7.2 mm mean error** (max 11 mm) vs
  truth, both cameras detect the ball ~100 % of in-flight frames; **vision-driven
  catch rate 9–10/10 across seeds**, clean grasps, ~11 cm mean reach — and **10/10
  even with +5 mm added sensor noise** (the Kalman filter absorbs it). Run with
  `openarm catch --vision` / `openarm catch --benchmark --vision`. 2 new tests
  (vision estimate accuracy + vision-driven catch); 50 headless tests total.
- Media: `media/catch_camera_view.png` (the robot's-eye camera view with the
  detected ball).

## Phase D2b — Bimanual reactive catching (best-arm selection, collision-free)  ✅
The robot now uses **both arms** and decides which one catches, without the arms
colliding — balls can be thrown toward either side.

- **Scene** `catch_bimanual_scene.xml`: both arms in a mirrored catch-ready pose
  (`q_left = MIRROR_R2L · q_right`, verified 0.0 mm symmetric — left grasp at
  (0.40, +0.22, 0.95), right at (0.40, −0.22, 0.95)), both grasp welds, and the
  two RGB-D cameras re-aimed at the **centre** (so left/right throws are both in
  view). `CatchController` was generalised to take an `ArmSpec` (works as either
  arm). `sample_throw`/`sample_throw_bimanual` throw toward left, right, or centre.
- **`BimanualCatchController`**: one **shared** ballistic Kalman estimator feeds a
  per-arm interception solver. Each step it solves the interception for *both*
  arms, then **selects** the feasible arm with the largest time margin whose
  ready→catch path is **collision-free w.r.t. the other arm** (existing
  `CollisionChecker(avoid_other_arm=True)`, `edge_clear`); the chosen arm runs the
  MPC catch, the other holds a safe ready pose. Estimation/control are shared with
  the single-arm catcher (`observe`/`control`/`hold` split out of `step`).
- **Verified headless:** ground truth **18/18 caught & clean**, vision-driven
  **12/12 caught & clean**; arm selection matches the thrown side (left→left,
  right→right; centre→either); **minimum inter-arm separation ~17–22 cm over all
  throws — zero collisions**. `openarm catch --bimanual [--vision]`. Test asserts
  correct-arm selection + no collision. **51 headless tests total.**
- Media: `media/catch_bimanual.gif` + `catch_bimanual_left/right.png` (the idle
  arm waits while the arm on the ball's side catches it).

## Phase D refinement — Two balls at once (multi-object tracking, dual catch)  ✅
Both balls thrown simultaneously (one per side); each arm catches one, in parallel.

- **Scene** `catch_twoball_scene.xml`: two identical (same-colour) balls — so this
  is *genuine anonymous* multi-object perception, not colour-keyed cheating — plus
  4 grasp welds (each arm × each ball).
- **Multi-ball perception** (`MultiBallPerception`, `detect_color_blobs` via
  connected components): each camera detects up to 2 blobs, deprojects each
  (radius-corrected), and points from both cameras are **clustered by proximity**
  into ≤2 anonymous 3D detections. Verified **both balls found in 12/12 frames,
  5.5 mm mean error**.
- **Multi-object tracking** (`MultiBallTracker`): one Kalman filter per ball;
  each frame the anonymous detections are **associated to the tracks by nearest
  predicted position** (small assignment problem); unmatched tracks coast. Unit
  test feeds shuffled-order detections and confirms each filter locks onto a
  distinct ball.
- **`TwoBallCatchController`**: once both tracks are confident, each arm is
  assigned the track on its side and the two single-arm catchers run in parallel;
  each arm welds whichever physical ball is actually between its pads (`_do_catch`
  grabs the nearest candidate). Throws are one-per-side, so the arms stay clear.
- **Verified headless:** ground truth **8/8 both caught**, vision-driven **8/8**
  (and 5/5 via CLI), each arm a distinct ball, **min inter-arm separation
  ~14–22 cm — no collisions**. `openarm catch --twoball [--vision]`. 3 new tests
  (tracker association, 2-ball perception, dual catch + no-collision). **54 tests
  total.** Media: `media/catch_twoball.gif` + `catch_twoball_held.png` (both arms
  each holding a ball).

## Phase F — Imitation learning (behavior cloning vs RL)  ✅
Learn the reach task *from demonstrations* and compare head-to-head with RL.

- **Scripted expert** (`imitation/expert.py`, `ReachExpert`): solves IK to the
  target once, then each step commands the joint-delta action that moves toward
  it — acting in the **same obs(23)/action(7) space as `OpenArmReachEnv`**, so the
  cloned policy is directly comparable to the SAC policy.
- **Demo collection** (`imitation/collect.py`): rolls out the expert, keeps the
  successful episodes, saves a simple `demos/reach.npz` (`obs`, `act`, `ep_lens`).
  171/200 successful episodes → 5431 transitions.
- **Behavior cloning** (`imitation/bc.py`, `BCPolicy`): a 2×256 MLP, obs→action,
  tanh-bounded. **Key fix: observation normalisation** (the obs mixes radians,
  velocities, and positions of very different scales) — without it BC reached only
  23 %; with it, MSE 0.24 → 0.0009.
- **Evaluation / comparison** (`imitation/eval.py`): runs BC in the env and, with
  `--compare-rl`, the trained SAC policy on the *same* seeds.

**Result (40 eval episodes, seed 123):** **BC 80–85 % success @ ~65 mm** vs
**SAC 47 % @ 63 mm** — behavior cloning from ~170 demos beats the 200k-step RL
policy's success rate at similar precision (an honest, reproducible head-to-head).
CLI: `openarm bc-collect|bc-train|bc-eval [--compare-rl]`. 3 new tests
(expert reliability, BC shapes/normalisation, end-to-end clone-and-reach).
**57 headless tests total.** Needs the `[rl]` extra (torch).

---

## Phase G — Webcam human-arm imitation (capstone)  ✅
The robot mirrors a human arm in real time. New package `openarm_control/teleop/`.

- **Pose source** (`teleop/pose.py`): `PoseSource.get()` returns one arm's
  shoulder/elbow/wrist (`ArmLandmarks`) in a **robot-aligned frame** (+x forward,
  +y left, +z up). `WebcamPoseSource` runs **MediaPipe Pose** on webcam frames and
  converts metric `pose_world_landmarks` to that frame (cv2/mediapipe imported
  *lazily* — no hard dependency); `ScriptedPoseSource` generates a deterministic
  gentle reaching motion so the whole stack is verifiable with no camera.
- **Retargeting** (`teleop/retarget.py`, `ArmRetargeter`): task-level and
  **relative-to-home** ("clutch"): on `calibrate()` it records the human wrist and
  the robot tool point as origins; thereafter `target = home + gain · Δwrist`
  (clamped to a reachable sphere), and the gripper orientation is `R_home` rotated
  by the *change* in forearm direction (`_rot_between`). This keeps targets inside
  the dexterous workspace and varies continuously. A single **warm-started LM
  descent** (`_solve_coherent`, seeded from the previous solution) replaces the
  multi-seed public IK so the joint trajectory is **temporally coherent — no
  configuration flips**.
- **Safe teleop** (`teleop/teleop.py`, `TeleopController`): each tick reads the
  pose, retargets, then applies **EMA smoothing → per-joint velocity limiting →
  joint-limit clamping** before writing the position actuators. Drives one arm or
  both. `teleop_scene.xml` (both arms, front camera, neutral mirrored ready pose).

**Debugging notes (verified numbers below were earned):**
- Naïve *absolute* mapping (coincident shoulders) put targets outside the arm's
  reachable cone (≈110 mm IK residual) → switched to relative-to-home.
- The public multi-seed IK flips between branches frame-to-frame (1.84 rad jumps)
  → single warm-started descent fixed coherence.
- `look_at_orientation` is discontinuous at its up-reference flip → made
  orientation **relative** (rotate `R_home` by Δforearm) instead.
- Left-arm home keyframe had joint-2 mis-mirrored (out of the left arm's
  `[-3.316, 0.175]` range) → arm started invalid and flipped; corrected the mirror.

**Result (synthetic source, headless, both arms):** tool point tracks the mapped
human wrist to **~9 mm mean (≤ 15 mm)**, gripper-along-forearm **~0.1°**, **zero IK
flips** (max frame jump ~0.02 rad), peak joint speed ~0.5 rad/s, commands always
in limits. `openarm mimic [--webcam] [--bimanual] [--headless N]`. 6 new tests
(pose validity, `_rot_between`, reachable+coherent retarget, safe tracking,
both arms, webcam import-safety). **63 headless tests total.** Live webcam needs
the `[vision]` extra (mediapipe, opencv-python).

### Phase G upgrade — full arm + hand mimicry
The first version tracked only the wrist, so the arm "felt like mostly the
shoulder moving" and couldn't grasp. Upgraded in three verified stages:

- **Stage 1 — whole-arm posture matching.** A 7-DOF arm tracking a 6-DOF wrist
  pose has one redundant DOF (the elbow *swivel*); wrist-only IK left it wherever,
  so the arm posture didn't copy the human's. Added a **weighted multi-objective
  IK** (`_solve_posture`) matching wrist position (w 1.0) + gripper orientation
  (w 0.30) + **elbow position toward the human elbow** (w 0.60, body `link4`).
  Bug found & fixed: the loop broke as soon as the *wrist* was within 2 mm
  (ignoring the elbow) — changed to converge on the *whole* objective (step→0).
  Verified: with the hand fixed and the elbow swivelling, the robot elbow now
  follows ≥2.5× more than wrist-only IK, wrist still held; on live motion the
  wrist tracks to ~1 cm.
- **Stage 2 — hand → gripper.** Added a second model, **MediaPipe HandLandmarker**
  (Tasks API, auto-downloaded), to `WebcamPoseSource`; `hand_closure()` turns the
  21 hand points into a [0,1] fist-closure signal (mean fingertip-to-wrist
  distance / hand size), associated to the tracked arm by nearest in-image wrist.
  `ArmLandmarks.grasp` carries it; `TeleopController` maps it (smoothed) to the
  gripper actuator. Verified: gripper follows the signal over its full travel.
- **Stage 3 — pick things up.** `teleop_pick_scene.xml` (table + three blocks +
  weld constraints). `TeleopController.enable_grasping()` + weld-on-grasp logic:
  close the hand near a block → weld it to the gripper; open the hand → release.
  Verified end-to-end: a block is grabbed, **lifted ~10 cm with the gripper, and
  dropped on release**. CLI: `openarm mimic --pick`.

3 new tests (posture matching, grasp→gripper, pick-up). Hand model auto-downloads
like the pose model.

### Phase G redesign — anatomical direction-based retargeting
Testing live revealed two issues the wrist-centric mapping caused:
1. **"It combines both my arms."** The wrist was mapped in **absolute,
   torso-anchored** coordinates, and MediaPipe's world landmarks are anchored at
   the body centre — so moving the *other* arm (or leaning) shifted the tracked
   arm's coordinates and moved the robot. The arms were coupled.
2. **"The upper arm (shoulder→elbow) stays in the air."** Once the wrist *pose*
   is pinned (6 DOF), a 7-DOF arm has only the 1-DOF elbow swivel left — so the
   upper-arm direction is nearly determined and *can't* follow you. (Confirmed
   with task-priority IK: wrist exact → upper-arm error stuck ~38°.)

Both are fixed by retargeting from **limb directions, anchored at the shoulder**
(`retarget.py`, full rewrite). Each frame: `û = elbow-shoulder`, `f̂ = wrist-elbow`
(in the robot frame, EMA-smoothed). The IK matches the **upper-arm direction** and
the **forearm direction** (each a tangential, radius-free objective) plus a
low-weight gripper orientation — so the *whole* arm reproduces your posture and
the hand position is *emergent* (follows your reach at robot scale). A Kabsch
rotation (`_best_rotation`) aligns your neutral arm to the robot's at
`calibrate()`. Because everything is built from shoulder-relative *directions*,
it's invariant to body translation → **the arms are decoupled**.

Checked a reference repo (`ntu-rris/google-mediapipe`): it computes joint
angles geometrically from landmarks (for ROM / KNN gestures) — no robot
retargeting, but it confirmed the geometric direction/angle approach.

**Coordinate-mapping bug (observed: "up→left, down→right, right→up").** Empirically
measured MediaPipe's world axes from body anatomy (shoulders → lateral,
shoulder→hip → up, nose → forward): **+x = subject left, +y = down, +z = away from
camera**. The `_to_robot_frame` map (fwd=−z, up=−y, left=±x) was already
axis-correct; the scrambling came from the **`R_cal` calibration rotation** —
aligning the human's neutral arm to the robot's *catch-ready* home (upper arm at
`[0.35,-0.85,0.40]`, ~90° off natural) rotated every subsequent motion. **Fix:
dropped `R_cal`** (identity) and map directions absolutely. Verified on a synthetic
axis test: a forearm pointing forward/up/down/left now maps to the robot pointing
forward/up/down/left (the robot's own *right* is workspace-limited for the right
arm — a reach limit, not a mapping error).

**Live preview** (`openarm mimic --webcam --preview`): an OpenCV window draws the
detected arm skeleton on the webcam feed plus a readout of the robot-frame
upper-arm/forearm directions (fwd/left/up + a word) and the hand state, so the
mapping can be verified visually. `WebcamPoseSource` now exposes `last_image_lms`
and `last_arm`.

**Verified (synthetic):** robot upper-arm direction follows within ~14°, forearm
tracks, zero IK flips, in limits; output unchanged under whole-body translation
(decoupling); axis mapping correct. Tests updated (whole-arm posture, decoupling,
coherence). **73 headless tests total.**

---

## Phase H — Full integration & showcase  ✅
Everything wired together as one system.

- **Grand-tour showcase** (`cli.py`, `SHOWCASE`): `openarm showcase` now walks the
  whole stack in order — `sort → plan → bimanual → servo → catch → mimic`
  (classical → planning → bimanual → vision → dynamic → learned/teleop). Each
  opens a viewer; close the window to advance.
- **Full-system integration tests** (`tests/test_integration.py`, 6 tests): rather
  than re-testing each capability, they lock the *wiring*: every `COMMANDS` entry
  imports and exposes `main()`; every `SCENES` entry compiles to a valid MuJoCo
  model (nq, nu > 0); the CLI dispatcher's `list`/`scenes` run; and the two
  end-to-end **headless** pipelines run without error — the catch pipeline
  (perception → Kalman → interception → MPC grasp) and the teleop pipeline
  (pose → retarget → smooth/limit → actuators). Also fixed `pyproject.toml` to
  package the new `imitation` and `teleop` subpackages.

**Result:** the `openarm` CLI exposes every capability; one command
(`openarm showcase`) tours them; the integration suite proves the platform holds
together end-to-end. **69 headless tests total** (`python -m pytest tests/`).

---

## Phase I — Intelligent manipulation (vision-grounded, language-commanded, reactive)
Startup-grade arc: *tell the robot what to do; it sees and understands the table,
plans collision-free, acts, adapts if things move, and can throw.* Built
foundation-first. New deps: **ultralytics** (YOLO-World open-vocab; torch upgraded
2.10→2.12, all prior tests still green). New package `openarm_control/agent/`.

- **M1 — multi-object perception & grounding.** `vision/detector.py`: a `Detector`
  interface with `OpenVocabDetector` (YOLO-World, lazy import — detect any object
  by text prompt) and a dependency-free `ColorShapeDetector` fallback (colour +
  coarse shape, for headless tests/baseline). `vision/scene_perception.py`
  (`ScenePerception`): render → detect → deproject to 3D → `ground("the red ball")`
  resolves a text query to a located `SceneObject`. `tabletop_scene.xml` (ball/box/
  can + bin + top camera; neutral-grey background so the colour fallback isn't
  fooled by a blue sky). **Verified:** fallback locates objects to ≤6 mm, grounding
  resolves; YOLO-World detects the right categories on the render but at *low
  confidence* (0.01–0.14) — the sim domain gap on plain primitives. Decision:
  build the system on the (detector-agnostic) interface now, harden detection
  (synthetic fine-tune) later. 4 tests.
- **M2 — lightweight NL command parser** (`agent/commands.py`): a keyword/grammar
  parser → `Intent(action, target, destination)`, e.g. "pick the red cube and put
  it in the bin" → `place / red cube / bin`. No heavy LLM. 3 tests (10 patterns).
- **M3 — task executor with obstacle avoidance** (`agent/executor.py`,
  `TaskExecutor`): ground the target → plan a **collision-free** free-space motion
  (RRT-Connect around clutter, target/carried excluded) → top-down weld grasp
  (reuses `PickPlaceController`) → transport → place. **Verified:** "put the green
  box in the bin" lands it in the bin while the other three objects move **0.0 m**.
  `openarm manipulate "..."` (+`--interactive`, `--vision`). 2 tests.
- **M4 — reactive grasping** (`agent/reactive.py`, `ReactivePicker`): closed loop —
  re-perceive every few ticks, re-solve a top-down grasp config for the target's
  *current* position, servo joints toward it (velocity-limited); if the object is
  nudged it re-tracks, then descends/closes/welds. (Resolved-rate-from-rest was
  fragile → joint-space tracking of `GraspSolver` configs.)

**Refinements addressed next:** (a) grasp at the object's *perceived top*
(tall cylinder was pressed, not gripped); (b) a **stateful multi-turn** command
session (remember the held object, resolve "it", primitives "go to"/"release");
(c) **carried-object collision avoidance** during transport (a held object that
hits the environment is a failure); (d) ready for custom object meshes.

### Phase I refinements + M5 (from live testing)
- **Adaptive grasp height** — grasp at the object's *perceived top* (depth), not a
  fixed table offset, so a tall cylinder is gripped near its top instead of being
  pressed. Stored grasp offset = grip height above the table, used to compute
  correct place heights (object clears the bin walls before dropping in — fixed a
  violent fling caused by the carried object penetrating the bin walls).
- **Stateful multi-turn session** (`agent/session.py`, `ManipulationSession`) +
  executor primitives (`grasp` / `go_to` / `release`). Remembers the held object,
  resolves "it"/"that"/generic referents, and supports decoupled steps. So
  "pick up the blue cylinder" then "put it in the bin" works (the "it" = the held
  object), and "go to the bin" → "release" works step by step. Parser gained
  `goto`/`release` actions and referent handling.
- **Carried-object collision avoidance** (`planners/collision.py`): the held object
  is rigidly **re-posed with the gripper at each checked config** (via the grasp
  transform) and treated as part of the arm, so it must avoid the environment and
  the other objects — but its contact with the gripper (the grasp) is ignored. The
  executor shares one `CollisionChecker` with the RRT planner; `go_to` sets the
  carried object during transport. A transport that would knock the held object
  into something is now a collision, like real robotics.
- **M5 — dynamic throwing** (`throwing.py`, `ThrowController`; `throw_scene.xml`,
  `openarm throw`): grasp a ball, then to land it in a bin: pick a release point
  (forward + high toward the bin), **invert the projectile equations** for the
  min-speed release velocity, solve **position-only** IK for a well-conditioned
  release config (orientation-free — the ball is a sphere; an orientation
  constraint forced a near-singular pose that inflated joint speeds), set
  `qd = J⁺v`, and **refuse if it exceeds the joint-speed budget** (out of the throw
  envelope). Swing = a quintic ending at the release config *with* the release
  velocity; **release by detaching the weld with the gripper open** (fingers clear
  of the 60 mm ball) so the follow-through can't drag it. First version landed
  ~90 mm from the bin centre — see the accuracy hardening below, which cut that to
  5–9 mm.

### Phase I — M5 throwing: accuracy hardening (advanced control)
The first throw had a stubborn, *consistent* ~93 mm overshoot in +x that a target-
shift correction couldn't remove (shifting the aim by 90 mm changed nothing; 130 mm
broke it). Instrumenting the swing exposed the real cause — two compounding bugs:

- **Over-powered swing.** A rest-to-rest quintic peaks at **1.875 × (amplitude/T)**
  at its midpoint, so commanding `q_wind/q_ft = q_rel ∓ qd_rel·0.30` swept the EE
  ~1.9× faster than the ballistic solution required. The ball always overshot the
  bin and *only the bin's far wall stopped it* — which is why the landing was pinned
  near the wall (~0.74–0.89 m) regardless of release timing, masquerading as a fixed
  "~93 mm" bias.
- **Saturated speed knob.** The fixed ±0.30 rad wind-up, with large `qd_rel`, clipped
  against the joint limits, so scaling the swing speed did *nothing* (the swing was
  always limit-to-limit).

**Fix 1 — size the swing from the desired velocity.** `_shape(q_rel, qd_rel, speed)`
sets the half-amplitude to `speed·qd_rel·(T_swing/3.75)`, so the quintic's midpoint
speed *equals* `speed·qd_rel` (the ballistic velocity at `speed=1`) and the swing
stays **off the joint limits** — making `speed` a genuine, continuous range knob.

**Fix 2 — simulation-in-the-loop release search** (`plan_release`). The analytic
welded-ball ballistic prediction under-reads the true release velocity, so we don't
trust it: `_true_landing` rolls the **real** swing+release+fly forward on a saved
state and returns the actual landing. The search winds up once, then evaluates the
true landing per release step (**coarse stride-4 then fine ±4 around the swing
midpoint**, where the swing peaks at `qd_rel` by design) and picks the step that
genuinely lands closest; if no step is within the gate it **escalates the swing
speed** (1.0 → 1.15 → 0.85 → 1.3 → 0.7) before refusing. Release **freezes the arm
at the release config** so the follow-through can't chase the ball, and the table
collision is disabled after the lift as a swing fail-safe.

**Result.** Across 7 narrow-bin targets (16 cm opening, was 26 cm) the ball lands
**5–9 mm from centre** (was ~93 mm); the predicted and executed landings match to
the millimetre (deterministic); bins outside the precise envelope (forward
x ≳ 0.70, where the achievable-landing set has a gap the bin wall would catch) are
correctly **refused**. The precise-throw envelope is forward **x ∈ [0.56, 0.67]**,
lateral **y ∈ [−0.32, +0.12]**.

- **Multi-bin benchmark** (`benchmarks/throwing_benchmark.py`): repositions the bin
  across `BENCH_BINS` (7 locations in the envelope), N perturbed trials each (ball
  start jittered ±5 mm for trial-to-trial variance), and reports per-bin precision
  (mean ± std error, in-bin success) + a reachability sweep marking the envelope
  edge → CSV (`results/`) + figures (`figures/`): a top-down landing map (bins as
  squares, landing scatter) and a per-bin precision bar chart. Each trial builds a
  **fresh model** (grasp_ball mutates the model: weld offset + disabled table
  collision). Smoke-checked (1 trial/bin: 7/7 in-bin, mean 5.2 mm); full multi-trial
  run deferred (slow — sim-in-the-loop search per trial).
- **Tests** (`tests/test_throwing.py`): lands-in-the-narrow-bin (<80 mm, plan
  matches execution), refuse-unreachable-pose, refuse-out-of-envelope (x=0.80 gap),
  the swing-speed-is-a-range-knob (amplitude scales with speed, off the limits), and
  the multi-bin throws (below). New deps unchanged.

### Phase I — M5 throwing: multi-target show (5 balls -> 5 bins)
`throw_multi_scene.xml` + `openarm throw --multi`: five balls on a table and five
narrow (12 cm) bins spread across the precise-throw envelope. The arm grasps each
ball and throws it into a *different* bin (a single fixed bin felt unimpressive).
**Lands 5/5.** Key engineering:
- **Table collision stays on** (the swing clears the table on its own — verified
  identical accuracy with/without the old fail-safe), so the remaining balls keep
  resting on it. `grasp_ball(disable_table=False)` skips the fail-safe; the
  single-bin demo still disables it.
- **Pristine reset between throws.** Throwing from the previous throw's frozen
  follow-through (stale ctrl/qvel/config) degrades accuracy to ~28 mm and even
  refuses some bins; resetting to the keyframe (clean arm pose + zero residual
  control) before each throw restores the clean ~20–29 mm that lands in a 12 cm bin.
  Resetting (vs. *sweeping* the arm back to ready) also avoids knocking the balls
  still on the table. The already-landed balls are kept in their bins across resets
  by saving/restoring their free-joint qpos, so the balls **accumulate** in the bins.
- Bins must sit inside the small precise envelope (x∈[0.56,0.67]); packed close,
  throws to a bin adjacent to another can clip it, so the five are spread in y.
  `tests/test_throwing.py::test_multi_bin_throws` throws to two bins (clean reset
  each) and asserts both settle inside.

### Phase I — M6: stacking (block-on-block + "stack X on Y")
`stack_scene.xml` (four 50 mm cubes, a table, a bin, a top-down camera) +
`ThrowController`-style `TaskExecutor.stack(target, support)` + `openarm stack`.
- **Place height from geometry.** Grasp the object (reusing the adaptive-height
  grasp), then place its **base on the support's perceived top**: place the grasp
  point at `support_top_z + grasp_offset + clearance` (the held object hangs
  `grasp_offset` below the grasp point). Result: a cube lands ~one cube height up,
  **~2 mm off-centre**, stable.
- **Point-down descent fix.** The place descent reused `_ik_line` with the fixed
  grasp orientation, whose yaw differs from the place pose's — the arm twisted at
  the end and the tool point **drifted ~3 cm** in xy (tolerable for a wide bin, why
  the manipulate tests never caught it; fatal for stacking a 50 mm cube). Switched
  `_carry_to`'s descent to the purpose-built **`_ik_line_down`** (vertical,
  point-down, free yaw → no xy drift). `go_to` (place-in-bin) now uses it too.
- **Language.** `commands.py` gained a `stack` action with a smart split: "stack X
  on Y" / "put X on top of Y" → `Intent('stack', target=X, destination=Y)` where Y
  is an *object* (not a destination keyword); "put it on the table" stays a place.
  `session.do` dispatches to `ex.stack`.
- **Clear-the-camera home.** After a stack the arm parked over the stack and
  occluded the top-down camera, so the next command couldn't perceive. Added
  `TaskExecutor.home()` (return to the ready/perch config captured at build time);
  `stack` calls it at the end so two independent stacks work in a row. (A 3-high
  tower still collapses — stacking on a 50 mm elevated cube is precarious — so the
  showcase does two independent stacks, not a tower.)
- **Detection note.** A bright orange *cube* top saturated toward yellow (g too
  high) and missed the orange colour rule, so its rgba was darkened to `0.7 0.22
  0.03` to land back in the orange band (the orange *ball* curves and shades
  differently). 4 stacking tests (`tests/test_stacking.py`): parse, block-on-block,
  by-language, refuse self-stack.

### Phase I — M6: peg-in-hole insertion
`peg_socket_scene.xml` (a cylindrical peg, radius 0.014, on a table; a socket =
four walls forming a 0.036 m square pocket) + `TaskExecutor.insert(peg, socket)` +
`openarm insert` / "insert the peg into the socket". The insertion reuses the
stacking machinery: grasp the peg, `_carry_to` above the socket (ignoring the
socket in collision so descending into it isn't a crash), then the **point-down
vertical descent** threads the peg through the opening down to the table
(`place_z = TABLE_TOP_Z + grasp_offset + 0.006`), release, `home()`. With ~2 mm
descent accuracy and ~4 mm radial clearance the straight descent threads cleanly —
the peg seats **1 mm off-centre, 0° tilt**, base on the table — so no compliant
spiral search is needed at this clearance (it remains the fallback for tighter
fits). The blue cylindrical peg reads as "blue ball" from top-down (its round top),
so it grounds by colour. Parser gained an `insert` action and `socket`/`hole`/`slot`
destinations; `session.do` dispatches to `ex.insert`. 3 tests
(`tests/test_insertion.py`): parse, insert primitive, by-language.

**This completes Phase I M1–M6.** Docs (ROADMAP + this log + memory) current.

### Phase I — Bimanual & reactive (live feedback: "make both arms work")
- **Reactive object-following (M4) — built then REMOVED.** `ReactivePicker` /
  `BimanualReactivePicker` (re-perceive → re-solve grasp → velocity-limited joint
  servo) were exposed via `openarm reactive`, but on live testing the single-arm
  servo was unreliable — the arm rotated in the air and got stuck reaching for the
  object. The whole reactive-following feature was **removed for now**
  (files, CLI command, scene, tests, exports all deleted) and will be re-implemented
  properly at the end, likely on the planned/MPC machinery (which is reliable) rather
  than a raw joint-space servo. The best-arm *idea* lives on in the coordinator below.
- **Bimanual simultaneous stacking** (`bimanual_stack_scene.xml`, a WIDE table;
  `bimanual.BimanualStack`; `openarm bimanual --mode stack`). Both arms build a tower
  on their own side at once. The left arm's plan hit an IK branch jump at the
  mirrored poses, so instead the left arm runs the **exact joint-space mirror**
  (`q_left = MIRROR_R2L · q_right`) of the right arm's known-good stack plan — driven
  through the existing `parallel_run`/`_ArmRunner` lockstep with per-arm grasp welds.
  Both towers land ~one cube up, ~2 mm aligned.
- **Intelligent dual-arm pick-and-place with hand-over** (`bimanual_handover_scene.xml`,
  a wide table with a bin only the right arm reaches + two cubes, both carrying both
  arms' welds; `bimanual.BimanualCoordinator`; `openarm bimanual --mode coordinate`).
  `pick_place(obj_xy, place_xy, block)`: choose the arm on the object's side (if a
  top-down grasp there is feasible); if **that arm can also reach the destination** it
  does the whole pick-and-place alone (the other arm holds); if it **can't reach the
  destination but the other arm can**, it carries the object to a centre **midpoint**,
  releases, steps home to clear, and the other arm picks it up there and places it.
  Feasibility = `GraspSolver.solve` success per arm. All motions are planned
  (`PickPlaceController.plan` + `parallel_run`), so it's reliable — the robust
  counterpart to the removed reactive servo. Verified: right arm bins the right cube
  alone; left arm hands the left cube over to the right arm, both end in the bin.
- **Interactive bimanual playground** (`bimanual_table_scene.xml` — a LARGE table,
  four colour blocks spread across both arms' reach, a grey bin on each side, all
  blocks with both arms' welds; `demos/demo_interactive.py`; `openarm interactive`).
  The viewer opens, the top-down camera detects the blocks, the **terminal lists
  them**, and you choose a block + a destination bin; the coordinator runs it
  (single-arm or hand-over). `--headless` runs a scripted self-test. Gotchas fixed:
  (a) the bins were tinted green/red and got **detected as objects** → made them
  neutral grey; (b) a far-corner block was occluded — the occluder is the right
  gripper's **camera shadow** at ~(0.227, −0.304) (not the gripper's xy), and the
  far-corner pose also branch-jumps `plan()`, so blocks live at x ≤ 0.32 clear of
  the shadow; (c) bodies named `block_<colour>` so the weld suffix matches; (d) arms
  reset to the ready pose between rounds so the camera re-perceives cleanly.
- **Bigger tables.** The bimanual scenes use wide tables (y ∈ [−0.42, 0.45]) so both
  arms have their own work area.
- Tests: `tests/test_bimanual.py` — `test_bimanual_stack_two_towers`,
  `test_bimanual_coordination_single_and_handover`, and
  `test_interactive_table_detection_and_delivery` (detects all four blocks; delivers
  a single-arm and a hand-over block into the bin). **103 headless tests.**

### Phase I — scanned objects + open-vocab backends (in progress)
- **YOLOE backend.** `OpenVocabDetector` now supports `backend="yolo-world"` (default)
  or `backend="yoloe"` (2025 "Real-Time Seeing Anything"; needs `get_text_pe` for the
  text prompt + a ~570 MB MobileCLIP encoder). On our top-down sim renders YOLO-World
  scored higher and is lighter (scanned mug → "cup" @ 0.48 vs YOLOE's 0.30), so it's
  the default; YOLOE is one flag away.
- **Scanned-object integration (Google Scanned Objects).** `build_scanned.py` curates a
  subset (copies `model.obj` + `texture.png`, auto-scales each to ~6 cm, computes the
  AABB centre + box half-extents); `gen_scanned_scene.py` writes `scanned_table_scene.xml`
  (each object = the scanned mesh as VISUAL, centred at the body origin, + a simple BOX
  collision for reliable grasping, free joint, both arms' welds). It loads and settles.
- **Localise + pick-by-number (the chosen, working approach).** Localising is reliable
  via `SegmentClassifyPerception` (segment the depth image for things above the table,
  filter the workspace + bin positions, deproject) — real geometric detection.
  **Classification is hard** (a *top-down* view is ambiguous — mug, bowl, clock all
  read as round → "clock"; the angled cam is too far; YOLO resizes to ~640 so the 6 cm
  objects are tiny), so the workflow **localises + lists by
  number + manipulates**, with the open-vocab label as a best-effort hint. Wired up:
  `openarm interactive --scanned` (`--detector yoloe` to switch backend). The dual-arm
  coordinator grasps a scanned object (box collision) and bins it with a hand-over when
  needed. The flat toy car (1.4 cm tall, not top-down graspable) was dropped; the kept
  set (mug, bowl, clock, elephant) is roughly equidimensional and tall enough.
- **Curation.** Only the selected meshes live in `assets/scanned/<name>/` (+ a license
  note); the 1030-model source dataset was removed. `gen_scanned_scene.py` rebuilds the
  scene from `_scanned_params.json`.
- 2 tests (`tests/test_scanned.py`): depth-localisation finds the objects (bins
  excluded), and the coordinator delivers a scanned object to a bin — both with
  `detector=None` so the suite needs no YOLO weights.

### Phase I — bimanual fixes (live feedback: arm start poses + object placement)
- **Left arm started parked low.** My new bimanual scenes set the left arm to all-zeros
  (a relaxed downward pose, EE below the table), so when it moved up it could clip the
  table — the bimanual coordinator executes planned trajectories **open-loop** (no
  per-trajectory collision check, unlike the single-arm RRT executor), so nothing
  caught it. Fixed by starting the left arm **raised**, as the joint-space mirror of the
  right (`q_left = MIRROR_R2L · q_right`, EE at (0.24, +0.23, 0.67)) in **all** bimanual
  scenes (stack, handover, table, scanned). With both arms raised, objects were also
  repositioned clear of **both** arms' top-down camera shadows (~(0.227, ±0.30)) so
  detection still sees them (|y| ≤ 0.20).
- **Object intersecting a bin.** A block sat on the bin's edge; objects are now ≥ 189 mm
  from either bin (bin half-width 70 mm). `SegmentClassifyPerception`'s bin-exclusion
  radius was raised to 0.15 m (the depth-segmented bin walls deproject ~0.12 m off the
  bin centre, so 0.10 m missed them).

**Full suite green: 105 headless tests pass** (verified after the bimanual start-pose
+ placement fixes and the scanned-object integration).

### Phase I — M6+: cuboid peg-in-hole (oriented insertion)
A harder insertion than the round peg: a **rectangular** block (36×22×70 mm) into a
**rotated rectangular** socket (`peg_cuboid_scene.xml`, 48×32 mm pocket at 45°). The
new challenge is **yaw alignment** — the block only fits when its cross-section
matches the hole's orientation. `TaskExecutor.insert_cuboid`:
- Grasps the peg with whatever **reachable** point-down yaw `GraspSolver` finds (the
  wrist can't achieve every yaw everywhere — at the socket only ~45–165° are
  reachable, so 30° failed; 45° works).
- Computes the insert gripper yaw `ψ = θ − peg_yaw + φ` that turns the block's long
  axis to the socket yaw `θ`, and — exploiting the gripper's and the rectangle's 180°
  symmetry — tries `ψ` and `ψ+180°`, using whichever is reachable.
- Threads it in with a new **`PickPlaceController._ik_line_oriented`**: a continuous
  6-DOF descent that holds the *full* orientation (point-down **and** the fixed yaw),
  seeded step-to-step so it can't branch-jump (the position-only / free-yaw lines
  twist or jump under a full-orientation constraint).
- Result: block seats **1–2 mm off-centre, ~0° yaw error, upright**. `openarm insert
  --cuboid`. 1 test (`tests/test_insertion.py::test_insert_cuboid_into_rotated_hole`).
  **106 headless tests.**

*(Next: a more advanced skill — cloth folding or similar — to be chosen.)*

### Phase I — conversational interaction polish
The session (`agent/session.py`) + parser (`agent/commands.py`) gained four
upgrades; `openarm manipulate` showcases them.
- **Multi-step commands.** `split_steps(text)` splits on "then"/"after that"/
  "next"/";" (not plain "and", which is part of one "pick X and put it in Y"
  intent); `session.run(text)` does each clause in order, stopping at the first
  failure. `demo_manipulate` now calls `run` (single commands still work).
- **State queries.** A `query` intent ("what are you holding?", "what's on the
  table?") → `session._answer`: the held label, or the labels perception sees. No
  motion.
- **Undo / "put it back".** An `undo` intent reverses the last relocation: the
  session records `(label, original_xy)` after each pick/place/move/stack/insert
  (the origin is the object's pre-grasp `pickup_xy`, captured in `grasp`), and
  `_do_undo` returns the object there — `_carry_to`+release if still held, else
  `put_at` (grasp it back). `put_at` **ignores the bin while extracting** (the
  object is coming *out*, so grazing the walls isn't a crash) — without that the
  transport planner reported "start in collision" pulling a cube back out of the bin.
- **Clarification.** On a "could not find X" failure the reply appends "I can see:
  …" (`session._clarify` + `executor.visible()`), so a missed object is actionable.
- Parser: `query`/`undo` actions ("what/which/where…" or holding-words → query;
  "undo"/"reverse"/"put it back" with no destination → undo; "put it back in the
  bin" stays a place). 3 tests (`tests/test_interaction.py`): parse+split,
  query+clarification, multi-step-then-undo. **(See the bimanual section for the
  current 102-test total — the reactive-following tests were removed with the
  feature.)**

### Dexterous hand (16-DOF Allegro) — explored, then REMOVED
A 16-DOF Allegro hand replacing the right parallel gripper was prototyped via MuJoCo's
`MjSpec` model-editing (compose a new model at load: remove the gripper, attach the hand,
contact-exclude it from the wrist links). It worked partially — the spurious self-collision
that locked the wrist was diagnosed and fixed, the mount was corrected to point the grasp
straight down, and a weld-assisted pick/place ran on hand-sized objects — but the overall
**gripper-replacement / hand placement / arm integration never looked or behaved right**,
and load-bearing **real finger contact** proved research-grade (the soft Allegro actuators
can't squeeze hard enough to lift by friction; stiffening diverges). The
whole effort was **removed**: deleted `hand.py`, `demos/demo_hand.py`, `tests/test_hand.py`,
`hand_scene.xml`, the `assets/allegro/` model, the `openarm hand` CLI command, and the
`hand`/`HAND_SCENE` config entry. **The parallel gripper remains the end-effector across
every skill.** (Lesson kept for any future end-effector work: a multi-finger hand needs the
mount orientation derived by *measuring* where the curled fingers close, and real-contact
grasping needs RL or heavy solver tuning — weld-assist alone looks fake.)

### Phase I — M6+: a family of matched insertion scenarios (selectable)
Generalized peg-in-hole into three **selectable** scenarios of increasing alignment
difficulty, distinguished by the peg's cross-section symmetry:
- **round** (`--shape round`, default): a **cylinder** into a **circular** hole. The hole
  is now a real round opening — an **octagonal ring** of eight tangent wall segments
  (inner apothem 20 mm), replacing the old square pocket. A cylinder is rotationally
  symmetric, so no yaw alignment is needed (the free-yaw vertical descent / `insert`).
- **square** (`--shape square`): a **40 mm square peg** into a **rotated (30°) square
  hole** (48 mm pocket). The cross-section is **4-fold symmetric**, so it fits at any 90°
  multiple — the grasp is snapped to the nearest reachable `{θ, θ+90, θ+180, θ+270}`.
  New scene `peg_square_scene.xml`. Seats **~4 mm off-centre, 0° tilt/yaw error**.
- **cuboid** (`--shape cuboid`, or `--cuboid`): a rectangular block into a rotated
  rectangular slot — **180-deg symmetric**, the tightest alignment (the existing case).
- **Unified executor.** `insert_cuboid`/`insert_square` both call a new
  `TaskExecutor._insert_aligned(peg, socket, n_fold)`: grasp at a reachable point-down
  yaw, then try the `n_fold` symmetry-equivalent socket orientations and thread in at the
  first reachable one with a fixed-yaw `_ik_line_oriented` descent. (`n_fold=2` rectangle,
  `4` square; round stays the free-yaw `insert`.)
- **CLI/demo/tests.** `openarm insert --shape round|square|cuboid` (`--cuboid` kept as an
  alias); `demo_insert.run_aligned` reports off-centre / tilt / yaw-error per scenario.
  `config.SCENES["peg_square"]`. `tests/test_insertion.py` +1 (square; the cuboid test
  refactored to share `_assert_aligned_insert(scene, method, sym)`). **112 headless tests.**

### Reliability — full suite made deterministically green (107/107)
After removing the hand experiment (107 tests), `test_executor::test_place_in_bin`
flaked **only in the full suite** (passed in isolation / per-file). Root-caused to two
real bugs, both fixed without removing features or masking the check:
- **Renderer GL-context leak (`vision/camera.py`).** `Camera` created a `mujoco.Renderer`
  (a GL context) and never freed it; across a long process / the full suite the contexts
  leak until renders return garbage and detection finds nothing ("could not find …").
  Added `Camera.close()` + `__del__` + context-manager, and `close()`/`__del__` on
  `ScenePerception`/`SegmentClassifyPerception`, so each perception frees its renderer.
- **Off-centre grasp capture (`pick_and_place.py`).** The grasp weld was applied *after*
  the close, capturing whatever pose the closing fingers had shoved the object into; the
  object was then carried and placed off-centre (missed the bin by ~17 cm). The trigger in
  the full suite was subtle: collecting all test modules imports `torch`, whose bundled MKL
  changes numpy's BLAS results, so the IK plan differs by ~1e-5 (the physics `mj_step` is
  bit-identical; thread-pinning doesn't help). That tiny difference tipped the marginal
  grasp. Fixes: weld at the **start** of the close (object not yet shoved), and in
  `attach()` **zero the lateral in-gripper offset** so a top-down-grasped object is held
  centred on the approach axis (grasp depth + orientation preserved). Clean grasps
  (stacking/insertion) are already centred → unchanged; the shoved box is corrected.
- Result: `place_in_bin` lands **6 mm** from the bin centre (was 160–171 mm under the
  torch-perturbed plan), and the **full suite is 107/107**, stably, with `torch` loaded.

### Phase 1 — bimanual best-arm + hand-over, driven by language
Connected the existing dual-arm intelligence to perception and the language parser so
ordinary relocation commands choose the right arm and hand the object over when needed.
- **`BimanualSession`** (`agent/bimanual_session.py`) mirrors the single-arm session's
  API on top of `BimanualCoordinator`: it grounds an object label (perceive → match by
  colour token → nearest graspable body) and a destination ("left/right bin", a generic
  "bin" = the nearer one, "table"), then routes the command:
  - "grab X" → `coordinator.pick` (the better-placed arm grabs and **holds** it);
  - "move/transfer X to Y" → one-shot `coordinator.pick_place` (best arm picks; **hands
    over** at the midpoint when only the other arm can reach Y);
  - "move it to Y" while holding → `coordinator.place_held` (holding arm places);
  - plus "which arm is holding it?" queries, undo, and "could not find … I can see …".
- **Coordinator additions (additive; `pick_place` untouched):** `pick(obj_xy, block)` /
  `place_held(place_xy)` with held-state, backed by new `PickPlaceController.plan_pick`
  (pick+lift only) and `plan_place_held` (transport from the current held config). The
  left arm reuses the exact mirror-of-a-right-arm-plan trick.
- **Parser:** `transfer`/`handover`/`relocate` added to the move verbs ("grab" and "move"
  already existed). **CLI:** `openarm bimanual --mode language ["<command>"] [--interactive]`.
- Verified: "grab the red block" → left arm holds it; "move it to the left bin" → 4 mm;
  "transfer the green block to the left bin" → right arm **hands over** to left → 2 mm.
  Tests: parse-transfer, grab→query→place-held, transfer-triggers-hand-over. **113 tests.**
  *(Known limit: undo of a completed cross-side bin transfer degrades gracefully — extract
  -from-bin + hand-over is a hard case; simpler undos work.)*

### Phase 2 — perception hardening (synthetic fine-tuning + multi-view)
Attacked the "perception is fragile" weakness with three additive pieces (the existing
colour/shape + open-vocab detectors and `ScenePerception` are unchanged):
- **Free auto-labeled data (`vision/synthgen.py`).** MuJoCo's **segmentation renderer**
  gives exact per-object masks → 2D boxes with **no manual labeling**. `SyntheticData
  Generator` renders a scene under **domain randomisation** (camera pose, lighting, object
  placement/yaw, colour-brightness jitter) and writes a **YOLO-format dataset** (images +
  normalized boxes + `data.yaml`). `openarm gen-data --out … --n …`.
- **Fine-tune / eval (`vision/finetune.py`).** `train` / `evaluate` wrap ultralytics
  (lazy/optional); `openarm detect train|eval`. Training is **user-run** (GPU), like the
  RL/BC pipelines. A fine-tuned checkpoint drops straight into the existing
  `OpenVocabDetector(model="…/best.pt")` — sim→deploy gap is tiny (same renderer).
- **Multi-view fusion (`vision/multiview.py`).** `MultiViewPerception` renders a top-down
  **and** an angled camera with a **single** renderer (never two live GL contexts),
  detects + deprojects per view, and **fuses by 3D proximity** keeping the most-confident
  label — so the angled view disambiguates what the top-down view can't. Same `perceive`/
  `ground` contract as `ScenePerception`, so it's a drop-in.
- Verified headless: segmentation boxes for all four blocks; a 6-image dataset writes valid
  YOLO labels; two views fuse to exactly 4 objects (no duplicates) and ground by colour.
  Tests in `tests/test_synth_perception.py` (3). **116 tests.**

### Phase 3a — non-prehensile pushing
First advanced-manipulation skill: move an object to a target **without grasping it**.
- **`pushing.py` `PushController.push(obj, target_xy)`.** Each stroke: find a *reachable*
  point-down config just behind the object (on the far side from the target) via the
  grasp-yaw search, then push along the object→target line with the **closed gripper** as
  a blunt pusher, holding that reachable orientation the whole stroke with the **continuous
  `_ik_line_oriented`** (seeded step-to-step, so the wrist can't rotate mid-push and fling
  the object). After each stroke it re-reads the object and re-aims — the closed-loop
  re-planning absorbs the slip/drift inherent to pushing. No weld, no grasp.
- **Two bugs found while tuning** (both fixed): (1) a free-yaw push line let the gripper
  rotate and bat the puck off the table; (2) forcing the push-direction yaw was unreachable
  (and seeding the line with a *different* yaw caused a ~117° wrist swing). The fix —
  reachable yaw + held continuously — gives a clean push.
- **`push_scene.xml`**: a low slippery puck (friction 0.9) + two goal regions; `push`
  parser verb (`push`/`nudge`/`shove`, distinct from grasp-and-move); `openarm push
  [--goal a|b]`. Lands the puck **~30–40 mm** from the goal across both goals. Tests in
  `tests/test_pushing.py` (3). **119 tests.**

### Phase 3b — tool use (reach extension)
The robot grasps a stick and uses it to move an object that is **beyond its bare
top-down reach** — picking up a tool to extend its workspace.
- **`pushing.py` `ToolController`.** Grasp the stick at its centre (weld), record the
  tool-tip offset (which stick end is nearest the target) and the held gripper
  orientation. To push: the gripper sweeps along a line **offset** from the desired tip
  line (`gripper = tip − offset`), holding the grasp orientation with `_ik_line_oriented`,
  so the stick's tip tracks the object even though the grasped stick is angled ~30° by the
  reachable grasp yaw; re-aim each stroke. No re-grasp between strokes (the tool weld stays
  active; `block=None` so the stroke doesn't touch it).
- **`tool_scene.xml`**: a 360 mm stick + a block + a goal, **all past the ~0.36 m bare
  reach**. `openarm tool`. The bare gripper can't reach the block or goal; with the stick
  the block is pushed **~84 mm onto the goal (~39 mm)**, deterministically (3/3).
- **Honest note:** reach-extension is kinematically tight for this 7-DOF arm — the limited
  forward reach + base proximity + the centred-weld (which prevents grasping the stick at
  its end) leave a narrow usable band, so the demo is tuned within it (a modest but genuine
  reach-extending push). Tests (`tests/test_tool.py`, 2) assert the targets are
  bare-unreachable (so the tool is *needed*) and that the block ends on the goal, on the
  table. **121 tests.**

### Phase 3c — deformable cloth folding
The frontier skill: fold a deformable cloth (classical control can't fully cover
deformables; this is the foundation + demo source for a learned policy).
- **Cloth = a MuJoCo `flexcomp` grid** (`cloth_scene.xml`): a 5x5 flex sheet on the table.
  Crucially, each grid **vertex is a body** (`cloth_0`..`cloth_24`, corners 0/4/20/24), so
  the gripper can **grasp a corner via a weld** (`grasp_right_<corner>`) -- no need to pin a
  soft-body vertex. Solver `CG`, edge-equality so it behaves like cloth.
- **`cloth.py` `ClothFoldController.fold(corner, target_xy)`**: solve a top-down pose at the
  corner, descend, **weld the corner** to the gripper, lift, and carry it to the opposite
  edge with the continuous `_ik_line_oriented` (a free-yaw point-down line branch-jumps at
  the workspace-edge corners), then lay it down and release. `set_ready` poses both arms +
  settles the cloth (the scene has no keyframe).
- Verified: folding corner 0 onto corner 4 lands it **~20 mm** from the target and shrinks
  the cloth's y-extent **148 mm -> 86 mm** (it folded), stably (finite). `openarm cloth
  [--corner 0|4|20|24]`. Tests (`tests/test_cloth.py`, 2): flex sim is stable + corners are
  weldable; the fold reduces the cloth's extent and brings the corner to the target.
- **Honest scope:** a *learned* ACT/Diffusion folding policy -- the original "standout
  learned skill" -- is a separate research effort (a cloth-fold env + demo collection +
  sequence-model training + eval). The scripted fold here is the deformable-manipulation
  foundation and a demonstration source; the `imitation/` package is the training hook.
  **123 tests.** Phase 3 (3a pushing, 3b tool use, 3c cloth) complete.

---

## Extension arc (post-v1): foundation phase F1

> Additive only — new packages/scenes/tests; the prior system is frozen and the
> existing test suite stays green (regression gate). See `docs/ROADMAP_EXTENSIONS.md`.

### Phase F1 — realistic objects, 6-DOF grasp, articulated assets
Foundation that unlocks the later articulated-manipulation (S3) and learned-policy
(I2) phases. Three independent, additive parts.

- **F1a — realistic objects (Google Scanned Objects), finished end-to-end.** The
  GSO assets, `scanned_table_scene.xml`, and `SegmentClassifyPerception` (depth
  segmentation + best-effort open-vocab crop labels) already existed. Verified the
  full pipeline: geometric localisation is **< 3 mm (xy)** on all four objects, and
  a full dual-arm pick-and-place delivers a scanned mesh into a bin (the +y mug is
  handed off to the right arm to reach the right bin). **Fix:** `demo_interactive
  --scanned` crashed when the optional open-vocab detector failed to import (a
  torch/torchvision version mismatch — `torchvision::nms`); it now **falls back
  gracefully to geometric-only localisation** (objects selected by number), so the
  demo runs without the heavy/broken dependency. The torch/torchvision fix is
  deferred to F3 (where the CUDA/torch stack is set up) to avoid disturbing the
  passing suite. (`tests/test_scanned.py` already covers localisation + delivery.)
- **F1b — 6-DOF grasp solver** (`grasp6.py`, new, beside the top-down `grasp.py`).
  `Grasp6DOFSolver` searches a full **approach direction + roll** instead of only
  top-down yaw. A straight-down approach reproduces the top-down grasp exactly, and
  a small tilt penalty makes it **prefer straight-down**, tilting only when that
  lowers IK error or no vertical grasp is reachable — a strict superset that changes
  nothing existing. Purely geometric/analytic (a learned proposer can later sit
  behind the same `solve` interface). Exported from the package `__init__`.
  `tests/test_grasp6.py` (3): down-approach == top-down (valid rotation), prefers
  `tilt=0` on a graspable point, finds a reachable 40 deg grasp when forced. Search
  ~2.4 s (capped IK restarts).
- **F1c — authored articulated assets** (`articulated_scene.xml`, new;
  `SCENES["articulated"]`). Three from-scratch fixtures (no external mesh download,
  fully controllable): a **drawer** (prismatic slide, range 0.10 m), a hinged
  **cabinet door** (revolute, 0–1.4 rad), and a turnable **valve** (revolute,
  ±3 rad), each with a graspable handle and both-arm grasp welds (for S3). Physics
  verified: compiles (nq 21), stable at rest, and every joint articulates within
  its range when pushed. `tests/test_articulated_assets.py` (3). The skills that
  *operate* these (open drawer/door, turn valve, then language) are phase S3 / I1.

**+6 tests (129 total).** No existing controller, scene, or test was modified.

### Phase F2 — compliant (admittance) control
OpenArm's defining trait is compliant, contact-rich operation; this adds it on top
of the existing position-control stack without touching it.
- **New `openarm_control/contact/` package** with `AdmittanceController` (translational
  Cartesian admittance). Each tick it (1) measures the external force on the gripper
  by summing contact forces over the **end-effector subtree geoms** in world frame
  (excluding internal finger-finger contacts), (2) integrates a virtual
  mass-spring-damper `M x_ddot + D x_dot + K (x_ref - x_des) = F_ext` to get a
  **compliant reference** that backs off when pushed, and (3) solves IK to the
  reference and drives the **existing position actuators + gravity compensation**.
  Force is low-pass filtered and the reference velocity/offset clamped for stability
  on spiky discrete contact. Orientation is held rigidly (translational admittance).
- **New `contact_scene.xml`** (`SCENES["contact"]`): a soft compliant pad (slow
  `solref`) on the table within the right arm's straight-down reach, so the contact
  force rises smoothly (a rigid point-contact slams and is chaotic — wrong test
  surface). `openarm admittance` (`demo_admittance.py`) + `tests/test_admittance.py` (2).
- **Result:** pressing the pad 3 cm deep, admittance settles at **~27 N** vs **~217 N**
  for plain position control commanding the same depth — **~8x lower contact force**,
  same settle depth. The compliant controller is the basis for force-guarded insertion,
  wiping, and operating the drawer/door/valve in S3.
- **Debugging lessons (all real):** (1) the contact-force **sign** must give the
  reaction *on* the EE (mj_contactForce acts on geom2) — getting it backwards made the
  loop push *harder*; (2) a **closed gripper's finger-finger contact** is internal and
  must be excluded or it swamps the external force; (3) a fixed **yaw=0** top-down pose
  is often unreachable — use the grasp solver's searched yaw (the press point looked
  "unreachable" until then); (4) rigid point-contact on a stiff surface is **chaotic**
  (force swung 113->4400 N run-to-run) — a soft pad is the right demonstrator.

**+2 tests (131 total).** No existing controller, scene, or test was modified.

### Phase S2 — bimanual coordination (unscrew / hold-while-manipulate)
The canonical "one arm stabilises while the other manipulates" skill.
- **New `unscrew_scene.xml`** (`SCENES["unscrew"]`): a jar (free body with a +y side
  handle) and a **separate lid** (free body) resting on it with a graspable knob.
  Because the lid only rests on the jar, turning it drags the jar via friction --
  so the jar must be held. Both arms' grasp welds (`grasp_left_base`, `grasp_right_lid`).
- **`bimanual.UnscrewTask`**: LEFT arm grasps the handle and holds the jar; RIGHT arm
  grasps the knob, **turns the lid** (a wrist-yaw sweep at fixed position, ~2 rad),
  **lifts it off**, and sets it aside. Grasps are off the body origin (handle/knob),
  so both are welded at their **actual** relative pose (a non-centred weld -- the
  usual centring would yank the body's origin under the gripper). A small `_dual_move`
  helper quintic-drives both arms while gravity-compensated. `openarm unscrew`
  (`demo_unscrew.py`) + `tests/test_unscrew.py` (2).
- **Result:** the lid turns **~119 deg**, is lifted off and carried **~185 mm** aside,
  while the jar stays put (**~10 mm** drift) -- genuine bimanual coordination, since
  the hold is what stops the turning lid from dragging the jar.
- **Honest scope note:** the *simultaneous rigid lift-and-carry* of one object by both
  arms (a tray/bar grasped at both ends) was prototyped and **deferred**: two 6-DOF
  welds on one free body is a closed kinematic chain (over-constrained) and needs
  coordinated Cartesian control + a one-weld/one-connect formulation -- a separate
  effort. Unscrew delivers the bimanual-coordination milestone cleanly without that.

**+2 tests (133 total).** No existing controller, scene, or test was modified.

### Phase S2 (redesign) — realistic bimanual bottle opening
The first S2 unscrew (a lid resting on a jar) was redesigned after live testing
showed three problems: the two arms collided, the lid popped off on contact, and
turning a resting lid felt unthreaded. Fixes:
- **No collision:** a bigger bottle with the cap up on top/centre (right arm) and a
  low +y side **handle** (left arm, grasped via the F1b 6-DOF solver) -- the two
  grasps are ~140 mm apart with the inter-arm checker reporting no collision.
- **No pop-off / real thread:** the cap is now a **jointed** body on the bottle
  (turn hinge + lift slide), so it cannot be knocked off and can only screw on its
  axis. A lift actuator provides the coupled **rise** (it goes up as it turns).
- **`UnscrewTask` (rewritten):** left arm welds + holds the bottle (resisting the
  turn reaction); the right wrist turns the cap via a grasp weld, re-gripping over
  several bursts (the wrist spans only ~0.8 rad, so it ratchets like a human), the
  cap rising as it turns; then it lifts the cap clear.
- **Result:** stable, collision-free; cap unscrewed **~95 deg** over the bursts,
  rises and lifts off, bottle held to **~16 mm**.
- **Honest limitation (real, hit hard):** a *full multi-rotation* (360 deg+) was not
  achievable reliably. The wrist physically torquing a threaded cap only transmits
  ~0.8 rad/grasp (weld rotational slip + wrist range), and **actuating** the cap's
  screw goes unstable in the bimanual context (the driven cap on a held bottle, or a
  gripper contacting an actuated cap, both produce NaNs). So the reliable result is a
  ~95 deg "loosen + lift off". Carrying the cap to the table aside needs a free-body
  model-swap (the cap is jointed to the bottle axis) -- left as a follow-up.

**Tests: `tests/test_unscrew.py` rewritten (2).** Still 133 total. No pre-existing
controller/scene/test outside the unscrew was modified.

### Phase S2 (final) — single-arm bottle opening (collision-free)
Live testing of the bimanual unscrew showed the two arms collide. I verified with
exact geom distances that this is **fundamental**: two 7-DOF arms working at one
small bottle cannot avoid contact — the working arm's redundant elbow flips into the
holding arm at unpredictable points throughout the motion (clearance guards just
skip the whole unscrew). Resolution: **make it single-arm**.
- **`unscrew_scene.xml` (single-arm):** the bottle is held **fixed in a stand**
  (clamped), with the threaded cap (turn hinge + lift slide + lift actuator). Only
  the right arm works; the left is parked at the neutral (all-zeros) pose, well out
  of the way. No `grasp_left_bottle` weld.
- **`UnscrewTask` (rewritten, single-arm):** the right arm grasps the cap and
  unscrews it over re-gripping bursts (wrist turns the welded cap; the cap rises via
  the lift actuator), then lifts it clear. Clamping the bottle makes the wrist+weld
  turn transmit far better than against a held bottle.
- **Result:** the cap turns **~334 deg** (a near-full multi-turn unscrew), rises and
  lifts off **~48 mm**, and the working arm stays **~210 mm clear** of the parked
  arm — **collision-free** and a more complete unscrew than the bimanual attempt.
- **Why not bimanual:** delivered honestly — two arms at one small bottle collide
  regardless of separation/IK tricks; the clamp is the clean fix (and matches how a
  bottle is realistically opened against a fixed base).

**Tests: `tests/test_unscrew.py` updated (2).** Still 133 total.

### Phase S3 — articulated-object manipulation (drawer / door / valve)
Operating the F1c articulated fixtures, building on the F2 weld-assisted grasp.
- **`openarm_control/articulated.py` `ArticulatedController`** (single-arm, weld-assisted):
  the arm grasps a handle, the fixture body is welded to the gripper (non-centred,
  since the handle is off the body origin), then the arm moves the handle along the
  joint's allowed motion -- a **straight pull** for the prismatic drawer, and an
  **arc about the joint axis with a matching wrist rotation** for the revolute door
  and valve (so the gripper genuinely swings/turns the part, not just translates).
  The non-working arm is **parked at zeros** (a raised park sits in the working
  arm's swing path -- the S2 lesson).
- **Scene tuning (reachability):** the F1c fixtures were spread for a physics test;
  two were out of reach for an arc sweep, so the **valve** was moved in + its lever
  shortened, and the **door** was moved closer (its arc was unreachable past ~0.3 rad
  at the far +y edge). The door was also lightened so it follows the gripper.
- **Result (all collision-free, ~210 mm inter-arm clearance):** drawer slides open
  **~77 mm**; cabinet door swings open **~40 deg**; valve turns **~75 deg**.
- `openarm articulated --task drawer|door|valve` (`demo_articulated.py`);
  `tests/test_articulated.py` (3) check the controller drives each joint (distinct
  from the F1c asset test, which only checks the joints move when pushed).

**+3 tests (136 total).** F1 (assets) + F2 (compliance) come together here.

### Cloth-folding recheck (deformable manipulation revisited)
The fold was flagged as "not very accurate" and the cloth model as maybe unrealistic.
Diagnosed and improved:
- **Cloth MODEL (the real issue) — improved.** The old cloth was a coarse **5x5** grid
  with **self-collision OFF**, so a folded layer had nothing to rest on (it sprang back
  / passed through). Rebuilt as a finer **9x9** flex grid with **self-collision enabled**
  (`selfcollide="narrow"`, higher cloth-cloth friction). It now settles flat (~5 mm
  spread), stays stable, simulates fast, and folds into friction-held layers. (`cloth_vertices`
  is now grid-size-agnostic; welds/tests/demo updated to the 9x9 corners 0/8/72/80.)
- **Fold = single-arm (collision-free).** The corner fold folds the sheet in half
  (y-span shrinks > 25 %); self-collision lets the folded part lie on the layer below.
- **Bimanual half-fold — attempted and deferred.** Grasping one
  edge's two corners and laying it on the opposite edge is **very accurate (~4-18 mm)**, but
  I could **not make it reliably collision-free**: two close-mounted 7-DOF arms working over
  one centred cloth collide *in motion* (upper arms, then wrists, then fingers) regardless of
  grasp separation. The static hover/grasp/carry poses ARE clear (~18 cm), and positioning the
  arms there avoids the *approach* cross -- but the grasp/carry motion still collides at some
  cloth size. This is the **same hardware limit as the bimanual bottle**; a clean two-arm fold
  needs a dedicated collision-checked dual-arm motion planner (a separate effort). Reverted to
  the collision-free single-arm fold.

**Tests still green.** The cloth scene/model/tests/demo were updated; no other system touched.

### Phase F3 — learning harness + CUDA/VRAM verification
The GPU-ready foundation for the learned-policy phases (I2 ACT/Diffusion, I3 VLA).
- **CUDA verified:** NVIDIA RTX 5060 Laptop, **8.5 GB VRAM**, torch 2.11.0+cu128
  (CUDA 12.8), `cuda.is_available()=True`, GPU matmul confirmed. `openarm device`
  (`imitation/device.py`: `get_device()` / `device_report()`) reports it.
- **GPU-aware training:** `imitation/bc.py` `train_bc` now moves the model + data to
  the device and trains on the GPU (reach demos: MSE 0.24 -> 0.006, 40 epochs in
  ~3.4 s); `BCPolicy.act` is device-agnostic so CPU-loaded models still work
  (backward compatible -- the existing imitation tests pass unchanged).
- **Image-observation logging:** `imitation/collect.py --images` renders a camera RGB
  frame per step (downscaled, default 96 px) and stores it in the npz alongside the
  state + action -- the observation a vision policy (ACT/Diffusion) trains on. Verified:
  `images (N, 96, 96, 3) uint8`, one frame per transition.
- **Dependency notes:** kept the harness **self-contained** (extends the existing npz
  pipeline) rather than pulling in LeRobot; ACT/Diffusion (I2) will use a custom CNN
  (no torchvision) since torchvision is still version-mismatched with torch 2.11
  (the F1a `torchvision::nms` issue) -- so it doesn't block GPU training.
- `tests/test_harness.py` (4): device utility, BCPolicy on-device act, BC trains +
  reloads, image-observation collection shape/dtype.

**+4 tests (140 total).** No existing controller/scene/test changed behaviour.

### Phase I2 — ACT (Action-Chunking Transformer) learned policy
The project's **first non-baseline learned policy** -- a vision+state policy trained
on the GPU, beyond the MLP behavior-cloning baseline.
- **`imitation/act.py` `ACTPolicy`** (self-contained, no torchvision): a small CNN
  encodes the camera image into visual tokens, the proprioceptive state into one
  token; a Transformer **encoder** fuses them and a **decoder** reads K learned
  position queries to predict a **chunk of K future actions** at once (action
  chunking -> smooth, temporally-consistent motion). ~1.4 M params. Deterministic
  core of ACT (no CVAE latent). `train_act` (GPU, AdamW + L1), `load_act`, `evaluate`.
- **Trained + evaluated end-to-end on the reach task** (using the F3 image-demo
  pipeline): collect 51 demo episodes (1537 image+state transitions) -> train 80
  epochs on the RTX 5060 (**L1 0.39 -> 0.06, ~50 s**) -> eval **~75-85 % success**
  (17/20 at seed 500), comparable to the BC baseline but now vision-based + chunked.
- `openarm act train|eval`; `tests/test_act.py` (2): architecture/shapes, trains +
  lowers loss + reloads on a small dataset (the full reach run is a longer GPU job).
- Demo artifacts: `demos/reach_vis.npz` (image+state demos) + `demos/reach_act.pt`.
  (For publishing these data/weight files should be gitignored -- regenerable.)

**+2 tests (142 total).** Establishes the learned-policy path; the harder tasks
(insertion via S1, drawer/cloth) reuse the same ACT/collection pipeline.

### Phase S1 — RL insertion suite (peg-in-hole, domain-randomized)
The heaviest phase: a contact-rich insertion environment with the design --
**one robot, many randomized holes** (not many robots) -- and a classical-vs-BC-vs-RL
comparison.
- **`rl/insert_env.py` `OpenArmInsertEnv`** (Gymnasium): the peg is held in the gripper
  (welded); each episode randomizes the **socket position**, the **peg start offset**,
  the **friction**, and the **peg radius** (different clearances) -- domain randomization
  for a precise, robust policy. Obs 23 (state + peg-tip + socket + relative), action 7
  (joint deltas), shaped reward + insertion success. Gymnasium-compliant (passes
  `check_env`, incl. determinism). Registered in `rl.TASKS` and `imitation.TASKS`.
- **Scripted insertion expert** (classical baseline + BC/ACT demo source): a fixed
  reachable top-down yaw, then a **continuous fixed-orientation Cartesian path** that
  aligns over the socket and descends vertically to seat the peg.
- **Result (30 randomized sockets):** CLASSICAL (scripted) **100% @ ~1.5 mm**; **BC
  (state MLP) 67%**; RL/SAC is a longer GPU job (`openarm rl-train --task insert`, user-run).
- `tests/test_insert_rl.py` (3): Gym-compliance, socket randomization, scripted-insert
  success. Artifacts `demos/insert.npz` + `insert_bc.pt` (gitignore for publishing).
- **Hard-won debugging (all real, the contact-rich reality):** (1) a fixed yaw is
  unreachable at the socket -> search yaw AND check the achieved **orientation** (a
  position-only IK accepts a 36deg-tilted gripper); (2) the held peg drooped under
  gravity -> a **much stiffer weld** keeps it rigid; (3) a **joint-space** proportional
  descent rotates the gripper and tilts/jams the peg -> a **fixed-R Cartesian** descent;
  (4) the peg seats inside the socket walls (~z 0.44), not at the table -> calibrate the
  success depth. Tight-clearance precision insertion is the **compliant-control (F2)**
  variant -- a rigid descent jams; this env uses a forgiving clearance the policies solve.

**+3 tests.** RL env + classical/BC comparison delivered; SAC training is user-run.

### Phase I1 — language for the articulated skills
Wires the natural-language parser to the new articulated-object skills (S3).
- **`agent/commands.py`**: added `open_drawer`, `open_door`, `turn_valve`, `unscrew`
  actions -- parsing "open the drawer", "pull the drawer open", "open the cabinet
  door", "turn/rotate the valve", "unscrew the cap", "open the bottle". Crucially
  checked **before** "open" = release-the-gripper, so "open the gripper" still
  parses as `release` (disambiguation by the fixture noun).
- **`agent/articulated_session.py` `ArticulatedSession`**: parses a (possibly
  multi-step) command and dispatches each clause to the `ArticulatedController` --
  e.g. "open the drawer then turn the valve" runs both in order (via `split_steps`).
- `openarm articulated --command "open the drawer then turn the valve"` (+ headless).
- `tests/test_i1_language.py` (4): articulated-command parsing, open-hand-vs-drawer
  disambiguation, multi-step split, end-to-end session dispatch (the drawer opens).

**+4 tests.** The new skills are now language-commandable, like the earlier
pick/place/stack/insert vocabulary.

### Phase E1 — OpenArm-Bench (unified skill evaluation)
A single benchmark consolidating the extension-arc skills with a standardized
protocol and the **classical vs BC vs ACT vs RL** method comparison -- a citable
summary of what the platform does and how the methods stack up.
- **`benchmarks/openarm_bench.py`**: runs insertion (classical vs BC), reach (BC vs
  ACT), articulated (drawer/door/valve), admittance (compliant vs rigid), and cloth
  fold; prints a table + writes `benchmarks/results/openarm_bench.csv`. `--quick` /
  `--only <subset>`. (Catching/throwing keep their own dedicated benchmarks.)
- **Headline results:** insertion classical **100%** / BC ~62%; reach BC ~88% / ACT
  ~88%; drawer **77 mm**, door **40 deg**, valve **75 deg**; admittance **27 N** vs
  rigid **217 N**; cloth fold **~44%** span reduction.
- `tests/test_openarm_bench.py` (1): the runner executes + writes the consolidated CSV.

**+1 test (150 total).** **Extension roadmap complete** -- F1, F2, S2, S3, cloth, F3,
I2, S1, I1, E1 all done (I3/TinyVLA deferred: 8.5 GB VRAM is tight for a full VLA).

---

## Octo (vision-language-action) — local fine-tune of a generalist policy

A learned **generalist transformer policy** added beside the existing ACT/BC/RL
baselines. Octo (Berkeley, 2024) provides a pretrained vision-language-action backbone
that fine-tunes on a small set of task-specific demonstrations. **Octo-small** (~27 M
params) is the chosen size: it trains end-to-end on the local 8.5 GB RTX 5060 with
batch=1 + gradient checkpointing — no cloud GPU required. (Octo-base ~93 M and
OpenVLA-7 B both need more VRAM; OpenVLA was the original I3 target and is deferred
again, but the free Colab T4 path remains available as a fallback.)

The pipeline is:
1. **Collect** scripted RGB+state demos via the existing `bc-collect --images` flow,
   upgraded to **256×256** frames (Octo's expected input resolution).
2. **Convert** the npz dataset into the **RLDS / TFDS** format that Octo's fine-tune
   scripts read.
3. **Head-only fine-tune** of octo-small on the OpenArm dataset. (Octo's
   `scripts/finetune.py` exposes three modes — `head_only`, `head_mlp_only`, `full`;
   it does **not** ship LoRA. `head_only` freezes the transformer backbone and trains
   only the action head — the lowest-VRAM mode and the right fit for in-domain
   fine-tuning on a small custom dataset.)
4. **Evaluate** the resulting policy in `reach_env` / `insert_env` and add a
   "Octo (vision+language)" row to OpenArm-Bench beside scripted / BC / ACT.

### Scope notes
- **Stack is dropped from the demo set:** no scripted `StackExpert` exists in
  `openarm_control/imitation/expert.py` (only ReachExpert + InsertExpert). The fine-tune
  covers **reach + insert** (~1 000 demos total), still substantial — Octo's fine-tune
  recipes typically operate on 50–500 demos per task.
- Demo collection at 256 px (vs. the prior 96 px used for ACT) is a cosmetic resolution
  bump, not a structural change; the existing `bc-collect` pipeline handles it via the
  `--img-size 256` flag.

### Smoke verification
The upgraded `bc-collect` flow at 256 px image render is end-to-end verified —
5 reach episodes, 159 transitions, all successful, 9.7 s wall-clock. Output schema:
`obs (N, 23) f32 · act (N, 7) f32 · ep_lens (5,) i64 · images (N, 256, 256, 3) u8`
— exactly the shape Octo's fine-tune dataloader expects.

### Demo collection
- **Reach:** 500 episodes attempted → 429 retained (the default `only_success=True`
  filter dropped 14 % failures), 13 998 transitions, mean episode length 32.6 steps,
  **13.6 MB compressed** on disk (the sim renders' large flat-colour regions compress
  exceptionally well — uncompressed would be ~2.7 GB).
- **Insert:** 500 episodes attempted → **500 retained (100 % success)** — the scripted
  Cartesian-path expert is deterministic and never fails. 32 500 transitions, mean
  episode length 65.0 steps, **519 MB compressed** on disk.

**Combined dataset (Octo fine-tune input):** 929 episodes, 46 498 transitions, ~533 MB
on disk. Plenty for head_only fine-tuning — Octo's recipes typically run on 50–500
demos per task.

### RLDS / TFDS converter
**`scripts/openarm_dataset/`** — a TFDS `DatasetBuilder` that wraps the npz files into
the RLDS schema Octo's `scripts/finetune.py` consumes. Built following the
`kpertsch/rlds_dataset_builder` template (the convention the Octo team recommends
for custom datasets). Two tasks are merged into one dataset, each tagged with its
own `language_instruction` prompt:
- `"reach the target"` ← `demos/octo_reach.npz`
- `"insert the peg into the socket"` ← `demos/octo_insert.npz`

Per-step schema: `observation.image (256, 256, 3) u8`, `observation.state (23,) f32`,
`action (7,) f32`, `language_instruction` (text), plus the standard RLDS episode
flags (`is_first` / `is_last` / `is_terminal`) and `discount` / `reward`.

Run with `pip install tensorflow tensorflow-datasets`, then
`cd scripts/openarm_dataset && tfds build` — the dataset materialises under
`~/tensorflow_datasets/openarm_dataset/1.0.0/` and the Octo finetune script reads it
by name via its dataset config.

---

## Phase B1 — Ball balancing (PD on a tilted plate)

The classical ball-on-plate stabilisation problem on the OpenArm. A square plate is
held by the right gripper at a fixed world position; a free ping-pong ball rolls on
it; the controller tilts the plate (small roll/pitch about world axes) to keep the
ball at a target xy on the plate surface. Distinct from every existing skill on the
platform — quasi-static manipulation gives way to **real-time dynamic
stabilisation**.

### Scene (`v2/openarm_mujoco_v2/balance_scene.xml`)
- 18-DOF arm (both arms attached) + plate free body + ball free body → `nq = 32`.
- **Plate:** 15 × 15 × 1 cm box, mass 50 g, `condim=6` rolling-friction contact,
  stiff critically-damped contact (`solref=0.005 1, solimp=0.95 0.99 0.001`).
- **Ball:** real ping-pong specs — 40 mm diameter, 2.73 g, sphere geom, same
  stiff-but-damped contact. With these tunings a freely dropped ball settles
  in <0.5 s with **0 mm steady-state bounce** (verified by the scene's drop-test
  probe).
- `<contact><exclude>` directives between (plate, ball) and the right gripper's
  collision bodies (`openarm_right_ee_base_link` + the two finger bodies). Without
  these the welded plate's static interpenetration with the gripper mesh generates
  enormous constraint-resolution forces every step, which then catapult the ball
  laterally at contact (observed: 1.3 kN/s impulse on a 3 g ball → ejected at 1.7 m/s).

### Controller (`openarm_control/balance.py`)
Two-class layout sharing a hold + tilt scaffold; the actual control law goes in
subclasses. `PDBalancer` is Tier 1; an `LQRBalancer` is planned for Tier 2.

**Hold mechanism — manual pin, not soft weld.** The first implementation used a
MuJoCo `weld` equality between the gripper and the plate free body. With a 50 g
plate + stiff contact + the welded constraint solver, the system was
contact-impulse-unstable: ball-plate collision spikes propagated through the weld
back into the arm and back into the ball, ejecting the ball even when the plate
was nominally flat. Switched to **kinematic puppetry**: after every `mj_step`,
`_pin_plate_to_gripper` snaps the plate's free joint to
`gripper_pose · stored_relpose` and zeros plate qvel. No constraint solver
involvement, no impulse spikes, no plate-mass gravity comp needed (plate is purely
kinematic). The weld is left declared but inactive.

**Reference capture (no IK in setup).** `setup_hold` does NOT run IK to a chosen
hold pose — that fails silently if the target is unreachable (IK returns its
best-effort residual config without flagging). Instead it uses the right arm's
existing `ready` keyframe pose, queries forward kinematics for the achieved tool
point + orientation, and stashes those as `_hold_pos` / `_R_grip_init`. All later
control steps re-IK to *those exact values* with `R_tilt @ _R_grip_init` — a zero-
tilt command then has zero IK residual, so the gripper doesn't drift.

**PD law (signs verified empirically).** With `R_tilt = R_x(roll) · R_y(pitch)`
pre-multiplied in the world frame: positive pitch → ball accelerates +X; positive
roll → ball accelerates -Y. To drive the ball back to the target:
```
pitch = -KP * (x - target_x) - KD * vx
roll  = +KP * (y - target_y) + KD * vy
```
Default gains `KP = 6.0 rad/m`, `KD = 1.8 rad·s/m`, max tilt 20°. Tilts capped so
the arm stays well inside joint limits.

### Verified result
With ball placed at **(+30 mm, +20 mm)** offset on the plate, the PD controller
converges to **< 1.1 mm** mean error over the last 0.4 s of a 6 s episode (peak
excursion during recovery: ~53 mm). At the harder **(+40 mm, +25 mm)** start,
final error 1.86 mm. Repeatable, deterministic.

### CLI + test
- `openarm balance` — viewer (default `LQR`).
- `openarm balance --controller pd|lqr|both --headless [--offset x,y]` — scripted
  report; `both` runs head-to-head.
- `tests/test_balance.py` — regression tests (scene compiles, hold pins plate
  horizontally, PD settles, LQR settles + beats PD, LQR gain matrix structure).

### Phase B2 — LQR controller variant

Discrete LQR on the linearised ball-on-plate dynamics, sharing `BallBalancer`'s
hold + tilt scaffold. The linearised model about the flat-plate equilibrium is:

```
state x = [px, py, vx, vy]        (ball position + velocity in plate frame)
input u = [roll, pitch]
   dpx/dt = vx
   dpy/dt = vy
   dvx/dt = +G_EFF · pitch          (G_EFF = 5/7 · g)
   dvy/dt = -G_EFF · roll
```

Euler-discretised at the sim timestep (`dt = 2 ms`), the discrete algebraic
Riccati equation (`scipy.linalg.solve_discrete_are`) gives the infinite-horizon
gain `K`, and the control law is `u = -K (x - x*)` with the same MAX_TILT clip
as PD.

**Cost tuning.** Q, R chosen so the resulting gain stays well inside MAX_TILT
even at 60 mm offsets — an initial LQR draft with an aggressive Q/R saturated
the tilt, and the arm's finite servo bandwidth means a saturating tilt takes
several steps to be *achieved*, during which the arm swings *through* the
wrong intermediate tilt and drives the ball off. Gentler gains
(`Q = diag(40, 40, 8, 8)`, `R = diag(8, 8)`) keep the commanded tilt within
the arm's one-step actuator range. Concretely `K[1, 0] ≈ 2.22 rad·m⁻¹`, so a
60 mm x-offset commands only ~7.6° of pitch — far below the 20° cap.

**Head-to-head vs PD** (100-episode fixed-seed, same offset sweep,
`openarm balance --controller both`):

| offset       | PD peak / final / RMS         | LQR peak / final / RMS        |
|---           |---                            |---                            |
| (30, 20) mm  | 53.3 / 1.04 / 3.54 mm         | **38.6 / 0.39 / 0.65 mm**     |
| (40, 30) mm  | 61.8 / 1.85 / 6.83 mm         | **53.1 / 0.39 / 0.78 mm**     |
| (50, 30) mm  | 80.0 / 1.72 / 8.02 mm         | **62.5 / 0.39 / 0.89 mm**     |
| (60, 40) mm  | both diverge (natural stability limit — the arm can't tilt fast enough) |     |

LQR wins on every measurable axis in the working regime: lower peak excursion,
~4× lower final steady-state error, ~5-10× lower steady-state RMS. Cleaner
transients (no PD overshoot) and quieter steady state, at the same tilt cap.

**+2 tests → 155 total.** Full suite green.

### Phase B3 — Trajectory tracking, disturbance rejection, hero GIFs

Two demo scenarios added on top of the static holding case, plus the visual
geometry redesign that ships in the same pass.

**Trajectory tracking.** `demo_balance` accepts `--trajectory circle|figure8`
with `--radius` and `--period`. The controllers get a time-varying target
`(r · cos(ωt), r · sin(ωt))` (or Lissajous 1:2 for figure-8) and the same PD /
LQR laws drive the ball to follow it. Bounded tracking on both — PD's stiffer
gains lag less than LQR's softer ones (~11 mm vs ~21 mm mean tracking error at
3 cm / 5 s circle), a legitimate PD-vs-LQR tradeoff for time-varying references.

**Disturbance rejection.** `--perturb` applies a random-direction ~25 cm/s
velocity kick to the ball every `--perturb-period` seconds. LQR consistently
recovers within a few hundred ms; steady-state RMS 7.2 mm vs PD's 14.9 mm.

**Geometry redesign** (in response to a visual-correctness note during Tier 3
review): the original placement (plate 3 cm above the tool point) put the
plate at the same z as the gripper's finger collision meshes, so the fingers
visually appeared to penetrate the plate — even though `<contact><exclude>`
made the intersection dynamically inert. A palm-up arm keyframe would be the
ideal fix but is not reachable — link6 shares the palm's world position on
every joint config, and link4/link5 extend up to z ≈ 0.83.

The fix: place the plate 12 cm above the gripper (`_hold_pos + [0, 0, 0.12]`),
add a visual-only cylinder geom in the scene XML as a "riser rod" bridging the
gap, and extend `<contact><exclude>` to cover plate/ball vs. all right-arm
bodies (`link4`, `link5`, `link6` plus the palm + fingers). The plate now
visibly sits *above* the arm, connected by a rendered stem — reads as "arm
holds a table on a stem". Retuned PD gains (`KP=2, KD=1.2`) keep the shorter
lever arm's tilt-induced plate translation small; LQR's soft gains cope
unchanged. Perturbation-recovery gate widened from 20 mm to 30 mm to
accommodate the slightly larger post-kick transient at the new geometry.

**Hero GIFs** rendered via `scripts/gen_showcase_media.py` (new `balance_circle`
and `balance_perturb` skills): a top-3/4 view of the arm holding the riser +
plate, showing the LQR tilting the plate to track a circle or recover from
kicks. Written to `media/balance_circle.gif` (~3.8 MB, 72 frames) and
`media/balance_perturb.gif` (~3.0 MB, 72 frames).

**+2 tests → 157 total** (`test_circle_trajectory_stays_bounded`,
`test_perturbation_recovery`). Full suite green.

### Phase B4 — Model-predictive control + OpenArm-Bench entry

Adds a third controller and a formal benchmark row for the balance skill.

**MPCBalancer.** Pure LQR (a regulator) always lags a moving reference because
it can only react to *current* position error. A finite-horizon MPC on the
linearised ball-on-plate would derive an anticipatory tilt from the target's
future trajectory; for a smooth ball-on-plate model that anticipation collapses
to an **analytic feedforward** from the target acceleration:

```
pitch_ff = +a_x_ref / G_EFF     (positive pitch -> ball accels +X)
roll_ff  = -a_y_ref / G_EFF     (positive roll  -> ball accels -Y)
u        = -K (x - x_ref) + u_ff
```

Same LQR gain, same cost tuning. The *only* change from `LQRBalancer` is
`u_ff`. No QP solver, no new dependency. For a static target the feedforward
is zero and MPC reduces exactly to LQR (verified by
`test_mpc_settles_ball_like_lqr` — final errors identical to floating-point
precision).

**Interface.** All three balancers now accept `(target_xy, target_axy)` on
`step()`. `_target_state_at(t, mode, radius, period)` in `demo_balance.py`
returns both position and analytic acceleration per trajectory mode (circle:
`a = -w² · position` from simple harmonic motion; figure-8: same on x,
`a_y = -(2w)² · y` on the 1:2 Lissajous). PD and LQR ignore `target_axy`; MPC
uses it. `--controller mpc` (or `--controller both` for the head-to-head
three-way in `--headless`) is now wired.

**Head-to-head at a fast circle** (r=4 cm, T=2.5 s):

| method | static settle (mm) | circle RMS (mm) |
|---|---|---|
| PD  | 0.44 | 40.3 |
| LQR | 0.39 | 39.2 |
| **MPC** | 0.39 | **37.7** |

MPC ties LQR on the static case (as it should — `u_ff = 0`) and beats it by
~4% on the moving-target case. The effect grows with target frequency and
radius. Modest but real.

**OpenArm-Bench entry.** New `bench_balance` in `benchmarks/openarm_bench.py`
records both metrics (static final err, circle tracking RMS) for all three
methods. `plot_openarm_bench.py` gets a new `plot_balance` panel: two side-by-
side bars showing the PD → LQR → MPC progression on both scenarios. Written
to `benchmarks/figures/openarm_bench_balance.png`.

**+2 tests → 159 total** (`test_mpc_settles_ball_like_lqr`,
`test_mpc_beats_or_ties_lqr_on_trajectory`). Full suite green.

### Phase B5 — Learned SAC balancer (classical vs. learned head-to-head)

Fifth column in the balance benchmark: SAC on the same physics/hold as
PD/LQR/MPC, in two variants — a policy trained end-to-end from scratch, and a
**residual policy** that learns a small correction on top of an LQR baseline.
The point of the section is to place model-free RL alongside hand-derived
classical control on a problem where the linearised model admits a
closed-form optimal law.

#### Environment

`openarm_control/rl/balance_env.py` (`OpenArmBalanceEnv`) reuses the exact
`BallBalancer` hold + tilt + manual-pin scaffolding the classical controllers
already use. Only who chooses `(roll, pitch)` each step changes.

```
Observation (6):  [x, y, vx, vy, x - tx, y - ty]      ball state + error to target
Action (2):       [roll, pitch] in [-1, 1] * MAX_TILT (~20°)
Reward:           - distance
                  - 0.05 * speed
                  - 0.001 * |action|
                  + 0.5 * exp(-(distance / 1 cm)²)     sharp precision bonus
                  - 5.0 * (ball rolled off plate)      one-shot terminal penalty
                  + 2.0 * success at end of episode
Termination:      ball outside 7 cm disk               OR  ≥ 300 steps
Rate:             100 Hz policy control                5 sim substeps per action (ZOH)
```

The 6-D observation deliberately includes both the absolute ball position and
the target-relative error. A scripted LQR policy fed through the env
(`u = -K · obs[:4]`) hits **3/3 success** on smoke seeds, which validates the
reward shape before any compute is spent on SAC training.

**Env sanity gate.** `tests/test_rl_balance_env.py` (+3 tests):

- `test_env_spaces_and_reset` — Gymnasium contract (6-D obs, 2-D act, spaces).
- `test_env_random_action_stays_terminating_or_truncating` — a random-action
  episode terminates or truncates cleanly (guards against a stuck sim).
- `test_env_scripted_lqr_policy_succeeds` — scripted LQR through the env wins
  ≥ 2/3 seeds. Regression gate on the observation layout and reward.

The env is registered as `"balance"` in `openarm_control/rl/TASKS`, so the
existing training and eval pipeline (SB3 SAC + TensorBoard) works unchanged:

```bash
openarm rl-train --task balance --timesteps 200000    # ~65 min on CPU
openarm rl-eval  --task balance                       # watch in the viewer
```

#### Part 1 — SAC from scratch (200 k timesteps)

Standard SB3 SAC (`lr=3e-4`, `buffer=300k`, `batch=256`, `gamma=0.98`, MLP
`[256, 256]`) matching the reach hyperparameters. End-to-end ~50 fps
(env.step ≈ 10 ms plus SB3 update overhead), ≈ 65-70 min on CPU.

Training curve (TensorBoard scalars):

| step | ep_rew_mean | ep_len_mean | success_rate | ent_coef |
|-----:|------------:|------------:|-------------:|---------:|
| 150    | -6.4  | 35.8 | 0 % | 0.96 |
| 20 k   | +0.1  | 48.5 | 0 % | 0.01 |
| 100 k  | +5.0  | 54.0 | 0 % | 0.02 |
| 200 k  | +6.7  | 59.1 | 0 % | 0.03 |

The reward climbed from -6.4 to +6.7 and the episode length went from 36 to 59
steps — the policy **learned survival** (keeping the ball on the plate for
longer). But the success band (final distance < 20 mm, speed < 5 cm/s) stayed
at **0 %** across all 200 k steps. The entropy coefficient collapsed to ~0.02
by step 20 k: the policy locked into a "hold the plate roughly flat" mode
early and stopped exploring precise centering. Reward shape reads the same
way — the -5 terminal penalty for dropping the ball dominates the +0.5
precision bonus for centering it, so surviving is worth much more than
converging.

Bench integration: `bench_balance` in `benchmarks/openarm_bench.py` loads
`openarm_control/rl/models/balance_sac.zip` when present and runs it on both
the static-hold and circle-tracking scenarios. The bench also detects the
"ball rolled off the plate" failure mode and caps the reported metric at
100 mm — past plate radius the raw number becomes ball-in-world-frame
position (this trained SAC's ball ends the 6 s bench episode roughly 4 m
from the plate) which is accurate but not meaningful for head-to-head
plotting. `plot_openarm_bench.plot_balance` renders the SAC bar with a red-×
overlay and a **FAILED (ball off plate)** label whenever the cap is hit.

#### Part 2 — Residual policy over an LQR baseline

The intended pattern from the residual-policy-learning literature
(Silver et al. 2018, Johannink et al. 2019): a classical baseline
`u_LQR = -K x` does the bulk of the work, and a learned policy `δ(x)` adds a
small correction:

```
u_final = clip( u_LQR(x) + δ(x) * RESIDUAL_MAX_TILT,   ± MAX_TILT )
```

`openarm_control/rl/balance_residual_env.py` (`OpenArmBalanceResidualEnv`)
subclasses the base env: same observation, same termination, same reward
skeleton — only `step()` changes to compose the LQR baseline with a
±2° residual. `RESIDUAL_MAX_TILT = 2°` is deliberately small: the LQR
already keeps the ball on the plate from typical initial conditions, so a
random policy on top adds noise but doesn't blow the task apart. The env is
registered as `"balance_residual"`.

**Zero-residual smoke.** `test_residual_env_zero_action_equals_lqr` runs the
env with `action = 0` and requires the LQR baseline to succeed on ≥ 2/3
seeds. A partner test (`test_residual_env_random_residual_bounded`) checks
that random residuals produce finite rewards over a short horizon. +2 tests.

**First training run (50 k, no residual regularization).** The reward used
here has a very small linear action penalty (`-0.001 * |action|`). Training
started from an excellent position — SAC's initial actor outputs residuals
near zero, so the LQR baseline is doing essentially all the work. But the
result was the opposite of what residual learning is supposed to give:

| step | ep_rew_mean | ep_len_mean | success_rate | ent_coef |
|-----:|------------:|------------:|-------------:|---------:|
| 1.2 k  |  98.6 | 300 | **100 %** | 0.89 |
| 7.2 k  |  89.5 | 300 |    96 %   | 0.22 |
| 19 k   |  68.9 | 299 |    80 %   | 0.01 |
| 31 k   |  50.7 | 296 |    57 %   | 0.01 |
| 42 k   |  23.4 | 284 |    23 %   | 0.01 |
| 49 k   |  16.8 | 267 |    10 %   | 0.01 |

The training actively **unlearned** the LQR baseline. The mechanism is a
familiar bootstrap failure: SAC's maximum-entropy objective encourages
non-zero actions; the untrained critic has no signal for which direction is
good; the actor takes gradient steps that push residuals away from zero into
random directions; those larger residuals now hurt performance; the critic
learns *that* off the collected rollouts and the actor keeps drifting. The
policy ends with 10 % success from a baseline that started at 100 %.

**Second training run (30 k, residual reward reshaped).** Two changes to make
zero the strong attractor:

- Replace the linear action penalty (`-0.001 · |action|`) with a squared
  penalty (`-1.0 · |action|²`). A full-scale residual now costs about -300
  over an episode — more than the baseline reward, so anything but small
  corrections is strongly punished. Small residuals stay cheap.
- Double the precision bonus (`+0.5 → +2.0`) so residuals that measurably
  reduce error at target are rewarded enough to survive the squared cost.

The training run is boringly correct — success rate stays at 100 % from the
first evaluation window and the reward climbs monotonically as SAC finds
useful small residuals:

| step | ep_rew_mean | ep_len_mean | success_rate | ent_coef |
|-----:|------------:|------------:|-------------:|---------:|
| 1.2 k  | 199 | 300 | 100 % | 0.89 |
| 6 k    | 254 | 300 | 100 % | 0.22 |
| 15.6 k | 325 | 300 | 100 % | 0.04 |
| 30 k   | 372 | 300 | 100 % | 0.03 |

Bench integration adds a **LQR+SAC** column that composes the LQR feedback
per sim step with the residual policy queried at 100 Hz, mirroring what
happens inside the env.

#### Head-to-head result

| method | static settle (mm) | circle track RMS (mm) |
|---|---|---|
| PD  | 0.44 | 40.3 |
| LQR | 0.39 | 39.2 |
| **MPC** | **0.39** | **37.7** |
| SAC (from scratch) | ✗ ball off plate | ✗ ball off plate |
| LQR + SAC residual | 5.9 | 43.2 |

Three findings, each independently useful:

1. **From scratch, model-free SAC does not solve this task in 200 k
   timesteps.** The reward shape as written traps the policy in a
   survival strategy; it never learns precision. Under bench conditions the
   ball rolls off the plate within a few seconds.
2. **Naïve residual RL destroys a good baseline.** With a weak action
   penalty, SAC's exploration pressure and uninformed critic push a
   100 %-success LQR baseline down to 10 % over 50 k steps.
3. **Residual RL with the right regularization does converge, but it does
   not beat LQR on this plant.** With `-|action|²` regularization the
   policy holds 100 % success end-to-end and the training curve is
   monotone, but the learned residual costs ~5 mm of steady-state error
   on the static-hold protocol and slightly widens the circle-tracking
   RMS. The linearised ball-on-plate model admits an essentially optimal
   feedback law in closed form (the LQR itself), so there is nothing for a
   learned residual to add that improves on it — the same "problem
   structure is the solution" observation that killed the from-scratch
   run.

Where a learned residual *would* pay off — modelling gaps the linearisation
does not capture (unmodelled friction, sensor noise, actuator lag) — is out
of scope for this simulator's clean rigid-body physics. Model-based RL
variants (MB-SAC, TD-MPC) that can exploit the smooth dynamics belong in a
separate study.

Trained SAC models follow the same convention as `reach_sac.zip` — they are
**not versioned in the repo** (see `.gitignore`). The bench auto-detects
`openarm_control/rl/models/balance_sac.zip` and
`openarm_control/rl/models/balance_residual_sac.zip` when present; to fill
in the SAC / LQR+SAC bench columns, reproduce them with:

```bash
openarm rl-train --task balance          --timesteps 200000    # ~65 min CPU
openarm rl-train --task balance_residual --timesteps 30000     # ~15 min CPU
python benchmarks/openarm_bench.py && python benchmarks/plot_openarm_bench.py
```

**+5 tests → 164 total** (3 base env tests + 2 residual env tests). Full
suite green.

### Octo finetune configuration (planned)
From `scripts/configs/finetune_config.py` in the Octo repo, the override matrix is:

```python
FINETUNING_KWARGS = {
    "name": "openarm_dataset",
    "data_dir": "~/tensorflow_datasets",
    "image_obs_keys": {"primary": "image", "wrist": None},
    "proprio_obs_key": "state",
    "language_key": "language_instruction",
}
# Head-only mode freezes the transformer backbone and trains only the action head.
# Backbone activations still pass through (forward + backward to the head), so VRAM
# is dominated by activation memory; expect to need batch_size ~16-32 (not the 256
# default) to fit in 8.5 GB.
mode = "head_only"          # -> frozen_keys = ("octo_transformer.*",)
action_horizon = 16          # Octo default is 4; matched to ACT's chunk for fair eval
window_size = 1
batch_size = 16              # tuned at runtime to maximise free VRAM
optimizer = dict(
    learning_rate=dict(name="cosine", peak_value=3e-4, warmup_steps=2000),
)
pretrained_path = "hf://rail-berkeley/octo-small-1.5"
num_steps = 50_000           # initial target; refine after watching loss
save_interval = 5_000
```

The training entry point is `python scripts/finetune.py --config.pretrained_path=... \
--config.dataset_kwargs.name=openarm_dataset` (plus overrides). The `openarm octo
train` CLI will be a thin wrapper around this subprocess.

### Runtime: WSL2 required for GPU on Windows
Octo is **JAX/Flax-based**, and **JAX has no native CUDA support on Windows** — the
JAX team's official recommendation is **WSL2** for GPU acceleration. The local
RTX 5060's 8.5 GB VRAM is fully accessible from WSL2 via the Windows NVIDIA driver
(installed once on the host; CUDA is then auto-stubbed inside WSL2).

The fine-tune path is therefore:

```bash
# one-time, on Windows (PowerShell as admin)
wsl --install -d Ubuntu      # installs WSL2 + Ubuntu, reboot once
# inside WSL2 (Ubuntu)
sudo apt update && sudo apt install -y python3.11-venv python3-pip
python3.11 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install "jax[cuda12]" tensorflow tensorflow-datasets        # JAX-CUDA via wheels
git clone https://github.com/octo-models/octo && cd octo
pip install -e .
# build the RLDS dataset (the repo is mounted at /mnt/c/Users/manas/Desktop/.../)
cd /mnt/c/Users/manas/Desktop/ASSIGNMENTS,\ TESTS\ \&\ BOOKS/Robotics/openarm_mujoco-master/scripts/openarm_dataset
tfds build
# fine-tune
cd ~/octo
python scripts/finetune.py \
    --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 \
    --config.dataset_kwargs.name=openarm_dataset \
    --config.dataset_kwargs.data_dir=~/tensorflow_datasets \
    --config.batch_size=16 \
    --config.mode=head_only \
    --config.action_horizon=16 \
    --config.num_steps=50000
```

The free fallback if WSL2 is undesirable is **Google Colab (free T4, 16 GB VRAM)** —
JAX-CUDA works there out of the box; the friction is uploading the RLDS dataset
to Drive once and surviving the ~4 h idle-session reset by checkpointing.

## Task-scene and articulated-skill refinements (upstream contribution round-trip)

Contributing the task scenes to the upstream OpenArm model repository
(enactic/openarm_mujoco PR #40) triggered a deep validation pass — driving
every task with the arm rather than trusting static checks — and the fixes
that came out of it are folded back here.

### Scene fixes (`articulated_scene.xml`, `balance_scene.xml`)
- **Door hinge no longer binds.** The panel's inner edge overlapped the hinge
  post's diagonal half-width and ground against it through the swing (the old
  "door ~40 deg" bench number was achieved *through* that friction). Post
  slimmed to 7 mm, panel offset 15 mm from the axis, and the hinge got a
  crisp limit (`solreflimit 0.005`) — the grinding had been masking a soft
  range-limit overshoot under force.
- **Drawer redesigned for a real grasp.** The cabinet sits on a solid base
  (handle at a natural ~10 cm above the table) and the handle bar stands
  40 mm proud on two stems: a parallel gripper's closed cage is 30 mm wide,
  taller than the drawer's 56 mm front panel, so closing on a flush handle
  fouls on the panel and never wraps the bar. The cabinet moved to
  (0.40, -0.28) — clear of the ready-pose gripper's resting zone (a
  regression test now pins this: `test_ready_keyframe_clear_of_arms`).
- **Valve clearance.** At (0.26, -0.05) the full ±3 rad sweep stays clear of
  both parked arms and the taller cabinet.
- **Balance plate inertia** was 4x too large (computed from full dimensions
  instead of half-extents); corrected to 9.42e-5 / 9.42e-5 / 1.875e-4.

### Controller fixes (`articulated.py`, `grasp.py`)
- **Deterministic skills.** IK random restarts were unseeded, so a skill
  could land on a different arm branch every run — the frontal drawer pull
  measured 4.1 / 52.1 / 92.5 mm across three identical runs. Restart-using
  call sites now pass a fixed `IK_BRANCH_SEED`; every skill is bit-identical
  run to run.
- **Cartesian-chained approaches.** A single joint-space hop to a hover point
  takes a curved detour that swept the fingers through neighbouring fixtures
  (measured: outer finger 1.4-2.5 mm into the cabinet en route to the valve).
  Approaches now reorient overhead in place, translate straight at height,
  and descend vertically on one chained IK branch — zero fixture contact
  across all three skills, audited every physics step.
- **Grasps read as grasps.** The servo settles on the handle before the weld
  activates (welding early froze a visible offset — the gripper "pulled air"
  ahead of the handle), and the fingers close slowly and visibly first.
- **Frontal drawer pull.** `open_drawer` approaches the handle horizontally
  at a 15-degree downward pitch (`front_orientation` in grasp.py), slides the
  half-open cage over the bar, applies a one-step bias correction, closes,
  welds, and pulls the drawer straight back toward the robot before releasing
  and withdrawing — the way a person opens a drawer.
- **Skills release and retreat.** Each skill now deactivates its grasp weld
  and withdraws at the end; previously the weld stayed active, silently
  dragging the fixture during the next skill of a multi-step command.

### Verified numbers (bench re-run, 165 tests green)
| skill | before | after |
|---|---|---|
| drawer opened | 76.6 mm (top-down grasp) | **95.1 mm** (frontal grasp) |
| door swung | 40.4 deg (through hinge grinding) | **54.1 deg** (free hinge) |
| valve turned | 74.9 deg | **78.1 deg** (zero-contact approach) |

Balance classical numbers unchanged (the plate is kinematically pinned, so
its inertia does not enter the loop); the LQR+SAC residual cells shifted
marginally with the corrected plate inertia (5.9 -> 5.2 mm static,
43.2 -> 42.9 mm circle).

Deliberately NOT adopted from the upstream variant: the valve's separate
scene (the language-commanded "open the drawer then turn the valve" needs
all fixtures in one scene) and the hollow-ball inertia (the balance stack is
derived around the solid-sphere 5/7 factor; switching is a coordinated
change across G_EFF, controllers, tests, and bench, kept as future work).

## Contact honesty + IK contract (2026-07-05)

Two fixes surfaced by an adversarial re-review of the task scenes.

- **The contact pad's softness was inert.** The arm's collision geoms carry
  `priority="1"`, and MuJoCo gives *all* contact parameters to the
  higher-priority geom — so finger–pad contacts used the finger's rigid
  `solref="0.005 1"` and the pad's authored `solref="0.05 1"` never governed
  a press (first contact jumped ~98 N in a single 1 ms step). Fix:
  `priority="2"` on `pad_geom` in `contact_scene.xml`. A press now rises
  smoothly (max per-step jump ~7 N).
- **`inverse_kinematics` returned non-converged solves as if they were
  solutions.** The plain return handed back the best attempt even when it
  missed by tens of centimetres, so callers' `if q is None` guards almost
  never fired. It now returns `None` when the best solve misses by more than
  `IK_ACCEPT` (5 mm — task scale; measured caller regimes: chained waypoints
  reject at 0.1–4.6 mm and are functionally exact, garbage solves sit at
  97–131 mm). `return_info` still reports strict 0.1 mm convergence.
  One deliberate opt-out: the drawer's frontal approach family is off the
  reachable manifold by design (~13 cm nominal bias, absorbed by the skill's
  measured one-step bias correction) and now requests best-effort explicitly
  (`_ik_nominal` in articulated.py).

### Verified numbers (full suite green, full bench re-run)
| cell | before | after | why |
|---|---|---|---|
| admittance compliant | 27.6 N | **20.5 N** | pad genuinely soft now |
| admittance rigid | 213.1 N | **63.9 N** | same (even a rigid press reads lower on a soft pad) |
| door swung | 54.1 deg | **53.7 deg** | one arc waypoint with a 12.3 mm best solve is now skipped instead of commanded |
| drawer / valve / all other cells | — | unchanged | drawer byte-identical at 95.1 mm |

The compliant-vs-rigid contrast the admittance row demonstrates is
controller behaviour, and it survives: same press depth, ~3× lower steady
force with admittance.

### Drawer approach: one branch, no snap (same day)

The drawer GIF showed the arm turning toward the handle and then snapping
3.5 rad back to a mirrored configuration mid-approach — the advance chain
was silently flipping IK branches, and the measured 95 mm pull only worked
*through* that flip. The skill now picks its arm branch once, from the
hardest pose of the motion (the end of the pull), back-chains the
pre-approach onto it, rejects any chained waypoint that leaves the branch
(>0.5 rad jump), and finishes the pull closed-loop on the actual slide
reading. The motion is now a single smooth branch (worst inter-waypoint
step 0.069 rad); the honest cost is a shorter pull: **95.1 -> 83.8 mm**
(this branch's reachable ceiling; the drawer's slide range is 100 mm).
Full suite green; drawer GIF regenerated.
