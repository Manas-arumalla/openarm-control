"""Dynamic manipulation: catch a ball thrown through the air.

This is a research-grade airborne ball-catching pipeline — the textbook robotic
catch loop, not a "park where the ball will be and wait":

    observe ball ->  BallisticKalmanFilter (constant-accel, gravity known)
                       |  estimates position + velocity, predicts the parabola
                       v
            InterceptionSolver  : search the predicted parabola for the EARLIEST
                       |          catch point that is (a) inside the arm's reach
                       |          and (b) reachable within the available flight
                       |          time, facing the incoming velocity.  IK at that
                       v          point gives the catch configuration q*.
        receding-horizon replanning (MPC) : every few ms, re-estimate and
                       |          re-generate a minimum-jerk JOINT trajectory that
                       |          arrives at q* exactly at the catch instant, with
                       v          the hand moving WITH the ball (velocity matched).
        soft catch     : when the ball enters the jaws, match its velocity, close
                         the fingers, weld (firm grasp), follow through, hold.

If no point on the trajectory is reachable in time, the solver returns ``None``
and the catch is honestly reported as a miss — nothing is faked.
"""

from itertools import permutations

import numpy as np
import mujoco

from .config import RIGHT_ARM, LEFT_ARM, GRASP_LOCAL_OFFSET, IK_TOLERANCE, IK_DAMPING
from .kinematics import OpenArmKinematics
from .pick_and_place import PickPlaceController
from .planners.collision import CollisionChecker

# --- planning constants ----------------------------------------------------
# Per-joint speed budget (rad/s) used to (a) test whether the arm can reach a
# candidate catch config in time and (b) bound the min-jerk peak velocity.
# Proximal joints are stiff/strong; distal joints (wrist) are soft -> slower.
QD_MAX = np.array([3.5, 3.5, 4.0, 4.0, 6.0, 6.0, 8.0])
MINJERK_PEAK = 1.875            # peak-velocity factor of a rest-to-rest quintic
REACT_MARGIN = 0.05             # s of slack added to every move-time estimate
SETTLE_LEAD = 0.05              # s: arrive at the catch pose this early, then wait
REACH_MIN, REACH_MAX = 0.20, 0.62   # reachable shell radius from the shoulder (m)
GRASP_TOL = 0.03                # m: ball is centred in the jaws -> grab now

# Verified airborne-throw envelope: the ball is launched from a sane airborne
# point and its parabola passes through the arm's reachable volume (the aim box).
# Used by the demo and tests so they exercise the same vetted distribution.
THROW_AIM_LOW = np.array([0.30, -0.35, 0.82])
THROW_AIM_HIGH = np.array([0.46, -0.05, 1.10])
THROW_LAUNCH_LOW = np.array([1.05, -0.55, 0.90])
THROW_LAUNCH_HIGH = np.array([1.60, 0.15, 1.40])


def sample_throw(rng, gravity, tf_range=(0.38, 0.60), aim=None, launch=None):
    """Sample a random but catchable airborne throw -> (launch_pos, launch_vel).

    Forward generation: pick an airborne launch point and an aim point inside the
    reachable volume, solve the launch velocity so the parabola passes through the
    aim, and reject underground/absurd launches or near-vertical ("cup") arrivals.
    ``aim``/``launch`` override the default (right-arm) boxes as (low, high) pairs.
    """
    g = np.asarray(gravity, dtype=float)
    aim_lo, aim_hi = aim if aim is not None else (THROW_AIM_LOW, THROW_AIM_HIGH)
    launch_lo, launch_hi = launch if launch is not None else (THROW_LAUNCH_LOW, THROW_LAUNCH_HIGH)
    while True:
        A = rng.uniform(aim_lo, aim_hi)
        L = rng.uniform(launch_lo, launch_hi)
        Tf = rng.uniform(*tf_range)
        v0 = (A - L - 0.5 * g * Tf ** 2) / Tf
        vc = v0 + g * Tf                          # arrival velocity at the aim
        descent = np.degrees(np.arctan2(-vc[2], np.linalg.norm(vc[:2])))
        if np.linalg.norm(v0) < 6.0 and vc[0] < -1.0 and -10 < descent < 60:
            return L, v0


# Aim/launch boxes per side for bimanual throws (left mirrors right in y).
_THROW_SIDES = {
    "right":  (([0.30, -0.35, 0.82], [0.46, -0.05, 1.10]), ([1.05, -0.55, 0.90], [1.60, 0.15, 1.40])),
    "left":   (([0.30,  0.05, 0.82], [0.46,  0.35, 1.10]), ([1.05, -0.15, 0.90], [1.60, 0.55, 1.40])),
    "center": (([0.32, -0.10, 0.85], [0.44,  0.10, 1.05]), ([1.05, -0.30, 0.95], [1.55, 0.30, 1.35])),
}


def sample_throw_bimanual(rng, gravity, side=None):
    """Random throw toward the left, right, or centre -> (launch, vel, side)."""
    side = side if side is not None else rng.choice(["right", "left", "center"])
    aim, launch = _THROW_SIDES[side]
    L, v0 = sample_throw(rng, gravity,
                         aim=(np.array(aim[0]), np.array(aim[1])),
                         launch=(np.array(launch[0]), np.array(launch[1])))
    return L, v0, side


# --------------------------------------------------------------------- maths
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def look_at_orientation(approach_dir, ref=(0.0, 0.0, 1.0)):
    """Gripper orientation whose approach axis (fingers, local -z) points along
    ``approach_dir`` in world coordinates.

    With ``approach_dir = (0, 0, -1)`` this reduces to a straight top-down grasp
    (matching ``grasp.topdown_orientation(0)``); pointing it at an incoming ball
    makes the ball fly into the open jaws.
    """
    a = _unit(np.asarray(approach_dir, dtype=float))
    zc = -a                                     # local +z in world
    ref = _unit(np.asarray(ref, dtype=float))
    if abs(float(np.dot(zc, ref))) > 0.95:      # degenerate -> pick another ref
        ref = np.array([1.0, 0.0, 0.0])
    yc = _unit(np.cross(zc, ref))               # local +y (closing axis)
    xc = _unit(np.cross(yc, zc))                # local +x
    return np.column_stack([xc, yc, zc])


def roll_about(R, angle):
    """Rotate orientation ``R`` about its own approach (local z) axis."""
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ Rz


def quintic(t, T, q0, qd0, qf, qdf):
    """Vectorised minimum-jerk quintic with position+velocity boundary
    conditions (zero boundary acceleration). Returns (q, qd, qdd) at time ``t``.
    """
    q0 = np.asarray(q0, float); qd0 = np.asarray(qd0, float)
    qf = np.asarray(qf, float); qdf = np.asarray(qdf, float)
    if T <= 1e-6:
        return qf.copy(), qdf.copy(), np.zeros_like(qf)
    t = float(np.clip(t, 0.0, T))
    d = qf - q0
    a0, a1, a2 = q0, qd0, np.zeros_like(q0)
    a3 = (20 * d - (8 * qdf + 12 * qd0) * T) / (2 * T ** 3)
    a4 = (-30 * d + (14 * qdf + 16 * qd0) * T) / (2 * T ** 4)
    a5 = (12 * d - 6 * (qdf + qd0) * T) / (2 * T ** 5)
    q = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    qd = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
    qdd = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
    return q, qd, qdd


# ----------------------------------------------------------- state estimation
class BallisticKalmanFilter:
    """Constant-acceleration Kalman filter for a projectile (gravity known).

    State x = [px, py, pz, vx, vy, vz]. Position is measured each step; gravity
    enters as a known control input, so the filter estimates the ball's velocity
    (and de-noises its position) from position observations alone, then predicts
    the parabola forward analytically.
    """

    def __init__(self, gravity, pos_noise=3e-3, accel_noise=0.5):
        self.g = np.asarray(gravity, dtype=float)
        self.R = (pos_noise ** 2) * np.eye(3)
        self.qa = accel_noise ** 2
        self.x = None
        self.P = None
        self.t = None
        self.n = 0
        self._last = None       # (t, p) for finite-difference velocity seed

    def observe(self, t, p):
        p = np.asarray(p, dtype=float)
        if self.x is None:
            v0 = np.zeros(3)
            if self._last is not None:
                dt = t - self._last[0]
                if dt > 1e-6:
                    v0 = (p - self._last[1]) / dt
                    self._init(t, p, v0)
                    return
            self._last = (t, p)
            return
        self._predict(t - self.t)
        self._update(p)
        self.t = t
        self.n += 1

    def _init(self, t, p, v0):
        self.x = np.concatenate([p, v0])
        self.P = np.diag([1e-3, 1e-3, 1e-3, 0.3, 0.3, 0.3])
        self.t = t
        self.n = 2

    def _predict(self, dt):
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        self.x = F @ self.x + np.concatenate([0.5 * dt**2 * self.g, dt * self.g])
        # Process noise from acceleration uncertainty (white-noise-accel model).
        G = np.concatenate([0.5 * dt**2 * np.ones(3), dt * np.ones(3)])
        self.P = F @ self.P @ F.T + self.qa * np.outer(G, G)

    def _update(self, z):
        H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    @property
    def ready(self):
        return self.x is not None and self.n >= 4

    @property
    def pos(self):
        return self.x[:3].copy()

    @property
    def vel(self):
        return self.x[3:].copy()

    def position_at(self, t_abs):
        """Predicted ball position at absolute time ``t_abs``."""
        tau = t_abs - self.t
        return self.x[:3] + self.x[3:] * tau + 0.5 * self.g * tau**2

    def velocity_at(self, t_abs):
        tau = t_abs - self.t
        return self.x[3:] + self.g * tau


# ------------------------------------------------------------- interception
class CatchPlan:
    """A reachable interception: where/when to catch and the catch config."""

    __slots__ = ("p", "v", "R", "q", "qd", "t_catch")

    def __init__(self, p, v, R, q, qd, t_catch):
        self.p = p; self.v = v; self.R = R
        self.q = q; self.qd = qd; self.t_catch = t_catch


class InterceptionSolver:
    """Finds the earliest reachable, in-time interception on the predicted arc."""

    def __init__(self, kinematics, shoulder, qd_max=QD_MAX,
                 horizon=1.2, dtau=0.03):
        self.king = kinematics
        self.shoulder = np.asarray(shoulder, dtype=float)
        self.qd_max = np.asarray(qd_max, dtype=float)
        self.horizon = horizon
        self.dtau = dtau

    def _in_reach(self, p):
        r = np.linalg.norm(p - self.shoulder)
        return REACH_MIN < r < REACH_MAX and p[2] > 0.45

    def _move_time(self, q_from, q_to):
        """Lower bound on the time to move (also bounds min-jerk peak velocity)."""
        dq = np.abs(q_to - q_from)
        return float(np.max(MINJERK_PEAK * dq / self.qd_max))

    def _orientations(self, v):
        """Candidate catch orientations facing the incoming ball (with fallbacks
        that tilt toward horizontal/forward, where the wrist can actually reach)."""
        approach = -_unit(v)                       # face where the ball comes from
        cands = []
        for blend in (0.0, 0.35, 0.6):             # tilt approach toward forward+up
            a = _unit((1 - blend) * approach + blend * np.array([0.4, 0.0, 0.6]))
            base = look_at_orientation(a)
            for roll in (0.0, np.pi / 2, -np.pi / 2):
                cands.append(roll_about(base, roll))
        return cands

    def _ik_fast(self, p, R, seed, max_iters=120):
        """One warm-started damped-least-squares descent (no extra seeds/restarts)
        -- fast enough to run every MPC replan. Restores qpos on exit."""
        king = self.king
        qsave = king.data.qpos.copy()
        q, err = king._solve_from_seed(seed, p, R, max_iters, IK_TOLERANCE,
                                       rest_weight=0.0, damping=IK_DAMPING)
        king.data.qpos[:] = qsave
        mujoco.mj_kinematics(king.model, king.data)
        return q, err < 5e-3        # 5 mm: a catch tolerance, not a placement one

    def solve(self, kf, t_now, q_now, q_warm=None):
        """Return a ``CatchPlan`` (earliest feasible interception) or ``None``."""
        if not kf.ready:
            return None
        seed = q_warm if q_warm is not None else q_now
        tau = self.dtau
        while tau <= self.horizon:
            t_catch = t_now + tau
            p = kf.position_at(t_catch)
            v = kf.velocity_at(t_catch)
            if self._in_reach(p):
                for R in self._orientations(v):
                    q, ok = self._ik_fast(p, R, seed)
                    if not ok:
                        continue
                    if self._move_time(q_now, q) + REACT_MARGIN <= tau:
                        qd = self._catch_jointvel(q, v)
                        return CatchPlan(p, v, R, q, qd, t_catch)
            tau += self.dtau
        return None

    def _catch_jointvel(self, q, v_ball):
        """Joint velocity that makes the hand move WITH the ball at the catch
        (velocity matching for a soft catch), via the damped Jacobian inverse."""
        J = self.king.jacobian(q)[:3]              # linear part, 3x7
        JT = J.T
        qd = JT @ np.linalg.solve(J @ JT + 1e-2 * np.eye(3), v_ball)
        s = np.linalg.norm(qd)
        cap = float(np.min(self.qd_max))
        return qd * (cap / s) if s > cap else qd


# --------------------------------------------------------------- controller
class CatchController:
    """MPC airborne catcher: estimate -> intercept -> re-plan -> soft catch."""

    def __init__(self, model, data, arm=RIGHT_ARM, ball="ball", balls=None,
                 catch_radius=0.07, replan_every=5, settle_duration=0.5,
                 hold_rise=0.05, perception=None, cam_period=5,
                 velocity_match=True, weld_mode="closest"):
        self.model = model
        self.data = data
        self.arm = arm
        self.ball = ball
        # Ablation switches (used by the benchmark suite).
        self.velocity_match = velocity_match    # match hand vel to the ball at catch
        self.weld_mode = weld_mode              # "closest" = grab at deepest; "entry" = at jaw boundary
        # Balls this gripper might grasp (>1 in the two-ball task); at the catch
        # it welds whichever one is actually between the pads.
        self.balls = list(balls) if balls is not None else [ball]
        self.catch_radius = catch_radius
        self.replan_every = replan_every
        self.settle_duration = settle_duration
        self.hold_rise = hold_rise
        self.dt = model.opt.timestep
        # Perception: if given, the ball is observed THROUGH CAMERAS (the
        # controller never reads its true pose); else ground-truth state is used.
        self.perception = perception
        self.cam_period = cam_period

        self.king = OpenArmKinematics(model, data, joint_names=arm.joints,
                                      site_name=arm.ee_site,
                                      tool_offset=GRASP_LOCAL_OFFSET)
        self.ppc = PickPlaceController(model, data, arm=arm)
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in arm.actuators]
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                          arm.gripper_actuator)
        self.qpos_i = self.king.qpos_indices
        self.dof_i = self.king.dof_indices

        # Body id + free-joint dof for every candidate ball.
        self._ball_info = {}
        for b in dict.fromkeys(self.balls + [ball]):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
            bjid = next(j for j in range(model.njnt)
                        if model.jnt_bodyid[j] == bid
                        and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
            self._ball_info[b] = (bid, int(model.jnt_dofadr[bjid]))
        self.ball_bid, self.ball_dof = self._ball_info[ball]

        # Shoulder = world position of this arm's first link, for the
        # reachable-shell prefilter.
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{arm.name}_link1")
        if b == -1:
            b = self.king.ee_body_id
        mujoco.mj_forward(model, data)
        self.shoulder = data.xpos[b].copy()

        self.solver = InterceptionSolver(self.king, self.shoulder)
        self.reset()

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        self.kf = BallisticKalmanFilter(self.model.opt.gravity)
        self.plan = None
        self.committed = False
        self.caught = False
        self.missed = False
        self.q_ref = self.data.qpos[self.qpos_i].copy()
        self.qd_ref = np.zeros(7)
        self.q_warm = None
        self.k = 0
        self.closing = False
        self.last_d = np.inf
        self.caught_ball = None
        # settle/hold trajectory after the catch
        self.q_catch = None
        self.qd_catch = None
        self.q_hold = None
        self.settle_t = 0.0
        self._open_gripper()

    def ball_pos(self):
        return self.data.xpos[self.ball_bid].copy()

    def grasp_pos(self):
        return self.king.forward_kinematics()[0]

    def grasp_vel(self):
        """Current world velocity of the grasp point (linear)."""
        J = self.king._jacobian_current()[:3]
        return J @ self.data.qvel[self.dof_i]

    # ------------------------------------------------------------- main step
    def observe(self, t):
        """Update the ball estimate (vision at the camera rate, else ground
        truth) and the current ball-position estimate ``self._ball_now``."""
        if self.perception is not None:
            # Camera-driven: observe only at the camera frame rate; the only ball
            # knowledge is the (noisy) vision estimate and the filter's prediction.
            if self.k % self.cam_period == 0:
                est = self.perception.observe()
                if est is not None:
                    self.kf.observe(t, est)
            self._ball_now = self.kf.position_at(t) if self.kf.ready else None
        else:
            self.kf.observe(t, self.ball_pos())
            self._ball_now = self.ball_pos()

    def control(self, t):
        """Track/catch toward the current plan (assumes the estimate is current).
        Separated from observation so a bimanual controller can share one
        estimator and drive only the chosen arm."""
        if self.caught:
            self._settle_step()
        else:
            self._track_step(t)
        self._gravity_comp()
        self.k += 1

    def hold(self):
        """Idle: keep the ready pose with the gripper open (gravity-compensated)."""
        self._command_arm(self.q_ref)
        self._open_gripper()
        self._gravity_comp()

    def _gravity_comp(self):
        """Cancel gravity on this arm's joints so it tracks the reference."""
        self.data.qfrc_applied[self.dof_i] = self.data.qfrc_bias[self.dof_i]

    def step(self):
        """Advance one control step (call once per ``mj_step``)."""
        t = self.data.time
        self.observe(t)
        self.control(t)

    def _track_step(self, t):
        # (Re)plan the interception at the MPC rate, then commit near the catch.
        if not self.committed and (self.plan is None or self.k % self.replan_every == 0):
            q_now = self.data.qpos[self.qpos_i].copy()
            plan = self.solver.solve(self.kf, t, q_now, q_warm=self.q_warm)
            if plan is not None:
                self.plan = plan
                self.q_warm = plan.q
                if plan.t_catch - t < 0.12:        # stop re-routing at the last moment
                    self.committed = True

        if self.plan is not None:
            # Receding-horizon min-jerk: re-fit from the current reference state
            # to q* over the remaining time, then advance one dt along it.
            T = max(self.plan.t_catch - t - SETTLE_LEAD, self.dt)
            qd_end = self.plan.qd if self.velocity_match else np.zeros(7)
            q1, qd1, _ = quintic(self.dt, T, self.q_ref, self.qd_ref,
                                 self.plan.q, qd_end)
            self.q_ref, self.qd_ref = q1, qd1
            self._command_arm(self.q_ref)
            # Two-phase soft catch: start CLOSING the fingers once the ball is in
            # the jaw zone, but WELD at the closest approach (when the ball is
            # centred and deepest in the hand) so the fingers actually clamp it
            # instead of grabbing it at the jaw edge. Uses the ball ESTIMATE (in
            # vision mode this is the Kalman prediction, never the true pose).
            d = np.linalg.norm(self._ball_now - self.grasp_pos())
            if self.closing or d < self.catch_radius:
                self.closing = True
                self._close_gripper()
                # "closest": grab when the ball is deepest in the jaws (centred);
                # "entry" (ablation): grab the instant it enters the jaw zone.
                if self.weld_mode == "entry" or d < GRASP_TOL or d > self.last_d + 1e-4:
                    self._do_catch()
                self.last_d = d
            else:
                self._open_gripper()
        else:
            # No feasible plan yet: hold the ready pose, gripper open.
            self._command_arm(self.q_ref)
            self._open_gripper()
            if self.kf.ready and self.kf.pos[0] < self.shoulder[0] - 0.05:
                self.missed = True              # ball passed the arm, never reachable

    # ----------------------------------------------------------- soft catch
    def _do_catch(self):
        """Velocity-matched soft catch: match the ball's velocity to the hand,
        close the fingers, weld for a firm hold, and plan a settle/hold move."""
        # The planning IK left cdof stale for a candidate config; refresh it for
        # the live state so the grasp-point Jacobian (hand velocity) is correct.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        # Grab whichever candidate ball is actually between the pads.
        gp0 = self.grasp_pos()
        caught = min(self.balls,
                     key=lambda b: np.linalg.norm(self.data.xpos[self._ball_info[b][0]] - gp0))
        self.caught_ball = caught
        _, dof = self._ball_info[caught]
        v_hand = self.grasp_vel()
        self.data.qvel[dof:dof + 3] = v_hand       # kill relative vel
        self.data.qvel[dof + 3:dof + 6] = 0.0      # no spin
        self._close_gripper()
        self.ppc.attach(caught)

        self.q_catch = self.data.qpos[self.qpos_i].copy()
        self.qd_catch = self.data.qvel[self.dof_i].copy()
        gp = self.grasp_pos()
        q_hold, info = self.king.inverse_kinematics(
            gp + np.array([0.0, 0.0, self.hold_rise]), target_mat=self.plan.R,
            q_init=self.q_catch, restarts=0, return_info=True)
        self.q_hold = q_hold if (info["success"]
                                 and np.max(np.abs(q_hold - self.q_catch)) < 1.0) else self.q_catch
        self.settle_t = 0.0
        self.caught = True

    def _settle_step(self):
        """Decelerate the follow-through and hold the ball (gripper closed)."""
        self.settle_t += self.dt
        q, _, _ = quintic(self.settle_t, self.settle_duration,
                          self.q_catch, self.qd_catch, self.q_hold, np.zeros(7))
        self._command_arm(q)
        self._close_gripper()

    # ------------------------------------------------------------- actuators
    def _command_arm(self, q):
        for i, a in enumerate(self.actuator_ids):
            self.data.ctrl[a] = q[i]

    def _open_gripper(self):
        self.data.ctrl[self.grip_act] = self.arm.gripper_open

    def _close_gripper(self):
        self.data.ctrl[self.grip_act] = self.ppc._grip_ctrl(True)


# ----------------------------------------------------------------- bimanual
class BimanualCatchController:
    """Catch with whichever arm reaches the ball best, collision-free.

    One shared ballistic estimator feeds a per-arm interception solver. When a
    feasible interception exists, the arm with the most comfortable, COLLISION-
    FREE catch (largest time margin whose ready->catch path clears the other arm)
    is chosen and driven to the catch; the other arm holds a safe ready pose.
    """

    def __init__(self, model, data, ball="ball", perception=None, cam_period=5):
        self.model = model
        self.data = data
        self.dt = model.opt.timestep
        self.perception = perception
        self.cam_period = cam_period
        # Per-arm catchers (their internal estimators are bypassed: a single
        # shared filter is assigned in reset()).
        self.cr = CatchController(model, data, arm=RIGHT_ARM, ball=ball, perception=None)
        self.cl = CatchController(model, data, arm=LEFT_ARM, ball=ball, perception=None)
        self.arms = {"right": self.cr, "left": self.cl}
        self.checker = {
            "right": CollisionChecker(model, data, self.cr.king, arm_name="right",
                                      avoid_other_arm=True),
            "left": CollisionChecker(model, data, self.cl.king, arm_name="left",
                                     avoid_other_arm=True),
        }
        self.ball_bid = self.cr.ball_bid
        self.reset()

    def reset(self):
        self.cr.reset()
        self.cl.reset()
        self.kf = BallisticKalmanFilter(self.model.opt.gravity)   # ONE shared estimator
        self.cr.kf = self.cl.kf = self.kf
        self.active = None
        self.k = 0

    def ball_pos(self):
        return self.data.xpos[self.ball_bid].copy()

    def _other(self, name):
        return self.cl if name == "right" else self.cr

    def _observe(self, t):
        if self.perception is not None:
            if self.k % self.cam_period == 0:
                est = self.perception.observe()
                if est is not None:
                    self.kf.observe(t, est)
            ball_now = self.kf.position_at(t) if self.kf.ready else None
        else:
            self.kf.observe(t, self.ball_pos())
            ball_now = self.ball_pos()
        self.cr._ball_now = ball_now
        self.cl._ball_now = ball_now

    def _select_arm(self, t):
        """Pick the feasible arm with the largest time margin whose ready->catch
        path is collision-free w.r.t. the other arm (held at its ready pose)."""
        best, best_margin = None, -1e9
        for name, c in self.arms.items():
            q_now = self.data.qpos[c.qpos_i].copy()
            plan = c.solver.solve(self.kf, t, q_now, q_warm=c.q_warm)
            if plan is None:
                continue
            margin = (plan.t_catch - t) - c.solver._move_time(q_now, plan.q)
            if not self.checker[name].edge_clear(c.q_ref, plan.q):
                continue                                  # would hit the other arm
            if margin > best_margin:
                best, best_margin = (name, plan), margin
        if best is not None:
            name, plan = best
            c = self.arms[name]
            c.plan, c.q_warm = plan, plan.q
            if plan.t_catch - t < 0.12:
                c.committed = True
            self.active = name

    def step(self):
        t = self.data.time
        self._observe(t)
        if self.active is None and self.kf.ready:
            self._select_arm(t)
        if self.active is None:
            self.cr.hold()
            self.cl.hold()
        else:
            self.arms[self.active].control(t)             # chosen arm catches
            self._other(self.active).hold()               # other stays safe/open
        self.k += 1

    # -- evaluation helpers ----------------------------------------------------
    @property
    def caught(self):
        return self.active is not None and self.arms[self.active].caught

    @property
    def active_arm(self):
        return self.arms[self.active] if self.active is not None else None

    def arm_separation(self):
        """Min distance between the two grippers' grasp points (collision proxy)."""
        return float(np.linalg.norm(self.cr.grasp_pos() - self.cl.grasp_pos()))


# ------------------------------------------------------- two-ball (multi-object)
class MultiBallTracker:
    """Track several balls from ANONYMOUS 3D detections (multi-object tracking).

    Holds one ballistic Kalman filter per ball and, each frame, associates the
    incoming detections to the existing tracks by nearest predicted position
    (a small assignment problem). Unmatched tracks simply coast on their model.
    """

    def __init__(self, gravity, n=2):
        self.kfs = [BallisticKalmanFilter(gravity) for _ in range(n)]
        self.n = n

    def _pred(self, kf, t):
        if kf.x is None:
            return None
        return kf.position_at(t) if kf.ready else kf.pos

    def update(self, t, detections):
        dets = list(detections)
        if not dets:
            return
        if all(kf.x is None for kf in self.kfs):
            # initialise in a stable order (by y) so track 0 is the -y/right ball
            for i, p in enumerate(sorted(dets, key=lambda p: p[1])[:self.n]):
                self.kfs[i].observe(t, p)
            return
        preds = [self._pred(kf, t) for kf in self.kfs]
        best, best_cost = None, np.inf
        for perm in permutations(range(self.n), len(dets)):
            cost = sum(0.5 if preds[ti] is None else np.linalg.norm(dets[di] - preds[ti])
                       for di, ti in enumerate(perm))
            if cost < best_cost:
                best, best_cost = perm, cost
        for di, ti in enumerate(best):
            self.kfs[ti].observe(t, dets[di])

    def ready(self):
        return all(kf.ready for kf in self.kfs)


class TwoBallCatchController:
    """Catch TWO balls thrown at once — one per arm, simultaneously.

    A ``MultiBallTracker`` keeps a filter per ball from anonymous detections; once
    both tracks are confident, each arm is assigned the track on its side and the
    two single-arm catchers run in parallel (each grabs whichever physical ball
    ends up between its pads). Throws are one-per-side, so the arms stay clear.
    """

    def __init__(self, model, data, balls=("ball", "ball2"), perception=None, cam_period=5):
        self.model = model
        self.data = data
        self.balls = list(balls)
        self.perception = perception
        self.cam_period = cam_period
        self.cr = CatchController(model, data, arm=RIGHT_ARM, ball=balls[0],
                                  balls=balls, perception=None)
        self.cl = CatchController(model, data, arm=LEFT_ARM, ball=balls[0],
                                  balls=balls, perception=None)
        self.ball_bids = {b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                          for b in balls}
        self.reset()

    def reset(self):
        self.cr.reset()
        self.cl.reset()
        self.tracker = MultiBallTracker(self.model.opt.gravity, n=len(self.balls))
        self.assign = None          # {"right": track_idx, "left": track_idx}
        self.k = 0

    def _detections(self, t):
        if self.perception is not None:
            return self.perception.observe() if self.k % self.cam_period == 0 else []
        return [self.data.xpos[b].copy() for b in self.ball_bids.values()]  # ground truth

    def step(self):
        t = self.data.time
        dets = self._detections(t)
        if dets:
            self.tracker.update(t, dets)

        # Once both tracks are confident, give each arm the track on its side.
        if self.assign is None and self.tracker.ready():
            tp = [kf.position_at(t) for kf in self.tracker.kfs]
            order = sorted(range(len(tp)), key=lambda i: tp[i][1])   # -y first
            self.assign = {"right": order[0], "left": order[-1]}

        for name, c in (("right", self.cr), ("left", self.cl)):
            if self.assign is None:
                c.hold()
            else:
                c.kf = self.tracker.kfs[self.assign[name]]           # track for this arm
                c._ball_now = c.kf.position_at(t) if c.kf.ready else None
                c.control(t)
        self.k += 1

    # -- evaluation helpers ----------------------------------------------------
    @property
    def num_caught(self):
        return int(self.cr.caught) + int(self.cl.caught)

    def arm_separation(self):
        return float(np.linalg.norm(self.cr.grasp_pos() - self.cl.grasp_pos()))
