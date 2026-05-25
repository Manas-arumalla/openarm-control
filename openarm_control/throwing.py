"""Throw a held ball into a bin: compute the release, swing, and let go.

The skill is the inverse of catching. Given the bin position (the target landing
point), it:

1. picks a **release point** in the workspace, toward the target and high;
2. inverts the **projectile equations** for the release velocity that lands the
   ball in the bin, choosing the flight time that minimises the launch speed;
3. solves IK for a **release configuration** and the joint velocity that produces
   that end-effector velocity (``qd = J⁺ v``) — and **refuses the throw** if it
   exceeds the joint speed limits (the bin is out of the throw envelope);
4. builds a wind-up → accelerating **swing** (a quintic ending at the release
   config with the release velocity) → follow-through, and **detaches the ball at
   the release instant** so it flies off with the arm's velocity.

Reuses the ballistic / minimum-jerk helpers from the catching module.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .config import RIGHT_ARM
from .grasp import GraspSolver, topdown_orientation
from .kinematics import OpenArmKinematics
from .catching import quintic, _unit, look_at_orientation

QD_MAX = np.array([4.5, 4.5, 5.0, 5.0, 8.0, 8.0, 10.0])    # per-joint speed budget (rad/s)
TABLE_TOP_Z = 0.40

# Bin locations (x, y on the floor) inside the precise-throw envelope (forward
# reach x in [0.56, 0.67], lateral y in [-0.32, +0.12]); used by the throwing
# benchmark to evaluate precision/consistency over a spread of targets. Bins
# beyond this envelope are correctly refused by ``plan_release`` (out of reach).
BENCH_BINS = [
    (0.56, -0.22), (0.60, -0.05), (0.62, -0.28), (0.64, 0.05),
    (0.66, -0.15), (0.58, 0.12), (0.67, -0.10),
]


class ThrowController:
    def __init__(self, model, data, arm=RIGHT_ARM, ball="ball_orange"):
        self.model, self.data, self.arm = model, data, arm
        self.gs = GraspSolver(model, data, arm=arm)            # grasp-point kinematics
        self.king = self.gs.king
        self.g = np.array(model.opt.gravity, dtype=float)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ball)
        self.ball_bid = bid
        j = next(jj for jj in range(model.njnt)
                 if model.jnt_bodyid[jj] == bid and model.jnt_type[jj] == mujoco.mjtJoint.mjJNT_FREE)
        self.ball_qadr = int(model.jnt_qposadr[j])
        self.ball_dofadr = int(model.jnt_dofadr[j])
        self.weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                      arm.weld(ball.split("_")[-1]))
        self.arm_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                         for n in arm.actuators]
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, arm.gripper_actuator)
        self.ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_body)
        self.dt = float(model.opt.timestep)
        self.plan_ = None
        self.released = False

    # ------------------------------------------------------------- planning
    def _release_point(self, target):
        sh = self.data.xpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{self.arm.name}_link1")]
        d = target[:2] - sh[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        return np.array([sh[0] + 0.42 * d[0], sh[1] + 0.42 * d[1], 0.82])

    def _release_velocity(self, p, target):
        """Velocity to land at ``target`` from ``p`` — the min-speed solution over
        flight time (a 1-parameter family)."""
        best = None
        for t in np.linspace(0.30, 1.1, 60):
            v = (target - p - 0.5 * self.g * t * t) / t
            s = np.linalg.norm(v)
            if v[2] > -0.2 and (best is None or s < best[0]):    # prefer an arc, min speed
                best = (s, v, t)
        return (best[1], best[2]) if best else (None, None)

    T_SWING = 0.7                                            # swing duration (gentle ramp)

    def _shape(self, q_rel, qd_rel, speed):
        """Size a rest-to-rest quintic ``q_wind -> q_rel -> q_ft`` so its peak
        velocity (at the midpoint, where we release) equals ``speed * qd_rel``.

        A rest-to-rest quintic of amplitude A over T peaks at ``1.875*A/T`` at the
        midpoint; with the release at the midpoint and a symmetric swing, the
        half-amplitude that yields a midpoint speed ``speed*qd_rel`` is
        ``speed*qd_rel*T/3.75``. Sizing the swing from the *desired* velocity (not a
        fixed sweep) keeps it off the joint limits, so ``speed`` is a genuine,
        continuous range knob (the old fixed ±0.30 sweep saturated and overshot)."""
        half = speed * qd_rel * (self.T_SWING / 3.75)
        lo, hi = self.king.jnt_low, self.king.jnt_high
        q_wind = np.clip(q_rel - half, lo, hi)
        q_ft = np.clip(q_rel + half, lo, hi)
        return q_wind, q_ft

    def plan(self, target, speed=1.0):
        """Plan a throw to ``target`` (3D landing point). Returns a plan dict, or
        ``None`` with ``self.reason`` if the bin is out of the throw envelope."""
        self.reason = ""
        p = self._release_point(target)
        v, t = self._release_velocity(p, target)
        if v is None:
            self.reason = "no ballistic solution"
            return None
        # Position-only IK (the ball is a sphere -> orientation free -> a natural,
        # well-conditioned config) gives the release config; qd_rel is the EE-velocity
        # joint solution that the swing peaks at (J⁺ v), scaled by ``speed``.
        q_rel, info = self.king.inverse_kinematics(p, None, return_info=True, restarts=12)
        if not info["success"]:
            self.reason = "release pose unreachable"
            return None
        J = self.king.jacobian(q_rel)[:3]
        qd_rel = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), v)
        if np.any(np.abs(speed * qd_rel) > 1.5 * QD_MAX):    # obviously out of range
            self.reason = f"throw too fast for the arm (need {np.max(np.abs(qd_rel)/QD_MAX):.1f}x limit)"
            return None
        q_wind, q_ft = self._shape(q_rel, qd_rel, speed)
        self.plan_ = dict(p=p, v=v, t=t, q_rel=q_rel, qd_rel=qd_rel, speed=speed,
                          q_wind=q_wind, q_ft=q_ft, T_swing=self.T_SWING, target=target)
        return self.plan_

    def _ballistic_landing(self, pos, vel, z_land=0.05):
        """Where a projectile from (pos, vel) crosses height z_land (the descending
        root), in the xy-plane -- or None if it never gets there."""
        a, b, c = 0.5 * self.g[2], vel[2], pos[2] - z_land
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        s = np.sqrt(disc)
        roots = sorted(r for r in ((-b + s) / (2 * a), (-b - s) / (2 * a)) if r > 1e-3)
        if not roots:
            return None
        tt = roots[-1]                                       # latest crossing (descending)
        return pos[:2] + vel[:2] * tt

    # ----------------------------------------------------------- execution
    def _q(self):
        return self.data.qpos[self.king.qpos_indices].copy()

    def _drive(self, q_cmd, grip_closed=True, gravity_comp=True):
        self.data.ctrl[self.arm_acts] = q_cmd
        if self.grip_act >= 0:
            self.data.ctrl[self.grip_act] = (self.arm.gripper_closed if grip_closed
                                             else self.arm.gripper_open)
        if gravity_comp:
            self.data.qfrc_applied[self.king.dof_indices] = \
                self.data.qfrc_bias[self.king.dof_indices]

    def _move_to(self, q_goal, T, grip_closed, viewer=None):
        q0 = self._q()
        n = max(1, int(T / self.dt))
        for k in range(n):
            q, _, _ = quintic(k * self.dt, T, q0, np.zeros(7), q_goal, np.zeros(7))
            self._drive(q, grip_closed)
            mujoco.mj_step(self.model, self.data)
            if viewer is not None and not viewer.is_running():
                return False
            if viewer is not None:
                viewer.sync()
        return True

    def _attach(self):
        p1, q1 = self.data.xpos[self.ee_bid].copy(), self.data.xquat[self.ee_bid].copy()
        p2, q2 = self.data.xpos[self.ball_bid].copy(), self.data.xquat[self.ball_bid].copy()
        nq = np.zeros(4); mujoco.mju_negQuat(nq, q1)
        rp = np.zeros(3); mujoco.mju_rotVecQuat(rp, p2 - p1, nq)
        rq = np.zeros(4); mujoco.mju_mulQuat(rq, nq, q2)
        self.model.eq_data[self.weld, 0:3] = 0.0
        self.model.eq_data[self.weld, 3:6] = rp
        self.model.eq_data[self.weld, 6:10] = rq
        self.model.eq_data[self.weld, 10] = 1.0
        self.model.eq_solref[self.weld] = [0.004, 1.0]       # stiff weld: ball tracks rigidly
        self.data.eq_active[self.weld] = 1

    def grasp_ball(self, viewer=None, disable_table=True):
        """Top-down grasp the ball from the table and lift it (so we start holding it).

        ``disable_table`` turns off the table's collision after the lift (a swing
        fail-safe). Set it False when other balls still rest on the table (a
        multi-throw scene) -- the swing clears the table on its own."""
        bp = self.data.xpos[self.ball_bid].copy()
        gq, info = self.gs.solve(np.array([bp[0], bp[1], bp[2]]), return_info=True)
        if not info["success"]:
            return False
        R = topdown_orientation(info["yaw"])
        above = self.king.inverse_kinematics(bp + [0, 0, 0.14], R, q_init=gq, restarts=8)
        self._move_to(above, 1.5, grip_closed=False, viewer=viewer)     # over the ball
        self._move_to(gq, 1.0, grip_closed=False, viewer=viewer)        # descend
        for _ in range(120):                                            # close + settle
            self._drive(gq, grip_closed=True); mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()
        self._attach()
        self._move_to(above, 1.0, grip_closed=True, viewer=viewer)      # lift
        # Fail-safe: the table only held the ball for the grasp; once the ball is
        # lifted, disable the table's collision so it can't obstruct the swing or
        # deflect the ball on release. (Skipped in multi-throw scenes, where other
        # balls still need the table to hold them up.)
        tg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        if disable_table and tg >= 0:
            self.model.geom_contype[tg] = 0
            self.model.geom_conaffinity[tg] = 0
            mujoco.mj_forward(self.model, self.data)
        return True

    WINDUP_T = 0.8                                           # settle time before the swing

    def _swing_q(self, pl, k):
        """Commanded arm config at swing step k (rest-to-rest quintic wind->ft)."""
        q, _, _ = quintic(k * self.dt, pl["T_swing"], pl["q_wind"], np.zeros(7),
                          pl["q_ft"], np.zeros(7))
        return q

    def _swing_release_fly(self, pl, release_step, viewer=None, fly_steps=700):
        """Sweep the swing from the current (wound-up) config, **release (detach the
        weld) at ``release_step``** with the gripper OPEN so the ball flies off with
        exactly the arm's velocity, then let it fly. Returns the landing position.
        Assumes the arm is already at ``q_wind``."""
        self.released = False
        n = max(1, int(pl["T_swing"] / self.dt))
        q_hold = self._swing_q(pl, release_step)             # config to freeze at on release
        for k in range(n):
            # Up to release: follow the swing. After release: HOLD the release config
            # so the gripper stops and the ball flies free (no chasing/knocking it).
            qc = self._swing_q(pl, k) if k <= release_step else q_hold
            self._drive(qc, grip_closed=False)
            mujoco.mj_step(self.model, self.data)
            if k == release_step and not self.released:
                self.release_vel = self.data.qvel[self.ball_dofadr:self.ball_dofadr + 3].copy()
                self.data.eq_active[self.weld] = 0           # let go at the optimal instant
                self.released = True
            if viewer is not None:
                if not viewer.is_running():
                    return self.data.xpos[self.ball_bid].copy()
                viewer.sync()
        for _ in range(fly_steps):                           # let it fly + settle
            self._drive(q_hold, grip_closed=False)
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            elif (self.data.xpos[self.ball_bid][2] < 0.16 and
                  np.linalg.norm(self.data.qvel[self.ball_dofadr:self.ball_dofadr + 3]) < 0.25):
                break
        return self.data.xpos[self.ball_bid].copy()

    def _run_throw(self, pl, release_step, viewer=None, fly_steps=700):
        """Wind up, then sweep + release + fly. Returns the landing position."""
        self._move_to(pl["q_wind"], self.WINDUP_T, grip_closed=False, viewer=viewer)
        return self._swing_release_fly(pl, release_step, viewer=viewer, fly_steps=fly_steps)

    # ------------------------------ optimal-release search (sim forward model)
    def _save(self):
        return (self.data.qpos.copy(), self.data.qvel.copy(), self.data.eq_active.copy(),
                self.data.qfrc_applied.copy(), self.data.ctrl.copy(), float(self.data.time))

    def _restore(self, s):
        (self.data.qpos[:], self.data.qvel[:], self.data.eq_active[:],
         self.data.qfrc_applied[:], self.data.ctrl[:]) = s[0], s[1], s[2], s[3], s[4]
        self.data.time = s[5]
        mujoco.mj_forward(self.model, self.data)

    def _true_landing(self, pl, release_step, from_state, fly_steps=900):
        """The **actual** simulated landing if we release at ``release_step`` —
        restore the wound-up state, run the real swing+release+fly. The forward
        model itself (no analytic shortcut), so it has no prediction bias."""
        self._restore(from_state)
        land = self._swing_release_fly(pl, release_step, viewer=None, fly_steps=fly_steps)
        return land.copy()

    def _search_release(self, target):
        """For the *current* plan/speed, find the release step whose **true
        simulated landing** is closest to the bin (coarse-to-fine around the swing
        midpoint, where the swing peaks at the ballistic velocity by design).
        Returns (err, step, landing) or None."""
        pl = self.plan_
        n = max(1, int(pl["T_swing"] / self.dt))
        s0 = self._save()
        self._move_to(pl["q_wind"], self.WINDUP_T, grip_closed=False)
        s_wind = self._save()

        def search(steps):
            best = None
            for k in steps:
                land = self._true_landing(pl, k, s_wind)
                e = float(np.linalg.norm(land[:2] - target[:2]))
                if best is None or e < best[0]:
                    best = (e, k, land)
            return best
        mid = n // 2
        coarse = search(range(max(0, mid - 70), min(n, mid + 16), 4))
        fine = search(range(max(0, coarse[1] - 4), min(n, coarse[1] + 4)))
        best = min(coarse, fine, key=lambda b: b[0])
        self._restore(s0)
        return best

    def plan_release(self, target, gate=0.06):
        """Plan the swing, then find the **optimal release** by simulation-in-the-
        loop. The analytic ballistic prediction (welded-ball velocity) under-reads
        the true release velocity, so we don't trust it: we evaluate the **true
        simulated landing** over the swing and pick the release step that actually
        lands closest to the bin. If the nominal swing can't get within ``gate``,
        scale the swing speed (a continuous range knob) and retry. Refuse if no
        speed lands within ``gate`` (the target is outside the throw envelope)."""
        target = np.asarray(target, float)
        best, best_speed = None, 1.0
        # Nominal speed first (the release-step sweep alone usually nails it); only
        # escalate to other swing speeds for bins near the envelope edge.
        for sp in (1.0, 1.15, 0.85, 1.3, 0.7):
            if self.plan(target, speed=sp) is None:
                continue
            b = self._search_release(target)
            if b is not None and (best is None or b[0] < best[0]):
                best, best_speed = b, sp
            if best is not None and best[0] <= gate:
                break
        if best is None:
            self.reason = self.reason or "no reachable release pose"
            return None
        if best[0] > gate:
            self.reason = f"out of throw envelope (best release misses by {best[0]*1000:.0f} mm)"
            return None
        self.plan(target, speed=best_speed)               # restore the winning plan
        self.pred_err, self.release_step, self.pred_landing = best[0], best[1], best[2]
        return self.plan_

    def execute(self, viewer=None):
        """Execute the planned throw for real. Returns the ball's landing position."""
        return self._run_throw(self.plan_, self.release_step, viewer=viewer)

    def throw(self, target, viewer=None):
        """Plan (with optimal-release search) and execute a throw. Returns
        (ok, landing) or (False, reason)."""
        if self.plan_release(target) is None:
            return False, self.reason
        return True, self.execute(viewer=viewer)
