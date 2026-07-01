"""Residual policy learning: SAC learns a small correction on top of an LQR baseline.

Standard "residual policy learning" pattern (Silver et al. 2018,
Johannink et al. 2019): a classical controller does the bulk of the work
and the learned policy adds fine-grained corrections. Same physics + hold as
``OpenArmBalanceEnv``. The tilt composition per step is::

    u_final = clip(u_LQR(state) + delta_SAC(state) * RESIDUAL_MAX_TILT,  ± MAX_TILT)

The reward is a reshaped version of the base env's -- the action term is
switched to a **squared penalty** (``-1.0 * |action|^2``) and the precision
bonus is quadrupled (``+2.0`` at target). Both are calibrated so that zero
residual becomes the strong attractor: a full-scale residual costs about
-300 over an episode -- more than the LQR baseline reward -- so anything
but small corrections is strongly punished, while small residuals that
measurably reduce error at the target are rewarded enough to survive the
squared cost. Without this reshaping SAC's max-entropy exploration and the
untrained critic drift the policy away from the good baseline (see
``docs/IMPLEMENTATION_LOG.md`` for the first training run's degradation
curve).

The 6-D observation, termination conditions, and off-plate handling are
identical to ``OpenArmBalanceEnv``.
"""

import numpy as np

from openarm_control.balance import LQRBalancer
from openarm_control.rl.balance_env import (
    OpenArmBalanceEnv, PLATE_RADIUS_OFF, PRECISION_SIGMA, SUCCESS_TOL_M, SUCCESS_TOL_V,
)


class OpenArmBalanceResidualEnv(OpenArmBalanceEnv):
    """SAC residual on top of an LQR baseline. Same env shape/obs/reward as
    ``OpenArmBalanceEnv``; the ``step()`` action is scaled to a *residual*
    correction (~2 deg) since the LQR is already doing the bulk of the work."""

    # Residual is a small correction: ~2 deg cap. Same-order as the typical
    # LQR command magnitude on this task -- big enough to matter, small enough
    # to keep exploration safe (the baseline keeps the ball on the plate).
    RESIDUAL_MAX_TILT = np.deg2rad(2.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # LQR gain for the linearised ball-on-plate at this timestep. Baseline
        # is u = -K state, same convention as LQRBalancer.
        self._K = LQRBalancer._compute_gain(self.model.opt.timestep)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # LQR baseline command from current state (target-relative).
        (x, y), (vx, vy) = self._bal.ball_state()
        tx, ty = float(self.target[0]), float(self.target[1])
        state = np.array([x - tx, y - ty, vx, vy])
        u_lqr = -self._K @ state                  # [roll, pitch] in rad

        # SAC residual: a small correction on top of the LQR baseline.
        roll_res  = float(action[0]) * self.RESIDUAL_MAX_TILT
        pitch_res = float(action[1]) * self.RESIDUAL_MAX_TILT

        # Composed command, clipped to the physical tilt cap.
        cap = self._bal.MAX_TILT
        roll_cmd  = float(np.clip(u_lqr[0] + roll_res,  -cap, cap))
        pitch_cmd = float(np.clip(u_lqr[1] + pitch_res, -cap, cap))

        # ZOH over the control substeps (matches the base env's tempo).
        for _ in range(self.control_substeps):
            self._bal._apply_tilt_and_step(roll_cmd, pitch_cmd)

        (x, y), (vx, vy) = self._bal.ball_state()
        err   = float(np.hypot(x - tx, y - ty))
        speed = float(np.hypot(vx, vy))
        off   = float(np.hypot(x, y)) > PLATE_RADIUS_OFF

        # Residual regularisation: without a strong pull toward zero, SAC's
        # max-entropy objective pushes actions away from zero and destroys
        # the LQR baseline (empirically observed on the first training run:
        # ep_rew_mean dropped 98 -> 17, success_rate 100% -> 10% as SAC
        # explored non-zero residuals with an untrained critic). Squared
        # penalty makes zero the strong attractor; the coefficient (1.0) is
        # calibrated so a full-scale residual costs -300 over an episode --
        # more than the baseline reward -- guaranteeing the policy stays
        # near the LQR feedback and only ever *adds* helpful corrections.
        reward = (- err
                  - 0.05 * speed
                  - 1.0 * float(np.sum(action ** 2))
                  + 2.0 * float(np.exp(-(err / PRECISION_SIGMA) ** 2)))
        if off:
            reward -= 5.0

        self._step += 1
        terminated = off
        truncated  = self._step >= self.max_steps
        success = (not off) and (err < SUCCESS_TOL_M) and (speed < SUCCESS_TOL_V)
        if (terminated or truncated) and success:
            reward += 2.0

        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, \
            {"distance": err, "is_success": success}
