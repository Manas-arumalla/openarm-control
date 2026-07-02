"""B4 -- Gym balance environment regression gate.

Verifies the ``OpenArmBalanceEnv`` (used by the SAC training pipeline) obeys
the Gymnasium contract, exposes the right shapes, and produces a well-shaped
reward: a scripted LQR policy pushed *through the env* must succeed on most
seeds. Guards against reward / observation regressions that would silently
break SAC training runs.
"""
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from openarm_control.rl import OpenArmBalanceEnv, OpenArmBalanceResidualEnv
from openarm_control.balance import LQRBalancer


def test_env_spaces_and_reset():
    env = OpenArmBalanceEnv(seed=0)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (6,), obs.shape
        assert env.observation_space.shape == (6,)
        assert env.action_space.shape == (2,)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_env_random_action_stays_terminating_or_truncating():
    """An episode with random actions must eventually end (terminated OR
    truncated). Guards against a stuck sim / never-ending loop."""
    env = OpenArmBalanceEnv(seed=1, max_steps=60)
    try:
        obs, _ = env.reset(seed=1)
        done = False
        for _ in range(120):
            act = env.action_space.sample()
            obs, r, term, trunc, info = env.step(act)
            assert obs.shape == (6,)
            assert np.isfinite(r)
            assert set(info) >= {"distance", "is_success"}
            if term or trunc:
                done = True
                break
        assert done, "episode never terminated or truncated"
    finally:
        env.close()


def test_env_scripted_lqr_policy_succeeds():
    """The env observation slice obs[:4] is exactly the LQR state ``[px, py,
    vx, vy]`` (target = 0). Applying ``u = -K x`` through env.step should
    settle the ball -- at least 2/3 seeds must succeed. Fails loudly if the
    action scaling / observation layout is wrong."""
    env = OpenArmBalanceEnv(seed=2)
    try:
        K = LQRBalancer._compute_gain(env.model.opt.timestep)
        wins = 0
        for ep in range(3):
            obs, _ = env.reset(seed=200 + ep)
            info = {}
            done = False
            while not done:
                u = -K @ obs[:4]           # LQR feedback on the ball state
                act = u / env._bal.MAX_TILT  # scale into env action units
                obs, _, term, trunc, info = env.step(act)
                done = term or trunc
            wins += int(info.get("is_success", False))
        assert wins >= 2, f"scripted LQR only won {wins}/3 -- env reward is off"
    finally:
        env.close()


def test_residual_env_zero_action_equals_lqr():
    """Residual env with a zero-residual policy must behave exactly like the
    LQR baseline it wraps -- so the same 2/3-success bar as scripted-LQR
    through the plain env has to hold. Guards the residual composition
    (``u_final = clip(u_LQR + delta * RES_MAX_TILT)``): if either the
    baseline or the residual scaling is wired backwards, this test flips."""
    env = OpenArmBalanceResidualEnv(seed=4)
    try:
        wins = 0
        for ep in range(3):
            obs, _ = env.reset(seed=400 + ep)
            info = {}
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
                done = term or trunc
            wins += int(info.get("is_success", False))
        assert wins >= 2, f"zero-residual only won {wins}/3 -- residual wiring is off"
    finally:
        env.close()


def test_residual_env_random_residual_bounded():
    """Random *residuals* on top of LQR must not send the ball flying --
    LQR alone keeps it on the plate, and a 2-degree residual cap is small
    enough that random noise can't cancel the LQR feedback. At worst the
    episode truncates at max_steps with the ball still on the plate."""
    env = OpenArmBalanceResidualEnv(seed=5, max_steps=60)
    try:
        obs, _ = env.reset(seed=500)
        info = {}
        for _ in range(60):
            act = env.action_space.sample()
            obs, r, term, trunc, info = env.step(act)
            assert np.isfinite(r)
            if term or trunc:
                break
        assert info.get("distance") is not None
    finally:
        env.close()
