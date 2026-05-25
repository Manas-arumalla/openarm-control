"""RL environment tests (fast; no policy training)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.rl.reach_env import OpenArmReachEnv
from openarm_control.rl.pick_place_env import OpenArmPickPlaceEnv, ACTION_SCALE


def test_env_passes_gym_checker():
    from stable_baselines3.common.env_checker import check_env
    check_env(OpenArmReachEnv())          # raises if non-compliant


def test_reset_step_contract():
    env = OpenArmReachEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (23,) and np.isfinite(obs).all()
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (23,)
    assert np.isfinite(r)
    assert "distance" in info and "is_success" in info


def test_moving_toward_target_reduces_distance():
    """A Jacobian step toward the target reduces EE distance (env dynamics sane)."""
    env = OpenArmReachEnv()
    env.reset(seed=3)
    d0 = np.linalg.norm(env.target - env._ee())
    for _ in range(60):
        err = env.target - env._ee()
        J = env.kin.jacobian()[:3]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-2 * np.eye(3), err)
        env.step(np.clip(dq / 0.04, -1, 1))
    d1 = np.linalg.norm(env.target - env._ee())
    assert d1 < d0 - 0.02, f"distance did not decrease ({d0:.3f} -> {d1:.3f})"


def test_success_terminates_episode():
    env = OpenArmReachEnv(max_steps=50)
    env.reset(seed=5)
    # place target exactly at current EE -> first step should already be success
    env.target = env._ee().copy()
    env.data.mocap_pos[env.mocap_id] = env.target
    _, r, term, trunc, info = env.step(np.zeros(7, dtype=np.float32))
    assert info["is_success"] and term


# ----------------------------------------------------- pick-and-place -----
def test_pick_env_gym_compliant():
    from stable_baselines3.common.env_checker import check_env
    env = OpenArmPickPlaceEnv()
    check_env(env)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (31,) and env.action_space.shape == (8,)


def test_pick_env_scripted_oracle_solves():
    """A Jacobian oracle can approach, grasp, and place — env mechanics are sound."""
    env = OpenArmPickPlaceEnv()

    def oracle(target_xyz, grip):
        err = target_xyz - env._grasp_pt()
        J = env.king.jacobian()[:3]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-2 * np.eye(3), err)
        return np.concatenate([np.clip(dq / ACTION_SCALE, -1, 1), [grip]])

    env.reset(seed=0)
    info = {}
    for _ in range(60):
        _, _, term, trunc, info = env.step(oracle(env._block(), -1.0))
        if np.linalg.norm(env._grasp_pt() - env._block()) < 0.03:
            break
    for _ in range(15):
        _, _, term, trunc, info = env.step(oracle(env._block(), 1.0))
    for _ in range(150):
        _, _, term, trunc, info = env.step(oracle(env.target + np.array([0, 0, 0.03]), 1.0))
        if term or trunc:
            break
    assert info["grasped"] and info["is_success"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
