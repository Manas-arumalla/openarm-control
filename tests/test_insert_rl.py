"""S1 — RL insertion suite (peg-in-hole with domain randomization).

The env holds a peg and each episode randomly repositions the socket, offsets the
peg start, and randomizes friction + peg radius (one robot, many holes -- the
domain randomization for a precise, robust insertion policy). These check the env
is well-formed (Gymnasium-compliant, deterministic) and that the scripted insertion
expert -- the classical baseline and BC/ACT demo source -- reliably inserts. The
SAC/BC training runs themselves are longer GPU jobs (user-run).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.rl.insert_env import OpenArmInsertEnv
from openarm_control.imitation.expert import make_env_and_expert


def test_insert_env_is_gym_compliant():
    from gymnasium.utils.env_checker import check_env
    env = OpenArmInsertEnv()
    check_env(env, skip_render_check=True)         # spaces, reset/step, determinism
    assert env.observation_space.shape == (23,)
    assert env.action_space.shape == (7,)
    env.close()


def test_socket_randomizes_across_episodes():
    env = OpenArmInsertEnv(seed=0)
    xys = []
    for ep in range(5):
        env.reset(seed=ep)
        xys.append(env.socket_xy.copy())
    env.close()
    xys = np.array(xys)
    assert xys.std(0).max() > 0.01, "socket position not randomized across episodes"


def test_scripted_insertion_inserts():
    """The classical scripted expert inserts reliably across randomized sockets."""
    env, expert = make_env_and_expert("insert", seed=0)
    n_success = 0
    for ep in range(8):
        obs, _ = env.reset(seed=ep)
        expert.reset()
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(expert.act(obs))
            done = term or trunc
        n_success += int(info.get("is_success", False))
    env.close()
    assert n_success >= 6, f"scripted insertion only {n_success}/8"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
