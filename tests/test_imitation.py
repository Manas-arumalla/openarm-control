"""Imitation-learning tests (scripted expert -> demos -> behavior cloning).

Needs the [rl] extra (torch, gymnasium). Mirrors the RL tests' dependencies.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytest.importorskip("torch")

from openarm_control.imitation.expert import make_env_and_expert
from openarm_control.imitation.collect import collect
from openarm_control.imitation.bc import BCPolicy, train_bc, load_bc
from openarm_control.imitation.eval import evaluate


def test_scripted_expert_is_reliable():
    """The expert that generates demonstrations actually reaches the target."""
    env, expert = make_env_and_expert("reach", seed=0)
    succ = 0
    for ep in range(8):
        obs, _ = env.reset(seed=ep)
        expert.reset()
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(expert.act(obs))
            done = term or trunc
        succ += int(info["is_success"])
    env.close()
    assert succ >= 6


def test_bc_policy_shapes_and_normalisation():
    p = BCPolicy(23, 7)
    p.set_norm(np.full(23, 2.0), np.full(23, 3.0))
    a = p.act(np.zeros(23, np.float32))
    assert a.shape == (7,) and np.all(np.abs(a) <= 1.0 + 1e-6)
    assert float(p.obs_std[0]) == 3.0


def test_bc_learns_and_reaches(tmp_path):
    """End-to-end: collect demos, behavior-clone, and reach with decent success."""
    npz = collect("reach", episodes=40, seed=0, out=str(tmp_path / "reach.npz"))
    model = train_bc(npz, epochs=120, out=str(tmp_path / "bc.pt"), verbose=False)
    # round-trips through disk (normalisation buffers included)
    assert load_bc(model).obs_dim == 23
    rate, _ = evaluate(model, "reach", episodes=12, seed=500)
    assert rate >= 0.4, f"BC reach success too low: {rate}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
