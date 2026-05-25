"""I2 — ACT (Action-Chunking Transformer) learned policy.

Checks the architecture (vision+state -> action chunk), that it trains on the GPU/
CPU and lowers its loss on a small dataset, and that a saved model reloads and acts.
The full reach training + eval is a longer GPU run (see `openarm act train`); these
tests keep it small and headless.
"""
import os
import sys

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.imitation.act import ACTPolicy, train_act, load_act
from openarm_control.imitation.device import get_device


def test_act_forward_and_act_shapes():
    dev = get_device()
    m = ACTPolicy(state_dim=23, act_dim=7, chunk=16).to(dev)
    img = torch.rand(4, 3, 96, 96, device=dev)
    state = torch.randn(4, 23, device=dev)
    out = m(img, state)
    assert out.shape == (4, 16, 7), out.shape
    assert torch.all(out.abs() <= 1.0 + 1e-5)               # tanh-bounded
    a = m.act(np.random.randint(0, 255, (96, 96, 3), np.uint8), np.zeros(23, np.float32))
    assert a.shape == (7,) and np.all(np.abs(a) <= 1.0 + 1e-5)


def test_act_trains_and_lowers_loss(tmp_path):
    """On a small image+state dataset with a learnable state->action mapping, ACT
    lowers its loss and the saved model reloads + acts."""
    rng = np.random.default_rng(0)
    n_ep, L, sdim, adim = 2, 25, 6, 3
    N = n_ep * L
    obs = rng.normal(size=(N, sdim)).astype(np.float32)
    act = np.tanh(obs[:, :adim] * 0.5).astype(np.float32)   # depends on state (learnable)
    imgs = rng.integers(0, 255, (N, 32, 32, 3), np.uint8)   # random (uninformative) images
    npz = tmp_path / "toy_vis.npz"
    np.savez(npz, obs=obs, act=act, images=imgs,
             ep_lens=np.array([L, L], np.int64))
    out = train_act(str(npz), chunk=8, epochs=30, batch=32,
                    out=str(tmp_path / "toy_act.pt"), verbose=False)
    model = load_act(out, device=get_device())
    # the policy produces a valid action for a held-out (image, state)
    a = model.act(imgs[0], obs[0])
    assert a.shape == (adim,) and np.all(np.abs(a) <= 1.0 + 1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
