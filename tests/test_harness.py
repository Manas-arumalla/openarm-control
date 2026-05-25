"""F3 — learning harness (GPU-aware training + image-observation logging).

Checks the device utility, that BC training/acting runs on the available device
(CPU or GPU) and learns, and that the collector can record camera image
observations alongside the proprioceptive state -- the dataset a vision policy
(ACT/Diffusion, phase I2) trains on. No long GPU training here (that is user-run).
"""
import os
import sys

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.imitation.device import get_device, device_report
from openarm_control.imitation.bc import BCPolicy, train_bc, load_bc
from openarm_control.imitation.collect import collect


def test_device_utility():
    assert get_device() in ("cuda", "cpu")
    rep = device_report()
    assert isinstance(rep, str) and ("CUDA" in rep or "CPU" in rep)


def test_bc_policy_runs_on_device_and_acts():
    """A BCPolicy moved to the available device produces an action of the right
    shape and bounded to [-1, 1] (tanh output)."""
    dev = get_device()
    pol = BCPolicy(obs_dim=23, act_dim=7).to(dev)
    a = pol.act(np.zeros(23, np.float32))
    assert a.shape == (7,)
    assert np.all(np.abs(a) <= 1.0 + 1e-5)


def test_bc_trains_and_lowers_loss(tmp_path):
    """Training on a tiny synthetic dataset lowers the loss and the saved model
    reloads + acts (the GPU/CPU training path end-to-end)."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(400, 6)).astype(np.float32)
    act = np.tanh(obs[:, :3] * 0.5).astype(np.float32)        # a learnable mapping
    npz = tmp_path / "toy.npz"
    np.savez(npz, obs=obs, act=act, ep_lens=np.array([400], np.int64))
    out = train_bc(str(npz), epochs=60, out=str(tmp_path / "toy_bc.pt"), verbose=False)
    model = load_bc(out)
    pred = np.array([model.act(o) for o in obs[:50]])
    mse = float(np.mean((pred - act[:50]) ** 2))
    assert mse < 0.02, f"BC did not fit the toy mapping: MSE {mse:.4f}"


def test_image_observation_collection(tmp_path):
    """The collector records camera RGB frames alongside the state."""
    out = collect("reach", episodes=2, images=True, img_size=84, out=str(tmp_path / "img.npz"))
    with np.load(out) as d:
        assert "images" in d.files
        imgs = d["images"]
        assert imgs.dtype == np.uint8
        assert imgs.ndim == 4 and imgs.shape[1:] == (84, 84, 3)
        assert imgs.shape[0] == d["obs"].shape[0], "one image per transition"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
