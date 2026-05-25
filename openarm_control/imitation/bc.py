"""Behavior cloning: an MLP that imitates the scripted demos (obs -> action).

    python -m openarm_control.imitation.bc --task reach --epochs 150
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import PROJECT_ROOT
from openarm_control.imitation.device import get_device, device_report

DEMO_DIR = os.path.join(PROJECT_ROOT, "demos")


class BCPolicy(nn.Module):
    """obs -> action MLP with tanh-bounded actions (matches the env's [-1,1] box)."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.obs_dim, self.act_dim = obs_dim, act_dim
        # Input normalisation (the obs mixes radians, velocities, positions of
        # very different scales) — set from the dataset; critical for BC.
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def set_norm(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(np.maximum(std, 1e-4), dtype=torch.float32))

    def forward(self, x):
        return self.net((x - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def act(self, obs):
        dev = next(self.parameters()).device           # works whether on CPU or GPU
        x = torch.as_tensor(np.asarray(obs, np.float32), device=dev).reshape(1, -1)
        return self.forward(x).cpu().numpy().reshape(-1)


def train_bc(npz_path, epochs=150, lr=1e-3, batch=256, seed=0, out=None, verbose=True,
             device=None):
    torch.manual_seed(seed)
    device = device or get_device()
    if verbose:
        print(f"  device: {device_report()}")
    d = np.load(npz_path)
    X = torch.as_tensor(d["obs"], dtype=torch.float32).to(device)
    Y = torch.as_tensor(d["act"], dtype=torch.float32).to(device)
    model = BCPolicy(X.shape[1], Y.shape[1])
    model.set_norm(X.mean(0).cpu().numpy(), X.std(0).cpu().numpy())   # normalise inputs
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    first = last = None
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        last = tot / n
        if ep == 0:
            first = last
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  MSE {last:.5f}")
    out = out or os.path.join(DEMO_DIR, os.path.basename(npz_path).replace(".npz", "_bc.pt"))
    torch.save({"state_dict": model.state_dict(),
                "obs_dim": model.obs_dim, "act_dim": model.act_dim}, out)
    print(f"trained BC: MSE {first:.5f} -> {last:.5f}; saved -> {out}")
    return out


def load_bc(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = BCPolicy(ck["obs_dim"], ck["act_dim"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a behavior-cloning policy.")
    ap.add_argument("--task", default="reach")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--demos", default=None, help="path to the .npz dataset")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    npz = a.demos or os.path.join(DEMO_DIR, f"{a.task}.npz")
    train_bc(npz, epochs=a.epochs, lr=a.lr, out=a.out)


if __name__ == "__main__":
    main()
