"""ACT -- Action-Chunking Transformer policy (vision + state) for the OpenArm.

A self-contained, GPU-trained learned policy (extension phase I2): the project's
first non-baseline learned result, beyond the MLP behavior-cloning baseline.

Architecture (the deterministic core of ACT, sans the CVAE latent):
  * a small CNN encodes the camera image into a sequence of visual tokens,
  * the proprioceptive state is encoded into one token,
  * a Transformer ENCODER fuses the tokens, and a Transformer DECODER reads K
    learned position queries to predict a **chunk of K future actions** at once.
Action chunking gives smooth, temporally-consistent motion (vs. one-step BC). No
torchvision dependency -- a custom conv stack -- so it trains on CUDA out of the box.

    python -m openarm_control.imitation.act train --demos demos/reach_vis.npz
    python -m openarm_control.imitation.act eval  --model demos/reach_act.pt --task reach
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


class _ImageEncoder(nn.Module):
    """Small conv stack: (B,3,H,W) in [0,1] -> a sequence of (B, h*w, d) visual tokens."""

    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),       # 96 -> 48
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),      # 48 -> 24
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),     # 24 -> 12
            nn.Conv2d(128, d_model, 4, 2, 1), nn.ReLU(),  # 12 -> 6
            nn.AdaptiveAvgPool2d(4),                    # -> 4x4 = 16 tokens
        )

    def forward(self, img):
        f = self.conv(img)                              # (B, d, 4, 4)
        return f.flatten(2).transpose(1, 2)             # (B, 16, d)


class ACTPolicy(nn.Module):
    def __init__(self, state_dim, act_dim, chunk=16, d_model=128, nhead=4, layers=2):
        super().__init__()
        self.state_dim, self.act_dim, self.chunk = state_dim, act_dim, chunk
        self.img_enc = _ImageEncoder(d_model)
        self.state_enc = nn.Linear(state_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, layers)
        dec = nn.TransformerDecoderLayer(d_model, nhead, d_model * 4, batch_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(dec, layers)
        self.query = nn.Parameter(torch.randn(chunk, d_model) * 0.02)
        self.head = nn.Linear(d_model, act_dim)
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    def set_norm(self, mean, std):
        self.state_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(np.maximum(std, 1e-4), dtype=torch.float32))

    def forward(self, img, state):
        """img (B,3,H,W) in [0,1], state (B, state_dim) -> action chunk (B, K, act_dim)."""
        vis = self.img_enc(img)                                       # (B, 16, d)
        st = self.state_enc((state - self.state_mean) / self.state_std).unsqueeze(1)
        memory = self.encoder(torch.cat([vis, st], dim=1))           # (B, 17, d)
        q = self.query.unsqueeze(0).expand(state.shape[0], -1, -1)   # (B, K, d)
        dec = self.decoder(q, memory)                                # (B, K, d)
        return torch.tanh(self.head(dec))                            # (B, K, act_dim)

    @torch.no_grad()
    def act(self, img, state):
        """Predict the chunk for the current (img, state); return the first action."""
        dev = next(self.parameters()).device
        im = torch.as_tensor(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0,
                             device=dev).unsqueeze(0)
        s = torch.as_tensor(np.asarray(state, np.float32), device=dev).reshape(1, -1)
        return self.forward(im, s)[0, 0].cpu().numpy()


def _build_chunks(d, chunk):
    """From an image+state demo npz, build (image, state, action-chunk) samples,
    respecting episode boundaries (the chunk is padded with the last action at the
    end of an episode)."""
    obs, act, imgs, ep_lens = d["obs"], d["act"], d["images"], d["ep_lens"]
    X_img, X_state, Y = [], [], []
    start = 0
    for L in ep_lens:
        end = start + int(L)
        for t in range(start, end):
            ch = act[t:min(t + chunk, end)]
            if len(ch) < chunk:
                ch = np.vstack([ch, np.repeat(ch[-1:], chunk - len(ch), axis=0)])
            X_img.append(imgs[t]); X_state.append(obs[t]); Y.append(ch)
        start = end
    return (np.asarray(X_img, np.uint8), np.asarray(X_state, np.float32),
            np.asarray(Y, np.float32))


def train_act(npz_path, chunk=16, epochs=60, lr=1e-4, batch=64, seed=0, out=None,
              device=None, verbose=True):
    torch.manual_seed(seed)
    device = device or get_device()
    d = np.load(npz_path)
    if "images" not in d.files:
        raise ValueError("ACT needs image observations -- collect with `--images`")
    Ximg, Xs, Y = _build_chunks(d, chunk)
    if verbose:
        print(f"  device: {device_report()}")
        print(f"  {len(Ximg)} samples, chunk={chunk}, image {Ximg.shape[1:]}")
    model = ACTPolicy(Xs.shape[1], Y.shape[2], chunk=chunk).to(device)
    model.set_norm(Xs.mean(0), Xs.std(0))
    # images stay on CPU (uint8) and move per batch; state/targets fit on the GPU
    img_cpu = torch.as_tensor(Ximg).permute(0, 3, 1, 2).float().div_(255.0)
    S = torch.as_tensor(Xs, device=device)
    T = torch.as_tensor(Y, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    n = len(Ximg)
    rng = np.random.default_rng(seed)
    first = last = None
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            im = img_cpu[idx].to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(im, S[idx]), T[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        last = tot / n
        first = last if ep == 0 else first
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  L1 {last:.5f}")
    out = out or os.path.join(DEMO_DIR, os.path.basename(npz_path).replace(".npz", "_act.pt"))
    torch.save({"state_dict": model.state_dict(), "state_dim": model.state_dim,
                "act_dim": model.act_dim, "chunk": model.chunk}, out)
    print(f"trained ACT: L1 {first:.5f} -> {last:.5f}; saved -> {out}")
    return out


def load_act(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ACTPolicy(ck["state_dim"], ck["act_dim"], chunk=ck["chunk"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model


def evaluate(model, task="reach", episodes=12, seed=100, camera=None, img_size=96):
    """Run the ACT policy in the env (rendering the camera each step) and report the
    success rate."""
    import mujoco
    from openarm_control.imitation.expert import make_env_and_expert
    env, _ = make_env_and_expert(task, seed=seed)
    renderer = mujoco.Renderer(env.model, img_size, img_size)
    camera = camera or mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
    n_success = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, info = False, {}
        while not done:
            renderer.update_scene(env.data, camera=camera)
            a = model.act(renderer.render(), obs)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        n_success += int(info.get("is_success", False))
    renderer.close()
    env.close()
    rate = n_success / episodes
    print(f"ACT eval on '{task}': {n_success}/{episodes} = {rate:.0%}")
    return rate


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACT action-chunking policy: train / eval.")
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--demos", default=os.path.join(DEMO_DIR, "reach_vis.npz"))
    ap.add_argument("--task", default="reach")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--model", default=os.path.join(DEMO_DIR, "reach_act.pt"))
    ap.add_argument("--episodes", type=int, default=12)
    a = ap.parse_args(argv)
    if a.mode == "train":
        train_act(a.demos, chunk=a.chunk, epochs=a.epochs, out=a.model)
    else:
        evaluate(load_act(a.model, device=get_device()), task=a.task, episodes=a.episodes)


if __name__ == "__main__":
    main()
