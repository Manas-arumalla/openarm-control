"""Collect scripted demonstrations into a simple .npz dataset.

    python -m openarm_control.imitation.collect --task reach --episodes 60

Saves obs/action arrays + episode lengths to demos/<task>.npz. A clean, zero-
dependency format; a LeRobot/HF exporter can be layered on later.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import PROJECT_ROOT
from openarm_control.imitation.expert import make_env_and_expert

DEMO_DIR = os.path.join(PROJECT_ROOT, "demos")


def collect(task="reach", episodes=60, seed=0, only_success=True, out=None,
            images=False, camera=None, img_size=96):
    """Collect scripted demos. With ``images=True`` also records a camera RGB frame
    per step (downscaled to ``img_size``) -- the observation a vision policy
    (ACT/Diffusion in I2) trains on, alongside the proprioceptive state."""
    env, expert = make_env_and_expert(task, seed=seed)
    renderer = None
    if images:
        import mujoco
        renderer = mujoco.Renderer(env.model, img_size, img_size)
        camera = camera or mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
    obs_all, act_all, img_all, ep_lens = [], [], [], []
    n_success = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        ep_obs, ep_act, ep_img, done, info = [], [], [], False, {}
        while not done:
            a = expert.act(obs)
            ep_obs.append(obs)
            ep_act.append(a)
            if renderer is not None:
                renderer.update_scene(env.data, camera=camera)
                ep_img.append(renderer.render().copy())
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        if (not only_success) or info.get("is_success"):
            obs_all.extend(ep_obs)
            act_all.extend(ep_act)
            img_all.extend(ep_img)
            ep_lens.append(len(ep_obs))
            n_success += int(info.get("is_success", False))
    if renderer is not None:
        renderer.close()
    env.close()

    out = out or os.path.join(DEMO_DIR, f"{task}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    arrays = dict(obs=np.asarray(obs_all, np.float32),
                  act=np.asarray(act_all, np.float32),
                  ep_lens=np.asarray(ep_lens, np.int64))
    if images:
        arrays["images"] = np.asarray(img_all, np.uint8)        # (N, H, W, 3)
    np.savez_compressed(out, **arrays)
    extra = f" + {len(img_all)} {img_size}px images" if images else ""
    print(f"saved {len(ep_lens)} demo episodes ({n_success} successful), "
          f"{len(obs_all)} transitions{extra} -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Collect scripted demonstrations.")
    ap.add_argument("--task", default="reach")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-all", action="store_true", help="keep failed episodes too")
    ap.add_argument("--images", action="store_true", help="also record camera RGB frames")
    ap.add_argument("--camera", default=None, help="camera name to render (default: first)")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    collect(a.task, a.episodes, a.seed, only_success=not a.keep_all, out=a.out,
            images=a.images, camera=a.camera, img_size=a.img_size)


if __name__ == "__main__":
    main()
