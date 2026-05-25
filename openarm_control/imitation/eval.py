"""Evaluate a behavior-cloning policy in its env — and compare to RL head-to-head.

    python -m openarm_control.imitation.eval --task reach
    python -m openarm_control.imitation.eval --task reach --compare-rl   # BC vs SAC
    python -m openarm_control.imitation.eval --task reach --render
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import PROJECT_ROOT
from openarm_control.imitation.expert import TASKS
from openarm_control.imitation.bc import load_bc

DEMO_DIR = os.path.join(PROJECT_ROOT, "demos")
RL_MODELS = os.path.join(os.path.dirname(__file__), "..", "rl", "models")


def _run(env, policy_fn, episodes, seed):
    """Run a deterministic policy; return (success_rate, mean_final_distance_m)."""
    succ, dists = 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(policy_fn(obs))
            done = term or trunc
        succ += int(info.get("is_success", False))
        dists.append(info.get("distance", np.nan))
    return succ / episodes, float(np.nanmean(dists))


def evaluate(model_path, task="reach", episodes=30, seed=123, render=False, compare_rl=None):
    env_cls = TASKS[task][0]
    env = env_cls(seed=seed, render_mode="human" if render else None)

    bc = load_bc(model_path)
    rate, md = _run(env, lambda o: bc.act(o), episodes, seed)
    print(f"BC  [{task}]: success {rate*100:4.0f}%  | mean final distance {md*1000:3.0f} mm  "
          f"({episodes} eps, seed {seed})")

    if compare_rl is not None:
        rl_path = compare_rl if isinstance(compare_rl, str) else os.path.join(RL_MODELS, f"{task}_sac")
        try:
            from stable_baselines3 import SAC
            sac = SAC.load(rl_path)
            r2, m2 = _run(env, lambda o: sac.predict(o, deterministic=True)[0], episodes, seed)
            print(f"RL  [{task}]: success {r2*100:4.0f}%  | mean final distance {m2*1000:3.0f} mm  "
                  f"(SAC @ {rl_path})")
        except Exception as e:
            print(f"RL comparison skipped (no trained SAC at {rl_path}): {e}")
    env.close()
    return rate, md


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a BC policy (optionally vs RL).")
    ap.add_argument("--task", default="reach")
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--compare-rl", nargs="?", const=True, default=None,
                    help="also evaluate the trained SAC policy on the same episodes")
    a = ap.parse_args(argv)
    model = a.model or os.path.join(DEMO_DIR, f"{a.task}_bc.pt")
    evaluate(model, a.task, a.episodes, a.seed, a.render, a.compare_rl)


if __name__ == "__main__":
    main()
