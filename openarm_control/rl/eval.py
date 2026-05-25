"""Watch a trained OpenArm policy in the MuJoCo viewer.

    openarm rl-eval --task reach            # loads models/reach_sac
    openarm rl-eval --task pick --episodes 5
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.rl import TASKS


def main(argv=None):
    p = argparse.ArgumentParser(prog="openarm rl-eval")
    p.add_argument("--task", choices=list(TASKS), default="reach")
    p.add_argument("--model", default=None)
    p.add_argument("--episodes", type=int, default=8)
    args = p.parse_args(argv)
    model_path = args.model or os.path.join(os.path.dirname(__file__), "models", f"{args.task}_sac")

    from stable_baselines3 import SAC
    model = SAC.load(model_path)
    env = TASKS[args.task](render_mode="human")

    print("=" * 50)
    print(f"Trained '{args.task}' policy — green marker is the target.")
    print("=" * 50)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done, info = False, {}
        while not done:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            time.sleep(max(0, env.model.opt.timestep * 10 - (time.time() - t0)))
        metric = info.get("distance", info.get("d_block_target", 0.0))
        print(f"  episode {ep+1}: metric = {metric*1000:.0f} mm, success = {info['is_success']}")
    env.close()


if __name__ == "__main__":
    main()
