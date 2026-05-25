"""Train an OpenArm policy with SAC (Stable-Baselines3).

    openarm rl-train --task reach --timesteps 300000     # train + save
    openarm rl-train --task pick  --timesteps 600000
    openarm rl-train --task reach --timesteps 0 --eval   # evaluate a saved model

Saves to openarm_control/rl/models/<task>_sac.zip; TensorBoard logs alongside.
Use `openarm rl-eval --task <task>` to watch the policy in the viewer.
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.rl import TASKS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def model_path(task):
    return os.path.join(MODEL_DIR, f"{task}_sac")


def evaluate(model, task, n_episodes=30, seed=10000):
    env = TASKS[task](seed=seed)
    succ, metric = 0, []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, info = False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        succ += int(info["is_success"])
        metric.append(info.get("distance", info.get("d_block_target", 0.0)))
    env.close()
    return succ / n_episodes, float(np.mean(metric))


def main(argv=None):
    p = argparse.ArgumentParser(prog="openarm rl-train")
    p.add_argument("--task", choices=list(TASKS), default="reach")
    p.add_argument("--timesteps", type=int, default=300000)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--logdir", default=os.path.join(MODEL_DIR, "tb"))
    args = p.parse_args(argv)
    out = args.out or model_path(args.task)

    from stable_baselines3 import SAC
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.timesteps > 0:
        env = TASKS[args.task]()
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                    buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                    train_freq=1, gradient_steps=1, learning_starts=2000,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    tensorboard_log=args.logdir)
        before = evaluate(model, args.task, n_episodes=20)
        print(f"[{args.task}] before training: success={before[0]:.0%} metric={before[1]*1000:.0f} mm")
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(out)
        print(f"saved -> {out}.zip   (TensorBoard: tensorboard --logdir {args.logdir})")
    else:
        model = SAC.load(out)

    rate, metric = evaluate(model, args.task, n_episodes=30)
    print(f"[{args.task}] evaluation: success={rate:.0%} metric={metric*1000:.0f} mm")


if __name__ == "__main__":
    main()
