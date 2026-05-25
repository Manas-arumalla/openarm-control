"""Non-prehensile pushing (Phase 3a): push a puck to a goal WITHOUT grasping it.

The closed gripper is used as a pusher; the controller approaches behind the puck,
pushes it toward the goal, and re-aims after each stroke until it lands on target.

    python -m openarm_control.demos.demo_push                  # push to each goal
    python -m openarm_control.demos.demo_push --headless
    python -m openarm_control.demos.demo_push --goal b
    python -m openarm_control.demos.demo_push "push the puck to goal a"
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import PUSH_SCENE
from openarm_control.pushing import PushController
from openarm_control.agent.commands import parse_command

GOALS = {"a": "goal_a", "b": "goal_b"}
ALIASES = {"green": "a", "blue": "b", "first": "a", "second": "b"}


def _load():
    model = mujoco.MjModel.from_xml_path(PUSH_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _goal_xy(model, data, key):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GOALS[key])
    return data.xpos[bid][:2].copy()


def _goal_from_text(text):
    """Pull a goal key ('a'/'b') out of a command string."""
    toks = set(text.lower().replace(".", " ").split())
    for k in ("a", "b"):
        if k in toks:
            return k
    for alias, k in ALIASES.items():
        if alias in toks:
            return k
    return None


def _push_to(model, data, key, viewer=None):
    pc = PushController(model, data)
    print(f"  > push the puck to goal {key}")
    ok, msg = pc.push("puck", _goal_xy(model, data, key), tol=0.05,
                      viewer=viewer)
    print(f"    {'OK' if ok else 'FAIL'}: {msg}")
    if viewer is None:
        for _ in range(150):
            mujoco.mj_step(model, data)
    return ok


def run(keys, headless=False, interactive=False):
    print("=" * 64)
    print("Non-prehensile pushing: the closed gripper nudges the puck onto a goal")
    print("(no grasp), re-aiming after each stroke.")
    print("=" * 64)
    if headless:
        for k in keys:
            model, data = _load()                # fresh per goal (puck resets)
            _push_to(model, data, k)
        return
    from mujoco import viewer as mjviewer
    model, data = _load()
    with mjviewer.launch_passive(model, data) as viewer:
        if interactive:
            print("Type a goal: 'push to a' / 'push to b' (blank to quit).")
            while viewer.is_running():
                try:
                    c = input("command> ").strip()
                except EOFError:
                    break
                if not c:
                    break
                key = _goal_from_text(c)
                if key is None:
                    print("    which goal? say 'a' or 'b'"); continue
                _push_to(model, data, key, viewer=viewer)
        else:
            for k in keys:
                if not viewer.is_running():
                    break
                _push_to(model, data, k, viewer=viewer)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Push a puck to a goal (non-prehensile).")
    ap.add_argument("command", nargs="?", help="e.g. 'push the puck to goal a'")
    ap.add_argument("--goal", choices=["a", "b"], help="push to this goal")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args(argv)
    if args.command:
        intent = parse_command(args.command)
        if intent is None or intent.action != "push":
            print("that doesn't look like a push command"); return
        key = _goal_from_text(args.command) or "a"
        keys = [key]
    elif args.goal:
        keys = [args.goal]
    else:
        keys = ["a", "b"]                        # showcase both
    run(keys, headless=args.headless, interactive=args.interactive)


if __name__ == "__main__":
    main()
