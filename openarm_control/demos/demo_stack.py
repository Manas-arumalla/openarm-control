"""Stacking: pick up a cube and set it on another (Phase I, M6).

Vision-grounded and language-commanded, building on the manipulation session: the
robot detects the cubes, grounds both the object and the support, and places one
on top of the other so it rests stably.

    python -m openarm_control.demos.demo_stack "stack the red cube on the green cube"
    python -m openarm_control.demos.demo_stack            # a stacking showcase
    python -m openarm_control.demos.demo_stack --interactive
    python -m openarm_control.demos.demo_stack "stack the red cube on the green cube" --headless
"""
import argparse
import os
import sys

import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import STACK_SCENE
from openarm_control.vision import ScenePerception, ColorShapeDetector
from openarm_control.agent import ManipulationSession

CUBES = ["cube_red", "cube_green", "cube_blue", "cube_orange"]
SHOWCASE = ["stack the red cube on the green cube",
            "stack the orange cube on the blue cube"]


def _build(vision=False):
    model = mujoco.MjModel.from_xml_path(STACK_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    if vision:
        from openarm_control.vision import OpenVocabDetector
        det = OpenVocabDetector(default_vocab=["cube", "block", "box"])
    else:
        det = ColorShapeDetector()
    perc = ScenePerception(model, data, "tablecam", detector=det)
    sess = ManipulationSession(model, data, perception=perc, graspables=CUBES)
    return model, data, sess


def _run(sess, command, viewer=None):
    print(f"  > {command!r}")
    ok, msg = sess.do(command, viewer=viewer)
    print(f"    {'OK' if ok else 'FAIL'}: {msg}")


def run_headless(commands, vision=False):
    _, _, sess = _build(vision=vision)
    for c in commands:
        _run(sess, c)


def run_viewer(commands, vision=False, interactive=False):
    model, data, sess = _build(vision=vision)
    from mujoco import viewer as mjviewer
    print("Stacking. Close the window to stop.")
    with mjviewer.launch_passive(model, data) as viewer:
        if interactive:
            print("Type commands, e.g. 'stack the red cube on the green cube'. Blank to quit.")
            while viewer.is_running():
                try:
                    c = input("command> ").strip()
                except EOFError:
                    break
                if not c:
                    break
                _run(sess, c, viewer=viewer)
        else:
            for c in commands:
                if not viewer.is_running():
                    break
                _run(sess, c, viewer=viewer)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stack one cube on another (language-commanded).")
    ap.add_argument("command", nargs="?", help="a command, e.g. 'stack the red cube on the green cube'")
    ap.add_argument("--headless", action="store_true", help="run without the viewer")
    ap.add_argument("--interactive", action="store_true", help="type commands live in the viewer")
    ap.add_argument("--vision", action="store_true",
                    help="use the open-vocab detector (YOLO-World) instead of the colour fallback")
    args = ap.parse_args(argv)

    print("=" * 64)
    print("See the cubes -> understand 'stack X on Y' -> pick, carry, and place on top.")
    print("=" * 64)
    commands = [args.command] if args.command else SHOWCASE
    if args.headless:
        run_headless(commands, vision=args.vision)
    else:
        run_viewer(commands, vision=args.vision, interactive=args.interactive)


if __name__ == "__main__":
    main()
