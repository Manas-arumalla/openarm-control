"""Language-commanded, vision-grounded manipulation (Phase I, M1-M3).

Type a command; the robot detects the objects with its camera, grounds the
target, and carries it out with a collision-free motion.

    python -m openarm_control.demos.demo_manipulate "put the green box in the bin"
    python -m openarm_control.demos.demo_manipulate            # a showcase sequence
    python -m openarm_control.demos.demo_manipulate --interactive
    python -m openarm_control.demos.demo_manipulate --vision   # open-vocab detector (YOLO-World)
    python -m openarm_control.demos.demo_manipulate "pick the red ball" --headless
"""
import argparse
import os
import sys

import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import TABLETOP_SCENE
from openarm_control.vision import ScenePerception, ColorShapeDetector
from openarm_control.agent import ManipulationSession

BODIES = ["ball_red", "box_green", "can_blue", "ball_orange"]
# Conversational showcase: queries, a multi-step command, and undo. State carries
# across turns ("it" = the held object).
SHOWCASE = ["what is on the table?",
            "pick up the green box and put it in the bin",
            "what are you holding?",
            "undo",
            "pick up the blue cylinder then put it in the bin"]


def _build(vision=False):
    model = mujoco.MjModel.from_xml_path(TABLETOP_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    if vision:
        from openarm_control.vision import OpenVocabDetector
        det = OpenVocabDetector(default_vocab=["ball", "box", "can", "bottle"])
    else:
        det = ColorShapeDetector()
    perc = ScenePerception(model, data, "tablecam", detector=det)
    sess = ManipulationSession(model, data, perception=perc, graspables=BODIES)
    return model, data, sess


def _run(sess, command, viewer=None):
    print(f"  > {command!r}")
    ok, msg = sess.run(command, viewer=viewer)            # run() handles multi-step + single
    held = f" [holding {sess.held}]" if sess.held else ""
    print(f"    {'OK' if ok else 'FAIL'}: {msg}{held}")


def run_headless(commands, vision=False):
    _, _, sess = _build(vision=vision)
    for c in commands:
        _run(sess, c)


def run_viewer(commands, vision=False, interactive=False):
    model, data, sess = _build(vision=vision)
    from mujoco import viewer as mjviewer
    print("Vision-grounded manipulation. Close the window to stop.")
    with mjviewer.launch_passive(model, data) as viewer:
        if interactive:
            print("Type commands; state carries over ('pick the blue cylinder' then")
            print("'put it in the bin', or 'go to the bin' then 'release'). Blank to quit.")
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
    ap = argparse.ArgumentParser(description="Language-commanded manipulation.")
    ap.add_argument("command", nargs="?", help="a command, e.g. 'put the green box in the bin'")
    ap.add_argument("--headless", action="store_true", help="run without the viewer")
    ap.add_argument("--interactive", action="store_true", help="type commands live in the viewer")
    ap.add_argument("--vision", action="store_true",
                    help="use the open-vocab detector (YOLO-World) instead of the colour fallback")
    args = ap.parse_args(argv)

    print("=" * 64)
    print("See the table -> understand the command -> plan around clutter -> act.")
    print("=" * 64)
    commands = [args.command] if args.command else SHOWCASE
    if args.headless:
        run_headless(commands, vision=args.vision)
    else:
        run_viewer(commands, vision=args.vision, interactive=args.interactive)


if __name__ == "__main__":
    main()
