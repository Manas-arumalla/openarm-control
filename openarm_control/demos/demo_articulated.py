"""Articulated-object manipulation demo — extension phase S3.

The arm operates the authored articulated fixtures: open a sliding DRAWER, swing a
hinged cabinet DOOR open, or turn a VALVE. Single-arm, weld-assisted: the arm
grasps the handle and moves it along the joint's allowed motion -- a straight pull
for the drawer, an arc with a matching wrist rotation for the door/valve so the
gripper genuinely swings/turns the part. The non-working arm is parked clear.

    python -m openarm_control.demos.demo_articulated --task drawer            # viewer
    python -m openarm_control.demos.demo_articulated --task door --headless   # self-test + report
    python -m openarm_control.demos.demo_articulated --task valve
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import ARTICULATED_SCENE
from openarm_control.articulated import ArticulatedController

TASKS = {
    "drawer": ("drawer_slide", "open_drawer", "drawer slid open"),
    "door":   ("door_hinge", "open_door", "door swung open"),
    "valve":  ("valve_turn", "turn_valve", "valve turned"),
}


def _load():
    model = mujoco.MjModel.from_xml_path(ARTICULATED_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def run_headless(task):
    joint, method, label = TASKS[task]
    m, d = _load()
    qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)]
    q0 = d.qpos[qadr]
    ok = getattr(ArticulatedController(m, d), method)()
    moved = d.qpos[qadr] - q0
    unit = "mm" if task == "drawer" else "deg"
    val = moved * 1000 if task == "drawer" else np.degrees(moved)
    print(f"Articulated '{task}' (ok={ok}): {label} by {abs(val):.0f} {unit}")


def run_interactive(task):
    from mujoco import viewer as mjviewer
    _, method, _ = TASKS[task]
    m, d = _load()
    print(f"Articulated manipulation: {task}. Close the window to quit.")
    with mjviewer.launch_passive(m, d) as viewer:
        getattr(ArticulatedController(m, d), method)(viewer=viewer, dt_realtime=True)
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()


def run_command(command, headless=False):
    """Language-commanded articulated manipulation, e.g. 'open the drawer then turn
    the valve'."""
    from openarm_control.agent.articulated_session import ArticulatedSession
    m, d = _load()
    print(f"command: {command!r}")
    if headless:
        sess = ArticulatedSession(m, d)
        for clause, ok, msg in sess.do(command):
            print(f"  {'OK ' if ok else 'FAIL'} {clause!r} -> {msg}")
    else:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(m, d) as viewer:
            sess = ArticulatedSession(m, d, viewer=viewer, dt_realtime=True)
            for clause, ok, msg in sess.do(command):
                print(f"  {'OK ' if ok else 'FAIL'} {clause!r} -> {msg}")
            while viewer.is_running():
                mujoco.mj_step(m, d); viewer.sync()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Articulated-object manipulation demo.")
    ap.add_argument("--task", default="drawer", choices=list(TASKS), help="which fixture to operate")
    ap.add_argument("--command", default=None,
                    help="natural-language command, e.g. 'open the drawer then turn the valve'")
    ap.add_argument("--headless", action="store_true", help="scripted self-test + report")
    args = ap.parse_args(argv)
    if args.command:
        run_command(args.command, headless=args.headless)
    elif args.headless:
        run_headless(args.task)
    else:
        run_interactive(args.task)


if __name__ == "__main__":
    main()
