"""Bimanual coordination demos. Run with the MuJoCo viewer:

    python control/demos/demo_bimanual.py                 # parallel sort (default)
    python control/demos/demo_bimanual.py --mode sync     # synchronized mirrored motion
    python control/demos/demo_bimanual.py --mode handoff  # right -> left object relay
    python control/demos/demo_bimanual.py --mode stack    # both arms stack a tower at once
    python control/demos/demo_bimanual.py --mode coordinate  # best arm picks; hands over if needed
    python control/demos/demo_bimanual.py --mode language     # natural-language: "transfer the red block to the left bin"
    python control/demos/demo_bimanual.py --mode language --interactive
"""
import os
import sys
import time
import argparse

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import (BIMANUAL_SCENE, BIMANUAL_STACK_SCENE,
                                     BIMANUAL_HANDOVER_SCENE, BIMANUAL_TABLE_SCENE)
from openarm_control.bimanual import (BimanualController, ParallelSort, RelayHandoff,
                                      BimanualStack, BimanualCoordinator, synchronized_move)
from openarm_control.grasp import topdown_orientation

LANGUAGE_GRASPABLES = ["block_red", "block_green", "block_blue", "block_orange"]
# A showcase that forces hand-overs: red sits on the +y (left) side but goes to the
# right bin (only the right arm reaches it), and vice-versa for green.
LANGUAGE_SHOWCASE = ["transfer the red block to the right bin",
                     "transfer the green block to the left bin"]


def _load(scene=BIMANUAL_SCENE):
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def run_sort():
    model, data = _load()
    task = ParallelSort(model, data)
    print("Bimanual PARALLEL SORT: both arms sort their side's blocks at once.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        task.run(viewer=viewer, dt_realtime=True)
        while viewer.is_running():
            task.bi.grav_comp(); mujoco.mj_step(model, data); viewer.sync(); time.sleep(model.opt.timestep)


def run_handoff():
    model, data = _load()
    relay = RelayHandoff(model, data)
    print("Bimanual HAND-OFF: right arm relays a block to the left arm's bin.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        relay.run(viewer=viewer, dt_realtime=True)
        while viewer.is_running():
            relay.bi.grav_comp(); mujoco.mj_step(model, data); viewer.sync(); time.sleep(model.opt.timestep)


def run_stack():
    model, data = _load(BIMANUAL_STACK_SCENE)
    task = BimanualStack(model, data)
    print("Bimanual STACK: both arms build a tower on their side, simultaneously.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        task.run(viewer=viewer, dt_realtime=True)
        while viewer.is_running():
            task.bi.grav_comp(); mujoco.mj_step(model, data); viewer.sync(); time.sleep(model.opt.timestep)


def run_coordinate():
    model, data = _load(BIMANUAL_HANDOVER_SCENE)
    co = BimanualCoordinator(model, data)
    binxy = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin")][:2].copy()
    print("Bimanual COORDINATION: the best-placed arm does each task; if it can't")
    print("reach the bin, it hands the object to the other arm.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        gxy = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_green")][:2].copy()
        ok, msg = co.pick_place(gxy, binxy, "box_green", place_z=0.52,
                                viewer=viewer, dt_realtime=True)
        print("  ->", msg)
        rxy = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_red")][:2].copy()
        ok, msg = co.pick_place(rxy, binxy, "box_red", place_z=0.52,
                                viewer=viewer, dt_realtime=True)
        print("  ->", msg)
        while viewer.is_running():
            co.bi.grav_comp(); mujoco.mj_step(model, data); viewer.sync(); time.sleep(model.opt.timestep)


def run_sync():
    model, data = _load()
    bi = BimanualController(model, data)
    bi.sync_ctrl()
    print("Bimanual SYNCHRONIZED motion: left arm exactly mirrors the right.")
    # Right-arm Cartesian waypoints (y<0 side); the left arm mirrors automatically.
    waypoints = [(0.25, -0.20, 0.50), (0.30, -0.20, 0.58), (0.22, -0.20, 0.58), (0.24, -0.23, 0.55)]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(1.0)
        for (x, y, z) in waypoints:
            rq = bi.right.gs.king.inverse_kinematics(
                np.array([x, y, z]), target_mat=topdown_orientation(1.5),
                q_init=data.qpos[bi.right.king.qpos_indices], restarts=0, rest_weight=0.0)
            if not synchronized_move(bi, np.array([rq]), 2.0, viewer=viewer, dt_realtime=True):
                break
        while viewer.is_running():
            bi.grav_comp(); mujoco.mj_step(model, data); viewer.sync(); time.sleep(model.opt.timestep)


def _language_setup():
    from openarm_control.vision import ScenePerception, ColorShapeDetector
    from openarm_control.agent.bimanual_session import BimanualSession
    model, data = _load(BIMANUAL_TABLE_SCENE)
    perc = ScenePerception(model, data, "tablecam", detector=ColorShapeDetector())
    sess = BimanualSession(model, data, perc, LANGUAGE_GRASPABLES)
    return model, data, sess


def run_language(command=None, headless=False, interactive=False):
    """Natural-language bimanual manipulation: the session grounds the object and
    destination, the coordinator picks the best arm and hands over when only the
    other arm can reach the bin."""
    print("=" * 64)
    print("Bimanual + language: 'transfer/move the <colour> block to the left/right")
    print("bin' -> best arm picks; hands over automatically when needed.")
    print("=" * 64)
    cmds = [command] if command else LANGUAGE_SHOWCASE
    if headless:
        _, _, sess = _language_setup()
        for c in cmds:
            print(f"  > {c!r}")
            ok, msg = sess.do(c)
            print(f"    {'OK' if ok else 'FAIL'}: {msg}")
        return
    model, data, sess = _language_setup()
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        if interactive:
            print("Type commands (e.g. 'transfer the red block to the right bin'). Blank to quit.")
            while viewer.is_running():
                try:
                    c = input("command> ").strip()
                except EOFError:
                    break
                if not c:
                    break
                ok, msg = sess.do(c, viewer=viewer)
                print(f"    {'OK' if ok else 'FAIL'}: {msg}")
        else:
            for c in cmds:
                if not viewer.is_running():
                    break
                print(f"  > {c!r}")
                ok, msg = sess.do(c, viewer=viewer)
                print(f"    {'OK' if ok else 'FAIL'}: {msg}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="openarm bimanual")
    p.add_argument("--mode", choices=["sort", "sync", "handoff", "stack", "coordinate", "language"],
                   default="sort")
    p.add_argument("command", nargs="?", help="a language command (language mode)")
    p.add_argument("--headless", action="store_true", help="run without the viewer (language mode)")
    p.add_argument("--interactive", action="store_true", help="type commands live (language mode)")
    args = p.parse_args(argv)
    if args.mode == "language":
        run_language(args.command, headless=args.headless, interactive=args.interactive)
        return
    {"sort": run_sort, "sync": run_sync, "handoff": run_handoff, "stack": run_stack,
     "coordinate": run_coordinate}[args.mode]()


if __name__ == "__main__":
    main()
