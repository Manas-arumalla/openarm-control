"""Peg-in-hole insertion (Phase I, M6) — a family of matched peg/hole shapes.

The robot grasps a peg and inserts it into a hole with a precise point-down
descent — a contact-rich skill solved with classical control. Three selectable
scenarios, in increasing alignment difficulty:

  * `--shape round`  : a cylinder into a **circular** hole — rotationally
    symmetric, so no yaw alignment is needed (free-yaw vertical descent).
  * `--shape square` : a square peg into a rotated **square** hole — 4-fold
    symmetric, so the grasp is snapped to the nearest reachable 90-deg multiple.
  * `--shape cuboid` : a rectangular block into a rotated **rectangular** slot —
    180-deg symmetric, the tightest alignment (one of two orientations fits).

    python -m openarm_control.demos.demo_insert                  # round, viewer
    python -m openarm_control.demos.demo_insert --shape square
    python -m openarm_control.demos.demo_insert --shape cuboid --headless
    python -m openarm_control.demos.demo_insert --cuboid         # alias for --shape cuboid
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import PEG_SOCKET_SCENE, PEG_SQUARE_SCENE, PEG_CUBOID_SCENE
from openarm_control.vision import ScenePerception, ColorShapeDetector
from openarm_control.agent import ManipulationSession
from openarm_control.agent.executor import TaskExecutor

GRASPABLES = ["peg"]
DEFAULT_CMD = "insert the blue peg into the socket"

# shape -> (scene, executor aligned-insert method). 'round' uses the free-yaw
# language path instead (see run()).
ALIGNED = {"square": (PEG_SQUARE_SCENE, "insert_square"),
           "cuboid": (PEG_CUBOID_SCENE, "insert_cuboid")}


def _build():
    model = mujoco.MjModel.from_xml_path(PEG_SOCKET_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    perc = ScenePerception(model, data, "tablecam", detector=ColorShapeDetector())
    sess = ManipulationSession(model, data, perception=perc,
                               graspables=GRASPABLES, bin_body="socket")
    return model, data, sess


def _report(model, data):
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "socket")
    pp, sp = data.xpos[pid], data.xpos[sid]
    dxy = np.linalg.norm(pp[:2] - sp[:2])
    R = data.xmat[pid].reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    print(f"  peg {dxy*1000:.0f} mm off-centre, {tilt:.0f} deg tilt, base z={pp[2]-0.045:.3f}  "
          f"INSERTED: {dxy < 0.015 and tilt < 15}")


def run(command, headless=False):
    model, data, sess = _build()
    print("=" * 64)
    print("Peg-in-hole: grasp the peg -> align over the socket -> thread it in.")
    print("=" * 64)
    print(f"  > {command!r}")
    if headless:
        ok, msg = sess.do(command)
        print(f"    {'OK' if ok else 'FAIL'}: {msg}")
        for _ in range(400):
            mujoco.mj_step(model, data)
        _report(model, data)
        return
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        ok, msg = sess.do(command, viewer=viewer)
        print(f"    {'OK' if ok else 'FAIL'}: {msg}")
        import time
        end = time.time() + 4.0
        while viewer.is_running() and time.time() < end:
            mujoco.mj_step(model, data)
            viewer.sync()
        _report(model, data)


def run_aligned(shape, headless=False):
    """Yaw-aligned insertion (square or cuboid): align the peg to the rotated hole's
    symmetry, then thread it straight in."""
    scene, method = ALIGNED[shape]
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    ex = TaskExecutor(model, data, graspables=["peg"], bin_body="socket")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "socket")
    Rs = data.xmat[sid].reshape(3, 3)
    syaw = np.degrees(np.arctan2(Rs[1, 0], Rs[0, 0]))
    sym = 90 if shape == "square" else 180             # cross-section symmetry (deg)
    half_h = 0.035
    print("=" * 64)
    print(f"{shape.capitalize()} peg-in-hole: align the block to the {syaw:.0f}-deg "
          f"rotated hole ({sym}-deg symmetry), thread in.")
    print("=" * 64)

    def report():
        for _ in range(400):
            mujoco.mj_step(model, data)
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        pp, sp = data.xpos[pid], data.xpos[sid]
        R = data.xmat[pid].reshape(3, 3)
        dxy = np.linalg.norm(pp[:2] - sp[:2])
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        pyaw = np.degrees(np.arctan2(R[1, 0], R[0, 0])) % sym
        yerr = min(abs(pyaw - syaw % sym), sym - abs(pyaw - syaw % sym))
        print(f"  block {dxy*1000:.0f} mm off-centre, {tilt:.0f} deg tilt, "
              f"{yerr:.0f} deg yaw-error, base z={pp[2]-half_h:.3f}  "
              f"INSERTED: {dxy < 0.013 and tilt < 15 and yerr < 15}")

    insert = getattr(ex, method)
    if headless:
        ok, msg = insert("peg", "socket")
        print(f"  {'OK' if ok else 'FAIL'}: {msg}")
        report()
        return
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        ok, msg = insert("peg", "socket", viewer=viewer)
        print(f"  {'OK' if ok else 'FAIL'}: {msg}")
        report()
        import time
        end = time.time() + 4.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Peg-in-hole insertion (round / square / cuboid).")
    ap.add_argument("command", nargs="?", default=DEFAULT_CMD,
                    help="a command, e.g. 'insert the blue peg into the socket' (round only)")
    ap.add_argument("--shape", choices=["round", "square", "cuboid"], default="round",
                    help="peg/hole shape: round (cylinder/circle), square (cube/square), "
                         "cuboid (rectangle/slot)")
    ap.add_argument("--cuboid", action="store_true", help="alias for --shape cuboid")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)
    shape = "cuboid" if args.cuboid else args.shape
    if shape == "round":
        run(args.command, headless=args.headless)
    else:
        run_aligned(shape, headless=args.headless)


if __name__ == "__main__":
    main()
