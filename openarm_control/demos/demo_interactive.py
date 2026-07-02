"""Interactive bimanual testing playground (Phase I).

The simulation opens; the camera detects the objects on a large table and the
terminal lists them. You pick an object (by number) and a destination bin, and the
dual-arm coordinator carries it out — the nearer arm picks, and if it can't reach
the chosen bin it hands the object to the other arm, which finishes.

    python -m openarm_control.demos.demo_interactive            # coloured-block scene
    python -m openarm_control.demos.demo_interactive --scanned  # real scanned objects
    python -m openarm_control.demos.demo_interactive --headless [--scanned]

Coloured blocks use the colour/shape detector. Scanned objects use depth-based
localisation (they stick up from the table) + open-vocabulary labelling on a tight
crop (YOLO-World by default, ``--detector yoloe`` for YOLOE); since a single
top-down view can't always name a small object, you select by **number** and the
label is a best-effort hint.
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import BIMANUAL_TABLE_SCENE, SCANNED_TABLE_SCENE
from openarm_control.bimanual import BimanualCoordinator
from openarm_control.pick_and_place import TABLE_TOP_Z

SCANNED_SCENE = SCANNED_TABLE_SCENE
PRIMITIVE_GRASPABLES = ["block_red", "block_green", "block_blue", "block_orange"]
SCANNED = [("mug", "cup"), ("bowl", "bowl"), ("clock", "clock"), ("elephant", "elephant")]
BINS = {"left": "bin_left", "right": "bin_right"}
PLACE_Z = 0.52


def _setup(scanned=False, detector="yolo-world"):
    scene = SCANNED_SCENE if scanned else BIMANUAL_TABLE_SCENE
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    co = BimanualCoordinator(model, data)
    binxy = {k: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)][:2].copy()
             for k, b in BINS.items()}
    if scanned:
        from openarm_control.vision.scene_perception import SegmentClassifyPerception
        # Open-vocab labels are best-effort; objects are localised geometrically
        # (depth) and selected by number, so fall back gracefully if the optional
        # open-vocab backend (ultralytics/torchvision) can't load.
        det, vocab = None, None
        try:
            from openarm_control.vision.detector import OpenVocabDetector
            det = OpenVocabDetector(conf=0.01, backend=detector)
            vocab = [c for _, c in SCANNED]
        except Exception as e:
            print(f"  [note] open-vocab detector unavailable ({type(e).__name__}); "
                  f"using geometric localisation only (objects listed by number).")
        perc = SegmentClassifyPerception(
            model, data, "tablecam", detector=det, vocab=vocab,
            exclude_xy=list(binxy.values()))
        graspables = [n for n, _ in SCANNED]
    else:
        from openarm_control.vision import ScenePerception, ColorShapeDetector
        perc = ScenePerception(model, data, "tablecam", detector=ColorShapeDetector())
        graspables = PRIMITIVE_GRASPABLES
    return model, data, co, perc, graspables, binxy


def _nearest(model, data, xy, graspables):
    best, bd = None, 0.12
    for b in graspables:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if bid >= 0:
            dist = np.linalg.norm(data.xpos[bid][:2] - xy)
            if dist < bd:
                best, bd = b, dist
    return best


def _detect(model, data, perc, graspables):
    out = []
    for o in perc.perceive():
        body = _nearest(model, data, o.position[:2], graspables)
        if body:
            out.append((o.label, body, o.position.copy()))
    return out


def _reset_arms(model, data, co):
    for name, ppc in (("right", co.bi.right), ("left", co.bi.left)):
        data.qpos[ppc.king.qpos_indices] = co.home[name]
        data.qvel[ppc.king.dof_indices] = 0.0
    data.qfrc_applied[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _do(model, data, co, body, pos, binxy, dest, viewer=None):
    grasp_z = float(np.clip(pos[2] - 0.01, TABLE_TOP_Z + 0.02, TABLE_TOP_Z + 0.12))
    ok, msg = co.pick_place(pos[:2], binxy[dest], body, grasp_z=grasp_z, place_z=PLACE_Z,
                            viewer=viewer, dt_realtime=viewer is not None, verbose=True)
    print(f"    {'OK' if ok else 'FAIL'}: {msg}")
    if viewer is None:
        for _ in range(150):
            mujoco.mj_step(model, data)
    return ok


def _list(objs):
    print("\nDetected objects:")
    for i, (label, body, pos) in enumerate(objs):
        print(f"  {i+1}. {label:10s} @ ({pos[0]:+.2f}, {pos[1]:+.2f})  [{body}]")


def run_interactive(scanned=False, detector="yolo-world"):
    model, data, co, perc, graspables, binxy = _setup(scanned, detector)
    from mujoco import viewer as mjviewer
    print("=" * 64)
    print("Interactive bimanual playground. Pick an object # + a bin; the best arm")
    print("does it (handing over if it can't reach the bin). Blank input quits.")
    print("=" * 64)
    with mjviewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            _reset_arms(model, data, co)
            viewer.sync()
            objs = _detect(model, data, perc, graspables)
            if not objs:
                print("no objects detected."); break
            _list(objs)
            try:
                sel = input("object # (blank to quit): ").strip()
            except EOFError:
                break
            if not sel or not sel.isdigit() or not (1 <= int(sel) <= len(objs)):
                if not sel:
                    break
                print("  ? pick a listed number"); continue
            label, body, pos = objs[int(sel) - 1]
            dest = input("destination bin [left/right]: ").strip().lower() or "right"
            if dest not in BINS:
                print("  ? unknown bin"); continue
            _do(model, data, co, body, pos, binxy, dest, viewer=viewer)


def run_headless(scanned=False, detector="yolo-world"):
    model, data, co, perc, graspables, binxy = _setup(scanned, detector)
    objs = _detect(model, data, perc, graspables)
    _list(objs)
    if not objs:
        return
    label, body, pos = objs[0]                            # deliver the first detected object
    print(f"  > deliver object 1 ({label} [{body}]) to the right bin")
    _do(model, data, co, body, pos, binxy, "right")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interactive bimanual testing playground.")
    ap.add_argument("--scanned", action="store_true", help="use real scanned objects")
    ap.add_argument("--detector", default="yolo-world", choices=["yolo-world", "yoloe"],
                    help="open-vocab backend for scanned objects")
    ap.add_argument("--headless", action="store_true", help="scripted self-test")
    args = ap.parse_args(argv)
    if args.headless:
        run_headless(scanned=args.scanned, detector=args.detector)
    else:
        run_interactive(scanned=args.scanned, detector=args.detector)


if __name__ == "__main__":
    main()
