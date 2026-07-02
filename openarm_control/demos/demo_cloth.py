"""Cloth folding (Phase 3c): grasp a corner of a deformable cloth and fold it over.

The cloth is a MuJoCo flex grid on the table. The robot grasps a corner and carries
it across to the opposite edge, folding the sheet -- deformable manipulation.

    python -m openarm_control.demos.demo_cloth
    python -m openarm_control.demos.demo_cloth --headless
    python -m openarm_control.demos.demo_cloth --corner 20    # fold a different corner
"""
import argparse
import os
import sys

import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import CLOTH_SCENE
from openarm_control.cloth import ClothFoldController, set_ready

# Fold a corner onto the opposite corner of its edge (a half-fold) on the 9x9 grid.
OPPOSITE = {"0": "cloth_8", "8": "cloth_0", "72": "cloth_80", "80": "cloth_72"}


def _load():
    model = mujoco.MjModel.from_xml_path(CLOTH_SCENE)
    data = mujoco.MjData(model)
    return model, data


def run(corner="0", headless=False):
    print("=" * 64)
    print("Cloth folding: grasp a corner of the cloth and fold it over to the")
    print("opposite edge (deformable manipulation).")
    print("=" * 64)
    corner_body = f"cloth_{corner}"
    target_body = OPPOSITE.get(corner, "cloth_4")
    if headless:
        model, data = _load()
        set_ready(model, data)
        cf = ClothFoldController(model, data)
        before = cf.cloth_vertices()
        ok, msg = cf.fold(corner_body, cf.corner_xy(target_body))
        after = cf.cloth_vertices()
        print(f"  > fold {corner_body} onto {target_body}")
        print(f"    {'OK' if ok else 'FAIL'}: {msg}")
        print(f"    cloth y-extent {1000*(before[:,1].max()-before[:,1].min()):.0f} mm "
              f"-> {1000*(after[:,1].max()-after[:,1].min()):.0f} mm (smaller = folded)")
        return
    model, data = _load()
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        set_ready(model, data)
        viewer.sync()
        cf = ClothFoldController(model, data)
        ok, msg = cf.fold(corner_body, cf.corner_xy(target_body), viewer=viewer)
        print(f"    {'OK' if ok else 'FAIL'}: {msg}")
        import time
        end = time.time() + 4.0
        while viewer.is_running() and time.time() < end:
            mujoco.mj_step(model, data)
            viewer.sync()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fold a deformable cloth by a corner.")
    ap.add_argument("--corner", choices=["0", "8", "72", "80"], default="0",
                    help="which cloth corner to grasp and fold")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)
    run(corner=args.corner, headless=args.headless)


if __name__ == "__main__":
    main()
