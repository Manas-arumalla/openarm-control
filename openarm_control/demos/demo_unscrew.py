"""Single-arm bottle-opening demo — extension phase S2.

The bottle is held fixed in a stand and the RIGHT arm grasps the threaded cap,
unscrews it over several re-gripping bursts (turning ~one full turn), and lifts it
clear. The cap is jointed (turn hinge + lift slide), so it cannot be knocked off
and rises as it turns -- a screw thread. (Two 7-DOF arms at one small bottle
unavoidably collide, so the holding is done by the clamp, which also gives the
working arm full room for a collision-free multi-turn unscrew.)

    python -m openarm_control.demos.demo_unscrew            # viewer
    python -m openarm_control.demos.demo_unscrew --headless # scripted self-test + report
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import UNSCREW_SCENE
from openarm_control.bimanual import UnscrewTask


def _load():
    model = mujoco.MjModel.from_xml_path(UNSCREW_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def run_headless():
    m, d = _load()
    cap = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cap")
    turn_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cap_turn")]
    capz0 = d.xpos[cap][2]

    ok = UnscrewTask(m, d).run()
    turned = abs(np.degrees(d.qpos[turn_qadr]))
    for _ in range(60):
        mujoco.mj_step(m, d)

    print(f"Single-arm bottle opening (ok={ok}):")
    print(f"  bottle           : clamped in a stand (collision-free, working arm has full room)")
    print(f"  cap unscrewed    : turned {turned:.0f} deg over several re-gripping bursts")
    print(f"  cap lifted off   : rose {(d.xpos[cap][2]-capz0)*1000:+.0f} mm clear of the bottle")


def run_interactive():
    from mujoco import viewer as mjviewer
    m, d = _load()
    print("Single-arm bottle opening: the bottle is clamped; the right arm unscrews the cap.")
    print("Close the window to quit.")
    with mjviewer.launch_passive(m, d) as viewer:
        UnscrewTask(m, d).run(viewer=viewer, dt_realtime=True)
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bimanual bottle-opening demo.")
    ap.add_argument("--headless", action="store_true", help="scripted self-test + report")
    args = ap.parse_args(argv)
    if args.headless:
        run_headless()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
