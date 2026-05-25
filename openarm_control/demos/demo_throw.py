"""Dynamic throwing: the arm grasps a ball and throws it into a bin (Phase I, M5).

It detects the bin, inverts the projectile equations for a release velocity,
finds a reachable swing, and throws — only if the bin is within the achievable
envelope (otherwise it reports the target is out of range).

    python -m openarm_control.demos.demo_throw                # one bin, viewer
    python -m openarm_control.demos.demo_throw --headless     # print landing
    python -m openarm_control.demos.demo_throw --bin-x 1.2    # try an out-of-range bin
    python -m openarm_control.demos.demo_throw --multi        # 5 balls -> 5 bins
    python -m openarm_control.demos.demo_throw --multi --headless
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import THROW_SCENE, THROW_MULTI_SCENE
from openarm_control.throwing import ThrowController

# (ball, bin) pairs for the multi-bin show (bins span the precise-throw envelope).
MULTI_PAIRS = [("ball_orange", "bin0"), ("ball_red", "bin1"), ("ball_green", "bin2"),
               ("ball_blue", "bin3"), ("ball_purple", "bin4")]


def _setup(bin_x=None):
    model = mujoco.MjModel.from_xml_path(THROW_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    if bin_x is not None:                          # move the bin (mocap-free: edit body pos)
        model.body_pos[bid][0] = bin_x
    mujoco.mj_forward(model, data)
    return model, data, bid


def run(headless=False, bin_x=None):
    model, data, bid = _setup(bin_x)
    tc = ThrowController(model, data, ball="ball_orange")
    binpos = data.xpos[bid].copy()
    target = np.array([binpos[0], binpos[1], 0.06])

    print("=" * 64)
    print("Throw a ball into a bin: ballistic inverse -> reachable swing -> release")
    print(f"bin at {np.round(binpos, 2)}")
    print("=" * 64)

    if headless:
        if not tc.grasp_ball():
            print("could not grasp the ball"); return
        ok, res = tc.throw(target)
        if not ok:
            print(f"REFUSED (out of throw envelope): {res}"); return
        err = np.linalg.norm(res[:2] - binpos[:2])
        print(f"landed {np.round(res, 2)}  |  {err*1000:.0f} mm from bin centre  |  "
              f"IN BIN: {err < 0.08 and res[2] < 0.2}")
        return

    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        if not tc.grasp_ball(viewer=viewer):
            print("could not grasp the ball"); return
        if tc.plan_release(target) is None:
            print(f"REFUSED (out of throw envelope): {tc.reason}")
            import time; time.sleep(2.0); return
        res = tc.execute(viewer=viewer)
        err = np.linalg.norm(res[:2] - binpos[:2])
        print(f"landed {np.round(res, 2)}  ({err*1000:.0f} mm from centre)  "
              f"IN BIN: {err < 0.08 and res[2] < 0.2}")
        import time
        end = time.time() + 3.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()


def run_multi(headless=False):
    """Grasp five balls in turn and throw each into a different bin.

    Between throws the scene is reset to a **pristine configuration** (clean arm
    pose, zero residual control/velocity) while the already-thrown balls are kept
    in their bins. That clean start is what makes every throw accurate (throwing
    from the previous frozen follow-through degrades it), and resetting instead of
    sweeping the arm back avoids knocking the balls still on the table."""
    model = mujoco.MjModel.from_xml_path(THROW_MULTI_SCENE)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    print("=" * 64)
    print("Multi-bin throwing: 5 balls -> 5 bins across the throw envelope")
    print("=" * 64)

    def do_round(viewer):
        landed, thrown = 0, {}                             # thrown: ball_qadr -> qpos(7)
        for ball, binname in MULTI_PAIRS:
            mujoco.mj_resetDataKeyframe(model, data, key)  # pristine arm + clean ctrl/qvel
            for qadr, q in thrown.items():                 # keep landed balls in their bins
                data.qpos[qadr:qadr + 7] = q
            mujoco.mj_forward(model, data)
            if viewer is not None and viewer.is_running():
                viewer.sync()
            tc = ThrowController(model, data, ball=ball)
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, binname)
            binpos = data.xpos[bid].copy()
            target = np.array([binpos[0], binpos[1], 0.06])
            if not tc.grasp_ball(viewer=viewer, disable_table=False):
                print(f"  {ball} -> {binname}: could not grasp"); continue
            if tc.plan_release(target) is None:
                print(f"  {ball} -> {binname}: REFUSED ({tc.reason})"); continue
            tc.execute(viewer=viewer)
            for _ in range(200):                           # settle in the bin
                mujoco.mj_step(model, data)
                if viewer is not None and viewer.is_running():
                    viewer.sync()
            bp = data.xpos[tc.ball_bid]
            err = np.linalg.norm(bp[:2] - binpos[:2])
            inbin = err < 0.06 and bp[2] < 0.10
            landed += int(inbin)
            thrown[tc.ball_qadr] = data.qpos[tc.ball_qadr:tc.ball_qadr + 7].copy()
            print(f"  {ball:12s} -> {binname}: {err*1000:3.0f} mm from centre  IN BIN: {inbin}")
        print(f"\nlanded {landed}/{len(MULTI_PAIRS)} balls in their bins")

    if headless:
        do_round(None)
        return
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        do_round(viewer)
        import time
        end = time.time() + 4.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Throw a ball into a bin.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--bin-x", type=float, default=None,
                    help="override the bin's forward distance (try 1.2 for out-of-range)")
    ap.add_argument("--multi", action="store_true",
                    help="throw five balls into five bins across the envelope")
    args = ap.parse_args(argv)
    if args.multi:
        run_multi(headless=args.headless)
    else:
        run(headless=args.headless, bin_x=args.bin_x)


if __name__ == "__main__":
    main()
