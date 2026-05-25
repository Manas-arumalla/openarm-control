"""Dynamic manipulation: the arm catches a ball thrown through the air.

A ball is launched on a ballistic arc from a random airborne point toward the
arm. The arm estimates the trajectory with a Kalman filter, solves for a
reachable interception, flies a re-planned minimum-jerk trajectory to meet the
ball mid-air facing the incoming velocity, and closes the gripper around it.

    python -m openarm_control.demos.demo_catch              # viewer demo
    python -m openarm_control.demos.demo_catch --benchmark  # headless catch-rate
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import CATCH_SCENE, CATCH_BIMANUAL_SCENE, CATCH_TWOBALL_SCENE
from openarm_control.catching import (CatchController, BimanualCatchController,
                                      TwoBallCatchController, sample_throw, sample_throw_bimanual)
from openarm_control.kinematics import orientation_error


def _ball_addrs(model):
    j = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    return model.jnt_qposadr[j], model.jnt_dofadr[j]


def _launch(model, data, catcher, bq, bdof, key, rng):
    mujoco.mj_resetDataKeyframe(model, data, key)
    L, v0 = sample_throw(rng, model.opt.gravity)
    data.qpos[bq:bq + 3] = L
    data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    data.qvel[bdof:bdof + 3] = v0
    catcher.reset()
    return L


def _make_catcher(model, data, vision):
    """Build a CatchController; with ``vision`` it is driven only by the two
    RGB-D cameras (no ground-truth ball state)."""
    if not vision:
        return CatchController(model, data)
    from openarm_control.vision import BallPerception
    perc = BallPerception(model, data, ["ballcam0", "ballcam1"])
    return CatchController(model, data, perception=perc)


def run_benchmark(n, seed, vision=False):
    """Headless: catch N random ballistic throws; print the catch/clean rate."""
    model = mujoco.MjModel.from_xml_path(CATCH_SCENE)
    data = mujoco.MjData(model)
    bq, bdof = _ball_addrs(model)
    fadr = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_right_finger_joint1")]
    catcher = _make_catcher(model, data, vision)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    rng = np.random.default_rng(seed)
    print(f"perception: {'two RGB-D cameras (vision)' if vision else 'ground-truth state'}")

    caught = clean = 0
    heights, oris, gaps = [], [], []
    for i in range(n):
        _launch(model, data, catcher, bq, bdof, key, rng)
        gp0 = catcher.grasp_pos()
        ori_at_catch = None
        for _ in range(800):
            catcher.step()
            mujoco.mj_step(model, data)
            if catcher.caught and ori_at_catch is None:   # measure facing AT the catch
                _, R = catcher.king.forward_kinematics()
                ori_at_catch = np.degrees(np.linalg.norm(orientation_error(R, catcher.plan.R)))
        if not catcher.caught:
            continue
        caught += 1
        gaps.append(np.linalg.norm(catcher.plan.p - gp0))
        near = np.linalg.norm(catcher.ball_pos() - catcher.grasp_pos())
        if catcher.ball_pos()[2] > 0.45 and near < 0.10 and data.qpos[fadr] > -0.45:
            clean += 1
            heights.append(catcher.ball_pos()[2]); oris.append(ori_at_catch)
    print(f"caught {caught}/{n}  ({100*caught/n:.0f}%)   clean grasps {clean}/{n}")
    if heights:
        print(f"mean catch height {np.mean(heights):.2f} m | mean orientation error "
              f"{np.mean(oris):.1f} deg | mean reach-to-intercept {np.mean(gaps)*100:.1f} cm")


def run_viewer(throws, seed, vision=False):
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_path(CATCH_SCENE)
    data = mujoco.MjData(model)
    bq, bdof = _ball_addrs(model)
    catcher = _make_catcher(model, data, vision)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    rng = np.random.default_rng(seed)
    dt = model.opt.timestep
    caught = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for throw in range(throws):
            L = _launch(model, data, catcher, bq, bdof, key, rng)
            for _ in range(900):
                t0 = time.time()
                catcher.step()
                mujoco.mj_step(model, data)
                if not viewer.is_running():
                    return
                viewer.sync()
                time.sleep(max(0, dt - (time.time() - t0)))
            ok = catcher.caught and catcher.ball_pos()[2] > 0.45
            caught += int(ok)
            print(f"  throw {throw+1}: launch={np.round(L,2)}  {'CAUGHT' if ok else 'missed'}")
            time.sleep(0.5)
    print(f"\ncaught {caught}/{throws}")


def run_bimanual(count, seed, vision, viewer):
    """Bimanual: throws toward either side; the system picks the best arm and
    catches collision-free. ``viewer`` shows it live, else headless benchmark."""
    model = mujoco.MjModel.from_xml_path(CATCH_BIMANUAL_SCENE)
    data = mujoco.MjData(model)
    bq, bdof = _ball_addrs(model)
    fing = {a: model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{a}_finger_joint1")] for a in ("right", "left")}
    perc = None
    if vision:
        from openarm_control.vision import BallPerception
        perc = BallPerception(model, data, ["ballcam0", "ballcam1"])
    bi = BimanualCatchController(model, data, perception=perc)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    rng = np.random.default_rng(seed)
    dt = model.opt.timestep
    print(f"perception: {'two RGB-D cameras (vision)' if vision else 'ground-truth state'}")

    vh = None
    if viewer:
        from mujoco import viewer as mjviewer
        vh = mjviewer.launch_passive(model, data).__enter__()
    caught = clean = matched = 0
    chosen = {"right": 0, "left": 0, "none": 0}
    min_sep_all = np.inf
    for i in range(count):
        mujoco.mj_resetDataKeyframe(model, data, key)
        L, v0, side = sample_throw_bimanual(rng, model.opt.gravity)
        data.qpos[bq:bq + 3] = L
        data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        data.qvel[bdof:bdof + 3] = v0
        bi.reset()
        min_sep = np.inf
        for _ in range(800 if not viewer else 900):
            t0 = time.time()
            bi.step()
            mujoco.mj_step(model, data)
            min_sep = min(min_sep, bi.arm_separation())
            if viewer:
                if not vh.is_running():
                    return
                vh.sync()
                time.sleep(max(0, dt - (time.time() - t0)))
        a = bi.active or "none"
        chosen[a] += 1
        ok = bi.caught
        held = grip = False
        if ok:
            c = bi.active_arm
            held = c.ball_pos()[2] > 0.45 and np.linalg.norm(c.ball_pos() - c.grasp_pos()) < 0.10
            grip = abs(data.qpos[fing[bi.active]]) < 0.45
        caught += int(ok); clean += int(held and grip)
        matched += int(side != "center" and bi.active == side)
        min_sep_all = min(min_sep_all, min_sep)
        print(f"  throw {i+1:2d}: thrown={side:6s} -> arm={a:5s}  "
              f"{'CAUGHT' if ok else 'missed'}  min_arm_gap={min_sep*100:.0f}cm")
        if viewer:
            time.sleep(0.4)
    if vh is not None:
        vh.__exit__(None, None, None)
    print(f"\ncaught {caught}/{count}, clean {clean}/{count} | arm {chosen} | "
          f"side-match {matched} | min inter-arm gap {min_sep_all*100:.0f} cm (>~0 = no collision)")


def _free_addr(model, body):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    j = next(j for j in range(model.njnt) if model.jnt_bodyid[j] == bid
             and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    return model.jnt_qposadr[j], model.jnt_dofadr[j]


def run_twoball(count, seed, vision, viewer):
    """Two balls thrown at once (one per side); each arm catches one, in parallel,
    via multi-object tracking — collision-free."""
    model = mujoco.MjModel.from_xml_path(CATCH_TWOBALL_SCENE)
    data = mujoco.MjData(model)
    q1, d1 = _free_addr(model, "ball")
    q2, d2 = _free_addr(model, "ball2")
    perc = None
    if vision:
        from openarm_control.vision import MultiBallPerception
        perc = MultiBallPerception(model, data, ["ballcam0", "ballcam1"], n_balls=2)
    bi = TwoBallCatchController(model, data, perception=perc)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    rng = np.random.default_rng(seed)
    dt = model.opt.timestep
    print(f"perception: {'two RGB-D cameras + multi-object tracking' if vision else 'ground-truth state'}")

    vh = None
    if viewer:
        from mujoco import viewer as mjviewer
        vh = mjviewer.launch_passive(model, data).__enter__()
    both = atleast1 = 0
    min_sep_all = np.inf
    for i in range(count):
        mujoco.mj_resetDataKeyframe(model, data, key)
        L1, v1, _ = sample_throw_bimanual(rng, model.opt.gravity, side="right")
        L2, v2, _ = sample_throw_bimanual(rng, model.opt.gravity, side="left")
        data.qpos[q1:q1 + 3] = L1; data.qpos[q1 + 3:q1 + 7] = [1, 0, 0, 0]
        data.qpos[q2:q2 + 3] = L2; data.qpos[q2 + 3:q2 + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        data.qvel[d1:d1 + 3] = v1
        data.qvel[d2:d2 + 3] = v2
        bi.reset()
        min_sep = np.inf
        for _ in range(820 if not viewer else 900):
            t0 = time.time()
            bi.step()
            mujoco.mj_step(model, data)
            min_sep = min(min_sep, bi.arm_separation())
            if viewer:
                if not vh.is_running():
                    return
                vh.sync()
                time.sleep(max(0, dt - (time.time() - t0)))
        nc = bi.num_caught
        both += int(nc == 2); atleast1 += int(nc >= 1)
        min_sep_all = min(min_sep_all, min_sep)
        print(f"  trial {i+1:2d}: caught {nc}/2  (right={bi.cr.caught_ball}, left={bi.cl.caught_ball})  "
              f"min_arm_gap={min_sep*100:.0f}cm")
        if viewer:
            time.sleep(0.4)
    if vh is not None:
        vh.__exit__(None, None, None)
    print(f"\nboth caught {both}/{count}, >=1 caught {atleast1}/{count} | "
          f"min inter-arm gap {min_sep_all*100:.0f} cm (>~0 = no collision)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Catch a ball thrown through the air.")
    ap.add_argument("--benchmark", nargs="?", const=40, type=int, metavar="N",
                    help="run N throws headless and print the catch rate (default 40)")
    ap.add_argument("--throws", type=int, default=8, help="number of viewer throws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vision", action="store_true",
                    help="observe the ball through two RGB-D cameras (no ground truth)")
    ap.add_argument("--bimanual", action="store_true",
                    help="both arms: throws toward either side, best-arm selection, collision-free")
    ap.add_argument("--twoball", action="store_true",
                    help="two balls at once (one per side); multi-object tracking, both arms catch")
    args = ap.parse_args(argv)

    print("=" * 64)
    print("Airborne catch: ball thrown on a ballistic arc; the arm predicts the")
    print("trajectory (Kalman), solves a reachable interception, and catches it")
    print("mid-air with a re-planned minimum-jerk trajectory (MPC).")
    print("=" * 64)
    if args.twoball:
        count = args.benchmark if args.benchmark is not None else args.throws
        run_twoball(count, args.seed, vision=args.vision, viewer=args.benchmark is None)
    elif args.bimanual:
        count = args.benchmark if args.benchmark is not None else args.throws
        run_bimanual(count, args.seed, vision=args.vision, viewer=args.benchmark is None)
    elif args.benchmark is not None:
        run_benchmark(args.benchmark, args.seed, vision=args.vision)
    else:
        run_viewer(args.throws, args.seed, vision=args.vision)


if __name__ == "__main__":
    main()
