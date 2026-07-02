"""Webcam human-arm imitation: the robot mirrors a human arm in real time.

A pose source yields the human's shoulder/elbow/wrist; the retargeter maps the
wrist into the robot's workspace (gripper pointed along the forearm) and solves
IK; the teleop controller smooths, velocity-limits, and clamps the joint command
before driving the arm. The same stack runs from a live webcam or a synthetic
pose source.

    python -m openarm_control.demos.demo_teleop                 # synthetic, viewer
    python -m openarm_control.demos.demo_teleop --webcam        # live webcam (MediaPipe)
    python -m openarm_control.demos.demo_teleop --webcam --preview  # + tracking/mapping window
    python -m openarm_control.demos.demo_teleop --bimanual      # both arms (synthetic)
    python -m openarm_control.demos.demo_teleop --headless 4    # 4 s headless self-check
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import (TELEOP_SCENE, TELEOP_PICK_SCENE,
                                     ARMS)
from openarm_control.teleop import (TeleopController, ScriptedPoseSource,
                                    WebcamPoseSource)

PICK_BLOCKS = ["block_red", "block_green", "block_blue"]


def _load(pick=False):
    model = mujoco.MjModel.from_xml_path(TELEOP_PICK_SCENE if pick else TELEOP_SCENE)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def _make_controller(model, data, arm, webcam=False, camera=0, mirror=True, pick=False):
    if webcam:
        source = WebcamPoseSource(side=arm.name, camera=camera, mirror=mirror)
    else:
        source = ScriptedPoseSource(side=arm.name)
    tc = TeleopController(model, data, arm=arm, source=source)
    if pick:
        tc.enable_grasping(PICK_BLOCKS)
    return tc


def run_headless(seconds, arm_name="right", seed=0, pick=False):
    """Drive the arm from the synthetic source headless; report tracking error.

    This is the verifiable core: the robot's tool point should follow the
    retargeter's mapped wrist target, with smooth, in-limit joint motion.
    """
    model, data = _load(pick=pick)
    arm = ARMS[arm_name]
    tc = _make_controller(model, data, arm, pick=pick)
    dt = model.opt.timestep
    sub = max(1, int(round(tc.dt / dt)))
    steps = int(seconds / tc.dt)

    errs, max_qd = [], 0.0
    cmd_ok, overshoot = True, 0.0
    q_prev = tc.q_now()
    for _ in range(steps):
        tc.step()
        cmd_ok = cmd_ok and bool(np.all((tc.q_cmd >= tc.jnt_low - 1e-9) &
                                        (tc.q_cmd <= tc.jnt_high + 1e-9)))
        for _ in range(sub):
            mujoco.mj_step(model, data)
        # tracking error: tool point vs the retargeter's mapped wrist target.
        if tc.retargeter.last_target is not None:
            tgt = tc.retargeter.last_target[0]
            errs.append(np.linalg.norm(tc.ee_pos() - tgt))
        q = tc.q_now()
        overshoot = max(overshoot, float(np.max(np.maximum(
            np.maximum(tc.jnt_low - q, q - tc.jnt_high), 0.0))))
        max_qd = max(max_qd, float(np.max(np.abs(q - q_prev) / tc.dt)))
        q_prev = q
    tc.close()
    errs = np.array(errs[5:])     # drop the initial slew-in
    print(f"arm={arm_name}  ticks={steps}  scale={tc.retargeter.scale:.2f}")
    print(f"tracking error: mean {errs.mean()*1000:.1f} mm  max {errs.max()*1000:.1f} mm")
    print(f"peak joint speed {max_qd:.2f} rad/s (limit {tc.qd_limit.max():.1f})")
    print(f"command always in joint limits: {cmd_ok} | peak physical overshoot "
          f"{np.degrees(overshoot):.2f} deg")


def _dir_word(v):
    """Human-readable label for a robot-frame direction [fwd, left, up]."""
    names = [("fwd", "back"), ("left", "right"), ("up", "down")]
    i = int(np.argmax(np.abs(v)))
    return names[i][0] if v[i] >= 0 else names[i][1]


def _draw_preview(source, retargeter):
    """Annotated webcam frame: detected arm skeleton + the robot-frame mapping."""
    import cv2
    frame = source.last_frame
    if frame is None:
        return None
    frame = frame.copy()
    h, w = frame.shape[:2]
    if source.last_image_lms:
        pts = [(int(x * w), int(y * h)) for (x, y) in source.last_image_lms]
        cv2.line(frame, pts[0], pts[1], (0, 255, 0), 3)      # upper arm
        cv2.line(frame, pts[1], pts[2], (0, 200, 255), 3)    # forearm
        for p, lab in zip(pts, ["shoulder", "elbow", "wrist"]):
            cv2.circle(frame, p, 7, (255, 255, 255), -1)
            cv2.putText(frame, lab, (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
    u, f = retargeter._u_s, retargeter._f_s
    lines = ["robot-frame mapping (fwd / left / up):"]
    if u is not None:
        lines.append("upper arm  [%+.2f %+.2f %+.2f]  -> %s" % (u[0], u[1], u[2], _dir_word(u)))
        lines.append("forearm    [%+.2f %+.2f %+.2f]  -> %s" % (f[0], f[1], f[2], _dir_word(f)))
    g = source.last_arm.grasp if source.last_arm else None
    lines.append("hand: %s" % ("--" if g is None else ("CLOSED" if g > 0.6 else "open" if g < 0.35 else "%.0f%%" % (g * 100))))
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (50, 255, 50), 1, cv2.LINE_AA)
    return frame


def run_viewer(arm_name="right", webcam=False, camera=0, mirror=True, seconds=30,
               pick=False, preview=False):
    model, data = _load(pick=pick)
    arm = ARMS[arm_name]
    tc = _make_controller(model, data, arm, webcam=webcam, camera=camera,
                          mirror=mirror, pick=pick)
    dt = model.opt.timestep
    sub = max(1, int(round(tc.dt / dt)))
    src = "webcam (MediaPipe)" if webcam else "synthetic pose"
    print(f"Mimic: {arm_name} arm follows a {src} source. Close the window to stop.")
    if webcam:
        print("Stand back so your whole arm is in frame; extend it to calibrate.")
    if pick:
        print("Reach to a block, CLOSE your hand to grab it, move it, OPEN to drop.")

    show_preview = preview and webcam
    if show_preview:
        import cv2
        print("A preview window shows your tracked arm + the robot-frame mapping.")

    # Grasp-point marker (pick scene): a sphere that tracks where a grab happens.
    marker_mid = -1
    if pick:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_marker")
        if bid >= 0:
            marker_mid = int(model.body_mocapid[bid])

    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        t_end = time.time() + seconds
        while viewer.is_running() and time.time() < t_end:
            t0 = time.time()
            tc.step()
            if marker_mid >= 0:
                data.mocap_pos[marker_mid] = tc.grasp_point()
            for _ in range(sub):
                mujoco.mj_step(model, data)
            viewer.sync()
            if show_preview:
                annotated = _draw_preview(tc.source, tc.retargeter)
                if annotated is not None:
                    cv2.imshow("OpenArm teleop — tracking & mapping", annotated)
                    if cv2.waitKey(1) & 0xFF == 27:    # Esc to quit
                        break
            time.sleep(max(0, tc.dt - (time.time() - t0)))
    if show_preview:
        cv2.destroyAllWindows()
    tc.close()


def run_bimanual(seconds=30):
    """Both arms mirror a (synthetic) human arm each, simultaneously."""
    model, data = _load()
    tcs = [_make_controller(model, data, ARMS[name]) for name in ("right", "left")]
    dt = model.opt.timestep
    sub = max(1, int(round(tcs[0].dt / dt)))
    print("Mimic (bimanual): both arms follow a synthetic pose source each.")
    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        t_end = time.time() + seconds
        while viewer.is_running() and time.time() < t_end:
            t0 = time.time()
            for tc in tcs:
                tc.step()
            for _ in range(sub):
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, tcs[0].dt - (time.time() - t0)))
    for tc in tcs:
        tc.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Webcam human-arm imitation (teleop).")
    ap.add_argument("--webcam", action="store_true",
                    help="use a live webcam via MediaPipe (else a synthetic pose source)")
    ap.add_argument("--camera", type=int, default=0, help="webcam device index")
    ap.add_argument("--arm", choices=["right", "left"], default="right")
    ap.add_argument("--bimanual", action="store_true", help="drive both arms (synthetic)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="follow directly instead of mirror-image (webcam only)")
    ap.add_argument("--pick", action="store_true",
                    help="manipulation scene: blocks on a table you can grab by closing your hand")
    ap.add_argument("--preview", action="store_true",
                    help="show a webcam window with the tracked arm + robot-frame mapping")
    ap.add_argument("--headless", nargs="?", const=4.0, type=float, metavar="SECONDS",
                    help="run headless for SECONDS and report tracking error (default 4)")
    ap.add_argument("--seconds", type=float, default=100000.0,
                    help="viewer run time in seconds (default: runs until you close the window)")
    args = ap.parse_args(argv)

    print("=" * 64)
    print("Human-arm imitation: pose -> retarget (wrist into robot workspace,")
    print("gripper along the forearm) -> safe real-time teleop (smooth, limited).")
    print("=" * 64)

    if args.headless is not None:
        run_headless(args.headless, arm_name=args.arm, pick=args.pick)
    elif args.bimanual:
        run_bimanual(seconds=args.seconds)
    else:
        run_viewer(arm_name=args.arm, webcam=args.webcam, camera=args.camera,
                   mirror=not args.no_mirror, seconds=args.seconds, pick=args.pick,
                   preview=args.preview)


if __name__ == "__main__":
    main()
