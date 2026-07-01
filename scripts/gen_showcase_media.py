#!/usr/bin/env python3
"""Render showcase screenshots + GIFs for the manipulation skills (headless).

Drives each scripted controller with a tiny *viewer shim* that captures frames
every few physics steps (the controllers call ``viewer.is_running()`` /
``viewer.sync()`` each step), then writes a downsampled GIF + a hero PNG into
``media/``. No controller code is touched.

    python scripts/gen_showcase_media.py --probe                 # initial-state PNGs (camera tuning)
    python scripts/gen_showcase_media.py --only drawer,cloth     # subset
    python scripts/gen_showcase_media.py                         # all skills
"""
import argparse
import os
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
MEDIA = os.path.join(ROOT, "media")
os.makedirs(MEDIA, exist_ok=True)

from openarm_control.config import (
    ARTICULATED_SCENE, CONTACT_SCENE, CLOTH_SCENE, UNSCREW_SCENE,
    PEG_SOCKET_SCENE, BALANCE_SCENE, RIGHT_ARM,
)

W, H = 720, 540          # render / hero-PNG resolution (crisp)
GIF_FRAMES = 72          # target frame count after subsampling
GIF_W = 600              # GIFs downscaled to this width for a README-friendly size
GIF_COLORS = 128         # palette size for the optimized GIF
GIF_FPS = 16


def free_cam(lookat, dist, azim, elev):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance = dist
    c.azimuth = azim
    c.elevation = elev
    return c


# Per-skill 3/4 free-camera presets (lookat xyz, distance, azimuth, elevation).
CAMS = {
    "drawer":     ([0.29, -0.20, 0.46], 0.58, 150, -13),
    "door":       ([0.30, 0.10, 0.48], 1.10, 150, -18),
    "valve":      ([0.28, -0.05, 0.47], 1.00, 150, -18),
    "cloth":      ([0.25, -0.10, 0.45], 0.95, 135, -22),
    "unscrew":    ([0.30, 0.00, 0.50], 0.95, 150, -16),
    "insert":     ([0.30, 0.00, 0.47], 0.95, 150, -18),
    "admittance": ([0.22, -0.16, 0.47], 0.90, 150, -16),
    # Balance scenes: plate is 12 cm above the gripper (via a visible riser
    # rod) at world z ~0.795. A moderate 3/4 view shows the arm below, riser
    # extending up, plate + ball on top.
    "balance_circle":  ([0.24, -0.22, 0.75], 0.85, 150, -18),
    "balance_perturb": ([0.24, -0.22, 0.75], 0.85, 150, -18),
}


class FrameCapture:
    """Viewer shim: render every ``stride``-th physics step into a frame buffer."""

    def __init__(self, model, data, cam, stride=6, cap=900):
        self.model, self.data, self.cam = model, data, cam
        self.r = mujoco.Renderer(model, height=H, width=W)
        self.stride, self.cap = stride, cap
        self.k = 0
        self.frames = []

    def is_running(self):
        return True

    def sync(self):
        if self.k % self.stride == 0 and len(self.frames) < self.cap:
            self.r.update_scene(self.data, camera=self.cam)
            self.frames.append(self.r.render().copy())
        self.k += 1

    def grab(self):
        self.r.update_scene(self.data, camera=self.cam)
        return self.r.render().copy()

    def close(self):
        self.r.close()


def _set_offscreen(model):
    """Enlarge the offscreen framebuffer in-memory so we can render at W x H
    (default is 640x480). Does not touch any scene XML."""
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, W)
    model.vis.global_.offheight = max(model.vis.global_.offheight, H)
    return model


def _keyframe_load(scene, key="ready"):
    model = _set_offscreen(mujoco.MjModel.from_xml_path(scene))
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    return model, data


# --------------------------------------------------------------------------
# Per-skill runners: build scene + run controller with the capture shim.
# Each returns (frames, hero_frame).
# --------------------------------------------------------------------------
def run_articulated(which):
    from openarm_control.articulated import ArticulatedController
    m, d = _keyframe_load(ARTICULATED_SCENE)
    cap = FrameCapture(m, d, free_cam(*CAMS[which]))
    method = {"drawer": "open_drawer", "door": "open_door", "valve": "turn_valve"}[which]
    getattr(ArticulatedController(m, d), method)(viewer=cap)
    hero = cap.grab()
    cap.close()
    return cap.frames, hero


def run_cloth():
    from openarm_control.cloth import ClothFoldController, set_ready
    m = _set_offscreen(mujoco.MjModel.from_xml_path(CLOTH_SCENE))
    d = mujoco.MjData(m)
    set_ready(m, d)
    cap = FrameCapture(m, d, free_cam(*CAMS["cloth"]), stride=8)
    cf = ClothFoldController(m, d)
    cf.settle(viewer=cap)
    cf.fold("cloth_0", cf.corner_xy("cloth_8"), viewer=cap)
    hero = cap.grab()
    cap.close()
    return cap.frames, hero


def run_unscrew():
    from openarm_control.bimanual import UnscrewTask
    m, d = _keyframe_load(UNSCREW_SCENE)
    cap = FrameCapture(m, d, free_cam(*CAMS["unscrew"]), stride=8)
    UnscrewTask(m, d).run(viewer=cap)
    hero = cap.grab()
    cap.close()
    return cap.frames, hero


def run_insert():
    from openarm_control.vision import ScenePerception, ColorShapeDetector
    from openarm_control.agent import ManipulationSession
    m, d = _keyframe_load(PEG_SOCKET_SCENE)
    perc = ScenePerception(m, d, "tablecam", detector=ColorShapeDetector())
    sess = ManipulationSession(m, d, perception=perc, graspables=["peg"], bin_body="socket")
    cap = FrameCapture(m, d, free_cam(*CAMS["insert"]))
    sess.do("insert the blue peg into the socket", viewer=cap)
    for _ in range(200):           # let it settle, keep filming
        mujoco.mj_step(m, d)
        cap.sync()
    hero = cap.grab()
    cap.close()
    return cap.frames, hero


def run_admittance():
    from openarm_control.demos import demo_admittance as da
    m, d = _keyframe_load(CONTACT_SCENE)
    from openarm_control.contact import AdmittanceController
    ac = AdmittanceController(m, d)
    R = da._reachable_R(ac)
    da._teleport_hover(ac, R)
    ac.reset([da.PX, da.PY, da.HOVER], R)
    cap = FrameCapture(m, d, free_cam(*CAMS["admittance"]), stride=6)
    n_desc, hold = 600, 400
    for k in range(n_desc + hold):
        da._hold_idle(ac)          # keep the idle left arm from sagging to the table
        ac.step([da.PX, da.PY, da._cmd_z(k, n_desc)], R_desired=R, grip=da.GRIP)
        cap.sync()
    hero = cap.grab()
    cap.close()
    return cap.frames, hero


def _run_balance(kind):
    """Ball-balance scenarios rendered from the LQR controller.

    `kind` selects the scenario:
      * ``"circle"``  -- LQR tracks a circular target (radius 3 cm, period 5 s).
        Ball visibly orbits the plate centre.
      * ``"perturb"`` -- static target; every 1.6 s, a 25 cm/s random-direction
        velocity kick is applied to the ball. LQR recovers before the next kick.
    """
    from openarm_control.balance import LQRBalancer
    from openarm_control.demos.demo_balance import _target_at, _apply_perturbation
    m, d = _keyframe_load(BALANCE_SCENE)
    bal = LQRBalancer(m, d)
    bal.setup_hold()
    bal.reset(ball_offset_xy=(0.0, 0.0), settle_steps=400)
    cam = free_cam(*CAMS[f"balance_{kind}"])
    cap = FrameCapture(m, d, cam, stride=6)
    np.random.seed(0)
    if kind == "circle":
        # 2 full circles for a satisfying loop.
        radius, period = 0.03, 5.0
        duration = 2 * period
    elif kind == "perturb":
        # LQR needs ~2 s to bring the ball back after a 20-25 cm/s kick, so
        # kicks are spaced 2.5 s apart: enough for the recovery to be visibly
        # complete before the next disturbance arrives. 3 kicks in 8 s reads
        # cleanly at ~30 fps.
        radius, period = 0.0, 5.0
        duration = 8.0
        perturb_interval = 2.5
        perturb_speed = 0.20
    else:
        raise ValueError(f"unknown balance kind {kind!r}")
    n = int(duration / m.opt.timestep)
    next_perturb = perturb_interval if kind == "perturb" else None
    for k in range(n):
        t = k * m.opt.timestep
        if kind == "circle":
            tx, ty = _target_at(t, "circle", radius, period)
        else:
            tx, ty = 0.0, 0.0
            if t >= next_perturb:
                _apply_perturbation(bal, perturb_speed)
                next_perturb += perturb_interval
        bal.step(target_xy=(tx, ty))
        cap.sync()
    hero = cap.grab()
    cap.close()
    return cap.frames, hero


RUNNERS = {
    "drawer": lambda: run_articulated("drawer"),
    "door": lambda: run_articulated("door"),
    "valve": lambda: run_articulated("valve"),
    "cloth": run_cloth,
    "unscrew": run_unscrew,
    "insert": run_insert,
    "admittance": run_admittance,
    "balance_circle":  lambda: _run_balance("circle"),
    "balance_perturb": lambda: _run_balance("perturb"),
}


def _subsample(frames, n):
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def write_gif(frames, name):
    """Downscale, then encode with frame-differencing (subrectangles) — the camera
    is fixed and the background static, so only the moving pixels are re-stored."""
    frames = _subsample(frames, GIF_FRAMES)
    h = round(H * GIF_W / W)
    small = [np.asarray(Image.fromarray(f).resize((GIF_W, h), Image.LANCZOS)) for f in frames]
    path = os.path.join(MEDIA, f"{name}.gif")
    imageio.mimsave(path, small, format="GIF", fps=GIF_FPS, loop=0, subrectangles=True)
    kb = os.path.getsize(path) / 1024
    print(f"  wrote {path}  ({len(small)} frames, {GIF_W}x{h}, {kb:.0f} KB)")


def write_png(frame, name):
    path = os.path.join(MEDIA, f"{name}.png")
    Image.fromarray(frame).save(path)
    print(f"  wrote {path}")


def probe(names):
    """Render each scene's initial state to media/_probe_<skill>.png for camera tuning."""
    loaders = {
        "drawer": (ARTICULATED_SCENE, "ready"), "door": (ARTICULATED_SCENE, "ready"),
        "valve": (ARTICULATED_SCENE, "ready"), "unscrew": (UNSCREW_SCENE, "ready"),
        "insert": (PEG_SOCKET_SCENE, "ready"), "admittance": (CONTACT_SCENE, "ready"),
        "balance_circle": (BALANCE_SCENE, "ready"),
        "balance_perturb": (BALANCE_SCENE, "ready"),
    }
    for n in names:
        if n == "cloth":
            from openarm_control.cloth import set_ready
            m = _set_offscreen(mujoco.MjModel.from_xml_path(CLOTH_SCENE)); d = mujoco.MjData(m); set_ready(m, d)
        else:
            scene, key = loaders[n]
            m, d = _keyframe_load(scene, key)
        r = mujoco.Renderer(m, height=H, width=W)
        r.update_scene(d, camera=free_cam(*CAMS[n]))
        Image.fromarray(r.render().copy()).save(os.path.join(MEDIA, f"_probe_{n}.png"))
        r.close()
        print(f"  probe media/_probe_{n}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate showcase GIFs + screenshots.")
    ap.add_argument("--only", default=None, help="comma list: " + ",".join(RUNNERS))
    ap.add_argument("--probe", action="store_true", help="render initial-state PNGs only (camera tuning)")
    args = ap.parse_args(argv)
    names = args.only.split(",") if args.only else list(RUNNERS)

    if args.probe:
        probe(names)
        return 0

    for n in names:
        print(f"[{n}]")
        frames, hero = RUNNERS[n]()
        if not frames:
            print("  (no frames captured!)")
            continue
        write_gif(frames, n)
        write_png(hero, f"{n}_hero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
