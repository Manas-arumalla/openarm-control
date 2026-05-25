"""Auto-labeled synthetic detection data via MuJoCo's segmentation renderer.

The open-vocab detector is low-confidence on plain sim primitives and a single
top-down view is ambiguous. To harden it we **fine-tune on the sim's own objects**
-- and we get the training labels *for free*: render a scene under domain
randomization (camera pose, lighting, object placement/yaw, background colour),
then read **exact 2D bounding boxes from the per-object segmentation masks**. No
manual labeling, and the train→deploy gap is tiny (same renderer). Output is a
YOLO-format dataset that `vision/finetune.py` trains on.

    gen = SyntheticDataGenerator(scene_xml, {"block_red": "red block", ...})
    gen.generate("datasets/openarm", n=2000, seed=0)
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import mujoco


def body_geom_ids(model, body):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        return []
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


def segmentation_bboxes(seg, geoms_by_body, min_px=18):
    """{body: (u0, v0, u1, v1)} for each body visible in a segmentation image
    ``seg`` (H,W,2: objid, objtype). Bodies with too few pixels are skipped."""
    objid, objtype = seg[..., 0], seg[..., 1]
    is_geom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    out = {}
    for body, geoms in geoms_by_body.items():
        if not geoms:
            continue
        mask = np.isin(objid, geoms) & is_geom
        if int(mask.sum()) < min_px:
            continue
        ys, xs = np.nonzero(mask)
        out[body] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return out


class SyntheticDataGenerator:
    """Render a scene under domain randomization and emit YOLO-format detection data
    with segmentation-derived boxes. ``classes`` maps body name -> class label."""

    def __init__(self, scene_xml, classes, width=640, height=480,
                 region=((0.15, 0.37), (-0.26, 0.26)), table_top=0.40):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_forward(self.model, self.data)
        self.classes = dict(classes)
        self.class_names = sorted(set(self.classes.values()))
        self.class_idx = {n: i for i, n in enumerate(self.class_names)}
        self.geoms_by_body = {b: body_geom_ids(self.model, b) for b in self.classes}
        self.width, self.height = width, height
        self.region, self.table_top = region, table_top
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        # qpos free-joint address per object body (for placement DR).
        self._qadr = {}
        for b in self.classes:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            jadr = self.model.body_jntadr[bid]
            if jadr >= 0 and self.model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                self._qadr[b] = int(self.model.jnt_qposadr[jadr])
        self._light = [int(i) for i in range(self.model.nlight)]
        self._base_rgba = self.model.geom_rgba.copy()

    def close(self):
        r = getattr(self, "renderer", None)
        if r is not None:
            try:
                r.close()
            except Exception:
                pass
            self.renderer = None

    def __del__(self):
        self.close()

    # ----------------------------------------------------- domain randomization
    def _randomize(self, rng):
        (x0, x1), (y0, y1) = self.region
        placed = []
        for b, qa in self._qadr.items():
            for _ in range(20):                       # rejection-sample a clear spot
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if all((x - px) ** 2 + (y - py) ** 2 > 0.075 ** 2 for px, py in placed):
                    break
            placed.append((x, y))
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            half_z = float(self.model.geom_size[self.geoms_by_body[b][0]][2]) if \
                self.model.geom_type[self.geoms_by_body[b][0]] == mujoco.mjtGeom.mjGEOM_BOX else \
                float(self.model.geom_size[self.geoms_by_body[b][0]][0])
            yaw = rng.uniform(-np.pi, np.pi)
            self.data.qpos[qa:qa + 3] = [x, y, self.table_top + half_z + 0.001]
            self.data.qpos[qa + 3:qa + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.data.qvel[self.model.jnt_dofadr[self.model.body_jntadr[bid]]:][:6] = 0.0
        # lighting + background brightness
        for li in self._light:
            self.model.light_diffuse[li] = rng.uniform(0.35, 0.85, 3)
            self.model.light_pos[li] = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1.5, 3)]
        # small per-object brightness jitter (keep hue/class identity)
        for b in self.classes:
            for g in self.geoms_by_body[b]:
                self.model.geom_rgba[g, :3] = np.clip(
                    self._base_rgba[g, :3] * rng.uniform(0.8, 1.2), 0.02, 1.0)
        mujoco.mj_forward(self.model, self.data)

    def _random_camera(self, rng):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cx, cy = np.mean(self.region[0]), np.mean(self.region[1])
        cam.lookat[:] = [cx + rng.uniform(-0.05, 0.05),
                         cy + rng.uniform(-0.05, 0.05), self.table_top + 0.02]
        cam.distance = rng.uniform(0.55, 0.9)
        cam.azimuth = rng.uniform(60, 120)
        cam.elevation = rng.uniform(-85, -45)         # top-ish to angled
        return cam

    # ----------------------------------------------------------------- render
    def sample(self, rng):
        """One randomized (rgb, boxes) sample. ``boxes`` = list of (class_idx, bbox)."""
        self._randomize(rng)
        cam = self._random_camera(rng)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=cam)
        rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=cam)
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        boxes = []
        for body, bb in segmentation_bboxes(seg, self.geoms_by_body).items():
            boxes.append((self.class_idx[self.classes[body]], bb))
        return rgb, boxes

    def generate(self, out_dir, n=1000, seed=0, val_frac=0.15):
        """Write a YOLO dataset: ``out_dir``/{images,labels}/{train,val}/ + data.yaml.
        Returns the path to data.yaml."""
        import imageio.v2 as imageio
        rng = np.random.default_rng(seed)
        n_val = max(1, int(n * val_frac))
        for split in ("train", "val"):
            os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
        for i in range(n):
            split = "val" if i < n_val else "train"
            rgb, boxes = self.sample(rng)
            h, w = rgb.shape[:2]
            imageio.imwrite(os.path.join(out_dir, "images", split, f"{i:06d}.png"), rgb)
            lines = []
            for cls, (u0, v0, u1, v1) in boxes:
                cx, cy = (u0 + u1) / 2 / w, (v0 + v1) / 2 / h
                bw, bh = (u1 - u0) / w, (v1 - v0) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            with open(os.path.join(out_dir, "labels", split, f"{i:06d}.txt"), "w") as f:
                f.write("\n".join(lines))
        yaml_path = os.path.join(out_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"path: {os.path.abspath(out_dir)}\n")
            f.write("train: images/train\nval: images/val\n")
            f.write(f"nc: {len(self.class_names)}\n")
            f.write("names: [" + ", ".join(f"'{n}'" for n in self.class_names) + "]\n")
        return yaml_path


# Default object->class map for the bimanual table scene (the main detection scene).
TABLE_CLASSES = {"block_red": "red block", "block_green": "green block",
                 "block_blue": "blue block", "block_orange": "orange block"}


def main(argv=None):
    """CLI: generate an auto-labeled synthetic detection dataset (`openarm gen-data`)."""
    from ..config import BIMANUAL_TABLE_SCENE
    ap = argparse.ArgumentParser(description="Generate synthetic, segmentation-labeled "
                                             "detection data under domain randomization.")
    ap.add_argument("--scene", default=BIMANUAL_TABLE_SCENE, help="scene XML to render")
    ap.add_argument("--out", default="datasets/openarm", help="output dataset dir")
    ap.add_argument("--n", type=int, default=1000, help="number of images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    gen = SyntheticDataGenerator(args.scene, TABLE_CLASSES)
    print(f"classes: {gen.class_names}")
    print(f"generating {args.n} images -> {args.out} ...")
    yaml_path = gen.generate(args.out, n=args.n, seed=args.seed)
    gen.close()
    print(f"done. dataset: {yaml_path}")
    print("next: openarm detect-train --data " + yaml_path)


if __name__ == "__main__":
    main()
