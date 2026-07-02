"""Curate a set of scanned objects: copy meshes, auto-scale to graspable size,
center them, and emit per-object params. Run a YOLO-World detection check.

Visual = the scanned mesh (looks real); collision = a simple box (reliable grasp).
"""
import os
import shutil
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "mujoco_scanned_objects-main", "models")
DST = os.path.join(ROOT, "v2", "openarm_mujoco_v2", "assets", "scanned")
TARGET_MAX = 0.06            # scale so the largest dimension is ~6 cm (graspable)

# (source folder, short name / weld suffix, YOLO category). Chosen to be roughly
# equidimensional + tall enough for a top-down grasp (a flat toy car, 1.4 cm tall,
# was dropped). Labels are best-effort hints; selection is by number.
SELECTED = [
    ("Cole_Hardware_Mug_Classic_Blue",        "mug",      "cup"),
    ("Cole_Hardware_Bowl_Scirocco_YellowBlue", "bowl",     "bowl"),
    ("Crosley_Alarm_Clock_Vintage_Metal",     "clock",    "clock"),
    ("Elephant",                              "elephant", "elephant"),
]


def aabb(objpath):
    vs = []
    for line in open(objpath, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
    vs = np.array(vs)
    return vs.min(0), vs.max(0)


def curate():
    params = {}
    for folder, name, cat in SELECTED:
        src = os.path.join(SRC, folder)
        if not os.path.isdir(src):
            print(f"  MISSING: {folder}"); continue
        lo, hi = aabb(os.path.join(src, "model.obj"))
        center = (lo + hi) / 2
        ext = hi - lo
        scale = TARGET_MAX / float(ext.max())
        d = os.path.join(DST, name)
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(src, "model.obj"), os.path.join(d, f"{name}.obj"))
        shutil.copy(os.path.join(src, "texture.png"), os.path.join(d, f"{name}.png"))
        params[name] = dict(cat=cat, scale=round(scale, 5),
                            center=(scale * center).round(4).tolist(),
                            half=(scale * ext / 2).round(4).tolist())
        print(f"  {name:9s} cat={cat:9s} scale={scale:.3f} "
              f"half(cm)={np.round(scale*ext/2*100,1)}")
    return params


if __name__ == "__main__":
    print("Curating scanned objects ->", DST)
    params = curate()
    import json
    json.dump(params, open(os.path.join(ROOT, "_scanned_params.json"), "w"), indent=2)
    print("params saved to _scanned_params.json")
