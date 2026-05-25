# Models, weights & datasets

This repo deliberately versions **only small, project-owned artifacts** and keeps
large, third-party, or regenerable files out of git. Everything below is either
**auto-downloaded on first use** or **reproducible with one command**, so a fresh
clone is fully functional.

## What ships in the repo

| Artifact | Size | Why it's versioned |
|---|---|---|
| `demos/reach_act.pt` | ~5 MB | Trained **ACT** vision policy — lets `openarm act eval` run out-of-the-box. |
| `demos/reach_bc.pt`, `demos/insert_bc.pt` | ~0.3 MB each | Trained **behaviour-cloning** policies — `openarm bc-eval` / insertion comparison run immediately. |
| `media/*.png`, `media/*.gif` | small | Showcase screenshots / GIFs. |
| `benchmarks/figures/*.png`, `benchmarks/results/*.csv` | small | Reproducible benchmark plots & tables. |

## What is NOT versioned (and how to get it)

### 1. Open-vocabulary detector weights (optional — vision feature)

The default object detector is a **dependency-free colour/shape heuristic**, so the
core platform needs none of these. They are only used by the *optional*
open-vocabulary path (`openarm interactive --detector yolo-world|yoloe`,
`openarm manipulate --vision`). [ultralytics](https://docs.ultralytics.com)
**auto-downloads them on first use** — no manual step required.

| File | Size | Fetched by | Source |
|---|---|---|---|
| `yolov8s-world.pt` | ~26 MB | `YOLOWorld("yolov8s-world.pt")` | ultralytics |
| `yoloe-11s-seg.pt` | ~27 MB | `YOLOE("yoloe-11s-seg.pt")` | ultralytics |
| `mobileclip_blt.ts` | ~572 MB | YOLOE text-prompt path (MobileCLIP encoder) | ultralytics |

To pre-fetch them ahead of time (e.g. for an offline machine):

```bash
python scripts/fetch_models.py            # downloads the YOLO-World + YOLOE weights
python scripts/fetch_models.py --yoloe    # also triggers the ~572 MB MobileCLIP encoder
```

### 2. Demonstration datasets (regenerable)

The `demos/*.npz` scripted-demo datasets are **not** versioned because they are
regenerated deterministically by the scripted experts:

```bash
openarm bc-collect --task reach  --episodes 200 --out demos/reach.npz
openarm bc-collect --task reach  --episodes 200 --images --out demos/reach_vis.npz   # +camera images, for ACT
openarm bc-collect --task insert --episodes 200 --out demos/insert.npz
```

Then retrain (optional — trained weights already ship, see above):

```bash
openarm bc-train  --task reach  --demos demos/reach.npz  --out demos/reach_bc.pt
openarm act train --demos demos/reach_vis.npz --out demos/reach_act.pt
openarm bc-train  --task insert --demos demos/insert.npz --out demos/insert_bc.pt
```

## Provenance

The **OpenArm v2 MuJoCo model** under `v2/openarm_mujoco_v2/` (meshes + MJCF) is by
**Enactic, Inc.** (Apache-2.0). The older `v0.3/` and `v1/` model trees are kept for
provenance. The YOLO-World / YOLOE / MobileCLIP weights are third-party
(ultralytics ecosystem) under their respective licenses.
