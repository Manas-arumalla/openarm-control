"""Fine-tune / evaluate a detector on the synthetic sim dataset (ultralytics).

The dataset comes from ``synthgen`` (segmentation-labeled, domain-randomized). The
actual training is **user-run** (it needs a GPU and minutes-to-hours, like the RL/BC
pipelines); ultralytics is a lazy, optional dependency. After training, point the
existing ``OpenVocabDetector(model="…/best.pt")`` (or a plain ``ultralytics.YOLO``)
at the fine-tuned weights -- detection of the sim's objects becomes high-confidence.

    openarm gen-data --out datasets/openarm --n 2000      # make the labeled data
    openarm detect-train --data datasets/openarm/data.yaml --epochs 80
    openarm detect-eval  --weights runs/openarm/.../best.pt --data datasets/openarm/data.yaml
"""
from __future__ import annotations

import argparse


def train(data_yaml, model="yolov8n.pt", epochs=80, imgsz=640, batch=16,
          project="runs/openarm", name="finetune", **kw):
    """Fine-tune ``model`` on the YOLO dataset at ``data_yaml``. Returns the run dir."""
    from ultralytics import YOLO            # lazy / optional
    yolo = YOLO(model)
    res = yolo.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                     project=project, name=name, **kw)
    return getattr(res, "save_dir", project)


def evaluate(weights, data_yaml, imgsz=640):
    """Validate ``weights`` on the dataset; returns a dict of detection metrics."""
    from ultralytics import YOLO            # lazy / optional
    m = YOLO(weights)
    metrics = m.val(data=data_yaml, imgsz=imgsz)
    box = metrics.box
    return {"map50_95": float(box.map), "map50": float(box.map50),
            "precision": float(box.mp), "recall": float(box.mr)}


def _train_main(argv):
    ap = argparse.ArgumentParser(prog="openarm detect train",
                                 description="Fine-tune a detector on synthetic sim data.")
    ap.add_argument("--data", required=True, help="path to data.yaml (from `openarm gen-data`)")
    ap.add_argument("--model", default="yolov8n.pt", help="base weights to fine-tune")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)
    out = train(args.data, model=args.model, epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch)
    print(f"trained -> {out}")


def _eval_main(argv):
    ap = argparse.ArgumentParser(prog="openarm detect eval",
                                 description="Evaluate a fine-tuned detector.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args(argv)
    m = evaluate(args.weights, args.data, imgsz=args.imgsz)
    print("detection metrics:")
    for k, v in m.items():
        print(f"  {k:10s} {v:.3f}")


def main(argv=None):
    """`openarm detect train|eval ...` -- fine-tune or evaluate on synthetic data."""
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    sub = argv[0] if argv else "train"
    rest = argv[1:] if argv and argv[0] in ("train", "eval") else argv
    if sub == "eval":
        return _eval_main(rest)
    return _train_main(rest)


if __name__ == "__main__":
    main()
