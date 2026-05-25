#!/usr/bin/env python3
"""Pre-fetch the optional open-vocabulary detector weights.

The core platform needs none of these — the default detector is a dependency-free
colour/shape heuristic. These are only for the optional open-vocab vision path
(`openarm interactive --detector yolo-world|yoloe`). ultralytics normally
auto-downloads them on first use; this script just triggers that ahead of time
(useful before going offline).

    python scripts/fetch_models.py            # YOLO-World + YOLOE weights (~53 MB)
    python scripts/fetch_models.py --yoloe    # also the ~572 MB MobileCLIP encoder

See docs/MODELS.md for the full picture (datasets, trained policies, provenance).
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yoloe", action="store_true",
                    help="also trigger YOLOE's ~572 MB MobileCLIP text encoder")
    args = ap.parse_args(argv)

    try:
        from ultralytics import YOLOWorld
    except ImportError:
        sys.exit("ultralytics not installed. Install the vision extra:\n"
                 "    pip install -e \".[vision]\"")

    print("Fetching YOLO-World weights (yolov8s-world.pt, ~26 MB)…")
    YOLOWorld("yolov8s-world.pt")
    print("  done.")

    if args.yoloe:
        from ultralytics import YOLOE
        print("Fetching YOLOE weights + MobileCLIP encoder (~600 MB total)…")
        model = YOLOE("yoloe-11s-seg.pt")
        # The MobileCLIP encoder downloads when text prompts are first embedded.
        try:
            model.set_classes(["cube"], model.get_text_pe(["cube"]))
        except Exception as e:  # noqa: BLE001 - best-effort pre-fetch
            print(f"  (text-encoder pre-fetch skipped: {e})")
        print("  done.")

    print("\nAll requested weights are cached. See docs/MODELS.md for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
