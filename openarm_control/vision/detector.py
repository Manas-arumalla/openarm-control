"""Object detectors behind one interface, so the perception stack is agnostic to
how objects are recognised.

* ``OpenVocabDetector`` — open-vocabulary detection (YOLO-World via ultralytics):
  detect *any* object named by a text prompt ("pencil", "mug", "banana"). This is
  the real everyday-object detector; ultralytics is imported lazily so the
  platform never hard-depends on it.
* ``ColorShapeDetector`` — dependency-free fallback that labels objects by colour
  + coarse shape ("red box", "green ball"). Lets the whole perceive→ground→plan
  pipeline be exercised headlessly without the heavy model (and is a sane baseline
  for the simple sim scenes).

Both return a list of ``Detection`` (label, score, pixel bbox). The
``ScenePerception`` layer turns those into 3D-located ``SceneObject``s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import COLOR_RULES


@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple                      # (umin, vmin, umax, vmax) in pixels

    @property
    def centroid(self):
        u0, v0, u1, v1 = self.bbox
        return (0.5 * (u0 + u1), 0.5 * (v0 + v1))

    @property
    def area(self):
        u0, v0, u1, v1 = self.bbox
        return max(0.0, (u1 - u0)) * max(0.0, (v1 - v0))


class ObjectDetector:
    """Interface: ``detect(rgb, queries=None) -> list[Detection]``."""

    def detect(self, rgb, queries=None):
        raise NotImplementedError


class OpenVocabDetector(ObjectDetector):
    """Open-vocabulary detection via ultralytics (lazy import), text-prompted.

    ``queries`` is a list of text prompts (the object names you care about);
    detection is restricted to them. Recognises arbitrary everyday objects, so a
    command like "pick the mug" can be grounded directly.

    Two backends:
      * ``"yolo-world"`` (default) — YOLO-World; light, and in our top-down sim
        renders it scored higher (e.g. a scanned mug -> "cup" @ 0.48).
      * ``"yoloe"`` — YOLOE (2025, "Real-Time Seeing Anything"); generally stronger
        on open-vocab benchmarks (esp. its visual-prompt / prompt-free modes), but
        its text-prompt path pulls a ~570 MB MobileCLIP encoder and scored a touch
        lower here (mug -> "cup" @ 0.30). Pick per your needs.
    """

    def __init__(self, model=None, conf=0.03, default_vocab=None, backend="yolo-world"):
        self.backend = backend.lower().replace("_", "-")
        if self.backend in ("yoloe", "yolo-e"):
            from ultralytics import YOLOE          # lazy, optional dependency
            self.model = YOLOE(model or "yoloe-11s-seg.pt")
            self.backend = "yoloe"
        else:
            from ultralytics import YOLOWorld      # lazy, optional dependency
            self.model = YOLOWorld(model or "yolov8s-world.pt")
            self.backend = "yolo-world"
        self.conf = float(conf)
        self._vocab = None
        if default_vocab:
            self.set_vocab(default_vocab)

    def set_vocab(self, classes):
        classes = list(classes)
        if self.backend == "yoloe":                # YOLOE needs text-prompt embeddings
            self.model.set_classes(classes, self.model.get_text_pe(classes))
        else:
            self.model.set_classes(classes)
        self._vocab = classes

    def detect(self, rgb, queries=None):
        if queries:
            self.set_vocab(queries)
        if self._vocab is None:
            raise ValueError("OpenVocabDetector needs a vocabulary (queries=[...])")
        res = self.model.predict(rgb, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            label = self._vocab[int(b.cls)] if self._vocab else res.names[int(b.cls)]
            out.append(Detection(label, float(b.conf[0]),
                                 tuple(float(x) for x in b.xyxy[0].tolist())))
        return out


def _fill_shape(mask_crop):
    """Coarse shape from how fully the blob fills its bbox: a square footprint
    (~1.0) reads as 'box', a round footprint (~pi/4) as 'ball'."""
    h, w = mask_crop.shape
    if h == 0 or w == 0:
        return "object"
    fill = float(mask_crop.sum()) / (h * w)
    aspect = max(h, w) / max(1, min(h, w))
    if aspect > 2.2:
        return "stick"                       # long thin object (pencil-like)
    return "box" if fill > 0.82 else "ball"


class ColorShapeDetector(ObjectDetector):
    """Dependency-free fallback: connected colour blobs labelled by colour + shape
    ("red box", "green ball"). Used for headless tests and as a baseline."""

    def __init__(self, colors=("red", "green", "blue", "orange"), min_area=40):
        self.colors = [c for c in colors if c in COLOR_RULES]
        self.min_area = min_area

    def detect(self, rgb, queries=None):
        from scipy import ndimage
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        masks = {c: COLOR_RULES[c](r, g, b) for c in self.colors}
        # Resolve overlap: orange also satisfies the red rule, so orange wins
        # those pixels (otherwise an orange object is double-detected as red too).
        if "red" in masks and "orange" in masks:
            masks["red"] = masks["red"] & ~masks["orange"]
        out = []
        for color in self.colors:
            mask = masks[color]
            labels, n = ndimage.label(mask)
            for i in range(1, n + 1):
                ys, xs = np.nonzero(labels == i)
                if xs.size < self.min_area:
                    continue
                u0, u1, v0, v1 = xs.min(), xs.max(), ys.min(), ys.max()
                shape = _fill_shape((labels == i)[v0:v1 + 1, u0:u1 + 1])
                out.append(Detection(f"{color} {shape}", 1.0,
                                     (float(u0), float(v0), float(u1), float(v1))))
        return out
