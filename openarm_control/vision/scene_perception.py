"""Scene perception: detect objects in a camera view and locate them in 3D.

Ties a :class:`Camera` (RGB + depth) to an :class:`ObjectDetector`: render the
view, detect objects (open-vocabulary or the colour/shape fallback), and
deproject each detection to a 3D world position. ``ground(query)`` resolves a
text request like "the pencil" or "red box" to a single located object — the
bridge from a language command to a graspable target.

The controller only ever sees what perception returns (label + 3D position),
never an object's true MuJoCo pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .detector import ColorShapeDetector


@dataclass
class SceneObject:
    label: str
    score: float
    position: np.ndarray             # 3D world position (m)
    bbox: tuple                      # pixel bbox (umin, vmin, umax, vmax)

    def __repr__(self):
        p = np.round(self.position, 3)
        return f"SceneObject({self.label!r}, score={self.score:.2f}, pos={p})"


class ScenePerception:
    def __init__(self, model, data, cam_name="tablecam", detector=None,
                 width=640, height=480):
        self.camera = Camera(model, data, cam_name, width=width, height=height)
        self.detector = detector if detector is not None else ColorShapeDetector()

    def close(self):
        """Free the camera's GL context (see Camera.close)."""
        cam = getattr(self, "camera", None)
        if cam is not None:
            cam.close()

    def __del__(self):
        self.close()

    def _deproject(self, depth, centroid, bbox):
        """3D world point for a detection, from the median depth of a small patch
        at its centroid (robust to a stray background pixel)."""
        u, v = centroid
        ui, vi = int(round(u)), int(round(v))
        h, w = depth.shape
        r = 3
        patch = depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
        vals = patch[np.isfinite(patch) & (patch > 1e-3)]
        if vals.size == 0:
            return None
        d = float(np.median(vals))
        return self.camera.deproject(u, v, d)

    def perceive(self, queries=None):
        """Detect objects and return them located in 3D (a list of SceneObject)."""
        rgb = self.camera.rgb()
        depth = self.camera.depth()
        objs = []
        for det in self.detector.detect(rgb, queries=queries):
            pos = self._deproject(depth, det.centroid, det.bbox)
            if pos is not None:
                objs.append(SceneObject(det.label, det.score, pos, det.bbox))
        return objs

    def ground(self, query, objects=None):
        """Resolve a text query to the single best-matching located object, or
        ``None``. Open-vocab: the query restricts detection; fallback: the query's
        words (colour/shape) are matched against the labels."""
        objs = objects if objects is not None else self.perceive(queries=[query])
        return _ground(query, objs)


def _ground(query, objs):
    """Pick the located object whose label best matches the query's words."""
    toks = [t for t in query.lower().replace("the", "").split() if t]
    scored = []
    for o in objs:
        label = o.label.lower()
        hits = sum(t in label for t in toks)
        if hits:
            scored.append((hits, o.score, o))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


class SegmentClassifyPerception:
    """Perception for small objects spread on a large table (e.g. scanned meshes).

    A single top-down view makes 6 cm objects tiny, which open-vocab detectors miss.
    Instead this **localises geometrically then classifies locally**: it segments the
    objects from the **depth** image (pixels that stick up above the table surface
    but below the arms), and for each region runs the open-vocabulary detector on a
    **tight crop** (where the object fills the frame and is recognised confidently).
    Bins (also raised) are excluded by their known positions.
    """

    def __init__(self, model, data, cam_name="tablecam", detector=None, vocab=None,
                 table_z=0.40, width=1280, height=960, exclude_xy=(), exclude_r=0.15,
                 min_area=150, x_range=(0.14, 0.44), y_abs=0.44):
        self.camera = Camera(model, data, cam_name, width=width, height=height)
        self.detector = detector
        self.vocab = list(vocab) if vocab else None
        self.table_z = table_z
        self.exclude_xy = [np.asarray(e, float) for e in exclude_xy]
        self.exclude_r = exclude_r
        self.min_area = min_area
        self.x_range = x_range            # keep only regions over the table (drop arm bases)
        self.y_abs = y_abs

    def close(self):
        """Free the camera's GL context (see Camera.close)."""
        cam = getattr(self, "camera", None)
        if cam is not None:
            cam.close()

    def __del__(self):
        self.close()

    def perceive(self, queries=None):
        from scipy import ndimage
        rgb = self.camera.rgb()
        depth = self.camera.depth()
        cam_z = float(self.camera.pose()[0][2])
        world_z = cam_z - depth                      # top-down camera: z ~ cam_z - depth
        mask = (world_z > self.table_z + 0.008) & (world_z < self.table_z + 0.18)
        labels, n = ndimage.label(mask)
        objs = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(labels == i)
            if xs.size < self.min_area:
                continue
            u0, u1, v0, v1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            dc = float(np.median(depth[labels == i]))
            pos = self.camera.deproject(uc, vc, dc)
            if not (self.x_range[0] < pos[0] < self.x_range[1] and abs(pos[1]) < self.y_abs):
                continue                              # off the table (arm base, etc.)
            if any(np.linalg.norm(pos[:2] - e) < self.exclude_r for e in self.exclude_xy):
                continue                              # skip bins / fixtures
            label, score = "object", 0.0
            if self.detector is not None:
                # a generous square crop centred on the region, so the WHOLE object
                # (not just the depth-masked top) is classified.
                half = int(max(u1 - u0, v1 - v0) * 0.9) + 30
                cu, cv = int(uc), int(vc)
                crop = rgb[max(0, cv - half):cv + half, max(0, cu - half):cu + half]
                dets = self.detector.detect(crop, queries=self.vocab)
                if dets:
                    best = max(dets, key=lambda d: d.score)
                    label, score = best.label, best.score
            objs.append(SceneObject(label, score, pos, (u0, v0, u1, v1)))
        return objs

    def ground(self, query, objects=None):
        objs = objects if objects is not None else self.perceive()
        return _ground(query, objs)
