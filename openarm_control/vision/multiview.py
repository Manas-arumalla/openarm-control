"""Multi-view perception: fuse detections from two cameras (top-down + angled).

A single top-down view is ambiguous -- a box and a ball look alike from straight
above, and small objects are tiny -- so identity from one view alone is unreliable.
This renders each camera, detects + deprojects per view, then **fuses by 3D
proximity**: detections of the same physical object across views are merged, and the
**most confident label wins** (the angled view usually disambiguates what the
top-down view can't). Drop-in alternative to ``ScenePerception`` (same ``perceive``
contract, same ``SceneObject``).

Uses a SINGLE MuJoCo renderer, rendering each camera in turn -- never two live GL
contexts at once (which can conflict), and cheaper than one renderer per camera.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .detector import ColorShapeDetector
from .scene_perception import SceneObject


class MultiViewPerception:
    def __init__(self, model, data, cam_names=("tablecam", "frontcam"), detector=None,
                 width=640, height=480, fuse_tol=0.08):
        self.model, self.data = model, data
        self.detector = detector if detector is not None else ColorShapeDetector()
        self.width, self.height = width, height
        self.fuse_tol = fuse_tol
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cams = []                                   # (cam_id, focal, cx, cy)
        for name in cam_names:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                continue
            fovy = float(model.cam_fovy[cid])
            f = (height / 2.0) / np.tan(np.radians(fovy) / 2.0)
            self.cams.append((cid, f, width / 2.0, height / 2.0))
        if not self.cams:
            raise ValueError(f"no cameras found among {cam_names}")

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

    # ----------------------------------------------------------- per-camera
    def _render(self, cid):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=cid)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=cid)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def _deproject(self, cid, f, cx, cy, u, v, depth):
        x = (u - cx) * depth / f
        y = -(v - cy) * depth / f
        z = -depth
        pos = self.data.cam_xpos[cid].copy()
        R = self.data.cam_xmat[cid].reshape(3, 3)
        return pos + R @ np.array([x, y, z])

    def _detect_view(self, cid, f, cx, cy, queries):
        rgb, depth = self._render(cid)
        out = []
        for det in self.detector.detect(rgb, queries=queries):
            u, v = det.centroid
            ui, vi = int(round(u)), int(round(v))
            r = 3
            patch = depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
            vals = patch[np.isfinite(patch) & (patch > 1e-3)]
            if vals.size == 0:
                continue
            pos = self._deproject(cid, f, cx, cy, u, v, float(np.median(vals)))
            out.append((det.label, det.score, pos, det.bbox))
        return out

    # ---------------------------------------------------------------- fuse
    def _fuse(self, dets):
        """Cluster detections by 3D proximity; per cluster keep the highest-score
        label (the more confident view decides identity)."""
        clusters = []                                    # each: list of (label, score, pos, bbox)
        for d in dets:
            for c in clusters:
                if np.linalg.norm(c[0][2][:2] - d[2][:2]) < self.fuse_tol:
                    c.append(d)
                    break
            else:
                clusters.append([d])
        objs = []
        for c in clusters:
            best = max(c, key=lambda x: x[1])            # highest-confidence label wins
            pos = np.mean([x[2] for x in c], axis=0)     # fuse position
            objs.append(SceneObject(best[0], best[1], pos, best[3]))
        return objs

    def perceive(self, queries=None):
        """Detect + 3D-locate objects, fused across all configured camera views."""
        dets = []
        for (cid, f, cx, cy) in self.cams:
            dets.extend(self._detect_view(cid, f, cx, cy, queries))
        return self._fuse(dets)

    def ground(self, query, objects=None):
        from .scene_perception import _ground
        objs = objects if objects is not None else self.perceive(queries=[query])
        return _ground(query, objs)
