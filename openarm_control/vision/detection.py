"""Simple, dependency-free color object detection on an RGB image.

Uses channel-dominance rules (robust for saturated colored objects on a neutral
background, which is what the sim scenes use). Returns the object's pixel
centroid, bounding box, and pixel area. A YOLO-based detector can slot in behind
the same interface later (Phase C stretch goal).
"""

import numpy as np

# (channel-dominance predicate) per color, on a uint8 HxWx3 RGB array.
COLOR_RULES = {
    "red":   lambda r, g, b: (r > 110) & (r.astype(int) - g > 50) & (r.astype(int) - b > 50),
    "green": lambda r, g, b: (g > 90) & (g.astype(int) - r > 30) & (g.astype(int) - b > 30),
    "blue":  lambda r, g, b: (b > 90) & (b.astype(int) - r > 30) & (b.astype(int) - g > 30),
    # warm orange ball: R high, R>G>B with a clear gap to blue (distinct from the
    # grey arm, blue sky, and checker floor).
    "orange": lambda r, g, b: (r > 140) & (r.astype(int) - b > 90)
              & (g.astype(int) - b > 30) & (r.astype(int) - g > 30),
}


def detect_color_blobs(rgb, color, k=2, min_area=6):
    """Detect up to ``k`` separate blobs of ``color`` -> list of (u, v) centroids,
    largest first. Connected-component labelling, so two same-coloured objects in
    the frame are returned as distinct detections (multi-object perception)."""
    from scipy import ndimage
    if color not in COLOR_RULES:
        raise ValueError(f"unknown color '{color}'")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = COLOR_RULES[color](r, g, b)
    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    idx = list(range(1, n + 1))
    areas = ndimage.sum(np.ones_like(labels), labels, index=idx)
    cents = ndimage.center_of_mass(mask, labels, index=idx)      # (row=v, col=u)
    blobs = [((float(u), float(v)), float(a))
             for (v, u), a in zip(cents, np.atleast_1d(areas)) if a >= min_area]
    blobs.sort(key=lambda ba: -ba[1])
    return [c for c, _ in blobs[:k]]


def detect_color(rgb, color, min_area=15):
    """Detect the largest blob of `color`. Returns dict or None.

    dict: {centroid: (u, v), bbox: (umin, vmin, umax, vmax), area: int, mask: HxW bool}
    """
    if color not in COLOR_RULES:
        raise ValueError(f"unknown color '{color}' (have {list(COLOR_RULES)})")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = COLOR_RULES[color](r, g, b)
    ys, xs = np.nonzero(mask)
    if xs.size < min_area:
        return None
    u = float(xs.mean())
    v = float(ys.mean())
    return {
        "centroid": (u, v),
        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        "area": int(xs.size),
        "mask": mask,
    }
