"""Position-based visual servoing (PBVS).

Estimates a target object's 3D world position purely from a camera (color
detection + depth deprojection) and drives a Cartesian controller toward it.
The controller never reads the object's true MuJoCo pose — only the
camera-derived estimate — so this is genuine vision-in-the-loop control.
"""

import numpy as np

from .detection import detect_color


class VisualServo:
    def __init__(self, model, data, camera, controller, color,
                 approach_offset=(0.0, 0.0, 0.0)):
        self.model = model
        self.data = data
        self.camera = camera
        self.controller = controller
        self.color = color
        self.approach_offset = np.asarray(approach_offset, dtype=float)
        self.last_estimate = None

    def estimate_target(self):
        """Return the camera-estimated 3D world position of the object, or None."""
        rgb = self.camera.rgb()
        det = detect_color(rgb, self.color)
        if det is None:
            return None
        u, v = det["centroid"]
        depth = self.camera.depth()
        z = float(depth[int(round(v)), int(round(u))])
        if not np.isfinite(z) or z <= 0.0 or z > 10.0:
            return None
        p = self.camera.deproject(u, v, z)
        self.last_estimate = p
        return p

    def step(self):
        """Re-estimate the target from vision and take one control step."""
        p = self.estimate_target()
        if p is not None:
            self.controller.set_target(p + self.approach_offset)
        return self.controller.step()

    def converged(self, tol=0.02):
        if self.last_estimate is None:
            return False
        cur, _ = self.controller.kin.forward_kinematics()
        return np.linalg.norm((self.last_estimate + self.approach_offset) - cur) < tol
