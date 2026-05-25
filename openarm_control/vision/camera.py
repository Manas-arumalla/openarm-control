"""Offscreen camera rendering + pinhole geometry for a MuJoCo camera.

Renders RGB and depth from a named camera and provides the intrinsics and the
deprojection (pixel + depth -> 3D world point) needed for position-based visual
servoing. The controller only ever sees what the camera produces — it never
reads an object's true MuJoCo pose.
"""

import mujoco
import numpy as np


class Camera:
    def __init__(self, model, data, cam_name, width=320, height=240):
        self.model = model
        self.data = data
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id < 0:
            raise ValueError(f"camera '{cam_name}' not found")
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        # Pinhole intrinsics from the vertical FOV.
        fovy = float(model.cam_fovy[self.cam_id])
        self.f = (height / 2.0) / np.tan(np.radians(fovy) / 2.0)
        self.cx = width / 2.0
        self.cy = height / 2.0

    def close(self):
        """Free the offscreen GL context. Without this, creating many cameras (one
        per perception, across a long-running process or a full test suite) leaks GL
        contexts/framebuffers until later renders return garbage. Safe to call twice."""
        r = getattr(self, "renderer", None)
        if r is not None:
            try:
                r.close()
            except Exception:
                pass
            self.renderer = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def rgb(self):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

    def depth(self):
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.cam_id)
        d = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return d

    def pose(self):
        """Camera world position and rotation (columns = camera x,y,z axes)."""
        return (self.data.cam_xpos[self.cam_id].copy(),
                self.data.cam_xmat[self.cam_id].reshape(3, 3).copy())

    def deproject(self, u, v, depth):
        """Pixel (u,v) + depth (m) -> 3D world point.

        MuJoCo camera convention: x right, y up, looking down -z.
        Image rows (v) increase downward, so v maps to -y.
        """
        x = (u - self.cx) * depth / self.f
        y = -(v - self.cy) * depth / self.f
        z = -depth
        pos, R = self.pose()
        return pos + R @ np.array([x, y, z])
