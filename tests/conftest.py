"""Shared pytest fixtures for the OpenArm control test suite."""
import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import SINGLE_ARM_SCENE


@pytest.fixture
def make_sim():
    """Factory returning a fresh (model, data) so tests never share mutable state."""
    def _make():
        model = mujoco.MjModel.from_xml_path(SINGLE_ARM_SCENE)
        data = mujoco.MjData(model)
        return model, data
    return _make


@pytest.fixture
def sim(make_sim):
    return make_sim()


def reset_to(model, data, keyframe="ready"):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)


def disable_obstacle(model):
    """No-op if the scene has no obstacle (the default sorting scene)."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_wall_geom")
    if gid != -1:
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
