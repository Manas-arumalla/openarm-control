"""Phase I / M1 — multi-object scene perception & grounding.

Headless tests use the dependency-free colour/shape fallback detector so the full
perceive -> locate -> ground pipeline is verifiable without the heavy open-vocab
model. The open-vocab detector (YOLO-World) is exercised only for import safety.
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import TABLETOP_SCENE
from openarm_control.vision import (ScenePerception, ColorShapeDetector,
                                    OpenVocabDetector, ObjectDetector)

# object body -> the colour word that identifies it
OBJECTS = {"ball_red": "red", "box_green": "green",
           "can_blue": "blue", "ball_orange": "orange"}


def _scene():
    model = mujoco.MjModel.from_xml_path(TABLETOP_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _perception(model, data):
    return ScenePerception(model, data, "tablecam", detector=ColorShapeDetector())


def test_tabletop_scene_loads():
    model, data = _scene()
    for body in OBJECTS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin") >= 0


def test_fallback_perception_locates_objects():
    """Every object is detected once and located in 3D near its true position."""
    model, data = _scene()
    objs = _perception(model, data).perceive()
    for body, color in OBJECTS.items():
        bp = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)]
        matches = [o for o in objs if color in o.label]
        assert matches, f"{color} object not detected"
        xy_err = np.linalg.norm(matches[0].position[:2] - bp[:2])
        assert xy_err < 0.02, f"{color}: xy error {xy_err*1000:.0f} mm"


def test_grounding_resolves_queries():
    """A text query resolves to the correct located object."""
    model, data = _scene()
    sp = _perception(model, data)
    objs = sp.perceive()
    for query, body in [("red ball", "ball_red"), ("the blue can", "can_blue"),
                        ("green box", "box_green"), ("orange ball", "ball_orange")]:
        bp = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)]
        obj = sp.ground(query, objects=objs)
        assert obj is not None, f"could not ground {query!r}"
        assert np.linalg.norm(obj.position[:2] - bp[:2]) < 0.03, \
            f"{query!r} grounded to the wrong object"


def test_openvocab_detector_is_an_object_detector():
    """The open-vocab detector slots into the same interface (not constructed —
    that downloads a large model; the live demo does that)."""
    pytest.importorskip("ultralytics")
    assert issubclass(OpenVocabDetector, ObjectDetector)
    assert hasattr(OpenVocabDetector, "set_vocab") and hasattr(OpenVocabDetector, "detect")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
