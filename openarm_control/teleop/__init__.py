"""Webcam human-arm imitation (teleoperation).

Pose source (webcam/synthetic) -> arm retargeting -> safe real-time teleop.

    from openarm_control.teleop import (
        ArmLandmarks, ScriptedPoseSource, WebcamPoseSource,
        ArmRetargeter, TeleopController,
    )
"""
from .pose import ArmLandmarks, PoseSource, ScriptedPoseSource, WebcamPoseSource
from .retarget import ArmRetargeter
from .teleop import TeleopController

__all__ = [
    "ArmLandmarks", "PoseSource", "ScriptedPoseSource", "WebcamPoseSource",
    "ArmRetargeter", "TeleopController",
]
