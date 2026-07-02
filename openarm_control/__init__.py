"""
OpenArm Control Package

Kinematics, Cartesian control, trajectory generation, grasping, autonomous
pick-and-place, and dynamics utilities for the Enactic OpenArm (v2, right arm).
"""

from .config import *  # noqa: F403  (deliberate re-export of scene paths + arm configs)
from .config import RIGHT_ARM, LEFT_ARM
from .kinematics import OpenArmKinematics, orientation_error
from .controller import CartesianController
from .trajectory import JointTrajectory, CartesianTrajectory, TrajectoryExecutor
from .grasp import GraspSolver, topdown_orientation
from .grasp6 import Grasp6DOFSolver, approach_orientation
from .pick_and_place import PickPlaceController
from .autonomy import SortingTask
from .bimanual import BimanualController, ParallelSort, RelayHandoff, synchronized_move

__all__ = [
    "OpenArmKinematics",
    "orientation_error",
    "CartesianController",
    "JointTrajectory",
    "CartesianTrajectory",
    "TrajectoryExecutor",
    "GraspSolver",
    "topdown_orientation",
    "Grasp6DOFSolver",
    "approach_orientation",
    "PickPlaceController",
    "SortingTask",
    "BimanualController",
    "ParallelSort",
    "RelayHandoff",
    "synchronized_move",
    "RIGHT_ARM",
    "LEFT_ARM",
]
