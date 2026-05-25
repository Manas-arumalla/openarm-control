"""Reinforcement-learning environments for the OpenArm (Gymnasium)."""

from .reach_env import OpenArmReachEnv
from .pick_place_env import OpenArmPickPlaceEnv
from .insert_env import OpenArmInsertEnv

# Task name -> Gymnasium env class (used by train.py / eval.py).
TASKS = {
    "reach": OpenArmReachEnv,
    "pick": OpenArmPickPlaceEnv,
    "insert": OpenArmInsertEnv,
}

__all__ = ["OpenArmReachEnv", "OpenArmPickPlaceEnv", "OpenArmInsertEnv", "TASKS"]
