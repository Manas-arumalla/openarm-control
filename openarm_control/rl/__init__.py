"""Reinforcement-learning environments for the OpenArm (Gymnasium)."""

from .reach_env import OpenArmReachEnv
from .pick_place_env import OpenArmPickPlaceEnv
from .insert_env import OpenArmInsertEnv
from .balance_env import OpenArmBalanceEnv
from .balance_residual_env import OpenArmBalanceResidualEnv

# Task name -> Gymnasium env class (used by train.py / eval.py).
TASKS = {
    "reach": OpenArmReachEnv,
    "pick": OpenArmPickPlaceEnv,
    "insert": OpenArmInsertEnv,
    "balance": OpenArmBalanceEnv,
    "balance_residual": OpenArmBalanceResidualEnv,
}

__all__ = ["OpenArmReachEnv", "OpenArmPickPlaceEnv", "OpenArmInsertEnv",
           "OpenArmBalanceEnv", "OpenArmBalanceResidualEnv", "TASKS"]
