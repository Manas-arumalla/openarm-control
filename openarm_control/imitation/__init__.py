"""Imitation learning: collect scripted demonstrations and train a behavior-
cloning policy that imitates them — comparable head-to-head with the RL policy
(same observation/action space as the Gymnasium envs)."""

from .expert import ReachExpert, make_env_and_expert, TASKS

__all__ = ["ReachExpert", "make_env_and_expert", "TASKS"]
