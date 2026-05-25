"""Intelligent manipulation layer: turn a spoken/typed command into action.

    parse_command("pick the red cube and put it in the bin")
        -> Intent(action='place', target='red cube', destination='bin')

The executor (M3) grounds the target via vision, plans a collision-free motion,
and acts; the session adds stateful, multi-step conversational control.

(The closed-loop reactive object-following layer was removed for now — it will be
re-implemented properly once the rest of the stack is stable.)
"""
from .commands import Intent, parse_command
from .executor import TaskExecutor
from .session import ManipulationSession

__all__ = ["Intent", "parse_command", "TaskExecutor", "ManipulationSession"]
