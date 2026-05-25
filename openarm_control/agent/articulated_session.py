"""Language-commanded articulated manipulation (extension phase I1).

Wires the natural-language parser to the articulated-object skills (S3): commands
like "open the drawer", "open the cabinet door", "turn the valve" -- and multi-step
sequences ("open the drawer then turn the valve") -- are parsed and dispatched to
the :class:`ArticulatedController`. The fixtures live in ``articulated_scene.xml``.
"""
from __future__ import annotations

from .commands import parse_command, split_steps
from ..articulated import ArticulatedController


class ArticulatedSession:
    def __init__(self, model, data, viewer=None, dt_realtime=False):
        self.ac = ArticulatedController(model, data)
        self.viewer = viewer
        self.dt_realtime = dt_realtime
        self._dispatch = {
            "open_drawer": self.ac.open_drawer,
            "open_door": self.ac.open_door,
            "turn_valve": self.ac.turn_valve,
        }

    def do(self, command):
        """Parse a (possibly multi-step) command and run each articulated skill in
        order. Returns a list of (clause, ok, message)."""
        results = []
        for clause in split_steps(command):
            intent = parse_command(clause)
            if intent is None:
                results.append((clause, False, "didn't understand that"))
                continue
            fn = self._dispatch.get(intent.action)
            if fn is None:
                results.append((clause, False,
                                f"'{intent.action}' isn't an articulated skill here"))
                continue
            ok = fn(viewer=self.viewer, dt_realtime=self.dt_realtime)
            results.append((clause, ok, f"{intent.action.replace('_', ' ')}"))
        return results
