"""Stateful, multi-turn manipulation session.

Keeps just enough conversational state — what the gripper is *holding*, the *last*
object referred to, and the *last move* (for undo) — so follow-up commands work
naturally and the robot can answer questions and reverse itself:

    sess.do("pick up the blue cylinder")     # grasps and holds it
    sess.do("put it in the bin")             # "it" -> the held cylinder
    sess.do("what are you holding?")         # -> "my hand is empty"
    sess.do("undo")                          # take it back out, return it
    sess.run("stack the red cube on the green cube then put the blue cube in the bin")

Pronouns ("it", "that", "the object") resolve to the held / last object. Commands
map to executor primitives (grasp / go_to / release / stack / insert), so the arm
carries state across turns. ``run`` splits a multi-step command and does each
clause in order. No large LLM — just the keyword parser plus this state machine.
"""
from __future__ import annotations

import numpy as np

from ..config import RIGHT_ARM
from ..pick_and_place import TABLE_TOP_Z
from .commands import parse_command, split_steps, _REFERENTS
from .executor import TaskExecutor


class ManipulationSession:
    def __init__(self, model, data, perception=None, graspables=None,
                 arm=RIGHT_ARM, bin_body="bin"):
        self.ex = TaskExecutor(model, data, arm=arm, perception=perception,
                               graspables=graspables, bin_body=bin_body)
        self.last_label = None
        self._undo = None                  # (label, original_xy) of the last relocation

    @property
    def held(self):
        return self.ex.held_label

    def _resolve(self, target):
        """Resolve a referent ('it', 'object', ...) to the held / last object."""
        if target in _REFERENTS:
            return self.ex.held_label or self.last_label or target
        return target

    def _clarify(self, msg):
        """On a 'could not find X' failure, add what the robot *can* see."""
        if "could not find" in msg or "not graspable" in msg:
            vis = sorted(set(self.ex.visible()))
            if vis:
                return f"{msg}. I can see: {', '.join(vis)}."
        return msg

    # ------------------------------------------------------------------ run
    def run(self, text, viewer=None):
        """Carry out a possibly multi-step command ('do A then B then C'). Runs each
        clause in order, stopping at the first failure. Returns (ok, message)."""
        steps = split_steps(text)
        if len(steps) <= 1:
            return self.do(text, viewer=viewer)
        msgs = []
        for clause in steps:
            ok, msg = self.do(clause, viewer=viewer)
            msgs.append(msg)
            if not ok:
                return False, f"stopped: {msg} (after: {'; '.join(msgs[:-1])})"
        return True, "; ".join(msgs)

    # ------------------------------------------------------------------- do
    def do(self, text, viewer=None):
        """Parse and carry out one command in context. Returns (ok, message)."""
        intent = parse_command(text)
        if intent is None:
            return False, f"didn't understand: {text!r}"
        target = self._resolve(intent.target)
        a = intent.action

        if a == "query":
            return True, self._answer(intent.target)

        if a == "undo":
            return self._do_undo(viewer=viewer)

        if a == "pick":
            ok, msg = self.ex.grasp(target, viewer=viewer)
            if ok:
                self.last_label = self.ex.held_label
                self._undo = (self.ex.held_label, self.ex.pickup_xy)
            return ok, self._clarify(msg)

        if a == "goto":
            tgt = None if intent.destination else target
            return self.ex.go_to(dest=intent.destination, target=tgt, viewer=viewer)

        if a == "release":
            if self.ex.held_label:
                self.last_label = self.ex.held_label
            return self.ex.release(viewer=viewer)

        if a == "stack":
            support = self._resolve(intent.destination) if intent.destination else None
            if not support:
                return False, "stack on what? (e.g. 'stack the red cube on the green cube')"
            ok, msg = self.ex.stack(target, support, viewer=viewer)
            if ok:
                self.last_label = target                  # gripper is empty after a stack
                self._undo = (target, self.ex.pickup_xy)
            return ok, self._clarify(msg)

        if a == "insert":
            ok, msg = self.ex.insert(target, socket_body="socket", viewer=viewer)
            if ok:
                self.last_label = target
                self._undo = (target, self.ex.pickup_xy)
            return ok, self._clarify(msg)

        if a in ("place", "move", "remove"):
            if self.ex.held_body is None:                 # not holding -> grab it first
                ok, msg = self.ex.grasp(target, viewer=viewer)
                if not ok:
                    return ok, self._clarify(msg)
                self.last_label = self.ex.held_label
            orig_label, orig_xy = self.ex.held_label, self.ex.pickup_xy
            dest = intent.destination or "table"
            ok, msg = self.ex.go_to(dest=dest, viewer=viewer)
            if not ok:
                return ok, msg
            ok, msg = self.ex.release(viewer=viewer)
            if ok:
                self._undo = (orig_label, orig_xy)
            return (ok, f"{a} -> {dest}") if ok else (ok, msg)

        if a == "throw":
            return False, "use `openarm throw` for throwing"
        return False, f"unsupported: {a}"

    # -------------------------------------------------------------- helpers
    def _answer(self, what):
        """Answer a query in plain language (no motion)."""
        if what == "held":
            return (f"I'm holding the {self.ex.held_label}." if self.ex.held_label
                    else "My hand is empty.")
        vis = sorted(set(self.ex.visible()))
        held = f" (holding the {self.ex.held_label})" if self.ex.held_label else ""
        return (f"I can see: {', '.join(vis)}.{held}" if vis
                else f"I don't see any objects.{held}")

    def _do_undo(self, viewer=None):
        """Reverse the last relocation: return that object to where it came from."""
        if self._undo is None:
            return False, "nothing to undo"
        label, xy = self._undo
        self._undo = None
        if xy is None:
            return False, "nothing to undo"
        if self.ex.held_label == label:                   # still in hand -> set it back down
            off = getattr(self.ex, "_grasp_offset", 0.04)
            ok, msg = self.ex._carry_to(np.asarray(xy, float), TABLE_TOP_Z + off + 0.02,
                                        viewer=viewer)
            if ok:
                ok, msg = self.ex.release(viewer=viewer)
            if ok:
                self.ex.home(viewer=viewer)
        else:                                             # already placed -> fetch it back
            ok, msg = self.ex.put_at(label, xy, viewer=viewer)
        return (True, f"put the {label} back") if ok else (False, self._clarify(msg))
