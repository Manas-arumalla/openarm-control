"""Language-driven *bimanual* manipulation session.

The single-arm `ManipulationSession` drives one arm; this drives **both**, deciding
which arm should act and handing the object over when only the other arm can reach
the destination -- all from natural-language commands:

    sess.do("grab the red block")                  # the better-placed arm grabs + holds it
    sess.do("move the green block to the left bin") # one-shot: best arm picks; hand-over if needed
    sess.do("transfer the red block to the right bin")
    sess.do("which arm is holding it?")
    sess.do("undo")                                 # put it back where it came from

It wraps `BimanualCoordinator` (best-arm selection + midpoint hand-over) and grounds
object labels + destinations through perception, mirroring the single-arm session's
state / queries / undo / clarification. Same keyword parser -- no large LLM.
"""
from __future__ import annotations

import numpy as np
import mujoco

from ..bimanual import BimanualCoordinator
from ..pick_and_place import TABLE_TOP_Z, GRASP_DEPTH
from ..vision.scene_perception import _ground as _ground_objs
from .commands import parse_command, split_steps, _REFERENTS


class BimanualSession:
    def __init__(self, model, data, perception, graspables,
                 bins=None, place_z=0.52):
        self.model, self.data = model, data
        self.co = BimanualCoordinator(model, data)
        self.perception = perception
        self.graspables = list(graspables)
        self.bins = dict(bins or {"left": "bin_left", "right": "bin_right"})
        self.place_z = place_z
        self.last_label = None
        self._undo = None                  # (block_body, origin_xy) of the last relocation

    # ----------------------------------------------------------- grounding
    def _bin_xy(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.bins[name])
        return self.data.xpos[bid][:2].copy()

    def _body_xy(self, body):
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)][:2].copy()

    def _nearest_body(self, xy, tol=0.12):
        """Map a perceived xy to the nearest graspable body (or None)."""
        best, bd = None, tol
        for b in self.graspables:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if bid >= 0:
                dist = float(np.linalg.norm(self.data.xpos[bid][:2] - xy))
                if dist < bd:
                    best, bd = b, dist
        return best

    def _ground(self, query):
        """Resolve a text query to a graspable (body, position), or (None, None).
        Matches the query's words against detected labels (colour is the reliable
        signal), then maps the best match to the nearest graspable body."""
        objs = self.perception.perceive()
        best = _ground_objs(query, objs)
        if best is None:
            return None, None
        body = self._nearest_body(best.position[:2])
        return body, best.position

    def visible(self):
        try:
            return [o.label for o in self.perception.perceive()]
        except Exception:
            return []

    def _resolve(self, target):
        if target in _REFERENTS:
            held = self.co.held["block"] if self.co.held else None
            return held or self.last_label or target
        return target

    def _dest_xy(self, dest, obj_xy=None):
        """Destination keyword -> (xy, place_z). Supports left/right bins, a generic
        'bin' (the one nearer the object), and 'table' (a clear default)."""
        if dest in ("left", "right") and dest in self.bins:
            return self._bin_xy(dest), self.place_z
        if dest == "bin":
            if len(self.bins) == 1:
                return self._bin_xy(next(iter(self.bins))), self.place_z
            ref = obj_xy if obj_xy is not None else np.zeros(2)
            name = min(self.bins, key=lambda k: np.linalg.norm(self._bin_xy(k) - ref[:2]))
            return self._bin_xy(name), self.place_z
        if dest == "table":
            return np.array([0.30, 0.0]), TABLE_TOP_Z + GRASP_DEPTH + 0.04
        return None, None

    # ------------------------------------------------------------------ run
    def run(self, text, viewer=None):
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
        """Parse and carry out one bimanual command. Returns (ok, message)."""
        intent = parse_command(text)
        if intent is None:
            return False, f"didn't understand: {text!r}"
        a = intent.action
        rt = viewer is not None                       # real-time playback when viewing

        if a == "query":
            return True, self._answer(intent.target)
        if a == "undo":
            return self._do_undo(viewer=viewer, rt=rt)

        target = self._resolve(intent.target)

        if a == "pick":                               # grab and hold
            body, pos = self._ground(target)
            if body is None:
                return False, self._clarify(f"could not find '{target}'")
            gz = float(np.clip(pos[2] - 0.01, TABLE_TOP_Z + 0.02, TABLE_TOP_Z + 0.12))
            ok, msg = self.co.pick(pos[:2], body, grasp_z=gz, viewer=viewer, dt_realtime=rt)
            if ok:
                self.last_label = body
                self._undo = (body, self._body_xy(body) if not self.co.held else
                              self.co.held["origin"])
            return ok, msg

        if a in ("move", "place", "transfer", "remove"):
            dest = intent.destination or "bin"
            # Already holding the target? -> place it (holding arm; hand-over via the
            # one-shot form). Otherwise do the one-shot best-arm pick+place (hand-over
            # automatically when only the other arm can reach the destination).
            held = self.co.held
            if held is not None and (target in (held["block"], "it") or self._resolve("it") == held["block"]):
                xy, pz = self._dest_xy(dest, self._body_xy(held["block"]))
                if xy is None:
                    return False, f"place where? ('{dest}')"
                origin = held["origin"]
                ok, msg = self.co.place_held(xy, place_z=pz, viewer=viewer, dt_realtime=rt)
                if ok:
                    self._undo = (target if target != "it" else self.last_label, origin)
                return ok, msg
            body, pos = self._ground(target)
            if body is None:
                return False, self._clarify(f"could not find '{target}'")
            xy, pz = self._dest_xy(dest, pos[:2])
            if xy is None:
                return False, f"move it where? ('{dest}')"
            gz = float(np.clip(pos[2] - 0.01, TABLE_TOP_Z + 0.02, TABLE_TOP_Z + 0.12))
            origin = self._body_xy(body)
            ok, msg = self.co.pick_place(pos[:2], xy, body, grasp_z=gz, place_z=pz,
                                         viewer=viewer, dt_realtime=rt)
            if ok:
                self.last_label = body
                self._undo = (body, origin)
            return ok, msg

        if a == "release":
            if self.co.held is None:
                return True, "my hands are empty"
            xy = self._body_xy(self.co.held["block"])
            return self.co.place_held(xy, viewer=viewer, dt_realtime=rt)

        if a == "goto":
            return False, "say 'move/transfer X to <left|right> bin'"
        return False, f"unsupported here: {a}"

    # -------------------------------------------------------------- helpers
    def _answer(self, what):
        held = self.co.held
        if what == "held" or (held is not None):
            if held is not None:
                return f"the {held['name']} arm is holding the {held['block']}."
            return "both hands are empty."
        vis = sorted(set(self.visible()))
        return f"I can see: {', '.join(vis)}." if vis else "I don't see any objects."

    def _clarify(self, msg):
        vis = sorted(set(self.visible()))
        return f"{msg}. I can see: {', '.join(vis)}." if vis else msg

    def _do_undo(self, viewer=None, rt=False):
        if self._undo is None:
            return False, "nothing to undo"
        body, origin = self._undo
        self._undo = None
        if origin is None:
            return False, "nothing to undo"
        held = self.co.held
        if held is not None and held["block"] == body:        # still held -> set it back
            ok, msg = self.co.place_held(np.asarray(origin, float), viewer=viewer, dt_realtime=rt)
        else:                                                 # already placed -> fetch + return
            cur = self._body_xy(body)
            gz = TABLE_TOP_Z + GRASP_DEPTH
            ok, msg = self.co.pick_place(cur, np.asarray(origin, float), body,
                                         grasp_z=gz, place_z=TABLE_TOP_Z + GRASP_DEPTH + 0.02,
                                         viewer=viewer, dt_realtime=rt)
        return (True, f"put the {body} back") if ok else (False, self._clarify(msg))
