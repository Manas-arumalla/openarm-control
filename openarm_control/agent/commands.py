"""Lightweight natural-language command parsing (no heavy LLM).

A compact keyword/grammar parser over the manipulation vocabulary turns commands
like:

    "pick the red cube and put it in the bin"  -> Intent('place', 'red cube', 'bin')
    "move the red cube to the left"            -> Intent('move',  'red cube', 'left')
    "take that object out"                     -> Intent('remove','object',   'out')
    "throw the ball in the bin"                -> Intent('throw', 'ball',      'bin')
    "pick up the green box"                    -> Intent('pick',  'green box', None)
    "stack the red cube on the green cube"     -> Intent('stack', 'red cube',  'green cube')
    "insert the peg into the socket"           -> Intent('insert','peg',       'socket')

into a structured ``Intent`` the executor can run. It's instant, robust, and
dependency-free; an optional LLM-API mode could parse free-form phrasing later
behind the same ``parse_command`` signature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Action verbs, in resolution priority (a "pick ... and put ..." command is a
# place; "take ... out" is a remove, checked before plain "take" = pick).
_THROW = ("throw", "toss", "lob", "chuck")
_STACK = ("stack", "pile")
_INSERT = ("insert", "plug")
_UNDO = ("undo", "reverse")
_QUERY = ("what", "which", "where")                      # question words (start of a query)
_HELD_WORDS = ("holding", "hold", "hand", "have", "carrying", "grasping")
_PLACE = ("put", "place", "drop", "set", "deposit")
_MOVE = ("move", "shift", "slide", "bring", "transfer", "handover", "handoff", "relocate")
_PUSH = ("push", "nudge", "shove")             # non-prehensile (push, don't grasp)
# articulated-object skills (checked before "open" = release-the-hand)
_TURN = ("turn", "rotate", "crank", "spin")
_UNSCREW = ("unscrew", "loosen")
_REMOVE = ("remove", "discard")
_PICK = ("pick", "grab", "grasp", "lift", "take", "get", "fetch")
_RELEASE = ("release", "open", "letgo")        # "release", "open", "let go", "drop it"
_REFERENTS = ("it", "that", "this", "one", "object", "thing", "item")

_COLORS = ("red", "green", "blue", "orange", "yellow", "purple", "black", "white")
# object nouns / shapes the perception layer may report (and generic referents)
_NOUNS = ("ball", "sphere", "box", "cube", "block", "can", "bottle", "cylinder",
          "cup", "mug", "pencil", "pen", "banana", "apple", "bowl", "peg", "puck",
          "object", "thing", "item", "one")

_DEST = {
    "bin": "bin", "basket": "bin", "container": "bin", "bucket": "bin",
    "left": "left", "right": "right", "table": "table",
    "socket": "socket", "hole": "socket", "slot": "socket",
}


@dataclass
class Intent:
    action: str                       # 'pick' | 'place' | 'move' | 'throw' | 'remove'
    target: str                       # text query for grounding, e.g. 'red cube'
    destination: str = None           # 'bin' | 'left' | 'right' | 'table' | 'out' | None
    raw: str = ""

    def __repr__(self):
        d = f", dest={self.destination!r}" if self.destination else ""
        return f"Intent({self.action!r}, target={self.target!r}{d})"


def _tokens(text):
    return re.findall(r"[a-z]+", text.lower())


_ON = ("on", "onto", "atop", "above", "over", "top")     # "stack X on (top of) Y"


def _is_stack(toks):
    """A stacking command: an explicit 'stack' verb, or 'put/place X on(to)/on top
    of Y' where Y is an *object* (not a destination like the bin/table)."""
    s = set(toks)
    if s & set(_STACK):
        return True
    if s & (set(_PLACE) | set(_MOVE)):
        split = next((i for i, t in enumerate(toks) if t in _ON), None)
        if split is not None:
            right = toks[split + 1:]
            has_obj = any(t in _COLORS for t in right) or \
                any(t in _NOUNS and t not in ("one", "it") for t in right)
            has_dest = any(t in _DEST for t in right)
            return has_obj and not has_dest
    return False


def _find_action(toks):
    s = set(toks)
    if (toks and toks[0] in _QUERY) or (s & set(_HELD_WORDS)):  # "what are you holding?"
        return "query"
    if (s & set(_UNDO)) or ("back" in s and not _has_dest(toks)):  # "undo" / "put it back"
        return "undo"
    if s & set(_THROW):
        return "throw"
    if s & set(_INSERT):                                 # "insert the peg into the socket"
        return "insert"
    # articulated-object skills (before "open" = release): drawer / door / valve / cap
    if "drawer" in s and (s & {"open", "pull", "slide"}):
        return "open_drawer"
    if (s & {"door", "cabinet"}) and ("open" in s):
        return "open_door"
    if "valve" in s and (s & set(_TURN)):
        return "turn_valve"
    if (s & set(_UNSCREW)) or ((s & {"cap", "lid", "jar", "bottle"}) and
                               (s & {"open", "twist", "off"})):
        return "unscrew"
    if _is_stack(toks):                                  # "stack the red cube on the green cube"
        return "stack"
    # release / open the hand (but "drop X in the bin" stays a place)
    if "release" in s or ("let" in s and "go" in s) or ("open" in s) \
            or ("drop" in s and "it" in s and not _has_dest(toks)):
        return "release"
    if "out" in toks and (s & set(_PICK) or s & set(_REMOVE)):  # "take it out"
        return "remove"
    if s & set(_REMOVE):
        return "remove"
    if "go" in s and _has_dest(toks):                           # "go to the bin"
        return "goto"
    if s & set(_PUSH):                                          # "push the puck to the goal"
        return "push"
    if s & set(_PLACE):                                          # "...and put it in the bin"
        return "place"
    if s & set(_MOVE):
        return "move"
    if s & set(_PICK):
        return "pick"
    return None


def _has_dest(toks):
    return ("out" in toks or "away" in toks or any(t in _DEST for t in toks))


def _find_target(toks):
    color = next((t for t in toks if t in _COLORS), None)
    noun = next((t for t in toks if t in _NOUNS and t not in ("one", "it")), None)
    if color and noun:
        return f"{color} {noun}"
    if color:
        return color
    if noun:
        return noun
    if any(t in ("it", "that", "this", "one") for t in toks):
        return "it"          # referent -> resolved to the held / last object
    return "object"


def _find_destination(toks):
    if "out" in toks or "away" in toks:
        return "out"
    for t in toks:
        if t in _DEST:
            return _DEST[t]
    return None


def _parse_stack(toks, text):
    """'stack X on (top of) Y' -> Intent('stack', target=X, destination=Y), where
    the destination is the *support object*'s query (resolved like a target)."""
    split = next((i for i, t in enumerate(toks) if t in _ON), None)
    if split is not None:
        target = _find_target(toks[:split])
        support = _find_target(toks[split + 1:])
    else:                                                # "stack the red cube" (no support)
        target, support = _find_target(toks), None
    return Intent(action="stack", target=target, destination=support, raw=text)


def split_steps(text):
    """Split a multi-step command into ordered clauses on sequence words
    ("... then ...", "after that", "next", ";"). 'pick X and put it in Y' stays one
    clause (the 'and' there is part of one pick-and-place intent)."""
    parts = re.split(r"\bthen\b|\bafter that\b|\bafterwards\b|\bnext\b|;", text.lower())
    return [p.strip(" ,.") for p in parts if p.strip(" ,.")]


def parse_command(text):
    """Parse a command string into an ``Intent`` (or ``None`` if no action found)."""
    toks = _tokens(text)
    action = _find_action(toks)
    if action is None:
        return None
    if action == "query":
        tgt = "held" if any(t in toks for t in _HELD_WORDS) else "scene"
        return Intent("query", target=tgt, raw=text)
    if action == "undo":
        return Intent("undo", target="it", raw=text)
    if action in ("open_drawer", "open_door", "turn_valve", "unscrew"):
        fixture = {"open_drawer": "drawer", "open_door": "door",
                   "turn_valve": "valve", "unscrew": "cap"}[action]
        return Intent(action=action, target=fixture, raw=text)
    if action == "stack":
        return _parse_stack(toks, text)
    target = _find_target(toks)
    dest = _find_destination(toks)
    # Sensible destination defaults per action.
    if action == "throw" and dest is None:
        dest = "bin"
    if action == "remove" and dest is None:
        dest = "out"
    if action == "insert" and dest is None:
        dest = "socket"
    return Intent(action=action, target=target, destination=dest, raw=text)
