"""Phase I / M2 — lightweight natural-language command parsing."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.agent import parse_command


CASES = [
    ("pick the red cube and put it in the bin", "place", "red cube", "bin"),
    ("move the red cube to the left",           "move",  "red cube", "left"),
    ("take that object out",                    "remove", "object",  "out"),
    ("throw the ball in the bin",               "throw", "ball",      "bin"),
    ("pick up the green box",                   "pick",  "green box", None),
    ("grab the blue can",                       "pick",  "blue can",  None),
    ("put the orange ball in the basket",       "place", "orange ball", "bin"),
    ("remove the red ball",                     "remove", "red ball", "out"),
    ("move the box to the right",               "move",  "box",       "right"),
    ("toss the ball into the bin",              "throw", "ball",      "bin"),
]


def test_command_cases():
    for text, action, target, dest in CASES:
        it = parse_command(text)
        assert it is not None, f"failed to parse: {text!r}"
        assert it.action == action, f"{text!r}: action {it.action!r} != {action!r}"
        assert it.target == target, f"{text!r}: target {it.target!r} != {target!r}"
        assert it.destination == dest, f"{text!r}: dest {it.destination!r} != {dest!r}"


def test_non_command_returns_none():
    assert parse_command("hello there how are you") is None


def test_throw_defaults_to_bin():
    assert parse_command("throw the orange ball").destination == "bin"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
