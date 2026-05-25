"""CLI smoke tests (headless; do not launch viewers)."""
import os
import sys
import importlib

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control import cli
from openarm_control.config import SCENES


def test_all_commands_import_and_have_main():
    for name, entry in cli.COMMANDS.items():
        module_path = entry[0]
        mod = importlib.import_module(module_path)
        assert hasattr(mod, "main"), f"{name} -> {module_path} has no main()"


def test_list_and_scenes_run(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "sort" in out and "bimanual" in out
    assert cli.main(["scenes"]) == 0
    assert "bimanual" in capsys.readouterr().out


def test_unknown_command_returns_2():
    assert cli.main(["nonsense-command"]) == 2


def test_registered_scenes_exist():
    for name, path in SCENES.items():
        assert os.path.exists(path), f"scene {name} missing: {path}"


def test_showcase_commands_are_valid():
    for c in cli.SHOWCASE:
        assert c in cli.COMMANDS


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
