"""Phase H — full-system integration smoke tests.

These don't re-test each capability (the other suites do); they prove the whole
platform is *wired together*: every CLI command resolves to a runnable module,
every registered scene loads, and the end-to-end headless entry points
(perception → predict → plan → act) run without error.
"""
import os
import sys
import importlib

import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.cli import COMMANDS, SHOWCASE
from openarm_control.config import SCENES


def test_every_cli_command_is_runnable():
    """Each command maps to an importable module exposing ``main``, with a help
    string and a (possibly empty) options field."""
    for cmd, entry in COMMANDS.items():
        module_path, help_ = entry[0], entry[1]
        assert len(entry) == 3, f"{cmd} entry must be (module, help, options)"
        mod = importlib.import_module(module_path)
        assert hasattr(mod, "main"), f"{cmd} -> {module_path} has no main()"
        assert help_, f"{cmd} has no help string"


def test_showcase_commands_exist():
    for cmd in SHOWCASE:
        assert cmd in COMMANDS, f"showcase references unknown command {cmd}"


def test_every_registered_scene_loads():
    """Every scene in the registry compiles to a valid MuJoCo model."""
    for name, path in SCENES.items():
        assert os.path.exists(path), f"scene {name} missing: {path}"
        model = mujoco.MjModel.from_xml_path(path)
        assert model.nq > 0 and model.nu > 0, f"scene {name} has no DOFs/actuators"


def test_cli_list_and_scenes(capsys):
    """The top-level CLI dispatcher runs its info commands."""
    from openarm_control import cli
    assert cli.main(["list"]) == 0
    assert cli.main(["scenes"]) == 0
    out = capsys.readouterr().out
    assert "mimic" in out and "catch" in out          # new + marquee commands listed


def test_headless_catch_pipeline_runs():
    """End-to-end dynamic catch (estimate → intercept → MPC) runs headless."""
    from openarm_control.demos import demo_catch
    demo_catch.run_benchmark(1, seed=0)               # smoke: must not raise


def test_headless_teleop_pipeline_runs():
    """End-to-end imitation (pose → retarget → safe teleop) runs headless."""
    from openarm_control.demos import demo_teleop
    demo_teleop.run_headless(1.0, arm_name="right")   # smoke: must not raise


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
