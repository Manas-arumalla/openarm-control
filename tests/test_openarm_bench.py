"""E1 — OpenArm-Bench unified evaluation runner.

Checks the benchmark runner executes and produces a consolidated results table.
Uses the fast, deterministic, model-free subset (articulated + admittance) so the
test is quick; the full benchmark (insertion/reach with trained policies) is run via
`python benchmarks/openarm_bench.py`.
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "benchmarks"))

import openarm_bench


def test_bench_runs_and_reports():
    rows = openarm_bench.main(["--only", "articulated,admittance"])
    skills = {r[0] for r in rows}
    assert {"articulated:drawer", "articulated:door", "articulated:valve"} <= skills
    assert "admittance" in skills
    # admittance: compliant force is well below rigid
    forces = {m: v for s, m, mt, v in rows if s == "admittance"}
    assert forces["compliant"] < forces["rigid"], forces
    # the results CSV was written
    assert os.path.exists(os.path.join(PROJECT_ROOT, "benchmarks", "results", "openarm_bench.csv"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
