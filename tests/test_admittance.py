"""F2 — Cartesian admittance (compliant) control.

Pressing a soft pad with admittance control yields on contact and settles with a
much smaller force than plain position control commanding the same depth, while
reaching the same place. Also checks the free-space case (no contact -> no force,
reference tracks the command). Reuses the demo helpers so the demo path is tested
too. Translational admittance reuses the existing position-control stack.
"""
import os
import sys

import numpy as np
import mujoco
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.contact import AdmittanceController
from openarm_control.demos.demo_admittance import (
    _load, _reachable_R, _teleport_hover, press_admittance, press_rigid,
    PX, PY, HOVER, SURF)


def test_admittance_reduces_contact_force():
    """Admittance yields on contact: far lower steady force than rigid control,
    both reaching the pad without punching through to the table."""
    R = _reachable_R(AdmittanceController(*_load()))
    f_adm, z_adm = press_admittance(*_load(), R)
    f_rig, z_rig = press_rigid(*_load(), R)
    assert f_rig > f_adm, f"rigid ({f_rig:.0f}) should exceed admittance ({f_adm:.0f})"
    assert f_adm < 0.6 * f_rig, f"admittance not clearly softer: {f_adm:.0f} vs {f_rig:.0f}"
    assert f_adm < 120.0, f"admittance contact force not bounded: {f_adm:.0f} N"
    # both settle on the pad, neither punches through to the table (~0.40)
    assert z_adm > 0.44 and z_rig > 0.44, f"EE punched through: {z_adm:.3f}, {z_rig:.3f}"


def test_no_force_and_tracks_in_free_space():
    """Hovering above the pad there is no external contact force, and the compliant
    reference tracks the commanded pose (no runaway)."""
    m, d = _load()
    ac = AdmittanceController(m, d)
    R = _reachable_R(ac)
    _teleport_hover(ac, R)
    ac.reset([PX, PY, HOVER], R)
    # a few steps holding a free-space target well above the pad
    for _ in range(200):
        F, ee = ac.step([PX, PY, HOVER], R_desired=R, grip=None)
    assert np.linalg.norm(F) < 1.0, f"unexpected contact force in free space: {F}"
    assert ee[2] > SURF + 0.03, f"reference drifted into the pad in free space: {ee[2]:.3f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
