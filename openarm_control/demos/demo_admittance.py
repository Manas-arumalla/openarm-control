"""Compliant (admittance) control demo — extension phase F2.

The right arm presses straight down onto a soft pad. With **admittance control**
the end-effector yields on contact and settles with a small, bounded force; with
plain **position control** commanding the same depth it pushes rigidly and the
contact force is far higher. The headless self-test prints both so the difference
is explicit.

    python -m openarm_control.demos.demo_admittance            # viewer: watch it yield
    python -m openarm_control.demos.demo_admittance --headless # admittance vs rigid report
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import CONTACT_SCENE, RIGHT_ARM
from openarm_control.grasp import GraspSolver, topdown_orientation
from openarm_control.contact import AdmittanceController

PX, PY = 0.22, -0.20            # press point (within the right arm's straight-down reach)
SURF = 0.46                     # soft pad top
HOVER = SURF + 0.06
TARGET = SURF - 0.03            # commanded 3 cm into the pad
GRIP = RIGHT_ARM.gripper_closed


def _load():
    model = mujoco.MjModel.from_xml_path(CONTACT_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready"))
    mujoco.mj_forward(model, data)
    return model, data


def _reachable_R(ctrl):
    """A reachable straight-down orientation at the press point (yaw is searched)."""
    gs = GraspSolver(ctrl.model, ctrl.data, arm=RIGHT_ARM)
    _, info = gs.solve([PX, PY, HOVER], return_info=True)
    return topdown_orientation(info["yaw"])


def _idle_dofs(ctrl):
    """DOF indices NOT belonging to the active arm — i.e. the idle left arm."""
    active = set(int(i) for i in ctrl.king.dof_indices)
    return np.array([d for d in range(ctrl.model.nv) if d not in active], dtype=int)


def _hold_idle(ctrl):
    """Gravity-compensate the idle (left) arm so it holds its ready pose instead of
    sagging to the table while the right arm works. Call once per step before
    stepping the sim."""
    idle = _idle_dofs(ctrl)
    ctrl.data.qfrc_applied[idle] = ctrl.data.qfrc_bias[idle]


def _teleport_hover(ctrl, R):
    q = ctrl.king.inverse_kinematics([PX, PY, HOVER], target_mat=R, restarts=6)
    ctrl.data.qpos[ctrl.king.qpos_indices] = q
    ctrl.data.qvel[ctrl.king.dof_indices] = 0.0
    mujoco.mj_forward(ctrl.model, ctrl.data)
    # Command every joint actuator to hold its current angle, so idle joints (the
    # left arm) keep the ready pose; the right arm + gripper are overridden below.
    for a in range(ctrl.model.nu):
        if ctrl.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            j = int(ctrl.model.actuator_trnid[a, 0])
            ctrl.data.ctrl[a] = ctrl.data.qpos[ctrl.model.jnt_qposadr[j]]
    for i, a in enumerate(ctrl.arm_acts):
        ctrl.data.ctrl[a] = q[i]
    if ctrl.grip_act != -1:
        ctrl.data.ctrl[ctrl.grip_act] = GRIP
    return q


def _cmd_z(k, n_desc):
    return HOVER + (TARGET - HOVER) * min(1.0, k / n_desc)


def press_admittance(model, data, R, n_desc=600, hold=600):
    """Press the pad with admittance; return (steady_force_N, settle_z)."""
    ac = AdmittanceController(model, data)
    _teleport_hover(ac, R)
    ac.reset([PX, PY, HOVER], R)
    forces, zs = [], []
    for k in range(n_desc + hold):
        _hold_idle(ac)
        F, ee = ac.step([PX, PY, _cmd_z(k, n_desc)], R_desired=R, grip=GRIP)
        if k >= n_desc + hold - 50:
            forces.append(np.linalg.norm(F)); zs.append(ee[2])
    return float(np.mean(forces)), float(np.mean(zs))


def press_rigid(model, data, R, n_desc=600, hold=600):
    """Press the pad with plain position control; return (steady_force_N, settle_z)."""
    rc = AdmittanceController(model, data)        # reuse IK + actuator wiring + force read
    _teleport_hover(rc, R)
    for k in range(n_desc + hold):
        x = np.array([PX, PY, _cmd_z(k, n_desc)])
        q = rc.king.inverse_kinematics(x, target_mat=R,
                                       q_init=data.qpos[rc.king.qpos_indices],
                                       restarts=0, rest_weight=0.0)
        if q is not None:
            for i, a in enumerate(rc.arm_acts):
                data.ctrl[a] = q[i]
        data.ctrl[rc.grip_act] = GRIP
        data.qfrc_applied[rc.king.dof_indices] = data.qfrc_bias[rc.king.dof_indices]
        _hold_idle(rc)
        mujoco.mj_step(model, data)
    return float(np.linalg.norm(rc.contact_force())), float(rc.ee_pos()[2])


def run_headless():
    m, d = _load()
    R = _reachable_R(AdmittanceController(m, d))
    f_adm, z_adm = press_admittance(*_load(), R)
    f_rig, z_rig = press_rigid(*_load(), R)
    print("Pressing a soft pad 3 cm deep (same command for both):\n")
    print(f"  admittance : steady force {f_adm:6.1f} N   (EE settles z={z_adm:.3f})")
    print(f"  position   : steady force {f_rig:6.1f} N   (EE settles z={z_rig:.3f})")
    ratio = f_rig / max(f_adm, 1e-6)
    print(f"\n  -> admittance yields on contact: {ratio:.1f}x lower contact force.")


def run_interactive():
    from mujoco import viewer as mjviewer
    m, d = _load()
    ac = AdmittanceController(m, d)
    R = _reachable_R(ac)
    _teleport_hover(ac, R)
    ac.reset([PX, PY, HOVER], R)
    print("Admittance demo: the arm presses the soft pad and yields on contact.")
    print("Watch the contact force stay low. Close the window to quit.")
    with mjviewer.launch_passive(m, d) as viewer:
        k = 0
        while viewer.is_running():
            t0 = time.time()
            F, _ = ac.step([PX, PY, _cmd_z(k, 600)], R_desired=R, grip=GRIP)
            if k % 100 == 0:
                print(f"  contact force = {np.linalg.norm(F):5.1f} N")
            viewer.sync()
            k += 1
            dt = m.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compliant (admittance) control demo.")
    ap.add_argument("--headless", action="store_true", help="print admittance-vs-rigid report")
    args = ap.parse_args(argv)
    if args.headless:
        run_headless()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
