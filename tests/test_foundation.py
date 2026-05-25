"""Foundation verification: kinematics, scene, control, trajectories.

Run headless with:   pytest tests/ -v       (or)   python tests/test_foundation.py
These tests encode the invariants the higher-level tracks build on, so a
regression in FK/IK/scene/control fails loudly here.
"""
import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import GRASP_LOCAL_OFFSET
from openarm_control.kinematics import OpenArmKinematics, orientation_error
from openarm_control.controller import CartesianController
from openarm_control.grasp import GraspSolver
from openarm_control.pick_and_place import PickPlaceController
from openarm_control.trajectory import quintic_polynomial, JointTrajectory
from conftest import reset_to, disable_obstacle

# Task-object positions in the corrected scene (60 mm blocks rest at z=0.43).
BLOCKS = {"block_red": (0.18, -0.12, 0.43),
          "block_green": (0.18, -0.25, 0.43),
          "block_blue": (0.18, -0.38, 0.43)}
BINS = {"bin_red": (0.36, -0.12, 0.42),
        "bin_green": (0.36, -0.25, 0.42),
        "bin_blue": (0.36, -0.38, 0.42)}


# --------------------------------------------------------------- scene ----
def test_scene_loads(sim):
    model, data = sim
    assert model.nq == 39 and model.nv == 36 and model.nu == 16


def test_keyframes_place_blocks_on_table(sim):
    """A full keyframe must place blocks on the table, not zero them to origin."""
    model, data = sim
    for kf in ("home", "ready"):
        reset_to(model, data, kf)
        for name, pos in BLOCKS.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert np.allclose(data.xpos[bid], pos, atol=1e-3), f"{name} misplaced in {kf}"


def test_blocks_settle_stably(sim):
    """Blocks rest on the table; <2 mm drift while the arm holds the ready pose."""
    model, data = sim
    reset_to(model, data, "ready")
    kin = OpenArmKinematics(model, data)
    for i, n in enumerate([f"right_joint{j}_ctrl" for j in range(1, 8)]):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        data.ctrl[aid] = data.qpos[kin.qpos_indices[i]]
    p0 = {n: data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)].copy()
          for n in BLOCKS}
    for _ in range(2000):
        mujoco.mj_step(model, data)
    for n, p in p0.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        assert np.linalg.norm(data.xpos[bid] - p) < 2e-3, f"{n} drifted"


# ----------------------------------------------------------- kinematics ---
def test_fk_matches_mujoco(sim):
    model, data = sim
    reset_to(model, data, "home")
    kin = OpenArmKinematics(model, data)
    p, _ = kin.forward_kinematics()
    assert np.allclose(p, data.site_xpos[kin.site_id])


def test_grasp_offset(sim):
    model, data = sim
    reset_to(model, data, "home")
    wrist = OpenArmKinematics(model, data)
    grasp = OpenArmKinematics(model, data, tool_offset=GRASP_LOCAL_OFFSET)
    pw, _ = wrist.forward_kinematics()
    pg, _ = grasp.forward_kinematics()
    assert abs(np.linalg.norm(pg - pw) - 0.135) < 1e-3


def test_ik_position_roundtrip(sim):
    """FK(IK(FK(q))) reproduces position for random reachable configs."""
    model, data = sim
    kin = OpenArmKinematics(model, data)
    ranges = np.column_stack([kin.jnt_low, kin.jnt_high])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(30):
        q = rng.uniform(ranges[:, 0], ranges[:, 1])
        target, _ = kin.forward_kinematics(q)
        qs = kin.inverse_kinematics(target)
        ach, _ = kin.forward_kinematics(qs)
        worst = max(worst, np.linalg.norm(target - ach))
    assert worst < 5e-4, f"worst position IK error {worst*1000:.3f} mm"


def test_ik_pose_roundtrip(sim):
    model, data = sim
    kin = OpenArmKinematics(model, data)
    ranges = np.column_stack([kin.jnt_low, kin.jnt_high])
    rng = np.random.default_rng(1)
    wp, wo = 0.0, 0.0
    for _ in range(20):
        q = rng.uniform(ranges[:, 0], ranges[:, 1])
        tp, tm = kin.forward_kinematics(q)
        qs = kin.inverse_kinematics(tp, target_mat=tm)
        ach, achm = kin.forward_kinematics(qs)
        wp = max(wp, np.linalg.norm(tp - ach))
        wo = max(wo, np.linalg.norm(orientation_error(achm, tm)))
    assert wp < 5e-4 and wo < np.deg2rad(0.5)


def test_ik_respects_joint_limits(sim):
    model, data = sim
    kin = OpenArmKinematics(model, data)
    rng = np.random.default_rng(2)
    ranges = np.column_stack([kin.jnt_low, kin.jnt_high])
    for _ in range(20):
        q = rng.uniform(ranges[:, 0], ranges[:, 1])
        target, _ = kin.forward_kinematics(q)
        qs = kin.inverse_kinematics(target)
        assert np.all(qs >= kin.jnt_low - 1e-6) and np.all(qs <= kin.jnt_high + 1e-6)


# ----------------------------------------------------- reachability -------
def test_task_objects_reachable_topdown(sim):
    """Every block (grasp z) and bin (place z) admits a reachable top-down grasp."""
    model, data = sim
    gs = GraspSolver(model, data)
    for name, p in BLOCKS.items():
        assert gs.is_reachable(np.array([p[0], p[1], 0.44])), f"{name} grasp not reachable"
    for name, p in BINS.items():
        assert gs.is_reachable(np.array([p[0], p[1], 0.48])), f"{name} place not reachable"


# -------------------------------------------------------- controller ------
@pytest.mark.parametrize("target", [
    (0.22, -0.10, 0.50), (0.22, -0.23, 0.52), (0.22, -0.36, 0.50),
    (0.30, -0.15, 0.58), (0.20, -0.25, 0.50), (0.25, -0.20, 0.55),
])
def test_controller_position_convergence(make_sim, target):
    model, data = make_sim()
    disable_obstacle(model)
    grasp = OpenArmKinematics(model, data, tool_offset=GRASP_LOCAL_OFFSET)
    ctrl = CartesianController(model, data, kinematics=grasp)
    reset_to(model, data, "ready")
    ctrl.reset()
    for i, aid in enumerate(ctrl.actuator_ids):
        data.ctrl[aid] = data.qpos[grasp.qpos_indices[i]]
    ctrl.set_target(np.array(target, float))
    for _ in range(4000):
        ctrl.step()
        mujoco.mj_step(model, data)
    cur, _ = grasp.forward_kinematics()
    assert np.linalg.norm(np.array(target) - cur) < 2e-3


# ----------------------------------------------------- pick & place -------
@pytest.mark.parametrize("block,pick,place", [
    ("block_red", (0.18, -0.12), (0.36, -0.12)),
    ("block_green", (0.18, -0.25), (0.36, -0.25)),
    ("block_blue", (0.18, -0.38), (0.36, -0.38)),
])
def test_pick_and_place_lands_in_target(make_sim, block, pick, place):
    model, data = make_sim()
    disable_obstacle(model)
    reset_to(model, data, "ready")
    ppc = PickPlaceController(model, data)
    for i, aid in enumerate(ppc.arm_acts):
        data.ctrl[aid] = data.qpos[ppc.king.qpos_indices[i]]
    ppc.execute(ppc.plan(pick_xy=pick, place_xy=place), block=block, gravity_comp=True)
    data.qfrc_applied[:] = 0
    for _ in range(800):
        mujoco.mj_step(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block)
    p = data.xpos[bid]
    assert abs(p[0] - place[0]) < 0.06 and abs(p[1] - place[1]) < 0.06 and p[2] > 0.41


# ---------------------------------------------------- motion planning -----
def test_rrt_plans_collision_free_path_around_obstacle():
    import mujoco as mj
    from openarm_control.config import OBSTACLE_SCENE
    from openarm_control.grasp import GraspSolver
    from openarm_control.planners.rrt import RRTPlanner
    from openarm_control.planners.collision import CollisionChecker
    model = mj.MjModel.from_xml_path(OBSTACLE_SCENE)
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "ready"))
    mj.mj_forward(model, data)
    gs = GraspSolver(model, data)
    kin = gs.king
    chk = CollisionChecker(model, data, kin)
    qs = gs.solve(np.array([0.18, -0.12, 0.55]))
    qg = gs.solve(np.array([0.36, -0.38, 0.55]))
    # The straight path must be blocked (otherwise the test proves nothing).
    assert not chk.edge_clear(qs, qg)
    path = RRTPlanner(model, data, kin, seed=1).plan(qs, qg)
    assert path is not None, "RRT failed to find a path"
    assert np.allclose(path[0], qs) and np.allclose(path[-1], qg)
    for a, b in zip(path[:-1], path[1:]):
        assert chk.edge_clear(a, b), "planned edge collides"


# -------------------------------------------------------- trajectory ------
def test_quintic_boundary_conditions():
    q, v, a = quintic_polynomial(0.0, 0.0, 2.0, 1.0, 3.0)
    assert abs(q - 1.0) < 1e-9 and abs(v) < 1e-9 and abs(a) < 1e-9
    q, v, a = quintic_polynomial(2.0, 0.0, 2.0, 1.0, 3.0)
    assert abs(q - 3.0) < 1e-9 and abs(v) < 1e-9 and abs(a) < 1e-9


def test_joint_trajectory_smoothness():
    """Velocity is continuous and starts/ends at zero (no jumps)."""
    q0 = np.zeros(7)
    q1 = np.array([0.5, -0.3, 0.2, 1.0, -0.4, 0.3, 0.1])
    traj = JointTrajectory(q0, q1, duration=2.0)
    ts = np.linspace(0, 2.0, 200)
    qs = np.array([traj.evaluate(t) for t in ts])
    vel = np.diff(qs, axis=0) / (ts[1] - ts[0])
    assert np.linalg.norm(vel[0]) < 1e-2 and np.linalg.norm(vel[-1]) < 1e-2
    assert np.max(np.abs(np.diff(vel, axis=0))) < 0.05  # no velocity jumps
    assert np.allclose(qs[0], q0) and np.allclose(qs[-1], q1, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
