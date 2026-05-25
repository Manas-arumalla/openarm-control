"""OpenArm-Bench -- unified evaluation across the manipulation skills (extension phase E1).

Runs a standardized protocol (fixed seeds, N episodes) over the extension-arc skills
and consolidates the results -- including the **classical vs BC vs ACT vs RL** method
comparison -- into one table + CSV. A single citable artifact summarizing what the
platform can do and how the methods compare. (The dynamic catching/throwing skills
have their own dedicated benchmarks -- catching_benchmark.py, throwing_benchmark.py.)

    python benchmarks/openarm_bench.py            # full
    python benchmarks/openarm_bench.py --quick    # fewer episodes
"""
import argparse
import csv
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openarm_control.config import PROJECT_ROOT, ARTICULATED_SCENE, CLOTH_SCENE

DEMO_DIR = os.path.join(PROJECT_ROOT, "demos")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# --------------------------------------------------------------------- skills
def bench_insertion(n, rows):
    """Peg-in-hole over randomized sockets: classical (scripted) vs BC."""
    from openarm_control.imitation.expert import make_env_and_expert
    env, expert = make_env_and_expert("insert", seed=0)

    def run(policy_act, seed0):
        s = 0
        for ep in range(n):
            obs, _ = env.reset(seed=seed0 + ep)
            if expert is not None:
                expert.reset()
            done, info = False, {}
            while not done:
                obs, _, term, trunc, info = env.step(policy_act(obs))
                done = term or trunc
            s += int(info.get("is_success", False))
        return s / n

    rows.append(("insertion", "classical", "success", run(lambda o: expert.act(o), 0)))
    bc_path = os.path.join(DEMO_DIR, "insert_bc.pt")
    if os.path.exists(bc_path):
        from openarm_control.imitation.bc import load_bc
        bc = load_bc(bc_path)
        rows.append(("insertion", "BC", "success", run(lambda o: bc.act(o), 1000)))
    env.close()


def bench_reach(n, rows):
    """Reach a random target: BC (state MLP) vs ACT (vision+state, chunked)."""
    from openarm_control.imitation.expert import make_env_and_expert
    bc_path = os.path.join(DEMO_DIR, "reach_bc.pt")
    if os.path.exists(bc_path):
        from openarm_control.imitation.bc import load_bc
        bc = load_bc(bc_path)
        env, _ = make_env_and_expert("reach", seed=0)
        s = 0
        for ep in range(n):
            obs, _ = env.reset(seed=700 + ep); done = False; info = {}
            while not done:
                obs, _, t, tr, info = env.step(bc.act(obs)); done = t or tr
            s += int(info.get("is_success", False))
        env.close()
        rows.append(("reach", "BC", "success", s / n))
    act_path = os.path.join(DEMO_DIR, "reach_act.pt")
    if os.path.exists(act_path):
        from openarm_control.imitation.act import load_act, evaluate
        from openarm_control.imitation.device import get_device
        rate = evaluate(load_act(act_path, device=get_device()), task="reach", episodes=n, seed=700)
        rows.append(("reach", "ACT", "success", rate))


def bench_articulated(rows):
    """Open the drawer / door, turn the valve (classical)."""
    from openarm_control.articulated import ArticulatedController
    specs = [("drawer", "drawer_slide", "open_drawer", "m"),
             ("door", "door_hinge", "open_door", "rad"),
             ("valve", "valve_turn", "turn_valve", "rad")]
    for name, joint, method, unit in specs:
        m = mujoco.MjModel.from_xml_path(ARTICULATED_SCENE)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready"))
        mujoco.mj_forward(m, d)
        qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)]
        q0 = d.qpos[qadr]
        getattr(ArticulatedController(m, d), method)()
        moved = abs(d.qpos[qadr] - q0)
        val = moved * 1000 if unit == "m" else np.degrees(moved)
        rows.append((f"articulated:{name}", "classical", f"opened ({'mm' if unit=='m' else 'deg'})",
                     round(val, 1)))


def bench_admittance(rows):
    """Compliant press: admittance vs rigid contact force (N)."""
    from openarm_control.demos.demo_admittance import _load, _reachable_R, press_admittance, press_rigid
    from openarm_control.contact import AdmittanceController
    R = _reachable_R(AdmittanceController(*_load()))
    f_adm, _ = press_admittance(*_load(), R)
    f_rig, _ = press_rigid(*_load(), R)
    rows.append(("admittance", "compliant", "contact force (N)", round(f_adm, 1)))
    rows.append(("admittance", "rigid", "contact force (N)", round(f_rig, 1)))


def bench_cloth(rows):
    """Single-arm fold: cloth span reduction (folded if much smaller)."""
    from openarm_control.cloth import ClothFoldController, set_ready
    m = mujoco.MjModel.from_xml_path(CLOTH_SCENE); d = mujoco.MjData(m)
    set_ready(m, d, settle=400)
    cf = ClothFoldController(m, d)
    before = cf.cloth_vertices(); y0 = before[:, 1].max() - before[:, 1].min()
    cf.fold("cloth_0", cf.corner_xy("cloth_8"))
    after = cf.cloth_vertices(); y1 = after[:, 1].max() - after[:, 1].min()
    rows.append(("cloth_fold", "classical", "y-span reduction (%)", round(100 * (1 - y1 / y0), 0)))


# --------------------------------------------------------------------- runner
def main(argv=None):
    ap = argparse.ArgumentParser(description="OpenArm-Bench: unified skill evaluation.")
    ap.add_argument("--quick", action="store_true", help="fewer episodes")
    ap.add_argument("--only", default=None, help="comma-separated subset: insertion,reach,articulated,admittance,cloth")
    a = ap.parse_args(argv)
    n = 8 if a.quick else 20
    which = a.only.split(",") if a.only else ["insertion", "reach", "articulated", "admittance", "cloth"]

    rows = []
    if "insertion" in which: bench_insertion(n, rows)
    if "reach" in which: bench_reach(n, rows)
    if "articulated" in which: bench_articulated(rows)
    if "admittance" in which: bench_admittance(rows)
    if "cloth" in which: bench_cloth(rows)

    print("\n=== OpenArm-Bench (manipulation skills) ===")
    print(f"{'skill':<20}{'method':<12}{'metric':<24}{'result'}")
    print("-" * 64)
    for skill, method, metric, result in rows:
        rv = f"{result:.0%}" if metric == "success" else f"{result}"
        print(f"{skill:<20}{method:<12}{metric:<24}{rv}")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "openarm_bench.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["skill", "method", "metric", "result"]); w.writerows(rows)
    print(f"\nresults -> {out}")
    return rows


if __name__ == "__main__":
    main()
