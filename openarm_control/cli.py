"""Unified command-line entry point for the OpenArm platform.

After `pip install -e .` this is available as the `openarm` command; it can also
be run as `python -m openarm_control.cli`.

    openarm list                 # all commands + their options
    openarm scenes               # registered scenes
    openarm <command> --help     # full options for one command

Classical control & planning
    openarm fk | ik | cartesian | trajectory | gripper
    openarm sort                              # single-arm colour sorting
    openarm plan [--planner rrt|prm]          # obstacle avoidance
    openarm bimanual --mode sort|sync|handoff|stack|coordinate|language

Perception, dynamics & learning
    openarm servo                             # visual servoing (see, reach, grab)
    openarm catch [--vision] [--bimanual] [--twoball] [--benchmark [N]]
    openarm rl-train [--task reach|pick|insert|balance] | rl-eval [--task reach|pick|insert|balance]
    openarm bc-collect | bc-train | bc-eval [--compare-rl]
    openarm mimic [--webcam] [--bimanual] [--pick] [--preview]
    openarm gen-data [--out DIR --n N]        # auto-labeled synthetic detection data
    openarm detect train|eval [--data … --weights …]   # fine-tune / eval a detector

Intelligent manipulation (vision-grounded, language-commanded)
    openarm manipulate "put the green box in the bin" [--interactive] [--vision]
    openarm throw [--multi]                   # ballistic throw into a bin / 5-bin show
    openarm stack "stack the red cube on the green cube"
    openarm insert [--shape round|square|cuboid]       # peg-in-hole (matched shapes)
    openarm push [--goal a|b]                          # non-prehensile pushing to a goal
    openarm tool                                       # tool use: grasp a stick, reach beyond the workspace
    openarm cloth [--corner 0|4|20|24]                 # deformable cloth folding (grasp a corner, fold over)
    openarm interactive [--scanned] [--detector yolo-world|yoloe]
    openarm bimanual --mode language "transfer the red block to the left bin"

    openarm showcase             # grand tour of the whole stack
    openarm test                 # run the headless test suite
"""
import sys
import importlib
import subprocess

from .config import SCENES

# command -> (module, one-line help, options). `options` documents the command's
# flags/positional args in the `openarm list` output (run `openarm <cmd> --help`
# for argparse's full per-flag descriptions). Each module exposes main(argv=None).
COMMANDS = {
    "fk":         ("openarm_control.demos.demo_fk", "Forward-kinematics viewer (move joint sliders)", ""),
    "ik":         ("openarm_control.demos.demo_ik", "Inverse-kinematics offline solve demo", ""),
    "cartesian":  ("openarm_control.demos.demo_cartesian_control", "Drag a target; arm follows (resolved-rate)", ""),
    "trajectory": ("openarm_control.demos.demo_trajectory", "End-effector traces a square (quintic)", ""),
    "gripper":    ("openarm_control.demos.demo_gripper", "Reach, grasp, and lift a block", ""),
    "sort":       ("openarm_control.demos.demo_pick_and_place", "Single-arm autonomous colour sorting", ""),
    "plan":       ("openarm_control.demos.demo_motion_planning", "RRT-Connect/PRM obstacle avoidance",
                   "--planner rrt|prm"),
    "bimanual":   ("openarm_control.demos.demo_bimanual", "Dual-arm coordination (incl. best-arm + hand-over via language)",
                   "--mode sort|sync|handoff|stack|coordinate|language  [\"<command>\"]  --interactive  --headless"),
    "servo":      ("openarm_control.demos.demo_visual_servo", "Visual servoing: see, reach, grab a cube", ""),
    "catch":      ("openarm_control.demos.demo_catch", "Catch a thrown ball (Kalman + MPC interception)",
                   "--vision  --bimanual  --twoball  --benchmark [N]  --throws N  --seed N"),
    "rl-train":   ("openarm_control.rl.train", "Train a SAC policy",
                   "--task reach|pick|insert|balance  --timesteps N  --eval  --out PATH  --logdir DIR"),
    "rl-eval":    ("openarm_control.rl.eval", "Watch a trained policy",
                   "--task reach|pick|insert|balance  --model PATH  --episodes N"),
    "bc-collect": ("openarm_control.imitation.collect", "Collect scripted demos -> npz dataset (state, or +images for vision policies)",
                   "--task NAME  --episodes N  --seed N  --keep-all  --images  --camera NAME  --img-size N  --out PATH"),
    "device":     ("openarm_control.imitation.device", "Report the training device (CUDA GPU / VRAM, or CPU)", ""),
    "bc-train":   ("openarm_control.imitation.bc", "Train a behavior-cloning policy on demos",
                   "--task NAME  --epochs N  --lr LR  --demos PATH  --out PATH"),
    "bc-eval":    ("openarm_control.imitation.eval", "Evaluate BC (--compare-rl for BC vs SAC)",
                   "--task NAME  --model PATH  --episodes N  --seed N  --render  --compare-rl [PATH]"),
    "act":        ("openarm_control.imitation.act", "ACT (action-chunking transformer) vision+state policy: train | eval (GPU)",
                   "train --demos PATH --epochs N --chunk N  |  eval --model PATH --task NAME --episodes N"),
    "mimic":      ("openarm_control.demos.demo_teleop", "Webcam human-arm imitation",
                   "--webcam  --camera IDX  --arm right|left  --bimanual  --no-mirror  "
                   "--pick  --preview  --headless [SECONDS]"),
    "manipulate": ("openarm_control.demos.demo_manipulate", "Language-commanded manipulation",
                   "\"<command>\"  --interactive  --vision  --headless"),
    "throw":      ("openarm_control.demos.demo_throw", "Throw a ball into a bin (ballistic inverse + swing)",
                   "--multi  --bin-x X  --headless"),
    "stack":      ("openarm_control.demos.demo_stack", "Stack one cube on another (language-commanded)",
                   "\"<command>\"  --interactive  --vision  --headless"),
    "insert":     ("openarm_control.demos.demo_insert", "Peg-in-hole: round / square / cuboid peg into a matching hole",
                   "--shape round|square|cuboid  --cuboid(alias)  --headless"),
    "push":       ("openarm_control.demos.demo_push", "Non-prehensile pushing: nudge a puck to a goal without grasping",
                   "[\"push the puck to goal a\"]  --goal a|b  --interactive  --headless"),
    "tool":       ("openarm_control.demos.demo_tool", "Tool use: grasp a stick and push a far (out-of-reach) block to a far goal",
                   "--headless"),
    "cloth":      ("openarm_control.demos.demo_cloth", "Cloth folding: grasp a corner of a 9x9 deformable cloth and fold it over",
                   "--corner 0|8|72|80  --headless"),
    "admittance": ("openarm_control.demos.demo_admittance", "Compliant control: press a soft pad and yield on contact (admittance vs rigid)",
                   "--headless"),
    "unscrew":    ("openarm_control.demos.demo_unscrew", "Bottle opening: unscrew a threaded cap on a clamped bottle (multi-turn) and lift it off",
                   "--headless"),
    "balance":    ("openarm_control.demos.demo_balance", "Ball balancing: keep or track a ping-pong ball on a plate the gripper holds (PD / LQR / MPC + trajectory + perturb)",
                   "--controller pd|lqr|mpc|both  --trajectory none|circle|figure8  --perturb  --headless  --offset x,y  --duration SEC  --radius M  --period SEC"),
    "articulated":("openarm_control.demos.demo_articulated", "Articulated manipulation: open a drawer / cabinet door, or turn a valve (incl. language commands)",
                   "--task drawer|door|valve  --command \"open the drawer then turn the valve\"  --headless"),
    "interactive":("openarm_control.demos.demo_interactive", "Bimanual playground: detect objects, pick one, dual-arm does it",
                   "--scanned  --detector yolo-world|yoloe  --headless"),
    "gen-data":   ("openarm_control.vision.synthgen", "Generate auto-labeled synthetic detection data (segmentation + domain randomisation)",
                   "--scene XML  --out DIR  --n N  --seed N"),
    "detect":     ("openarm_control.vision.finetune", "Fine-tune / evaluate a detector on the synthetic data (ultralytics)",
                   "train --data data.yaml [--model --epochs --imgsz --batch]  |  eval --weights W --data data.yaml"),
}

# Curated grand tour for `openarm showcase` (each opens a viewer; close the
# window to advance). Ordered classical -> planning -> bimanual -> vision ->
# dynamic -> manipulation -> learned/teleop, so it walks the whole stack.
SHOWCASE = ["sort", "plan", "bimanual", "servo", "catch", "stack", "insert", "mimic"]


def _print_list():
    print("OpenArm commands:\n")
    for name, entry in COMMANDS.items():
        help_ = entry[1]
        opts = entry[2] if len(entry) > 2 else ""
        print(f"  {name:<11} {help_}")
        if opts:
            print(f"  {'':<11}   options: {opts}")
    print(f"\n  {'showcase':<11} run a curated sequence: {', '.join(SHOWCASE)}")
    print(f"  {'scenes':<11} list registered scenes")
    print(f"  {'test':<11} run the headless test suite (pass pytest args through)")
    print("\nRun `openarm <command> --help` for full per-flag descriptions.")


def _print_scenes():
    print("Registered scenes:\n")
    for name, path in SCENES.items():
        print(f"  {name:<18} {path}")


def _dispatch(cmd, argv):
    module_path = COMMANDS[cmd][0]
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "main"):
        raise SystemExit(f"'{cmd}' module has no main()")
    try:
        mod.main(argv)            # preferred: main(argv=None)
    except TypeError:
        mod.main()                # fallback: legacy no-arg main()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_list()
        return 0
    cmd, rest = argv[0], argv[1:]

    if cmd == "list":
        _print_list(); return 0
    if cmd == "scenes":
        _print_scenes(); return 0
    if cmd == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "tests/", *rest])
    if cmd == "showcase":
        print(f"Showcase: {', '.join(SHOWCASE)} (close each viewer window to advance)\n")
        for c in SHOWCASE:
            print(f"=== {c} ===")
            _dispatch(c, [])
        return 0
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}\n")
        _print_list()
        return 2
    _dispatch(cmd, rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
