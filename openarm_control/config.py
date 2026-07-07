import os
from dataclasses import dataclass, field
from typing import List, Tuple

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V2_MODEL_DIR = os.path.join(PROJECT_ROOT, "v2", "openarm_mujoco_v2")

# Scene files
SINGLE_ARM_SCENE = os.path.join(V2_MODEL_DIR, "single_arm_scene.xml")
OBSTACLE_SCENE = os.path.join(V2_MODEL_DIR, "single_arm_scene_obstacle.xml")
BIMANUAL_SCENE = os.path.join(V2_MODEL_DIR, "bimanual_scene.xml")
REACH_SCENE = os.path.join(V2_MODEL_DIR, "reach_scene.xml")
VISION_SCENE = os.path.join(V2_MODEL_DIR, "vision_scene.xml")
RL_PICK_SCENE = os.path.join(V2_MODEL_DIR, "rl_pick_scene.xml")
CATCH_SCENE = os.path.join(V2_MODEL_DIR, "catch_scene.xml")
CATCH_BIMANUAL_SCENE = os.path.join(V2_MODEL_DIR, "catch_bimanual_scene.xml")
CATCH_TWOBALL_SCENE = os.path.join(V2_MODEL_DIR, "catch_twoball_scene.xml")
TELEOP_SCENE = os.path.join(V2_MODEL_DIR, "teleop_scene.xml")
TELEOP_PICK_SCENE = os.path.join(V2_MODEL_DIR, "teleop_pick_scene.xml")
TABLETOP_SCENE = os.path.join(V2_MODEL_DIR, "tabletop_scene.xml")
THROW_SCENE = os.path.join(V2_MODEL_DIR, "throw_scene.xml")
THROW_MULTI_SCENE = os.path.join(V2_MODEL_DIR, "throw_multi_scene.xml")
STACK_SCENE = os.path.join(V2_MODEL_DIR, "stack_scene.xml")
PEG_SOCKET_SCENE = os.path.join(V2_MODEL_DIR, "peg_socket_scene.xml")
PEG_SQUARE_SCENE = os.path.join(V2_MODEL_DIR, "peg_square_scene.xml")
PEG_CUBOID_SCENE = os.path.join(V2_MODEL_DIR, "peg_cuboid_scene.xml")
MOVE_PUCK_SCENE = os.path.join(V2_MODEL_DIR, "move_puck_scene.xml")
PUSH_SCENE = MOVE_PUCK_SCENE   # legacy alias (scene renamed to match upstream)
TOOL_SCENE = os.path.join(V2_MODEL_DIR, "tool_scene.xml")
CLOTH_SCENE = os.path.join(V2_MODEL_DIR, "cloth_scene.xml")
BIMANUAL_STACK_SCENE = os.path.join(V2_MODEL_DIR, "bimanual_stack_scene.xml")
BIMANUAL_HANDOVER_SCENE = os.path.join(V2_MODEL_DIR, "bimanual_handover_scene.xml")
BIMANUAL_TABLE_SCENE = os.path.join(V2_MODEL_DIR, "bimanual_table_scene.xml")
SCANNED_TABLE_SCENE = os.path.join(V2_MODEL_DIR, "scanned_table_scene.xml")
ARTICULATED_SCENE = os.path.join(V2_MODEL_DIR, "articulated_scene.xml")
CONTACT_SCENE = os.path.join(V2_MODEL_DIR, "contact_scene.xml")
UNSCREW_SCENE = os.path.join(V2_MODEL_DIR, "unscrew_scene.xml")
BALANCE_SCENE = os.path.join(V2_MODEL_DIR, "balance_scene.xml")

# Registry of named scenes (used by the `openarm scenes` CLI command).
SCENES = {
    "single": SINGLE_ARM_SCENE,
    "obstacle": OBSTACLE_SCENE,
    "bimanual": BIMANUAL_SCENE,
    "reach": REACH_SCENE,
    "vision": VISION_SCENE,
    "rl_pick": RL_PICK_SCENE,
    "catch": CATCH_SCENE,
    "catch_bimanual": CATCH_BIMANUAL_SCENE,
    "catch_twoball": CATCH_TWOBALL_SCENE,
    "teleop": TELEOP_SCENE,
    "teleop_pick": TELEOP_PICK_SCENE,
    "tabletop": TABLETOP_SCENE,
    "throw": THROW_SCENE,
    "throw_multi": THROW_MULTI_SCENE,
    "stack": STACK_SCENE,
    "peg_socket": PEG_SOCKET_SCENE,
    "peg_square": PEG_SQUARE_SCENE,
    "peg_cuboid": PEG_CUBOID_SCENE,
    "move_puck": MOVE_PUCK_SCENE,
    "push": MOVE_PUCK_SCENE,   # legacy alias
    "tool": TOOL_SCENE,
    "cloth": CLOTH_SCENE,
    "bimanual_stack": BIMANUAL_STACK_SCENE,
    "bimanual_handover": BIMANUAL_HANDOVER_SCENE,
    "bimanual_table": BIMANUAL_TABLE_SCENE,
    "scanned_table": SCANNED_TABLE_SCENE,
    "articulated": ARTICULATED_SCENE,
    "contact": CONTACT_SCENE,
    "unscrew": UNSCREW_SCENE,
    "balance": BALANCE_SCENE,
}

# Right arm identifiers
RIGHT_ARM_JOINTS = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7"
]

RIGHT_ARM_ACTUATORS = [
    "right_joint1_ctrl",
    "right_joint2_ctrl",
    "right_joint3_ctrl",
    "right_joint4_ctrl",
    "right_joint5_ctrl",
    "right_joint6_ctrl",
    "right_joint7_ctrl"
]

RIGHT_GRIPPER_ACTUATOR = "right_finger1_ctrl"
RIGHT_EE_SITE = "right_ee_control_point"

# Grasp point expressed in the end-effector body frame (meters).
# The wrist site (right_ee_control_point) sits at the ee_base_link origin; the
# finger pads close ~0.135 m further along the local -z (gripper approach) axis,
# with the fingertips reaching ~0.164 m. Targeting this point makes the gripper
# close *around* an object instead of in mid-air above it.
GRASP_LOCAL_OFFSET = (0.0, 0.0, -0.135)

# Gripper control: actuator ctrl range is [-0.7854, 0]. Measured finger gap:
#   ctrl =  0.0    -> ~30 mm  (narrow / closed onto a block)
#   ctrl = -0.7854 -> ~134 mm (wide open)
# So the polarity is the opposite of what the name suggests: 0 closes, -0.785 opens.
GRIPPER_OPEN_CTRL = -0.7854
GRIPPER_CLOSED_CTRL = 0.0

# IK & Control defaults
IK_MAX_ITERS = 200
IK_TOLERANCE = 1e-4          # 0.1 mm position convergence
IK_ACCEPT = 5e-3             # 5 mm: worst best-effort solve still returned as a
                             # solution (task scale -- few-mm chained waypoints are
                             # functionally exact; a solve tens of cm off is a
                             # failure and returns None)
IK_DAMPING = 5e-2            # initial Levenberg-Marquardt damping (adaptive)
IK_RESTARTS = 24             # random restarts before giving up
IK_REST_WEIGHT = 0.02        # nullspace pull toward rest pose (redundancy resolution)
IK_MAX_STEP = 0.3            # max joint-space step per iteration (rad)

CARTESIAN_POS_GAIN = 4.0     # resolved-rate position error gain (1/s); stable across workspace
CARTESIAN_ORI_GAIN = 3.0     # resolved-rate orientation error gain (1/s)


# ---------------------------------------------------------------------------
# Per-arm specification. The model is bimanual; the left arm mirrors the right
# (works on +y instead of -y) and its gripper opens at +0.7854 instead of
# -0.7854. Everything arm-specific is captured here so the kinematics, grasp,
# controller, and pick-and-place code can drive either arm.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmSpec:
    name: str                          # "right" or "left"
    gripper_open: float                # gripper actuator ctrl for fully open
    gripper_closed: float = 0.0        # gripper actuator ctrl for closed
    grasp_offset: Tuple[float, float, float] = GRASP_LOCAL_OFFSET
    joints: List[str] = field(default_factory=list)
    actuators: List[str] = field(default_factory=list)

    @property
    def gripper_actuator(self) -> str:
        return f"{self.name}_finger1_ctrl"

    @property
    def ee_site(self) -> str:
        return f"{self.name}_ee_control_point"

    @property
    def ee_body(self) -> str:
        return f"openarm_{self.name}_ee_base_link"

    def weld(self, obj: str) -> str:
        """Weld name attaching this arm's gripper to object suffix, e.g. 'red'."""
        return f"grasp_{self.name}_{obj}"


def _make_arm(name: str, gripper_open: float) -> ArmSpec:
    return ArmSpec(
        name=name,
        gripper_open=gripper_open,
        joints=[f"openarm_{name}_joint{i}" for i in range(1, 8)],
        actuators=[f"{name}_joint{i}_ctrl" for i in range(1, 8)],
    )


RIGHT_ARM = _make_arm("right", gripper_open=-0.7854)
LEFT_ARM = _make_arm("left", gripper_open=0.7854)
ARMS = {"right": RIGHT_ARM, "left": LEFT_ARM}

# Joint-space sagittal mirror: q_left = MIRROR_R2L * q_right reproduces the
# right arm's pose reflected across the y=0 plane EXACTLY (verified 0.000 mm /
# 0.000 deg). Derived from the left arm's mirrored joint axes; only joint 4 keeps
# its sign. Used for perfectly synchronized mirrored bimanual motion.
MIRROR_R2L = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
