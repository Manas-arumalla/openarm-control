import os
import sys
import time
import mujoco
import mujoco.viewer

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import SINGLE_ARM_SCENE, RIGHT_ARM_ACTUATORS
from openarm_control.kinematics import OpenArmKinematics
from openarm_control.controller import CartesianController

def main():
    model = mujoco.MjModel.from_xml_path(SINGLE_ARM_SCENE)
    data = mujoco.MjData(model)

    kinematics = OpenArmKinematics(model, data)
    controller = CartesianController(model, data)

    # Get mocap body id
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap")

    # Load ready keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)

    # Sync initial ctrl with qpos
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in RIGHT_ARM_ACTUATORS]
    for i, act_id in enumerate(actuator_ids):
        data.ctrl[act_id] = data.qpos[kinematics.qpos_indices[i]]

    # Place mocap target at current EE position
    current_pos, current_mat = kinematics.forward_kinematics()
    data.mocap_pos[model.body_mocapid[mocap_id]] = current_pos
    # Note: For full pose control, we would sync orientation too, but let's stick to 3D position tracking here.

    print("=" * 50)
    print("Cartesian Control Demo (Resolved-Rate IK)")
    print("1. Alt+LeftClick in the viewer to select the red target sphere (target_mocap).")
    print("2. Ctrl+RightClick to drag the target around.")
    print("The arm will automatically follow the target sphere in real-time.")
    print("=" * 50)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Read target from mocap body
            target_pos = data.mocap_pos[model.body_mocapid[mocap_id]]

            # Step controller
            controller.set_target(target_pos)
            controller.step()

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time synchronization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
