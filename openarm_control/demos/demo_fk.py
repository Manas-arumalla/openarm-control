import os
import sys
import time
import mujoco
import mujoco.viewer

# Add the project root to sys.path so we can import 'control'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openarm_control.config import SINGLE_ARM_SCENE, RIGHT_ARM_ACTUATORS
from openarm_control.kinematics import OpenArmKinematics

def main():
    model = mujoco.MjModel.from_xml_path(SINGLE_ARM_SCENE)
    data = mujoco.MjData(model)
    kinematics = OpenArmKinematics(model, data)
    
    # Load home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        
    # Sync initial ctrl with qpos to prevent sudden jump
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in RIGHT_ARM_ACTUATORS]
    for i, act_id in enumerate(actuator_ids):
        data.ctrl[act_id] = data.qpos[kinematics.qpos_indices[i]]

    mujoco.mj_forward(model, data)
    
    print("=" * 50)
    print("Forward Kinematics Demo")
    print("Use the 'Control' panel in the viewer to move joint sliders.")
    print("The terminal will print the end-effector (EE) position.")
    print("=" * 50)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = time.time()
        while viewer.is_running():
            step_start = time.time()
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Print EE pos every 1 second
            if time.time() - last_print > 1.0:
                pos, mat = kinematics.forward_kinematics()
                print(f"EE Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
                last_print = time.time()
                
            viewer.sync()
            
            # Real-time synchronization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
