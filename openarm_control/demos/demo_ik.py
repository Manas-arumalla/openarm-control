import os
import sys
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import mujoco
from openarm_control.config import SINGLE_ARM_SCENE
from openarm_control.kinematics import OpenArmKinematics

def main():
    print("=" * 50)
    print("Inverse Kinematics Offline Demo")
    print("=" * 50)

    model = mujoco.MjModel.from_xml_path(SINGLE_ARM_SCENE)
    data = mujoco.MjData(model)
    kinematics = OpenArmKinematics(model, data)

    # Load home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Current EE position
    start_pos, start_mat = kinematics.forward_kinematics()
    print(f"Initial EE Position: {start_pos}")

    # Define a reachable target in the manipulation workspace
    target_pos = np.array([0.28, -0.22, 0.50])
    print(f"Target EE Position:  {target_pos}")

    # Solve IK
    print("\nSolving IK...")
    q_sol, info = kinematics.inverse_kinematics(target_pos, return_info=True)
    print(f"  converged={info['success']} (seeds tried: {info['seeds_tried']})")

    # Verify solution
    # We pass q_sol to forward_kinematics to see where it actually puts the EE
    achieved_pos, _ = kinematics.forward_kinematics(q_sol)

    print("\nResults:")
    print(f"Joint Angle Solution: {np.round(q_sol, 3)}")
    print(f"Achieved Position:    {achieved_pos}")
    error = np.linalg.norm(target_pos - achieved_pos)
    print(f"Position Error:       {error*1000:.2f} mm")

if __name__ == "__main__":
    main()
