import mujoco
import numpy as np
from .config import RIGHT_ARM_JOINTS

class DynamicsUtils:
    """Utilities for dynamics computations (gravity, Coriolis, mass matrix)."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in RIGHT_ARM_JOINTS]
        
        if -1 in self.joint_ids:
            raise ValueError(f"One or more joints not found in model: {RIGHT_ARM_JOINTS}")
            
        self.dof_indices = [model.jnt_dofadr[jid] for jid in self.joint_ids]
        
    def get_gravity_coriolis_torques(self):
        """
        Returns the bias forces (gravity + Coriolis + centrifugal) for the arm joints.
        Must be called after mujoco.mj_forward or mj_rne.
        """
        # mj_rne computes qfrc_bias which includes gravity and Coriolis/centrifugal
        return self.data.qfrc_bias[self.dof_indices].copy()
        
    def get_mass_matrix(self):
        """
        Computes the 7x7 mass (inertia) matrix for the arm joints.
        Must be called after mujoco.mj_kinematics or mj_forward.
        """
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        
        # Extract submatrix for the arm DOFs
        ix = np.ix_(self.dof_indices, self.dof_indices)
        return M[ix].copy()
