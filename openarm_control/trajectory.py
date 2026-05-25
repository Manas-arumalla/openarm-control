import numpy as np

def quintic_polynomial(t, t_0, t_f, q_0, q_f, v_0=0, v_f=0, a_0=0, a_f=0):
    """
    Computes a quintic polynomial trajectory at time t.
    Returns (position, velocity, acceleration)
    """
    if t <= t_0:
        return q_0, v_0, a_0
    if t >= t_f:
        return q_f, v_f, a_f
        
    T = t_f - t_0
    tau = (t - t_0) / T
    
    # Base quintic spline for 0 to 1 with zero boundary derivatives
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / (T**2)
    
    q = q_0 + (q_f - q_0) * s
    v = (q_f - q_0) * ds
    a = (q_f - q_0) * dds
    
    return q, v, a

class JointTrajectory:
    """Smooth joint-space trajectory (quintic polynomial)."""
    def __init__(self, q_start, q_end, duration):
        self.q_start = np.array(q_start)
        self.q_end = np.array(q_end)
        self.duration = duration
        
    def evaluate(self, t):
        """Returns joint positions at time t (0 <= t <= duration)."""
        if t <= 0: return self.q_start.copy()
        if t >= self.duration: return self.q_end.copy()
        
        q = np.zeros_like(self.q_start)
        for i in range(len(q)):
            q[i], _, _ = quintic_polynomial(t, 0, self.duration, self.q_start[i], self.q_end[i])
        return q

class CartesianTrajectory:
    """Straight-line Cartesian trajectory with optional gripper state."""
    def __init__(self, pos_start, pos_end, duration, gripper_start=0.0, gripper_end=0.0):
        self.pos_start = np.array(pos_start)
        self.pos_end = np.array(pos_end)
        self.duration = duration
        self.gripper_start = gripper_start
        self.gripper_end = gripper_end
        
    def evaluate(self, t):
        """Returns (Cartesian position, gripper state) at time t (0 <= t <= duration)."""
        if t <= 0: return self.pos_start.copy(), self.gripper_start
        if t >= self.duration: return self.pos_end.copy(), self.gripper_end
        
        pos = np.zeros_like(self.pos_start)
        for i in range(3):
            pos[i], _, _ = quintic_polynomial(t, 0, self.duration, self.pos_start[i], self.pos_end[i])
            
        # Linear interpolation for gripper
        tau = t / self.duration
        gripper = self.gripper_start + (self.gripper_end - self.gripper_start) * tau
        
        return pos, gripper

class TrajectoryExecutor:
    """Executes trajectories on the model."""
    def __init__(self, model, data, kinematics, actuator_ids):
        self.model = model
        self.data = data
        self.kinematics = kinematics
        self.actuator_ids = actuator_ids
        
    def step_joint(self, traj, t):
        """Sets actuator commands for joint trajectory at time t."""
        q_cmd = traj.evaluate(t)
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = q_cmd[i]
            
    def step_cartesian(self, traj, t, current_q=None):
        """
        Sets actuator commands for Cartesian trajectory at time t.
        Solves IK at each step. Returns the IK solution to use as 
        current_q in the next step to avoid flips.
        """
        pos_target, gripper_target = traj.evaluate(t)
        # Solve IK for pos_target
        q_cmd = self.kinematics.inverse_kinematics(pos_target, q_init=current_q)
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = q_cmd[i]
            
        # Command gripper
        try:
            gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger1_ctrl")
            if gripper_act_id != -1:
                self.data.ctrl[gripper_act_id] = gripper_target * -0.785
        except:
            pass
            
        return q_cmd
