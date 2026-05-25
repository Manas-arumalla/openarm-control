"""Cartesian admittance control for the OpenArm.

Admittance control makes the end-effector behave like a virtual mass-spring-damper
in response to the external (contact) force, instead of rigidly tracking a
commanded pose. Per control tick we

1. measure the external force on the gripper (summed contact forces over the
   end-effector subtree, in world frame),
2. integrate the virtual dynamics ``M x_ddot + D x_dot + K (x_ref - x_des) =
   F_ext`` to get a *compliant* reference ``x_ref`` (a soft K backs the reference
   off from the commanded pose when it pushes into something), and
3. solve IK to ``x_ref`` and drive the **existing** position actuators with
   gravity compensation -- exactly the low-level stack the rest of the codebase
   uses.

The orientation is held rigidly (translational admittance), which is what guarded
insertion, wiping and drawer/door/valve operation need. This adds a new control
mode beside position control; it changes nothing existing.
"""

import numpy as np
import mujoco

from ..config import RIGHT_ARM
from ..kinematics import OpenArmKinematics


def _in_subtree(model, body, root):
    """True if ``body`` is ``root`` or a descendant of it."""
    a = body
    while True:
        if a == root:
            return True
        if a <= 0:
            return False
        a = model.body_parentid[a]


def _subtree_geoms(model, root_body):
    """Set of geom ids belonging to ``root_body`` or any descendant body."""
    return {g for g in range(model.ngeom)
            if _in_subtree(model, int(model.geom_bodyid[g]), root_body)}


def ee_contact_force(model, data, ee_geoms):
    """Net external contact force (world frame, N) on the given end-effector geoms.

    Sums every active contact that involves an EE geom, converting each contact's
    force from the contact frame to world and applying Newton's third law so the
    result is the force acting *on* the end-effector.
    """
    f = np.zeros(3)
    buf = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        on1, on2 = g1 in ee_geoms, g2 in ee_geoms
        if on1 == on2:                     # neither, or internal (finger-finger): skip
            continue
        mujoco.mj_contactForce(model, data, i, buf)
        R_c = np.array(c.frame, dtype=float).reshape(3, 3)
        # mj_contactForce gives the wrench acting on geom2 (reaction on geom1 is
        # equal and opposite); take the component acting on the EE geom.
        f_world = R_c.T @ buf[:3]
        f += f_world if on2 else -f_world
    return f


class AdmittanceController:
    """Translational Cartesian admittance on top of the position-control stack."""

    def __init__(self, model, data, arm=RIGHT_ARM, stiffness=400.0, mass=1.5,
                 damping_ratio=1.0, tool_offset=None, max_offset=0.06,
                 force_alpha=0.15, max_speed=0.3):
        self.model = model
        self.data = data
        self.arm = arm
        offset = arm.grasp_offset if tool_offset is None else tool_offset
        self.king = OpenArmKinematics(model, data, joint_names=arm.joints,
                                      site_name=arm.ee_site, tool_offset=offset)
        self.arm_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                         for n in arm.actuators]
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                          arm.gripper_actuator)
        self.ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_body)
        self.ee_geoms = _subtree_geoms(model, self.ee_body)

        # virtual dynamics (isotropic): M x_ddot + D x_dot + K (x_ref - x_des) = F
        self.K = float(stiffness)
        self.M = float(mass)
        self.D = float(damping_ratio) * 2.0 * np.sqrt(self.M * self.K)
        self.max_offset = float(max_offset)   # clamp |x_ref - x_des| for safety
        self.force_alpha = float(force_alpha)  # EMA on the (spiky) contact force
        self.max_speed = float(max_speed)      # clamp |reference velocity|

        self._x_ref = None
        self._v = np.zeros(3)
        self._f_filt = np.zeros(3)
        self._R = None

    # -- queries -----------------------------------------------------------
    def ee_pos(self):
        pos, _ = self.king.forward_kinematics()
        return pos

    def contact_force(self):
        return ee_contact_force(self.model, self.data, self.ee_geoms)

    def _arm_q(self):
        return self.data.qpos[self.king.qpos_indices].copy()

    # -- control -----------------------------------------------------------
    def reset(self, x_desired, R_desired):
        """Initialise the compliant reference at the commanded pose."""
        self._x_ref = np.asarray(x_desired, dtype=float).copy()
        self._v = np.zeros(3)
        self._f_filt = np.zeros(3)
        self._R = np.asarray(R_desired, dtype=float).reshape(3, 3)

    def step(self, x_desired, R_desired=None, grip=None, gravity_comp=True):
        """Advance one simulation step toward ``x_desired`` with admittance.

        Returns ``(force_world, ee_position)`` measured this step.
        """
        x_desired = np.asarray(x_desired, dtype=float)
        if R_desired is not None:
            self._R = np.asarray(R_desired, dtype=float).reshape(3, 3)
        if self._x_ref is None:
            _, R_now = self.king.forward_kinematics()
            self.reset(x_desired, self._R if self._R is not None else R_now)

        dt = self.model.opt.timestep
        F = self.contact_force()
        # low-pass the spiky contact force before it drives the integrator
        self._f_filt += self.force_alpha * (F - self._f_filt)

        # integrate the virtual mass-spring-damper for the compliant reference
        acc = (self._f_filt - self.D * self._v - self.K * (self._x_ref - x_desired)) / self.M
        self._v += acc * dt
        sp = float(np.linalg.norm(self._v))
        if sp > self.max_speed:                # clamp reference speed for stability
            self._v *= self.max_speed / sp
        self._x_ref += self._v * dt

        # clamp the compliant offset so a runaway force can't fling the reference
        off = self._x_ref - x_desired
        d = float(np.linalg.norm(off))
        if d > self.max_offset:
            self._x_ref = x_desired + off * (self.max_offset / d)
            self._v *= 0.5

        q = self.king.inverse_kinematics(self._x_ref, target_mat=self._R,
                                         q_init=self._arm_q(), restarts=0,
                                         rest_weight=0.0)
        if q is not None:
            for i, a in enumerate(self.arm_acts):
                self.data.ctrl[a] = q[i]
        if self.grip_act != -1 and grip is not None:
            self.data.ctrl[self.grip_act] = grip
        if gravity_comp:
            self.data.qfrc_applied[self.king.dof_indices] = \
                self.data.qfrc_bias[self.king.dof_indices]

        mujoco.mj_step(self.model, self.data)
        return F, self.ee_pos()
