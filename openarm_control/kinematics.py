"""Forward / inverse kinematics for one OpenArm arm.

Wraps MuJoCo's kinematics as ground truth and adds a robust damped
least-squares IK with adaptive Levenberg-Marquardt damping, nullspace
redundancy resolution (pull toward a rest pose / away from joint limits),
and random restarts so it reliably converges for reachable targets.

The "tool point" can be offset from the control site (e.g. the gripper's
grasp point), so IK and the controller can target where the fingers actually
close instead of the wrist.
"""

import mujoco
import numpy as np

from .config import (
    RIGHT_ARM_JOINTS, RIGHT_EE_SITE, IK_MAX_ITERS, IK_TOLERANCE,
    IK_DAMPING, IK_RESTARTS, IK_REST_WEIGHT, IK_MAX_STEP,
)


def orientation_error(R_cur, R_des):
    """World-frame rotation vector taking R_cur to R_des (axis * angle)."""
    q_cur = np.zeros(4)
    q_des = np.zeros(4)
    mujoco.mju_mat2Quat(q_cur, np.asarray(R_cur, dtype=float).reshape(9))
    mujoco.mju_mat2Quat(q_des, np.asarray(R_des, dtype=float).reshape(9))
    # error quaternion = q_des * conj(q_cur)  (rotation expressed in world frame)
    q_cur_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, q_des, q_cur_inv)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, q_err, 1.0)
    return vel


class OpenArmKinematics:
    """Forward and inverse kinematics for one OpenArm arm (default: right)."""

    def __init__(self, model, data, joint_names=RIGHT_ARM_JOINTS,
                 site_name=RIGHT_EE_SITE, tool_offset=None):
        self.model = model
        self.data = data

        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in joint_names]
        if -1 in self.joint_ids:
            raise ValueError(f"One or more joints not found: {joint_names}")

        self.dof_indices = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.qpos_indices = np.array([model.jnt_qposadr[j] for j in self.joint_ids])

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"Site {site_name} not found in model.")
        self.ee_body_id = int(model.site_bodyid[self.site_id])

        self.tool_offset = (np.zeros(3) if tool_offset is None
                            else np.asarray(tool_offset, dtype=float))

        # Joint limits + rest pose (range centers) for nullspace redundancy.
        self.jnt_low = np.array([model.jnt_range[j][0] for j in self.joint_ids])
        self.jnt_high = np.array([model.jnt_range[j][1] for j in self.joint_ids])
        self.rest_pose = 0.5 * (self.jnt_low + self.jnt_high)

    # ------------------------------------------------------------------ FK ---
    def _tool_pose(self):
        """Tool-point world position and EE-site orientation (uses current data)."""
        p_site = self.data.site_xpos[self.site_id]
        R = self.data.site_xmat[self.site_id].reshape(3, 3)
        return p_site + R @ self.tool_offset, R

    def forward_kinematics(self, q=None):
        """FK to the tool point. Returns (position, 3x3 rotation). Restores state."""
        if q is None:
            return (lambda p, R: (p.copy(), R.copy()))(*self._tool_pose())

        qpos_orig = self.data.qpos.copy()
        self.data.qpos[self.qpos_indices] = q
        mujoco.mj_kinematics(self.model, self.data)
        p, R = self._tool_pose()
        p, R = p.copy(), R.copy()
        self.data.qpos[:] = qpos_orig
        mujoco.mj_kinematics(self.model, self.data)
        return p, R

    def jacobian(self, q=None):
        """6x7 tool-point Jacobian (rows 0-2 linear, 3-5 angular). Restores state."""
        if q is not None:
            qpos_orig = self.data.qpos.copy()
            self.data.qpos[self.qpos_indices] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

        J = self._jacobian_current()

        if q is not None:
            self.data.qpos[:] = qpos_orig
            mujoco.mj_kinematics(self.model, self.data)
        return J

    def _jacobian_current(self):
        """Tool-point Jacobian for the arm DOFs at the current configuration.

        Requires mj_comPos (cdof) to be current for the configuration in
        data.qpos -- callers must run mj_kinematics + mj_comPos first.
        """
        point, _ = self._tool_pose()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.ee_body_id)
        return np.vstack([jacp[:, self.dof_indices], jacr[:, self.dof_indices]])

    # ------------------------------------------------------------------ IK ---
    def _solve_from_seed(self, q0, target_pos, target_mat, max_iters, tol,
                         rest_weight, damping):
        """One LM/DLS descent from seed q0. Returns (q, error_norm)."""
        q = np.clip(np.asarray(q0, dtype=float).copy(), self.jnt_low, self.jnt_high)
        lam = damping
        prev = np.inf
        ne = np.inf
        n_task = 6 if target_mat is not None else 3
        I_task = np.eye(n_task)
        I_joint = np.eye(7)

        for _ in range(max_iters):
            self.data.qpos[self.qpos_indices] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            p_tool, R = self._tool_pose()

            err = target_pos - p_tool
            if target_mat is not None:
                err = np.concatenate([err, orientation_error(R, target_mat)])
            ne = np.linalg.norm(err)
            if ne < tol:
                return q, ne

            # Adaptive damping: shrink on progress, grow when stuck/overshooting.
            if ne < prev:
                lam = max(lam * 0.7, 1e-4)
            else:
                lam = min(lam * 2.0, 10.0)
            prev = ne

            J = self._jacobian_current()
            if target_mat is None:
                J = J[:3]

            H = J @ J.T + (lam ** 2) * I_task
            J_pinv = J.T @ np.linalg.solve(H, I_task)   # damped pseudo-inverse
            dq = J_pinv @ err

            # Nullspace bias toward rest pose (redundancy: keeps joints natural).
            if rest_weight > 0.0:
                N = I_joint - J_pinv @ J
                dq += N @ (rest_weight * (self.rest_pose - q))

            step = np.linalg.norm(dq)
            if step > IK_MAX_STEP:
                dq *= IK_MAX_STEP / step
            q = np.clip(q + dq, self.jnt_low, self.jnt_high)

        return q, ne

    def inverse_kinematics(self, target_pos, target_mat=None, q_init=None,
                           max_iters=IK_MAX_ITERS, tol=IK_TOLERANCE,
                           damping=IK_DAMPING, restarts=IK_RESTARTS,
                           rest_weight=IK_REST_WEIGHT, return_info=False, seed=None):
        """Robust IK to the tool point.

        Tries the supplied seed, the current pose, and the rest pose first, then
        random restarts until a seed converges within ``tol``. Returns the joint
        solution (and, if ``return_info``, a dict with success/error/iterations).
        State is restored on exit.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        if target_mat is not None:
            target_mat = np.asarray(target_mat, dtype=float).reshape(3, 3)

        qpos_orig = self.data.qpos.copy()
        rng = np.random.default_rng(seed)

        seeds = []
        if q_init is not None:
            seeds.append(np.asarray(q_init, dtype=float))
        seeds.append(self.data.qpos[self.qpos_indices].copy())
        seeds.append(self.rest_pose.copy())

        best_q, best_err, n_tried = None, np.inf, 0
        for i in range(3 + max(0, restarts)):
            seed = (seeds[i] if i < len(seeds)
                    else rng.uniform(self.jnt_low, self.jnt_high))
            q, err = self._solve_from_seed(seed, target_pos, target_mat,
                                           max_iters, tol, rest_weight, damping)
            n_tried += 1
            if err < best_err:
                best_q, best_err = q, err
            if best_err < tol:
                break

        self.data.qpos[:] = qpos_orig
        mujoco.mj_kinematics(self.model, self.data)

        if return_info:
            return best_q, {"success": bool(best_err < tol),
                            "error": float(best_err), "seeds_tried": n_tried}
        return best_q
