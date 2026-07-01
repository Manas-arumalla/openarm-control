"""Ball-on-plate balancing on top of the right gripper (Phase B1).

The classical ball-on-plate stabilisation problem, lifted onto a 7-DOF
manipulator. A square plate is welded to the right gripper at a fixed
hold pose; a ping-pong ball rolls on the plate. Tilting the plate by
small angles tilts the gravity component along the plate surface and
rolls the ball -- a textbook 2-axis tilting-platform control problem.

Only **2 axes of plate tilt** (roll about world X, pitch about world Y)
do the balancing; the arm's other DOF hold the plate centred in space.
We rotate the **gripper** by the desired tilt in the world frame, then
solve IK at the fixed hold position; the rigidly welded plate follows.

Why pre-multiply the world tilt onto the initial gripper orientation
(rather than command an absolute orientation in some plate-local
frame): the weld stores plate-vs-gripper as a constant relative pose
at attach time, so a world-frame rotation of the gripper produces an
equal world-frame rotation of the plate. The plate stays horizontal
when the gripper is at its initial orientation, and tilts by exactly
(roll, pitch) when the gripper does -- which is all the controller
needs to reason about.

Two controllers ship here, sharing the same hold + IK + step loop:
  - ``PDBalancer``  -- analytic PD on ball position + velocity (Tier 1).
  - ``LQRBalancer`` -- discrete LQR on the linearised ball-on-plate
    dynamics (Tier 2; added once PD is verified).
"""

import numpy as np
import mujoco

from .config import RIGHT_ARM, LEFT_ARM
from .pick_and_place import PickPlaceController


# Ball-on-plate effective gravity along the surface: a rolling sphere of
# uniform density has a (5/7) factor on translational acceleration vs.
# tilt angle. The signed axis convention used below (verified empirically):
#   positive pitch about world +Y -> plate +X edge goes DOWN -> ball accels +X
#   positive roll  about world +X -> plate +Y edge goes UP   -> ball accels -Y
# PD laws follow these signs (see PDBalancer.step).
G_EFF = (5.0 / 7.0) * 9.81


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _tilt_to_R(roll, pitch):
    """World-frame rotation that applies ``pitch`` (about world Y) then ``roll``
    (about world X). For small angles the order is irrelevant."""
    return _rotx(roll) @ _roty(pitch)


def _mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


class BallBalancer:
    """Shared hold + plate-tilt scaffolding. The actual control law lives
    in subclasses (``PDBalancer``, ``LQRBalancer``)."""

    # IK *target* (where to aim the gripper tool point). The hold reference used
    # in subsequent control steps is the ACHIEVED pose after the initial IK,
    # captured into ``_hold_pos`` + ``_R_grip_init`` -- so re-IKing with zero
    # tilt produces zero motion (the target equals the current pose). Re-targeting
    # the *commanded* pose every step instead would drag the gripper toward an
    # unachievable point and fling the welded plate.
    HOLD_POS = np.array([0.32, -0.12, 0.55])
    # Cap commanded tilt so the arm always stays well inside joint limits and
    # the plate never gets so steep that the ball goes ballistic.
    MAX_TILT = np.deg2rad(20.0)

    def __init__(self, model, data, hold_pos=None, max_tilt=None):
        self.model, self.data = model, data
        if hold_pos is not None: self.HOLD_POS = np.asarray(hold_pos, float)
        if max_tilt is not None: self.MAX_TILT = float(max_tilt)
        # Reuse the existing pick-place infra for IK + actuator ids on both arms.
        self.right = PickPlaceController(model, data, arm=RIGHT_ARM)
        self.left  = PickPlaceController(model, data, arm=LEFT_ARM)
        # Plate + ball + weld ids.
        self.plate_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self.ball_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.plate_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")]
        self.plate_dadr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_free")]
        self.ball_qadr  = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
        self.ball_dadr  = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
        self.weld_eid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "plate_to_right_ee")
        # Achieved hold pose -- captured AFTER the multi-restart IK in
        # setup_hold(). All subsequent IK calls target ``_hold_pos`` (achieved
        # tool position) with ``R_tilt @ _R_grip_init`` (achieved orientation
        # pre-multiplied by the commanded world-frame tilt) so a zero tilt
        # command produces zero motion -- the IK target equals the current pose.
        self._hold_pos = None
        self._R_grip_init = None
        self._left_q_park = None
        # Pre-allocated Jacobian buffers for the per-step welded-mass gravity
        # comp (the plate is a free body, so its weight doesn't appear in the
        # arm's qfrc_bias -- without explicit comp the arm sags under it).
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._plate_total_mass = float(model.body_mass[self.plate_bid])

    # -- setup ------------------------------------------------------------
    def setup_hold(self):
        """Use the right arm's existing 'ready' pose as the hold reference --
        no IK in setup, so no IK-convergence failures here. We query forward
        kinematics for where the gripper actually is, place the plate
        horizontally just above the tool point, weld, and stash the achieved
        (tool_point, orientation) as the IK *target* for every subsequent
        control step. A zero-tilt command then re-IKs to the exact current
        pose -> zero motion -> no drift."""
        # Park the LEFT arm at zeros (out of the way), leave the right arm at
        # whatever the keyframe set it to ('ready' for the balance scene).
        self.data.qpos[self.left.king.qpos_indices] = 0.0
        self.data.qvel[self.left.king.dof_indices]  = 0.0
        self.data.qvel[self.right.king.dof_indices] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Achieved tool pose at the keyframe's right-arm config.
        achieved_pos, achieved_R = self.right.king.forward_kinematics()
        self._hold_pos    = achieved_pos.copy()
        self._R_grip_init = achieved_R.copy()
        # Place the plate 6 cm ABOVE the gripper ee_body. The <contact><exclude>
        # entries in the scene XML make plate/ball collisions vs. the arm's
        # link4/5/6/finger bodies dynamically inert -- the plate CAN sit inside
        # the wrist volume without the constraint solver ejecting it. Visually
        # the riser rod (defined in balance_scene.xml, contype/conaffinity=0)
        # bridges the gap between palm and plate so the reading is still
        # "arm below, plate on top". A 6 cm offset (vs. the 20 cm needed to
        # physically clear link4's collision) keeps the lever-arm short so
        # IK-residual errors don't amplify into ball kicks.
        grip_world_pos = self.data.xpos[self.right.ee_body].copy()
        plate_pos = grip_world_pos + np.array([0.0, 0.0, 0.12])
        self.data.qpos[self.plate_qadr:self.plate_qadr + 3] = plate_pos
        self.data.qpos[self.plate_qadr + 3:self.plate_qadr + 7] = [1, 0, 0, 0]
        # Ball just above plate (will be moved by .reset()).
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = plate_pos + [0, 0, 0.025]
        self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.plate_dadr:self.plate_dadr + 6] = 0.0
        self.data.qvel[self.ball_dadr:self.ball_dadr + 6]   = 0.0
        mujoco.mj_forward(self.model, self.data)
        q_grip = self.data.qpos[self.right.king.qpos_indices].copy()
        # Compute plate-vs-gripper relative pose from the current state.
        # Used by the manual pin AND (when active) stored into the weld's eq_data.
        p1 = self.data.xpos[self.right.ee_body].copy()
        q1 = self.data.xquat[self.right.ee_body].copy()
        p2 = self.data.xpos[self.plate_bid].copy()
        q2 = self.data.xquat[self.plate_bid].copy()
        nq1 = np.zeros(4); mujoco.mju_negQuat(nq1, q1)
        relpos  = np.zeros(3); mujoco.mju_rotVecQuat(relpos, p2 - p1, nq1)
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, nq1, q2)
        self._gripper_to_plate_relpos  = relpos.copy()
        self._gripper_to_plate_relquat = relquat.copy()
        # MANUAL PINNING (robust path): stiff contact + light free-body plate +
        # weld constraint creates contact-impulse instability at ball-plate
        # collision (the ball gets a kilo-g lateral kick from a tiny normal
        # asymmetry). Keep the weld DISABLED and instead pin the plate's free
        # joint each step from the gripper pose -- no constraint solver, no
        # impulse spikes, no plate-mass gravity-comp needed (plate is purely
        # kinematic). The weld's eq_data is still seeded for completeness.
        self.model.eq_data[self.weld_eid, 0:3]  = 0.0
        self.model.eq_data[self.weld_eid, 3:6]  = relpos
        self.model.eq_data[self.weld_eid, 6:10] = relquat
        self.model.eq_data[self.weld_eid, 10]   = 1.0
        self.data.eq_active[self.weld_eid] = 0
        self._pin_manually = True
        self.model.eq_data[self.weld_eid, 10]   = 1.0
        self.data.eq_active[self.weld_eid] = 1
        # Stash the parked left-arm config for the per-step hold.
        self._left_q_park = self.data.qpos[self.left.king.qpos_indices].copy()
        # Stash the achieved plate world centre + the vector from gripper to
        # plate centre. On every tilt step we now command the gripper to
        # TRANSLATE so the plate centre stays put (only rotating). Without
        # this the plate (20 cm above the gripper via the riser rod) would
        # swing through a big arc for even a small tilt, and the "moving
        # plate under the ball" effect swamps the tilt-driven ball roll.
        self._plate_centre_fixed = self.data.xpos[self.plate_bid].copy()
        self._plate_offset_in_grip_frame = relpos.copy()
        # Initial ctrl targets so the actuators don't snap on the first step.
        for i, a in enumerate(self.right.arm_acts):
            self.data.ctrl[a] = q_grip[i]
        for i, a in enumerate(self.left.arm_acts):
            self.data.ctrl[a] = self._left_q_park[i]
        return q_grip

    def reset(self, ball_offset_xy=(0.0, 0.0), settle_steps=400):
        """Place the ball ~3 cm above the plate (NOT touching) so it free-falls
        onto the plate -- contact engages with a smooth, well-damped landing
        rather than the violent overshoot you get when the ball is placed
        exactly at plate-top with a stiff solver. Settle for ``settle_steps``
        so the ball is resting before control starts."""
        plate_pos = self.data.xpos[self.plate_bid].copy()
        bx = plate_pos[0] + float(ball_offset_xy[0])
        by = plate_pos[1] + float(ball_offset_xy[1])
        # plate body z is plate-centre; plate top is body_z + half_thickness (0.005).
        # Place the ball with its CENTRE 5 cm above plate body  => ball bottom is
        # 3 cm above plate top (a ~25 cm/s landing speed -- enough for smooth
        # contact resolution, low enough to not bounce visibly).
        bz = plate_pos[2] + 0.05
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = [bx, by, bz]
        self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.ball_dadr:self.ball_dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Settle: hold the plate level by re-issuing the no-tilt command each step.
        for _ in range(settle_steps):
            self._apply_tilt_and_step(0.0, 0.0)

    # -- queries ----------------------------------------------------------
    def plate_pose(self):
        """(centre_xyz, R_3x3) of the plate in the world frame."""
        c = self.data.xpos[self.plate_bid].copy()
        R = self.data.xmat[self.plate_bid].reshape(3, 3).copy()
        return c, R

    def ball_state(self):
        """(xy_local_on_plate, vxy_local_on_plate). For small tilts the plate
        frame ≈ world frame -- we use that approximation here; the controllers
        are robust to its small error within MAX_TILT (~20°)."""
        c, _ = self.plate_pose()
        b_pos = self.data.xpos[self.ball_bid].copy()
        b_vel = self.data.qvel[self.ball_dadr:self.ball_dadr + 3].copy()  # linear vel only
        xy   = b_pos[:2] - c[:2]
        vxy  = b_vel[:2]
        return xy, vxy

    # -- inner step -------------------------------------------------------
    def _apply_tilt_and_step(self, roll, pitch):
        """Solve IK for a gripper rotated by (roll, pitch) in world frame, drive
        the right-arm actuators, hold the left arm parked, gravity-compensate
        both, and step the sim once."""
        roll  = float(np.clip(roll,  -self.MAX_TILT, self.MAX_TILT))
        pitch = float(np.clip(pitch, -self.MAX_TILT, self.MAX_TILT))
        R_target = _tilt_to_R(roll, pitch) @ self._R_grip_init
        # Keep the gripper at the achieved hold pose while only rotating; the
        # plate lever-arm is now short enough (6 cm) that the small plate
        # translation from gripper rotation doesn't destabilise the ball.
        q = self.right.king.inverse_kinematics(
            self._hold_pos, target_mat=R_target,
            q_init=self.data.qpos[self.right.king.qpos_indices],
            restarts=0, rest_weight=0.0,
        )
        if q is not None:
            for i, a in enumerate(self.right.arm_acts):
                self.data.ctrl[a] = q[i]
        # Hold the left arm at its parked config (no servo drift).
        for i, a in enumerate(self.left.arm_acts):
            self.data.ctrl[a] = self._left_q_park[i]
        # Gravity-compensate both arms so the position servos don't sag.
        self.data.qfrc_applied[self.right.king.dof_indices] = \
            self.data.qfrc_bias[self.right.king.dof_indices]
        self.data.qfrc_applied[self.left.king.dof_indices] = \
            self.data.qfrc_bias[self.left.king.dof_indices]
        mujoco.mj_step(self.model, self.data)
        # Manual pin: after physics step, snap the plate's free-joint qpos to
        # the gripper-relative pose captured at attach time. The plate is then
        # a pure kinematic puppet of the gripper -- no constraint solver, no
        # contact-impulse instability, the welded plate's mass doesn't even
        # need gravity comp (it never moves under physics). The ball's contact
        # with the plate is resolved against the updated plate pose on the
        # NEXT step via mj_forward below.
        if self._pin_manually:
            self._pin_plate_to_gripper()
            mujoco.mj_forward(self.model, self.data)

    def _pin_plate_to_gripper(self):
        """Snap the plate's free-joint qpos to ``gripper_pose * relpose``,
        i.e. wherever the weld would have held it. Zeros plate qvel so the
        constraint solver sees a static plate next step."""
        grip_pos  = self.data.xpos[self.right.ee_body]
        grip_quat = self.data.xquat[self.right.ee_body]
        # plate world pos = grip + R_grip @ relpos
        R_grip = self.data.xmat[self.right.ee_body].reshape(3, 3)
        new_plate_pos = grip_pos + R_grip @ self._gripper_to_plate_relpos
        # plate world quat = grip_quat * relquat
        new_plate_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_plate_quat, grip_quat, self._gripper_to_plate_relquat)
        self.data.qpos[self.plate_qadr:self.plate_qadr + 3]      = new_plate_pos
        self.data.qpos[self.plate_qadr + 3:self.plate_qadr + 7]  = new_plate_quat
        self.data.qvel[self.plate_dadr:self.plate_dadr + 6]      = 0.0


class LQRBalancer(BallBalancer):
    """Discrete LQR controller on the linearised ball-on-plate dynamics.

    Ball-on-plate about the flat-plate equilibrium (state ``[px, py, vx, vy]``
    in plate frame, action ``[roll, pitch]``, using the sign convention verified
    in ``PDBalancer``: positive pitch -> ball accels +X; positive roll -> ball
    accels -Y). The continuous-time model is::

        dpx/dt = vx
        dpy/dt = vy
        dvx/dt = +G_EFF * pitch
        dvy/dt = -G_EFF * roll

    Euler-discretised at the sim timestep, then the discrete algebraic Riccati
    equation gives the infinite-horizon gain ``K``. Control is ``u = -K (x - x*)``
    where ``x*`` is the desired ball state (``target_xy`` + zero velocity).

    A cleaner, more principled controller than PD: it jointly weights position
    and velocity across both axes and returns the *optimal* linear feedback for
    the chosen ``Q``, ``R`` cost. For this SIMO problem the practical difference
    versus a well-tuned PD is modest, but the derivation is worth showing.
    """

    # Q, R tuned so tilts stay well inside MAX_TILT (20 deg) even at 60 mm
    # offsets -- the arm's position-servo bandwidth means a tilt commanded
    # too aggressively takes several steps to be achieved, and the transient
    # of the arm swinging THROUGH the wrong tilt on the way to the right one
    # can drive the ball off before the correct tilt actually helps. Gentler
    # gains keep the commanded tilt within one-step actuator range.
    Q = np.diag([40.0, 40.0, 8.0, 8.0])    # position penalty >> velocity penalty
    R = np.diag([8.0, 8.0])                 # generous control penalty (soft LQR)

    def __init__(self, model, data, Q=None, R=None, **kw):
        super().__init__(model, data, **kw)
        if Q is not None: self.Q = np.asarray(Q, float)
        if R is not None: self.R = np.asarray(R, float)
        self._K = self._compute_gain(model.opt.timestep)

    @classmethod
    def _compute_gain(cls, dt):
        """Discrete LQR gain (2 x 4) for the linearised model at timestep ``dt``."""
        from scipy.linalg import solve_discrete_are
        # Continuous state matrix: state = [px, py, vx, vy]
        A_c = np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]], dtype=float)
        # Continuous input matrix: input = [roll, pitch]
        # dvx = +G_EFF * pitch  (column 1 = pitch)
        # dvy = -G_EFF * roll   (column 0 = roll)
        B_c = np.array([[0.0,     0.0],
                        [0.0,     0.0],
                        [0.0,    +G_EFF],
                        [-G_EFF,  0.0]], dtype=float)
        # Euler discretisation.
        A_d = np.eye(4) + A_c * dt
        B_d = B_c * dt
        P = solve_discrete_are(A_d, B_d, cls.Q, cls.R)
        # K = (R + B^T P B)^-1  B^T P A
        return np.linalg.solve(cls.R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)

    def step(self, target_xy=(0.0, 0.0), target_axy=(0.0, 0.0)):
        _ = target_axy   # LQR ignores it; MPC uses it. Kept for signature uniformity.
        (x, y), (vx, vy) = self.ball_state()
        tx, ty = float(target_xy[0]), float(target_xy[1])
        # State offset relative to (target_xy, zero velocity).
        state = np.array([x - tx, y - ty, vx, vy])
        roll, pitch = (-self._K @ state).tolist()
        self._apply_tilt_and_step(roll, pitch)
        (x2, y2), (vx2, vy2) = self.ball_state()
        return float(np.hypot(x2 - tx, y2 - ty)), float(np.hypot(vx2, vy2))


class MPCBalancer(LQRBalancer):
    """Model-based balancer that adds **trajectory feedforward** to LQR feedback.

    Rationale: plain LQR (regulator about a fixed target) always LAGS a moving
    reference -- for a circle, LQR traces a smaller-radius orbit inside the
    intended path because it can only react to position error. A finite-horizon
    MPC on the same linearised model would derive an anticipatory tilt from the
    target's future trajectory; for a smooth ball-on-plate model, that
    anticipation collapses to an analytic feedforward:

        pitch_ff = +ax_target / G_EFF   (positive pitch -> ball accels +X)
        roll_ff  = -ay_target / G_EFF   (positive roll  -> ball accels -Y)

    Combined control:  ``u = -K (x - x_ref) + u_ff``.

    Same LQR gain K, same cost tuning -- the ONLY change vs. ``LQRBalancer`` is
    the added feedforward. Cheap analytically, no QP solver, no new deps. For
    a static target (target_axy = 0) it reduces exactly to LQR.
    """

    def step(self, target_xy=(0.0, 0.0), target_axy=(0.0, 0.0)):
        (x, y), (vx, vy) = self.ball_state()
        tx, ty = float(target_xy[0]), float(target_xy[1])
        ax_ref, ay_ref = float(target_axy[0]), float(target_axy[1])
        # LQR feedback.
        state = np.array([x - tx, y - ty, vx, vy])
        u_fb  = -self._K @ state
        # Analytic feedforward from the linearised model:
        # accel = [-G_EFF * roll,  +G_EFF * pitch]  => invert.
        roll_ff  = -ay_ref / G_EFF
        pitch_ff = +ax_ref / G_EFF
        roll, pitch = float(u_fb[0] + roll_ff), float(u_fb[1] + pitch_ff)
        self._apply_tilt_and_step(roll, pitch)
        (x2, y2), (vx2, vy2) = self.ball_state()
        return float(np.hypot(x2 - tx, y2 - ty)), float(np.hypot(vx2, vy2))


class PDBalancer(BallBalancer):
    """Analytic PD controller on ball position + velocity. Two diagonal gains
    (no cross-coupling): pitch tracks ball-x, roll tracks ball-y, each with a
    derivative term on the corresponding ball velocity component.

    Signs (verified empirically, matching the convention noted at the top of
    this module: positive pitch -> ball accels +X; positive roll -> ball
    accels -Y, so to *return* the ball to the target we use opposite signs):
        pitch = -KP * (x - target_x) - KD * vx
        roll  = +KP * (y - target_y) + KD * vy
    """

    # KP is intentionally modest: the plate sits 6 cm above the gripper's
    # rotation axis, so a commanded tilt drags the plate horizontally by
    # ~6 cm * sin(tilt). At KP=6 this translation swamps the tilt-driven
    # ball roll and PD diverges. KP=2 keeps the tilt small enough that the
    # rolling response dominates.
    KP = 2.0     # rad of tilt per metre of ball offset
    KD = 1.2     # rad of tilt per m/s of ball velocity

    def __init__(self, model, data, kp=None, kd=None, **kw):
        super().__init__(model, data, **kw)
        if kp is not None: self.KP = float(kp)
        if kd is not None: self.KD = float(kd)

    def step(self, target_xy=(0.0, 0.0), target_axy=(0.0, 0.0)):
        """One PD control step. Returns the current (xy_err_norm, speed).
        ``target_axy`` is accepted for signature uniformity with MPCBalancer
        and ignored -- PD is a pure feedback law."""
        _ = target_axy
        (x, y), (vx, vy) = self.ball_state()
        tx, ty = float(target_xy[0]), float(target_xy[1])
        pitch = -self.KP * (x - tx) - self.KD * vx
        roll  =  self.KP * (y - ty) + self.KD * vy
        self._apply_tilt_and_step(roll, pitch)
        # Re-read after stepping.
        (x2, y2), (vx2, vy2) = self.ball_state()
        return float(np.hypot(x2 - tx, y2 - ty)), float(np.hypot(vx2, vy2))
