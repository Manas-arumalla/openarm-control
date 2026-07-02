"""Bimanual coordination for the OpenArm (right + left).

Holds a PickPlaceController per arm sharing one model/data.

  - ParallelSort: both arms sort their side's blocks into bins SIMULTANEOUSLY.
  - synchronized_move(): both arms move at once along given joint polylines.
  - RelayHandoff: right arm relays a block to the left arm via a shared midpoint.

The core mechanism is `_ArmRunner`, a per-arm state machine that advances one
timestep at a time through a flattened list of motion segments (with grasp-weld
attach/detach), so two arms can be stepped together in lockstep. Both arms are
gravity-compensated every step.
"""

import time

import mujoco
import numpy as np

from .config import RIGHT_ARM, LEFT_ARM, MIRROR_R2L
from .grasp import topdown_orientation
from .pick_and_place import PickPlaceController, TABLE_TOP_Z, GRASP_DEPTH
from .planners.collision import CollisionChecker
from .trajectory import quintic_polynomial


def mirror_config(q_right):
    """Left-arm joints that mirror a right-arm config across the y=0 plane."""
    return np.asarray(MIRROR_R2L, dtype=float) * np.asarray(q_right, dtype=float)


class BimanualController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.right = PickPlaceController(model, data, arm=RIGHT_ARM)
        self.left = PickPlaceController(model, data, arm=LEFT_ARM)
        # Each checker treats the OTHER arm (at its current pose) as an obstacle.
        self.right_checker = CollisionChecker(model, data, self.right.king,
                                              arm_name="right", avoid_other_arm=True)
        self.left_checker = CollisionChecker(model, data, self.left.king,
                                             arm_name="left", avoid_other_arm=True)

    def path_blocked(self, checker, knots_list, samples=40):
        """True if any config along the segments collides (incl. the other arm)."""
        configs = np.vstack([np.atleast_2d(k) for k, _, _ in knots_list])
        step = max(1, len(configs) // samples)
        return any(checker.in_collision(configs[i]) for i in range(0, len(configs), step))

    def sync_ctrl(self):
        for ppc in (self.right, self.left):
            for i, a in enumerate(ppc.arm_acts):
                self.data.ctrl[a] = self.data.qpos[ppc.king.qpos_indices[i]]

    def grav_comp(self):
        for ppc in (self.right, self.left):
            self.data.qfrc_applied[ppc.king.dof_indices] = \
                self.data.qfrc_bias[ppc.king.dof_indices]


class _ArmRunner:
    """Advances one arm through a flat list of (knots, closed, duration, block)."""

    def __init__(self, ppc, jobs):
        self.ppc = ppc
        self.data = ppc.data
        self.steps = []
        for block, segs in jobs:
            for knots, closed, dur in segs:
                self.steps.append((np.asarray(knots, float), bool(closed), float(dur), block))
        self.idx = 0
        self.t = 0.0
        self.prev = self.data.qpos[ppc.king.qpos_indices].copy()
        self.welded = False
        self.kk = None
        self._begin()

    def _begin(self):
        if self.idx >= len(self.steps):
            self.kk = None
            return
        knots, closed, _, block = self.steps[self.idx]
        if (not closed) and self.welded and block is not None:
            self.ppc.detach(block)
            self.welded = False
        self.kk = (np.vstack([self.prev, knots])
                   if not np.allclose(knots[0], self.prev) else knots)

    @property
    def done(self):
        return self.kk is None

    def step(self, dt):
        ppc = self.ppc
        if self.kk is None:                       # finished: hold pose
            for i, a in enumerate(ppc.arm_acts):
                self.data.ctrl[a] = self.data.qpos[ppc.king.qpos_indices[i]]
            return
        knots, closed, dur, block = self.steps[self.idx]
        s, _, _ = quintic_polynomial(self.t, 0.0, dur, 0.0, 1.0)
        q = ppc._sample(self.kk, s)
        for i, a in enumerate(ppc.arm_acts):
            self.data.ctrl[a] = q[i]
        if ppc.grip_act != -1:
            self.data.ctrl[ppc.grip_act] = ppc._grip_ctrl(closed)
        self.t += dt
        if self.t >= dur:
            self.prev = self.kk[-1]
            if closed and not self.welded and block is not None:
                ppc.attach(block)
                self.welded = True
            self.idx += 1
            self.t = 0.0
            self._begin()


def parallel_run(bi, right_jobs, left_jobs, viewer=None, dt_realtime=False):
    """Step both arms' job lists in lockstep until both finish."""
    rr = _ArmRunner(bi.right, right_jobs)
    lr = _ArmRunner(bi.left, left_jobs)
    dt = bi.model.opt.timestep
    while not (rr.done and lr.done):
        t0 = time.time()
        rr.step(dt)
        lr.step(dt)
        bi.grav_comp()
        mujoco.mj_step(bi.model, bi.data)
        if viewer is not None:
            if not viewer.is_running():
                return False
            viewer.sync()
            if dt_realtime:
                time.sleep(max(0, dt - (time.time() - t0)))
    return True


def synchronized_move(bi, right_path, duration, viewer=None, dt_realtime=False):
    """Move both arms simultaneously, the left arm exactly mirroring the right.

    `right_path` is a polyline of right-arm joint configs; the left arm follows
    the joint-space mirror at the same timing, so the motions are perfectly
    symmetric (no per-arm IK deviation).
    """
    model, data = bi.model, bi.data
    dt = model.opt.timestep
    right_path = np.atleast_2d(right_path)
    left_path = np.array([mirror_config(q) for q in right_path])
    rprev = data.qpos[bi.right.king.qpos_indices].copy()
    lprev = data.qpos[bi.left.king.qpos_indices].copy()
    rk = np.vstack([rprev, right_path]) if not np.allclose(right_path[0], rprev) else right_path
    lk = np.vstack([lprev, left_path]) if not np.allclose(left_path[0], lprev) else left_path
    for k in range(max(1, int(duration / dt))):
        t0 = time.time()
        s, _, _ = quintic_polynomial(k * dt, 0.0, duration, 0.0, 1.0)
        for i, a in enumerate(bi.right.arm_acts):
            data.ctrl[a] = bi.right._sample(rk, s)[i]
        for i, a in enumerate(bi.left.arm_acts):
            data.ctrl[a] = bi.left._sample(lk, s)[i]
        bi.grav_comp()
        mujoco.mj_step(model, data)
        if viewer is not None:
            if not viewer.is_running():
                return False
            viewer.sync()
            if dt_realtime:
                time.sleep(max(0, dt - (time.time() - t0)))
    return True


class ParallelSort:
    """Both arms sort their side's blocks into the matching bins, simultaneously."""

    def __init__(self, model, data):
        self.bi = BimanualController(model, data)
        # (block, pick_xy, bin_xy) per arm
        self.right_jobs = [("block_r1", (0.18, -0.16), (0.36, -0.16)),
                           ("block_r2", (0.18, -0.30), (0.36, -0.30))]
        self.left_jobs = [("block_l1", (0.18, 0.16), (0.36, 0.16)),
                          ("block_l2", (0.18, 0.30), (0.36, 0.30))]

    def run(self, viewer=None, dt_realtime=False):
        bi = self.bi
        bi.sync_ctrl()
        rjobs = [(b, bi.right.plan(pick_xy=p, place_xy=q)) for b, p, q in self.right_jobs]
        ljobs = [(b, bi.left.plan(pick_xy=p, place_xy=q)) for b, p, q in self.left_jobs]
        return parallel_run(bi, rjobs, ljobs, viewer=viewer, dt_realtime=dt_realtime)


class BimanualStack:
    """Both arms stack a cube onto a base cube on their own side, simultaneously,
    building two towers side by side. Each arm picks its top cube and places it on
    its base cube at the stacking site (place height = base-cube top + grasp depth,
    so the held cube's base lands on the base cube)."""

    BLOCK_HALF = 0.025

    def __init__(self, model, data):
        self.bi = BimanualController(model, data)
        # (top_block, pick_xy, base_xy)
        self.right_job = ("box_green", (0.18, -0.22), (0.32, -0.22))
        self.left_job = ("box_orange", (0.18, 0.22), (0.32, 0.22))
        # grasp the 50 mm cube near its top; place its base on the base cube's top.
        self.grasp_z = TABLE_TOP_Z + GRASP_DEPTH
        self.place_z = TABLE_TOP_Z + 2 * self.BLOCK_HALF + GRASP_DEPTH + 0.005

    def run(self, viewer=None, dt_realtime=False):
        bi = self.bi
        bi.sync_ctrl()
        br, pr, qr = self.right_job
        bl, _, _ = self.left_job
        # Plan the right arm's stack, then drive the left arm with the exact
        # joint-space MIRROR of it. The scene is symmetric, so the mirrored motion
        # stacks the left tower simultaneously -- and mirroring a known-good path
        # sidesteps the left arm's own IK branch jumps at the mirrored poses.
        right_segs = bi.right.plan(pick_xy=pr, place_xy=qr,
                                   grasp_z=self.grasp_z, place_z=self.place_z)
        mirror = np.asarray(MIRROR_R2L, dtype=float)
        left_segs = [(np.atleast_2d(k) * mirror, c, d) for k, c, d in right_segs]
        return parallel_run(bi, [(br, right_segs)], [(bl, left_segs)],
                            viewer=viewer, dt_realtime=dt_realtime)


class RelayHandoff:
    """Collision-aware right->left object hand-off via a shared midpoint.

    Right arm carries block_r1 to the midpoint and releases. Before the left arm
    moves in, the system CHECKS whether the right arm (where it ended up) blocks
    the left arm's planned path -- using inter-arm collision detection. If so, it
    intelligently moves the right arm to a clear pose first, then the left arm
    picks up the block and places it in the left bin.
    """

    def __init__(self, model, data, block="block_r1",
                 pick_xy=(0.18, -0.16), midpoint_xy=(0.24, -0.05), place_xy=(0.36, 0.16)):
        self.bi = BimanualController(model, data)
        self.block = block
        self.pick_xy = pick_xy
        self.midpoint_xy = midpoint_xy
        self.place_xy = place_xy
        # Right arm's "clear" pose = its ready keyframe configuration.
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        self.right_clear = model.key_qpos[kid][self.bi.right.king.qpos_indices].copy()

    def run(self, viewer=None, dt_realtime=False):
        bi = self.bi
        bi.sync_ctrl()

        # Phase 1: right carries the block to the midpoint and releases.
        seg_r = bi.right.plan(pick_xy=self.pick_xy, place_xy=self.midpoint_xy)
        if not parallel_run(bi, [(self.block, seg_r)], [], viewer=viewer, dt_realtime=dt_realtime):
            return False

        # Phase 2: plan the left arm's pick-up; check if the right arm blocks it.
        seg_l = bi.left.plan(pick_xy=self.midpoint_xy, place_xy=self.place_xy)
        if bi.path_blocked(bi.left_checker, seg_l):
            print("  [coordination] right arm blocks the left arm's path -> clearing it")
            clear_seg = [(np.array([self.right_clear]), False, 2.0)]
            if not parallel_run(bi, [(None, clear_seg)], [], viewer=viewer, dt_realtime=dt_realtime):
                return False
            if bi.path_blocked(bi.left_checker, seg_l):
                print("  [coordination] warning: left path still blocked after clearing")
        else:
            print("  [coordination] left arm's path is clear")

        # Phase 3: left picks up the block and places it in the left bin.
        return parallel_run(bi, [], [(self.block, seg_l)], viewer=viewer, dt_realtime=dt_realtime)


class BimanualCoordinator:
    """Intelligent dual-arm manipulation for ordinary pick-and-place tasks.

    Given an object and a destination, it figures out *which arm should do it*:
    the arm on the object's side (and feasible) picks it; if that arm can also reach
    the destination it finishes the task alone; if it **can't reach the destination
    but the other arm can**, it hands the object over at a centre midpoint and the
    other arm completes the place. Built on planned, collision-checked motions
    (`PickPlaceController` + `parallel_run`), so it's reliable — not a servo.

    Objects handled this way need *both* arms' grasp welds (either may carry them).
    """

    MIDPOINT = (0.27, 0.0)             # handover point on the centreline (both arms reach)

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.bi = BimanualController(model, data)
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
        self.home = {"right": model.key_qpos[kid][self.bi.right.king.qpos_indices].copy(),
                     "left": model.key_qpos[kid][self.bi.left.king.qpos_indices].copy()}
        self.held = None             # {'ppc','name','block','origin'} when an arm holds an object

    # ----------------------------------------------------------- reachability
    def _feasible(self, ppc, xy, z):
        _, info = ppc.gs.solve(np.array([xy[0], xy[1], z]), return_info=True)
        return bool(info["success"])

    def _pick_arm(self, obj_xy, z):
        """The arm on the object's side if feasible, else the other if feasible."""
        order = ((self.bi.right, "right"), (self.bi.left, "left"))
        if obj_xy[1] > 0:                                  # +y is the left arm's side
            order = order[::-1]
        for ppc, name in order:
            if self._feasible(ppc, obj_xy, z):
                return ppc, name
        return None, None

    def _plan(self, ppc, pick_xy, place_xy, grasp_z, place_z):
        """Pick->place segments for one arm, or ``None`` if the carry path is
        infeasible (e.g. an IK branch jump) -- callers degrade gracefully instead of
        crashing. The left arm is planned by **mirroring** a right-arm plan across
        y=0 (the left arm's own IK branch-jumps on far +y poses; the right arm plans
        cleanly on its side, and the mirror is exact)."""
        try:
            if ppc is self.bi.left:
                rp = (pick_xy[0], -pick_xy[1])
                rq = (place_xy[0], -place_xy[1])
                segs = self.bi.right.plan(pick_xy=rp, place_xy=rq, grasp_z=grasp_z, place_z=place_z)
                mirror = np.asarray(MIRROR_R2L, dtype=float)
                return [(np.atleast_2d(k) * mirror, c, d) for k, c, d in segs]
            return ppc.plan(pick_xy=pick_xy, place_xy=place_xy, grasp_z=grasp_z, place_z=place_z)
        except ValueError:
            return None

    def _run_arm(self, ppc, jobs, viewer=None, dt_realtime=False):
        """Run one arm's job list while the other arm holds its pose."""
        if ppc is self.bi.right:
            return parallel_run(self.bi, jobs, [], viewer=viewer, dt_realtime=dt_realtime)
        return parallel_run(self.bi, [], jobs, viewer=viewer, dt_realtime=dt_realtime)

    def _name(self, ppc):
        return "right" if ppc is self.bi.right else "left"

    # ------------------------------------------------------------- the task
    def pick_place(self, obj_xy, place_xy, block, grasp_z=None, place_z=None,
                   viewer=None, dt_realtime=False, verbose=True):
        """Pick ``block`` (at ``obj_xy``) and place it at ``place_xy``, choosing the
        arm(s) intelligently. Returns (ok, message)."""
        bi = self.bi
        bi.sync_ctrl()
        grasp_z = (TABLE_TOP_Z + GRASP_DEPTH) if grasp_z is None else grasp_z
        place_z = (TABLE_TOP_Z + GRASP_DEPTH + 0.04) if place_z is None else place_z

        pick, pname = self._pick_arm(obj_xy, grasp_z)
        if pick is None:
            return False, "neither arm can reach the object"
        other = bi.left if pick is bi.right else bi.right

        if self._feasible(pick, place_xy, place_z):
            # One arm does the whole task; the other holds.
            segs = self._plan(pick, obj_xy, place_xy, grasp_z, place_z)
            if segs is not None:
                if verbose:
                    print(f"  [coordination] {pname} arm can reach both -> it does the task")
                ok = self._run_arm(pick, [(block, segs)], viewer=viewer, dt_realtime=dt_realtime)
                return ok, f"{pname} arm picked and placed the {block}"
            # plan infeasible for the single arm -> fall through to a hand-over

        if not self._feasible(other, place_xy, place_z):
            return False, f"can't plan a path to the destination for the {block}"

        # Hand-over: pick arm -> midpoint -> other arm -> destination.
        if verbose:
            print(f"  [coordination] {pname} can't reach/plan the destination -> "
                  f"hand off to {self._name(other)} arm at the midpoint")
        mid = np.array(self.MIDPOINT)
        seg_a = self._plan(pick, obj_xy, mid, grasp_z, TABLE_TOP_Z + GRASP_DEPTH + 0.02)
        seg_b = self._plan(other, mid, place_xy, grasp_z, place_z)
        if seg_a is None or seg_b is None:
            return False, f"couldn't plan a hand-over for the {block}"
        if not self._run_arm(pick, [(block, seg_a)], viewer=viewer, dt_realtime=dt_realtime):
            return False, "hand-off (carry to midpoint) failed"
        # Clear the handing arm out of the way before the other moves in.
        clear = [(np.array([self.home[pname]]), False, 1.5)]
        self._run_arm(pick, [(None, clear)], viewer=viewer, dt_realtime=dt_realtime)
        ok = self._run_arm(other, [(block, seg_b)], viewer=viewer, dt_realtime=dt_realtime)
        return ok, (f"{pname} arm handed the {block} to the {self._name(other)} arm, "
                    f"which placed it")

    # ------------------------------------------ stateful grab / hold / place
    def _plan_pick(self, ppc, pick_xy, grasp_z):
        """Pick+lift segments for one arm (mirror a right-arm plan for the left), or
        ``None`` if unreachable / discontinuous."""
        try:
            if ppc is self.bi.left:
                segs = self.bi.right.plan_pick((pick_xy[0], -pick_xy[1]), grasp_z=grasp_z)
                mirror = np.asarray(MIRROR_R2L, dtype=float)
                return [(np.atleast_2d(k) * mirror, c, d) for k, c, d in segs]
            return ppc.plan_pick(pick_xy, grasp_z=grasp_z)
        except ValueError:
            return None

    def pick(self, obj_xy, block, grasp_z=None, viewer=None, dt_realtime=False, verbose=True):
        """Grab ``block`` (at ``obj_xy``) with the better-placed arm and HOLD it (no
        place). The choice is the arm on the object's side if it can reach, else the
        other. A later ``place_held`` puts it down. Returns (ok, message)."""
        if self.held is not None:
            return False, f"already holding the {self.held['block']} -- place it first"
        self.bi.sync_ctrl()
        grasp_z = (TABLE_TOP_Z + GRASP_DEPTH) if grasp_z is None else grasp_z
        ppc, name = self._pick_arm(obj_xy, grasp_z)
        if ppc is None:
            return False, "neither arm can reach the object"
        segs = self._plan_pick(ppc, obj_xy, grasp_z)
        if segs is None:
            return False, f"couldn't plan a grasp for the {block}"
        if verbose:
            print(f"  [coordination] {name} arm grabs the {block}")
        ok = self._run_arm(ppc, [(block, segs)], viewer=viewer, dt_realtime=dt_realtime)
        if ok:
            self.held = {"ppc": ppc, "name": name, "block": block, "origin": tuple(obj_xy)}
        return ok, (f"{name} arm grabbed the {block}" if ok else f"failed to grab the {block}")

    def place_held(self, place_xy, place_z=None, viewer=None, dt_realtime=False, verbose=True):
        """Put down the currently-held object at ``place_xy`` with the holding arm.
        If that arm can't reach the destination, it reports so (use the one-shot
        ``pick_place`` form for an automatic hand-over). Returns (ok, message)."""
        if self.held is None:
            return False, "not holding anything"
        ppc, name, block = self.held["ppc"], self.held["name"], self.held["block"]
        place_z = (TABLE_TOP_Z + GRASP_DEPTH + 0.04) if place_z is None else place_z
        if not self._feasible(ppc, place_xy, place_z):
            return False, (f"the {name} arm can't reach there -- say "
                           f"'move the {block} to ...' so I can hand it over")
        try:
            segs = ppc.plan_place_held(place_xy, place_z)
        except ValueError:
            return False, f"couldn't plan a path to put the {block} down there"
        if verbose:
            print(f"  [coordination] {name} arm places the {block}")
        ok = self._run_arm(ppc, [(block, segs)], viewer=viewer, dt_realtime=dt_realtime)
        if ok:
            self.held = None
        return ok, (f"{name} arm placed the {block}" if ok else f"failed to place the {block}")


class UnscrewTask:
    """Single-arm bottle opening: the RIGHT arm unscrews a threaded cap on a clamped
    bottle over several re-gripping bursts and lifts it clear.

    The bottle is held fixed in a stand. Two 7-DOF arms working at one small bottle
    unavoidably collide (the working arm's redundant elbow swings into the holding
    arm), so the holding is done by the clamp -- this also gives the working arm full
    room and a collision-free, proper multi-turn unscrew. The cap is jointed (turn
    hinge + lift slide on the bottle axis), so it cannot be knocked off and only
    screws on its axis. The right wrist turns it via a grasp weld, re-gripping between
    bursts like a human; a lift actuator provides the coupled rise (lift = pitch *
    turn). The left arm is parked, out of the way.
    """

    def __init__(self, model, data, cap="cap", cap_xy=(0.30, -0.08), cap_z=0.522,
                 n_bursts=10, turn_per_burst=0.8, pitch=0.0008, lift_clear=0.06):
        self.bi = BimanualController(model, data)
        self.model, self.data = model, data
        self.cap = cap
        self.cap_xy, self.cap_z = cap_xy, cap_z
        self.n_bursts, self.turn_per_burst = n_bursts, turn_per_burst
        self.pitch, self.lift_clear = pitch, lift_clear
        self.lift_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cap_lift_drive")
        self.turn_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cap_turn")]

    def _weld_cap(self):
        """Weld the cap to the right gripper at the actual current relative pose."""
        ppc = self.bi.right
        eid = ppc._weld_id(self.cap)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cap)
        if eid < 0 or bid < 0:
            return
        p1, q1 = self.data.xpos[ppc.ee_body].copy(), self.data.xquat[ppc.ee_body].copy()
        p2, q2 = self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
        nq1 = np.zeros(4); mujoco.mju_negQuat(nq1, q1)
        relpos = np.zeros(3); mujoco.mju_rotVecQuat(relpos, p2 - p1, nq1)
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, nq1, q2)
        self.model.eq_data[eid, 0:3] = 0.0
        self.model.eq_data[eid, 3:6] = relpos
        self.model.eq_data[eid, 6:10] = relquat
        self.model.eq_data[eid, 10] = 1.0
        self.data.eq_active[eid] = 1

    def _move(self, qr, dur, grip=None, cap_lift=None, viewer=None, dt_realtime=False):
        """Quintic-drive the RIGHT arm to ``qr`` while the left arm holds its parked
        pose; command the cap lift actuator if given. Both arms gravity-compensated."""
        bi, model, data = self.bi, self.model, self.data
        dt = model.opt.timestep
        if cap_lift is not None and self.lift_act >= 0:
            data.ctrl[self.lift_act] = cap_lift
        rprev = data.qpos[bi.right.king.qpos_indices].copy()
        lhold = data.qpos[bi.left.king.qpos_indices].copy()
        qr = np.asarray(qr, float)
        for k in range(max(1, int(dur / dt))):
            t0 = time.time()
            s, _, _ = quintic_polynomial(k * dt, 0.0, dur, 0.0, 1.0)
            qrc = rprev + (qr - rprev) * s
            for i, a in enumerate(bi.right.arm_acts):
                data.ctrl[a] = qrc[i]
            for i, a in enumerate(bi.left.arm_acts):       # hold the left arm parked
                data.ctrl[a] = lhold[i]
            if grip is not None and bi.right.grip_act != -1:
                data.ctrl[bi.right.grip_act] = grip
            if cap_lift is not None and self.lift_act >= 0:
                data.ctrl[self.lift_act] = cap_lift
            bi.grav_comp()
            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    return False
                viewer.sync()
                if dt_realtime:
                    time.sleep(max(0, dt - (time.time() - t0)))
        return True

    def run(self, viewer=None, dt_realtime=False):
        bi, Rt = self.bi, self.bi.right
        bi.sync_ctrl()
        kx, ky = self.cap_xy

        def knob(lift):
            return np.array([kx, ky, self.cap_z + lift])

        qg, iR = Rt.gs.solve(knob(0.0), return_info=True)
        if qg is None:
            return False
        yawR = iR["yaw"]
        openg, closeg = Rt.arm.gripper_open, Rt.arm.gripper_closed

        def ik_cap(lift, dyaw, seed):
            return Rt.king.inverse_kinematics(knob(lift), target_mat=topdown_orientation(yawR + dyaw),
                                              q_init=seed, restarts=0, rest_weight=0.0)

        # 1) hover, descend onto the cap, grasp (weld)
        qh = ik_cap(0.10, 0.0, qg)
        self._move(qh, 2.0, grip=openg, cap_lift=0.0, viewer=viewer, dt_realtime=dt_realtime)
        self._move(qg, 1.5, grip=openg, cap_lift=0.0, viewer=viewer, dt_realtime=dt_realtime)
        self._weld_cap()

        # 2) unscrew over several re-gripping bursts: the wrist turns the cap +0.8 rad,
        #    then the gripper releases, rotates back, and re-grasps -- the cap rises via
        #    the lift actuator. (The left arm stays parked; only the right arm works.)
        q, cum = qg, 0.0
        for i in range(self.n_bursts):
            cum += self.turn_per_burst
            lift = min(self.pitch * cum, 0.10)
            q_turn = ik_cap(lift, self.turn_per_burst, q)
            if q_turn is None:
                break
            self._move(q_turn, 0.5, grip=closeg, cap_lift=lift, viewer=viewer, dt_realtime=dt_realtime)
            q = q_turn
            if i < self.n_bursts - 1:
                Rt.detach(self.cap)
                q_up = ik_cap(lift + 0.03, 0.0, q)
                if q_up is not None:
                    self._move(q_up, 0.3, grip=openg, cap_lift=lift, viewer=viewer, dt_realtime=dt_realtime)
                    q = q_up
                q_dn = ik_cap(lift, 0.0, q)
                if q_dn is not None:
                    self._move(q_dn, 0.3, grip=closeg, cap_lift=lift, viewer=viewer, dt_realtime=dt_realtime)
                    q = q_dn
                self._weld_cap()

        # 3) lift the now-unscrewed cap clear of the bottle (still gripped)
        q_off = ik_cap(self.lift_clear, 0.0, q)
        if q_off is not None:
            self._move(q_off, 1.5, grip=closeg, cap_lift=self.lift_clear, viewer=viewer, dt_realtime=dt_realtime)
        return True
