"""Task executor: run an Intent on the robot, with obstacle avoidance.

Pipeline per command:  perceive -> ground the target -> plan a **collision-free**
motion (RRT-Connect around the other objects / bin) -> grasp (top-down weld) ->
transport (collision-free) -> place / release.

Free-space moves go through the RRT planner so the arm avoids clutter; the
grasp/lift/descend mechanics reuse the proven ``PickPlaceController`` (continuous
on-branch Cartesian IK + weld-assisted grasp). The target/carried object is
excluded from collision so reaching and carrying it isn't treated as a crash.
"""
from __future__ import annotations

import numpy as np
import mujoco

from ..config import RIGHT_ARM
from ..pick_and_place import PickPlaceController, TABLE_TOP_Z, GRASP_DEPTH
from ..planners import CollisionChecker, RRTPlanner

OPEN, CLOSE = False, True


class TaskExecutor:
    def __init__(self, model, data, arm=RIGHT_ARM, perception=None,
                 graspables=None, bin_body="bin", hover=0.12, rrt_seed=0):
        self.model, self.data, self.arm = model, data, arm
        self.perception = perception
        self.ppc = PickPlaceController(model, data, arm, hover=hover)
        self.king = self.ppc.king
        self.checker = CollisionChecker(model, data, self.king, arm_name=arm.name)
        self.rrt = RRTPlanner(model, data, self.king, seed=rrt_seed, checker=self.checker)
        self.hover = hover
        self.graspables = list(graspables) if graspables else []
        self.bin_body = bin_body
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bin_body)
        self.bin_xy = data.xpos[bid][:2].copy() if bid >= 0 else np.array([0.1, -0.46])
        self.last_status = ""
        # ready/perch arm config (clear of the table camera) -- captured at build
        # time from the scene's keyframe, so the arm can step out of view to
        # re-perceive between commands.
        self.q_home = self._q_now()
        # held-object state (for stateful multi-turn commands)
        self.held_body = None
        self.held_label = None
        self.R_grasp = None
        self._place_lift = None
        self.pickup_xy = None              # where the last grasped object came from (for undo)

    # ------------------------------------------------------------- helpers
    def _q_now(self):
        return self.data.qpos[self.king.qpos_indices].copy()

    def _target_body(self, position):
        """Nearest graspable body to a perceived 3D position (for the weld)."""
        best, best_d = None, 0.12
        for b in self.graspables:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if bid < 0:
                continue
            d = np.linalg.norm(self.data.xpos[bid][:2] - position[:2])
            if d < best_d:
                best, best_d = b, d
        return best

    def _destination_xy(self, dest, pick_xy):
        if dest == "bin":
            return self.bin_xy.copy()
        if dest == "left":                       # robot left = +y
            return np.array([pick_xy[0], pick_xy[1] + 0.16])
        if dest == "right":                      # robot right = -y
            return np.array([pick_xy[0], pick_xy[1] - 0.16])
        if dest == "out":                        # off to the near side of the table
            return np.array([0.12, -0.05])
        return pick_xy.copy()                    # 'table' / default: same spot

    def _plan_path(self, q_start, q_goal, ignore):
        """Collision-free joint path q_start->q_goal: a direct edge if clear, else
        RRT-Connect, else None. (CollisionChecker/RRT set qpos internally for
        checks; the caller restores the real state before executing.)"""
        if self.checker.edge_clear(q_start, q_goal, ignore_bodies=ignore):
            return np.array([q_start, q_goal])
        path = self.rrt.plan(q_start, q_goal, ignore_bodies=ignore)
        return np.array(path) if path else None

    def _grasp_geom(self, obj, body):
        """Top-down grasp config gq + orientation R + the lift polyline (gq->hover)
        for a perceived object. Grasp height = just below the object's perceived
        top (so a tall object is gripped near its top, not pressed)."""
        from ..grasp import topdown_orientation
        pick_xy = obj.position[:2]
        grasp_z = float(np.clip(obj.position[2] - 0.025,
                                TABLE_TOP_Z + 0.035, TABLE_TOP_Z + 0.16))
        pick = np.array([pick_xy[0], pick_xy[1], grasp_z])
        gq, info = self.ppc.gs.solve(pick, return_info=True)
        if not info["success"]:
            return None
        R = topdown_orientation(info["yaw"])
        lift = self.ppc._ik_line(gq, pick, pick + np.array([0, 0, self.hover]), R)
        # grasp offset = how far the grasp point sits above the table (the held
        # object's bottom is ~TABLE_TOP_Z, so it hangs this far below the gripper).
        return gq, R, lift, grasp_z - TABLE_TOP_Z

    # ----------------------------------------------------- primitive actions
    def grasp(self, target, viewer=None):
        """Find, approach (collision-free), and pick up ``target``; lift clear and
        hold it. Sets ``held_body``/``held_label``. Returns (ok, msg)."""
        if self.perception is None:
            return self._fail("no perception")
        obj = self.perception.ground(target)
        if obj is None:
            return self._fail(f"could not find '{target}'")
        self.pickup_xy = obj.position[:2].copy()           # remember its origin (for undo)
        body = self._target_body(obj.position)
        if body is None:
            return self._fail(f"'{target}' not graspable")
        g = self._grasp_geom(obj, body)
        if g is None:
            return self._fail(f"'{target}' not reachable")
        gq, R, lift, self._grasp_offset = g
        qpos_save = self.data.qpos.copy()
        approach = self._plan_path(self._q_now(), lift[-1], ignore=(body,))
        if approach is None:
            return self._fail("no collision-free path to the object", qpos_save)
        self.data.qpos[:] = qpos_save
        mujoco.mj_forward(self.model, self.data)
        self.ppc.execute([(approach, OPEN, 2.5), (lift[::-1], OPEN, 1.5),
                          (np.array([gq, gq]), CLOSE, 1.5), (lift, CLOSE, 2.0)],
                         block=body, viewer=viewer)
        self.held_body, self.held_label, self.R_grasp = body, obj.label, R
        return True, f"picked '{target}'"

    def _carry_to(self, xy, place_z, ignore=(), viewer=None):
        """Carry the held object to *above* ``(xy, place_z)`` collision-free, and
        remember the descent so ``release()`` can lower + drop. ``ignore`` excludes
        bodies from collision at the destination (e.g. the support we stack onto, or
        a target we descend onto). Returns (ok, msg)."""
        place = np.array([xy[0], xy[1], place_z])
        pq, info = self.ppc.gs.solve(place, return_info=True)
        if not info["success"]:
            return self._fail("can't reach the place pose")
        # A point-down, free-yaw vertical descent reference (place -> hover): the
        # gripper stays vertical so the held object hangs straight below and lands
        # on target, with no xy drift (a fixed-orientation line twists at the end
        # and wanders ~3 cm -- tolerable for a wide bin, fatal for stacking).
        place_lift = self.ppc._ik_line_down(pq, place, place + np.array([0, 0, self.hover]))
        pre_place = place_lift[-1]
        qpos_save = self.data.qpos.copy()
        if self.held_body:                                 # carried object must avoid
            others = [b for b in self.graspables
                      if b != self.held_body and b not in ignore]   # other objects
            self.checker.set_carried(self.held_body, also_avoid=others)
            transport = self._plan_path(self._q_now(), pre_place, ignore=tuple(ignore))
            self.checker.set_carried(None)
        else:
            transport = self._plan_path(self._q_now(), pre_place, ignore=tuple(ignore))
        if transport is None:
            return self._fail("no collision-free path there", qpos_save)
        self.data.qpos[:] = qpos_save
        mujoco.mj_forward(self.model, self.data)
        self._place_lift = place_lift                      # remembered for release()
        self.ppc.execute([(transport, self.held_body is not None, 3.0)],
                         block=None, viewer=viewer)
        return True, "carried"

    def go_to(self, dest=None, target=None, viewer=None):
        """Move to *above* a destination (bin/left/right/table) or a target object,
        carrying the held object collision-free, and stop there. Returns (ok, msg)."""
        if target is not None and self.perception is not None:
            o = self.perception.ground(target)
            if o is None:
                return self._fail(f"could not find '{target}'")
            xy = o.position[:2]
        else:
            xy = self._destination_xy(dest, self._ee_xy())
        off = getattr(self, "_grasp_offset", 0.04)        # held object hangs this far below the grip
        if dest == "bin":
            place_z = 0.52 + off + 0.05                   # clear the ~0.52 bin walls, then drop in
        else:
            place_z = TABLE_TOP_Z + off + 0.02            # just above the surface for a gentle place
        ok, msg = self._carry_to(xy, place_z, ignore=(), viewer=viewer)
        return (True, f"at {dest or target}") if ok else (ok, msg)

    def stack(self, target, support, viewer=None):
        """Pick up ``target`` and place it **on top of** ``support`` so it rests
        stably. Locates the support first, grasps the target, then carries it above
        the support and lowers so the held object's base lands on the support's top
        (place height = support top + grasp offset). Returns (ok, msg)."""
        if self.perception is None:
            return self._fail("no perception")
        sup = self.perception.ground(support)              # locate support before grasping
        if sup is None:
            return self._fail(f"could not find '{support}'")
        sup_body = self._target_body(sup.position)
        sup_top_z = float(sup.position[2])
        sup_xy = sup.position[:2].copy()
        tobj = self.perception.ground(target)              # refuse a self-stack before grasping
        if tobj is not None and sup_body is not None and self._target_body(tobj.position) == sup_body:
            return self._fail("can't stack an object on itself")
        ok, msg = self.grasp(target, viewer=viewer)
        if not ok:
            return ok, msg
        off = getattr(self, "_grasp_offset", 0.04)
        place_z = sup_top_z + off + 0.012                  # held base ~just above the support top
        ignore = (sup_body,) if sup_body else ()
        ok, msg = self._carry_to(sup_xy, place_z, ignore=ignore, viewer=viewer)
        if not ok:
            return ok, msg
        ok, msg = self.release(viewer=viewer)
        if not ok:
            return ok, msg
        self.home(viewer=viewer)                           # step clear so the camera sees again
        return True, f"stacked '{target}' on '{support}'"

    def insert(self, peg, socket_body="socket", viewer=None):
        """Grasp ``peg`` and insert it into the socket: carry it above the socket,
        then a precise **vertical descent** threads it through the opening down to
        the table (the point-down descent has no xy drift, so a few mm of clearance
        is enough). The socket location is the named body's position. (ok, msg)."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, socket_body)
        if sid < 0:
            return self._fail(f"no socket '{socket_body}'")
        socket_xy = self.data.xpos[sid][:2].copy()
        ok, msg = self.grasp(peg, viewer=viewer)
        if not ok:
            return ok, msg
        off = getattr(self, "_grasp_offset", 0.04)
        place_z = TABLE_TOP_Z + off + 0.006                # peg base just above the table, in the socket
        ok, msg = self._carry_to(socket_xy, place_z, ignore=(socket_body,), viewer=viewer)
        if not ok:
            return ok, msg
        ok, msg = self.release(viewer=viewer)
        if not ok:
            return ok, msg
        self.home(viewer=viewer)
        return True, f"inserted '{peg}' into the socket"

    def home(self, viewer=None):
        """Return the arm to its ready/perch config so it's clear of the table
        camera (lets the next command perceive an unobstructed scene). Carries the
        held object if holding one. Returns (ok, msg)."""
        qpos_save = self.data.qpos.copy()
        if self.held_body:
            others = [b for b in self.graspables if b != self.held_body]
            self.checker.set_carried(self.held_body, also_avoid=others)
            path = self._plan_path(self._q_now(), self.q_home, ignore=())
            self.checker.set_carried(None)
        else:
            path = self._plan_path(self._q_now(), self.q_home, ignore=())
        if path is None:
            return self._fail("no clear path home", qpos_save)
        self.data.qpos[:] = qpos_save
        mujoco.mj_forward(self.model, self.data)
        self.ppc.execute([(path, self.held_body is not None, 2.0)], block=None, viewer=viewer)
        return True, "home"

    def release(self, viewer=None):
        """Lower (if a place target was set), open the gripper, drop the held
        object, and retreat. Returns (ok, msg)."""
        held = self.held_body
        lift = getattr(self, "_place_lift", None)
        if lift is not None:
            self.ppc.execute([(lift[::-1], CLOSE, 1.5)], block=None, viewer=viewer)
        if held is not None:
            self.ppc.detach(held)
        # open the gripper in place, then retreat up
        q = self._q_now()
        self.ppc.execute([(np.array([q, q]), OPEN, 0.8)], block=None, viewer=viewer)
        if lift is not None:
            self.ppc.execute([(lift, OPEN, 1.5)], block=None, viewer=viewer)
        self.held_body, self.held_label, self._place_lift = None, None, None
        return True, "released"

    def put_at(self, target, xy, viewer=None):
        """Grasp ``target`` and set it down on the table at ``xy`` (used by 'undo' /
        'put it back' to return an object to where it came from). Returns (ok, msg)."""
        ok, msg = self.grasp(target, viewer=viewer)
        if not ok:
            return ok, msg
        off = getattr(self, "_grasp_offset", 0.04)
        # ignore the bin while extracting: the object is coming *out* of it, so
        # grazing its walls on the way up isn't a crash to avoid.
        ignore = (self.bin_body,) if self.bin_body else ()
        ok, msg = self._carry_to(np.asarray(xy, float), TABLE_TOP_Z + off + 0.02,
                                 ignore=ignore, viewer=viewer)
        if not ok:
            return ok, msg
        ok, msg = self.release(viewer=viewer)
        if ok:
            self.home(viewer=viewer)
        return ok, msg

    def insert_cuboid(self, peg="peg", socket_body="socket", viewer=None):
        """Insert a **rectangular** peg into a rotated **rectangular** slot. The
        cross-section is 180-deg symmetric, so two equivalent alignments are tried."""
        return self._insert_aligned(peg, socket_body, n_fold=2, viewer=viewer)

    def insert_square(self, peg="peg", socket_body="socket", viewer=None):
        """Insert a **square** peg into a rotated **square** hole. The cross-section
        is 4-fold symmetric, so four equivalent alignments are tried (snap to the
        nearest reachable 90-deg multiple)."""
        return self._insert_aligned(peg, socket_body, n_fold=4, viewer=viewer)

    def _insert_aligned(self, peg, socket_body, n_fold, viewer=None):
        """Insert a prismatic peg whose cross-section has ``n_fold`` rotational
        symmetry (2 = rectangle, 4 = square) into a matching rotated hole.

        Beyond the round peg, the block must be **yaw-aligned** to the hole. The peg
        is grasped top-down at whatever yaw is reachable; the gripper is then rotated
        to whichever of the ``n_fold`` symmetry-equivalent socket orientations is
        reachable, and a **fixed-yaw, point-down** descent (consistent orientation
        throughout, so no wrist twist / xy drift) threads it in. Returns (ok, msg)."""
        from ..grasp import topdown_orientation
        OPEN, CLOSE = False, True
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, socket_body)
        pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, peg)
        if sid < 0 or pid < 0:
            return self._fail("peg or socket not found")
        Rs = self.data.xmat[sid].reshape(3, 3)
        theta = float(np.arctan2(Rs[1, 0], Rs[0, 0]))      # socket yaw
        socket_xy = self.data.xpos[sid][:2].copy()
        peg_pos = self.data.xpos[pid].copy()
        Rp = self.data.xmat[pid].reshape(3, 3)
        peg_yaw = float(np.arctan2(Rp[1, 0], Rp[0, 0]))     # peg yaw

        # Grasp near the top with whatever point-down yaw is reachable here.
        grasp_z = float(np.clip(peg_pos[2] + 0.015, TABLE_TOP_Z + 0.04, TABLE_TOP_Z + 0.12))
        off = grasp_z - TABLE_TOP_Z
        pick = np.array([peg_pos[0], peg_pos[1], grasp_z])
        gq, info = self.ppc.gs.solve(pick, return_info=True)
        if not info["success"]:
            return self._fail("peg grasp pose unreachable")
        phi = info["yaw"]                                  # gripper yaw at the grasp
        R_grasp = topdown_orientation(phi)
        lift = self.ppc._ik_line(gq, pick, pick + [0, 0, self.hover], R_grasp)

        # The peg's axis sits at (peg_yaw - phi) relative to the gripper. To put it at
        # the socket yaw the gripper turns to psi = theta - peg_yaw + phi; the peg's
        # cross-section is n_fold-symmetric (and so is the parallel gripper), so any of
        # psi + k*(360/n_fold) deg fits -- try each and use the first reachable one.
        ins = np.array([socket_xy[0], socket_xy[1], TABLE_TOP_Z + off + 0.006])
        hov = ins + [0, 0, self.hover]
        step = 2.0 * np.pi / n_fold
        pre = descend = None
        for k in range(n_fold):
            psi = theta - peg_yaw + phi + k * step
            R_ins = topdown_orientation(psi)
            p = self.king.inverse_kinematics(hov, R_ins, restarts=12, return_info=True)
            if not p[1]["success"]:
                continue
            dd = self.ppc._ik_line_oriented(p[0], hov, ins, R_ins)
            if self.ppc._max_jump(dd) <= np.deg2rad(18.0):
                pre, descend = p[0], dd
                break
        if descend is None:
            return self._fail("no reachable aligned insertion pose")

        self.ppc.execute([(np.array([lift[-1]]), OPEN, 2.5),     # over the peg
                          (lift[::-1], OPEN, 1.5),               # descend onto it
                          (np.array([gq, gq]), CLOSE, 1.5),      # close + grip
                          (lift, CLOSE, 1.5),                    # lift
                          (np.array([pre]), CLOSE, 2.5),         # carry + rotate over the socket
                          (descend, CLOSE, 2.0),                 # thread in (fixed yaw)
                          (np.array([descend[-1], descend[-1]]), OPEN, 1.0),   # release
                          (descend[::-1], OPEN, 1.5)],           # retreat
                         block=peg, viewer=viewer)
        self.held_body, self.held_label = None, None
        return True, f"inserted the {peg} into the rotated socket"

    def visible(self):
        """Labels perception currently sees (for queries and clarification)."""
        if self.perception is None:
            return []
        return [o.label for o in self.perception.perceive()]

    # ------------------------------------------------------------- execute
    def execute(self, intent, viewer=None):
        """Atomic command: ground the target and carry out the action end-to-end
        (used by the one-shot demo/tests). Returns (ok, msg)."""
        ok, msg = self.grasp(intent.target, viewer=viewer)
        if not ok or intent.action == "pick":
            return ok, msg
        ok, msg = self.go_to(dest=intent.destination, viewer=viewer)
        if not ok:
            return ok, msg
        return self.release(viewer=viewer)

    def _ee_xy(self):
        return self.king.forward_kinematics()[0][:2]

    def _carry_ignore(self):
        return (self.held_body,) if self.held_body else ()

    def _fail(self, msg, restore=None):
        if restore is not None:
            self.data.qpos[:] = restore
            mujoco.mj_forward(self.model, self.data)
        self.last_status = msg
        return False, msg
