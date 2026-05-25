"""Bidirectional RRT-Connect joint-space planner with path shortcutting.

Grows two trees (from start and goal) that reach toward each other, which
threads narrow passages far more efficiently than a single goal-biased tree.
The resulting path is then shortcut (connect non-adjacent waypoints with
collision-free straight lines) for a short, smooth executed motion.
"""

import numpy as np

from .collision import CollisionChecker


class RRTPlanner:
    def __init__(self, model, data, kinematics, step_size=0.2, max_iters=6000, seed=0,
                 checker=None):
        self.model = model
        self.data = data
        self.kin = kinematics
        self.step_size = step_size
        self.max_iters = max_iters
        self.checker = checker if checker is not None else CollisionChecker(model, data, kinematics)
        self.lo = kinematics.jnt_low
        self.hi = kinematics.jnt_high
        self.rng = np.random.default_rng(seed)

    def _steer(self, q_from, q_to):
        d = np.linalg.norm(q_to - q_from)
        if d <= self.step_size:
            return q_to.copy()
        return q_from + (q_to - q_from) * (self.step_size / d)

    def _extend(self, tree, parents, q_target, ignore):
        """Add one step from the nearest node toward q_target. Returns new node idx or None."""
        d = np.linalg.norm(np.array(tree) - q_target, axis=1)
        i_near = int(np.argmin(d))
        q_new = self._steer(tree[i_near], q_target)
        if self.checker.in_collision(q_new, ignore):
            return None
        if not self.checker.edge_clear(tree[i_near], q_new, ignore_bodies=ignore):
            return None
        tree.append(q_new); parents.append(i_near)
        return len(tree) - 1

    def _connect(self, tree, parents, q_target, ignore):
        """Repeatedly extend toward q_target until reached or blocked."""
        idx = None
        while True:
            new = self._extend(tree, parents, q_target, ignore)
            if new is None:
                return idx, False
            idx = new
            if np.allclose(tree[idx], q_target, atol=1e-9):
                return idx, True

    def plan(self, q_start, q_goal, ignore_bodies=()):
        q_start = np.asarray(q_start, float)
        q_goal = np.asarray(q_goal, float)
        if self.checker.in_collision(q_start, ignore_bodies):
            print("RRT: start in collision."); return None
        if self.checker.in_collision(q_goal, ignore_bodies):
            print("RRT: goal in collision."); return None

        ta, pa = [q_start], [-1]    # tree from start
        tb, pb = [q_goal], [-1]     # tree from goal
        for _ in range(self.max_iters):
            q_rand = self.rng.uniform(self.lo, self.hi)
            a_new = self._extend(ta, pa, q_rand, ignore_bodies)
            if a_new is not None:
                b_idx, reached = self._connect(tb, pb, ta[a_new], ignore_bodies)
                if reached:
                    path_a = self._branch(ta, pa, a_new)            # start..meet
                    path_b = self._branch(tb, pb, b_idx)[::-1]      # meet..goal
                    full = path_a + path_b[1:]
                    return self._shortcut(full, ignore_bodies)
            ta, pa, tb, pb = tb, pb, ta, pa     # swap trees
        print("RRT: no path found."); return None

    @staticmethod
    def _branch(tree, parents, idx):
        path, i = [], idx
        while i != -1:
            path.append(tree[i]); i = parents[i]
        return path[::-1]

    def _shortcut(self, path, ignore, iters=300):
        path = [np.asarray(p, float) for p in path]
        for _ in range(iters):
            if len(path) <= 2:
                break
            i = int(self.rng.integers(0, len(path) - 2))
            j = int(self.rng.integers(i + 2, len(path)))
            if self.checker.edge_clear(path[i], path[j], ignore_bodies=ignore):
                path = path[:i + 1] + path[j:]
        return path
