"""High-level autonomous tasks built on the pick-and-place controller.

SortingTask reads block positions live from the simulation and sorts each block
into its color-matched bin, so it adapts to wherever the blocks currently are.
"""

import mujoco
import numpy as np

from .pick_and_place import PickPlaceController


# Which bin each colored block belongs to (body names).
DEFAULT_SORT = {
    "block_red": "bin_red",
    "block_green": "bin_green",
    "block_blue": "bin_blue",
}


class SortingTask:
    """Sort colored blocks into their matching bins, one at a time."""

    def __init__(self, model, data, sort_map=None):
        self.model = model
        self.data = data
        self.ppc = PickPlaceController(model, data)
        self.sort_map = dict(sort_map) if sort_map else dict(DEFAULT_SORT)

    def _xy(self, body_name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[bid][:2].copy()

    def run(self, viewer=None, dt_realtime=False):
        """Execute the full sort. Returns the number of blocks successfully placed."""
        placed = 0
        for block, bin_name in self.sort_map.items():
            pick_xy = self._xy(block)
            place_xy = self._xy(bin_name)
            try:
                segs = self.ppc.plan(pick_xy=tuple(pick_xy), place_xy=tuple(place_xy))
            except ValueError as e:
                print(f"  [{block}] skipped: {e}")
                continue
            print(f"  [{block}] -> {bin_name}  pick={np.round(pick_xy,3)} place={np.round(place_xy,3)}")
            if not self.ppc.execute(segs, block=block, viewer=viewer, dt_realtime=dt_realtime):
                break
            placed += 1
        return placed
