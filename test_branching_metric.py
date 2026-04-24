#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.2.0",
#   "scipy>=1.13.0",
#   "scikit-image>=0.23.0",
#   "tifffile>=2024.5.0",
#   "matplotlib>=3.9.0",
#   "edt>=2.4.0",
# ]
# ///

import unittest

import numpy as np

from analyze_cell import count_branch_points


class BranchingMetricTest(unittest.TestCase):
    def test_straight_line_has_one_branch_segment(self) -> None:
        vol = np.zeros((9, 9, 9), dtype=bool)
        vol[4, 2:7, 4] = True
        self.assertEqual(count_branch_points(vol), 1)

    def test_y_shape_has_three_branch_segments(self) -> None:
        vol = np.zeros((11, 11, 11), dtype=bool)
        # trunk
        vol[5, 2:6, 5] = True
        # left arm
        vol[5, 5, 3:6] = True
        # right arm
        vol[5, 5, 5:8] = True
        self.assertEqual(count_branch_points(vol), 3)

    def test_loop_has_one_branch_segment(self) -> None:
        vol = np.zeros((11, 11, 11), dtype=bool)
        z = 5
        vol[z, 3:8, 3] = True
        vol[z, 3:8, 7] = True
        vol[z, 3, 3:8] = True
        vol[z, 7, 3:8] = True
        self.assertEqual(count_branch_points(vol), 1)


if __name__ == "__main__":
    unittest.main()
