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

import json
import math
import unittest

import numpy as np

from analyze_cell import per_label_metrics


class PerLabelMetricsTest(unittest.TestCase):
    def test_returns_expected_rows_and_distance_columns(self) -> None:
        labels = np.zeros((8, 8, 8), dtype=np.int32)
        labels[1:3, 1:3, 1:3] = 1
        labels[5:7, 5:7, 5:7] = 2

        dt_membrane = np.full(labels.shape, 100.0, dtype=np.float32)
        dt_membrane[labels == 1] = 2.0
        dt_membrane[labels == 2] = 7.0

        df = per_label_metrics(
            labels=labels,
            voxel_size_zyx=(1.0, 1.0, 1.0),
            distance_transforms={"membrane": dt_membrane},
        )

        self.assertEqual(len(df), 2)
        self.assertIn("distance_to_membrane_um", df.columns)
        self.assertIn("distance_to_closest_same_type_um", df.columns)
        self.assertIn("volume_um3", df.columns)
        self.assertIn("surface_area_um2", df.columns)
        self.assertIn("sphericity", df.columns)
        self.assertIn("aspect_ratio_major_minor", df.columns)
        self.assertIn("branches", df.columns)

        row1 = df[df["label"] == 1].iloc[0]
        row2 = df[df["label"] == 2].iloc[0]

        self.assertAlmostEqual(float(row1["volume_um3"]), 8.0, places=6)
        self.assertAlmostEqual(float(row2["volume_um3"]), 8.0, places=6)
        self.assertAlmostEqual(float(row1["distance_to_membrane_um"]), 2.0, places=6)
        self.assertAlmostEqual(float(row2["distance_to_membrane_um"]), 7.0, places=6)

        expected_nn = math.sqrt(4.0**2 + 4.0**2 + 4.0**2)
        self.assertAlmostEqual(float(row1["distance_to_closest_same_type_um"]), expected_nn, places=6)
        self.assertAlmostEqual(float(row2["distance_to_closest_same_type_um"]), expected_nn, places=6)


    def test_dist_histogram_columns_added_when_flagged(self) -> None:
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[1:4, 1:4, 1:4] = 1  # 27 pixels

        # Give each pixel a different distance so the histogram has real spread
        dt = np.zeros(labels.shape, dtype=np.float32)
        coords = np.argwhere(labels == 1)
        for i, (z, y, x) in enumerate(coords):
            dt[z, y, x] = float(i + 1)  # distances 1..27

        df = per_label_metrics(
            labels=labels,
            voxel_size_zyx=(1.0, 1.0, 1.0),
            distance_transforms={"other": dt},
            compute_dist_histogram=True,
            dist_histogram_bins=5,
        )

        self.assertEqual(len(df), 1)
        row = df.iloc[0]

        # Existing min column must still be present and correct
        self.assertAlmostEqual(float(row["distance_to_other_um"]), 1.0, places=6)

        # New histogram columns must exist
        self.assertIn("distance_to_other_mean_um", df.columns)
        self.assertIn("distance_to_other_hist_min_um", df.columns)
        self.assertIn("distance_to_other_hist_max_um", df.columns)
        self.assertIn("distance_to_other_hist_um", df.columns)

        self.assertAlmostEqual(float(row["distance_to_other_hist_min_um"]), 1.0, places=6)
        self.assertAlmostEqual(float(row["distance_to_other_hist_max_um"]), 27.0, places=6)
        self.assertAlmostEqual(float(row["distance_to_other_mean_um"]), 14.0, places=6)  # mean of 1..27

        counts = json.loads(row["distance_to_other_hist_um"])
        self.assertEqual(len(counts), 5)
        self.assertEqual(sum(counts), 27)  # all pixels accounted for

    def test_dist_histogram_not_added_when_not_flagged(self) -> None:
        labels = np.zeros((6, 6, 6), dtype=np.int32)
        labels[1:3, 1:3, 1:3] = 1
        dt = np.full(labels.shape, 5.0, dtype=np.float32)

        df = per_label_metrics(
            labels=labels,
            voxel_size_zyx=(1.0, 1.0, 1.0),
            distance_transforms={"other": dt},
            compute_dist_histogram=False,
        )

        self.assertNotIn("distance_to_other_mean_um", df.columns)
        self.assertNotIn("distance_to_other_hist_um", df.columns)


if __name__ == "__main__":
    unittest.main()
