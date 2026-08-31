"""Check the measurements against closed forms, and the distance transform against scipy.

The measuring is ITK's, so comparing to ITK would be circular. What is worth checking is that
we hand it the right thing and read the right numbers back: a sphere and a disc of known size,
measured through our own entry points, in µm with anisotropic sampling.
"""

import numpy as np
import pytest

from pixel_patrol_anatomy.distances import distance_transform_um
from pixel_patrol_anatomy.geometry import label_metrics, region_metrics

VOXEL = (0.1, 0.05, 0.05)     # z, y, x, anisotropic as real data is
PIXEL = (0.05, 0.05)          # y, x


def ball(radius_um: float, shape=(41, 81, 81)) -> np.ndarray:
    grid = np.indices(shape) - (np.array(shape) // 2)[:, None, None, None]
    return ((grid * np.array(VOXEL)[:, None, None, None]) ** 2).sum(0) <= radius_um ** 2


def disc(radius_um: float, shape=(81, 81)) -> np.ndarray:
    grid = np.indices(shape) - (np.array(shape) // 2)[:, None, None]
    return ((grid * np.array(PIXEL)[:, None, None]) ** 2).sum(0) <= radius_um ** 2


def test_a_sphere_measures_like_a_sphere():
    measured = region_metrics(ball(1.0), VOXEL)

    # 1% covers the sampling: a voxelised sphere is not a sphere.
    assert measured["volume_um3"] == pytest.approx(4 / 3 * np.pi, rel=0.01)
    assert measured["surface_area_um2"] == pytest.approx(4 * np.pi, rel=0.02)
    assert measured["sphericity"] == pytest.approx(1.0, abs=0.02)
    assert measured["aspect_ratio_major_minor"] == pytest.approx(1.0, abs=0.02)


def test_a_disc_measures_like_a_disc():
    measured = region_metrics(disc(1.0), PIXEL)

    assert measured["area_um2"] == pytest.approx(np.pi, rel=0.01)
    assert measured["perimeter_um"] == pytest.approx(2 * np.pi, rel=0.02)
    assert measured["circularity"] == pytest.approx(1.0, abs=0.03)


def test_a_cube_gets_its_volume_exactly_and_its_faces_a_little_short():
    """The flip side of a boundary estimator built for curves.

    ITK's is a Crofton estimator: right on a smooth surface, and about 10% short on flat
    axis-aligned faces, closing slowly as the shape gets bigger. Segmented organelles are
    curved, so this is the right way round to be wrong, but it is worth knowing.
    """
    cube = np.zeros((30, 30, 30), dtype=bool)
    cube[10:20, 10:20, 10:20] = True          # 10 samples per side, 5 µm

    measured = region_metrics(cube, (0.5, 0.5, 0.5))

    assert measured["volume_um3"] == pytest.approx(5.0 ** 3)
    assert 0.85 < measured["surface_area_um2"] / (6 * 5.0 ** 2) < 0.95
    assert measured["aspect_ratio_major_minor"] == pytest.approx(1.0, abs=0.02)


def test_an_elongated_shape_reports_its_axis_ratio():
    grid = np.indices((41, 81, 81)) - np.array([20, 40, 40])[:, None, None, None]
    physical = grid * np.array(VOXEL)[:, None, None, None]
    rod = ((physical[0] / 2.4) ** 2 + (physical[1] / 0.8) ** 2
           + (physical[2] / 0.8) ** 2) <= 1

    measured = region_metrics(rod, VOXEL)

    # 2.4 µm over 0.8 µm, and still round: an ellipsoid has no roughness.
    assert measured["aspect_ratio_major_minor"] == pytest.approx(3.0, rel=0.1)
    assert measured["sphericity"] == pytest.approx(1.0, abs=0.05)


def test_every_instance_is_measured_in_one_pass():
    labels = np.zeros((20, 60, 60), dtype=np.uint16)
    labels[5:15, 5:15, 5:15] = 1
    labels[5:10, 30:40, 30:40] = 2

    measured = label_metrics(labels, VOXEL)

    assert set(measured) == {1, 2}
    assert measured[1]["n_samples"] == 10 * 10 * 10
    assert measured[1]["volume_um3"] > measured[2]["volume_um3"]
    # The centroid comes back in array order, in µm.
    assert measured[1]["centroid_um"] == pytest.approx(
        (9.5 * VOXEL[0], 9.5 * VOXEL[1], 9.5 * VOXEL[2]), rel=0.02)


def test_the_distance_transform_matches_scipy():
    from scipy.ndimage import distance_transform_edt

    target = np.zeros((30, 60, 60), dtype=bool)
    target[15, 30, 30] = True

    ours = distance_transform_um(target, VOXEL)
    reference = distance_transform_edt(~target, sampling=VOXEL)

    # A different implementation of the same transform: the `edt` package against scipy.
    assert np.abs(ours - reference).max() < 1e-5


# ── distances, against a different algorithm and against the closed form ───────

def test_the_instance_to_target_distance_matches_a_kd_tree():
    """The distance the report carries, checked without a distance transform at all.

    We take the minimum of a Euclidean transform over an instance's samples. A KD-tree over
    the target's sample coordinates answers the same question by a different route, so the two
    agreeing is worth something.
    """
    from scipy.spatial import cKDTree

    from pixel_patrol_anatomy.distances import distance_target

    target = np.zeros((30, 60, 60), dtype=np.int32)
    target[10:20, 10:20, 10:20] = 1
    instance = np.zeros_like(target, dtype=bool)
    instance[22:25, 40:45, 40:45] = True

    field = distance_transform_um(distance_target(target, "mito", "label"), VOXEL)
    ours = float(field[instance].min())

    scale = np.array(VOXEL)
    tree = cKDTree(np.argwhere(target > 0) * scale)
    reference = float(tree.query(np.argwhere(instance) * scale)[0].min())

    assert ours == pytest.approx(reference, rel=1e-4)


def test_two_spheres_are_their_gap_apart():
    """Surface to surface, in µm, against the arithmetic."""
    from pixel_patrol_anatomy.distances import distance_target

    shape = (61, 61, 61)
    grid = np.indices(shape) * np.array(VOXEL)[:, None, None, None]
    left = ((grid[0] - 3.0) ** 2 + (grid[1] - 1.0) ** 2 + (grid[2] - 1.0) ** 2) <= 0.5 ** 2
    right = ((grid[0] - 3.0) ** 2 + (grid[1] - 2.0) ** 2 + (grid[2] - 1.0) ** 2) <= 0.5 ** 2

    field = distance_transform_um(distance_target(right.astype(np.int32), "x", "label"), VOXEL)
    measured = float(field[left].min())

    # Centres 1 µm apart, radius 0.5 each, so the surfaces meet: one sample step, not zero,
    # because the transform measures centre to centre.
    assert measured < 2 * max(VOXEL)


def test_the_object_mask_target_is_the_distance_to_the_boundary():
    """For the object mask the target is inverted, so the distance is to the boundary."""
    from pixel_patrol_anatomy.distances import distance_target

    mask = np.zeros((30, 60, 60), dtype=np.int32)
    mask[5:25, 10:50, 10:50] = 1

    inverted = distance_target(mask, "pm", "mask", object_mask_name="pm")
    field = distance_transform_um(inverted, VOXEL)

    # A sample in the middle of the mask is far from the boundary; one on its face is one
    # step away, in that axis's own sample size.
    assert field[15, 30, 30] > field[5, 30, 30]
    assert field[5, 30, 30] == pytest.approx(VOXEL[0])          # z face, 0.1 µm
    assert field[15, 10, 30] == pytest.approx(VOXEL[1])         # y face, 0.05 µm
