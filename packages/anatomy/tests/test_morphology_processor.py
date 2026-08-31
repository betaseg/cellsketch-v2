import numpy as np
import pytest
from pixel_patrol_base.core.record import record_from

from pixel_patrol_anatomy.plugins.loaders.object_loader import OBJECT_KIND
from pixel_patrol_anatomy.plugins.processors.morphology import MorphologyProcessor

VOXEL = (0.1, 0.02, 0.02)


def _leaf(volume: np.ndarray, *, name: str, kind: str, object_shape=None, center=None) -> object:
    """A record shaped like the leaf block the pipeline hands to a LEAF processor."""
    data = volume[np.newaxis, ...]
    meta = {
        **({"object_center_z_um": center[0], "object_center_y_um": center[1],
            "object_center_x_um": center[2]} if center else {}),
        "dim_order": "CZYX",
        "dim_names": ["C", "Z", "Y", "X"],
        "shape": list(data.shape),
        "ndim": 4,
        "dim_c": 0,
        "channel_names": [name],
        "entity_kinds": [kind],
        "object_shape": list(object_shape or volume.shape),
        "pixel_size_Z": VOXEL[0],
        "pixel_size_Y": VOXEL[1],
        "pixel_size_X": VOXEL[2],
    }
    return record_from(data, meta, kind=OBJECT_KIND)


def _cube(shape, origin, size, value=1) -> np.ndarray:
    vol = np.zeros(shape, dtype=np.int32)
    sl = tuple(slice(o, o + s) for o, s in zip(origin, size))
    vol[sl] = value
    return vol


def test_mask_entity_reports_whole_structure_morphology():
    volume = _cube((10, 20, 20), (2, 5, 5), (4, 6, 6))
    row = MorphologyProcessor().run_chunk(_leaf(volume, name="nucleus", kind="mask"))

    voxel_um3 = np.prod(VOXEL)
    assert row["entity_name"] == "nucleus"
    assert row["entity_kind"] == "mask"
    # A mask is one structure, not one instance — instance_count stays null so the
    # object-row sum counts label instances only.
    assert "instance_count" not in row
    assert row["volume_um3"] == pytest.approx(4 * 6 * 6 * voxel_um3)
    assert row["total_volume_um3"] == pytest.approx(row["volume_um3"])
    assert row["surface_area_um2"] > 0
    assert 0 < row["sphericity"] <= 1.2


def test_mask_polarity_is_measured_from_the_object_centre_in_the_record():
    # Centroid at voxel (5, 10, 15): level with the centre in Z and Y, displaced in +X.
    volume = _cube((10, 20, 20), (4, 9, 14), (3, 3, 3))
    centre = (5 * VOXEL[0], 10 * VOXEL[1], 10 * VOXEL[2])
    row = MorphologyProcessor().run_chunk(
        _leaf(volume, name="nucleus", kind="mask", center=centre)
    )

    assert row["polar_dist_um"] > 0
    assert row["polar_az_deg"] == pytest.approx(0.0, abs=1.0)   # +X is azimuth 0
    assert row["polar_el_deg"] == pytest.approx(0.0, abs=1.0)


def test_mask_polarity_is_absent_without_an_object_centre():
    volume = _cube((10, 20, 20), (2, 5, 5), (4, 6, 6))
    row = MorphologyProcessor().run_chunk(_leaf(volume, name="nucleus", kind="mask"))

    assert "polar_dist_um" not in row


def test_label_entity_reports_counts_and_totals_only():
    volume = _cube((10, 20, 20), (2, 2, 2), (3, 3, 3), value=1)
    volume += _cube((10, 20, 20), (5, 12, 12), (4, 4, 4), value=2)
    row = MorphologyProcessor().run_chunk(_leaf(volume, name="mito", kind="label"))

    voxel_um3 = np.prod(VOXEL)
    assert row["instance_count"] == 2
    assert row["total_volume_um3"] == pytest.approx((27 + 64) * voxel_um3)
    # Per-instance measurements need the rest of the object (distances, polarity), so they
    # are the object-level processor's job, not this one's.
    assert "instance_volume_um3" not in row
    # And a label entity has no whole-structure morphology: the "sphericity" of a union
    # of unrelated objects is meaningless.
    assert "sphericity" not in row


def test_empty_label_entity_counts_zero():
    row = MorphologyProcessor().run_chunk(
        _leaf(np.zeros((10, 20, 20), dtype=np.int32), name="mito", kind="label")
    )

    assert row["instance_count"] == 0
    assert row["total_volume_um3"] == 0.0


def test_memory_chunked_fragment_is_refused():
    volume = _cube((10, 20, 20), (2, 5, 5), (4, 6, 6))
    record = _leaf(volume, name="mito", kind="label", object_shape=(10, 40, 40))

    with pytest.raises(ValueError, match="measured whole"):
        MorphologyProcessor().run_chunk(record)


def test_a_single_plane_leaf_blames_the_leaf_configuration_not_memory():
    """Forgetting --slice-size Z=-1 gives per-plane leaves; say so, don't blame memory."""
    plane = _cube((1, 20, 20), (0, 5, 5), (1, 6, 6))
    record = _leaf(plane, name="mito", kind="label", object_shape=(10, 20, 20))

    with pytest.raises(ValueError, match=r"--slice-size Z=-1"):
        MorphologyProcessor().run_chunk(record)


def test_aggregation_sums_counts_and_drops_per_entity_scalars():
    proc = MorphologyProcessor()
    rows = [
        {"instance_count": 4, "sphericity": 0.8, "total_volume_um3": 2.0, "entity_name": "mito"},
        {"instance_count": 3, "sphericity": 0.5, "total_volume_um3": 5.0, "entity_name": "nucleus"},
    ]

    assert proc.get_aggregation("instance_count")(rows, {}) == 7
    # Entity volumes overlap (organelles sit inside the membrane), so no object-level sum.
    assert proc.get_aggregation("total_volume_um3")(rows, {}) is None
    assert proc.get_aggregation("sphericity")(rows, {}) is None
    assert proc.get_aggregation("sphericity")(rows[:1], {}) == 0.8
    assert proc.get_aggregation("entity_name")(rows[:1], {}) == "mito"
