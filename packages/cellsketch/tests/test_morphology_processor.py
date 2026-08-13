import importlib.util

import numpy as np
import pytest
from pixel_patrol_base.core.record import record_from

from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND
from pixel_patrol_cellsketch.plugins.processors.morphology import MorphologyProcessor

VOXEL = (0.1, 0.02, 0.02)


def _leaf(volume: np.ndarray, *, name: str, kind: str, cell_shape=None) -> object:
    """A record shaped like the leaf block the pipeline hands to a LEAF processor."""
    data = volume[np.newaxis, ...]
    meta = {
        "dim_order": "CZYX",
        "dim_names": ["C", "Z", "Y", "X"],
        "shape": list(data.shape),
        "ndim": 4,
        "dim_c": 0,
        "channel_names": [name],
        "entity_kinds": [kind],
        "cell_shape_zyx": list(cell_shape or volume.shape),
        "pixel_size_Z": VOXEL[0],
        "pixel_size_Y": VOXEL[1],
        "pixel_size_X": VOXEL[2],
    }
    return record_from(data, meta, kind=CELL_KIND)


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
    # cell-row sum counts label instances only.
    assert "instance_count" not in row
    assert row["volume_um3"] == pytest.approx(4 * 6 * 6 * voxel_um3)
    assert row["total_volume_um3"] == pytest.approx(row["volume_um3"])
    assert row["surface_area_um2"] > 0
    assert 0 < row["sphericity"] <= 1.2
    assert "instance_volume_um3" not in row


def test_label_entity_reports_one_list_element_per_instance():
    volume = _cube((10, 20, 20), (2, 2, 2), (3, 3, 3), value=1)
    volume += _cube((10, 20, 20), (5, 12, 12), (4, 4, 4), value=2)
    row = MorphologyProcessor().run_chunk(_leaf(volume, name="mito", kind="label"))

    voxel_um3 = np.prod(VOXEL)
    assert row["instance_count"] == 2
    assert row["instance_label"] == [1, 2]
    assert row["instance_volume_um3"] == pytest.approx([27 * voxel_um3, 64 * voxel_um3])
    assert row["total_volume_um3"] == pytest.approx(sum(row["instance_volume_um3"]))
    for col in ("instance_surface_area_um2", "instance_sphericity", "instance_aspect_ratio_major_minor"):
        assert len(row[col]) == 2
    # Whole-structure columns stay absent for label entities — a label entity's
    # "sphericity" would be the sphericity of a union of unrelated objects.
    assert "sphericity" not in row


def test_empty_label_entity_emits_null_lists_not_empty_ones():
    row = MorphologyProcessor().run_chunk(
        _leaf(np.zeros((10, 20, 20), dtype=np.int32), name="mito", kind="label")
    )

    assert row["instance_count"] == 0
    # None, not []: an empty list would type the parquet column List(Null) and clash
    # with the List(Double) written for cells that do have instances.
    assert row["instance_volume_um3"] is None


def test_spatial_fragment_is_refused():
    volume = _cube((10, 20, 20), (2, 5, 5), (4, 6, 6))
    record = _leaf(volume, name="mito", kind="label", cell_shape=(10, 40, 40))

    with pytest.raises(ValueError, match="mb-per-task"):
        MorphologyProcessor().run_chunk(record)


def test_aggregation_sums_counts_and_drops_per_entity_scalars():
    proc = MorphologyProcessor()
    rows = [
        {"instance_count": 4, "sphericity": 0.8, "total_volume_um3": 2.0, "entity_name": "mito"},
        {"instance_count": 3, "sphericity": 0.5, "total_volume_um3": 5.0, "entity_name": "nucleus"},
    ]

    assert proc.get_aggregation("instance_count")(rows, {}) == 7
    # Entity volumes overlap (organelles sit inside the membrane), so no cell-level sum.
    assert proc.get_aggregation("total_volume_um3")(rows, {}) is None
    assert proc.get_aggregation("sphericity")(rows, {}) is None
    assert proc.get_aggregation("sphericity")(rows[:1], {}) == 0.8
    assert proc.get_aggregation("entity_name")(rows[:1], {}) == "mito"
    # Group size decides, not how many rows carry the column: 'nucleus' has no
    # instance list, but the cell row must still not inherit 'mito's.
    assert proc.get_aggregation("instance_volume_um3")(
        [{"instance_volume_um3": [1.0, 2.0]}, {"entity_name": "nucleus"}], {}
    ) is None


@pytest.mark.skipif(
    importlib.util.find_spec("kimimaro") is not None, reason="kimimaro is installed"
)
def test_skeleton_metrics_are_nan_when_kimimaro_is_missing():
    volume = _cube((10, 20, 20), (2, 2, 2), (3, 3, 3), value=1)
    row = MorphologyProcessor().run_chunk(_leaf(volume, name="mito", kind="label"))

    # Not measured, not zero-length: the columns exist so a run with the optional
    # extra installed writes the same schema.
    assert np.isnan(row["instance_branches"]).all()
    assert np.isnan(row["instance_length_um"]).all()
