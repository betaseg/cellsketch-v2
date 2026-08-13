import numpy as np
import pytest
from pixel_patrol_base.core.record import record_from

from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND
from pixel_patrol_cellsketch.plugins.processors.contacts import ContactsProcessor

SHAPE = (10, 20, 20)
VOXEL = (0.1, 0.02, 0.02)   # anisotropic, as in real data


def _cell(**entities: tuple[np.ndarray, str]):
    """A cell record whose channels are the given name → (volume, kind) entities."""
    names = list(entities)
    stack = np.stack([entities[n][0] for n in names], axis=0)
    meta = {
        "dim_order": "CZYX",
        "dim_names": ["C", "Z", "Y", "X"],
        "shape": list(stack.shape),
        "ndim": 4,
        "channel_names": names,
        "entity_kinds": [entities[n][1] for n in names],
        "cell_id": "cell_a",
        "cell_shape_zyx": list(SHAPE),
        "pixel_size_Z": VOXEL[0],
        "pixel_size_Y": VOXEL[1],
        "pixel_size_X": VOXEL[2],
    }
    return record_from(stack, meta, kind=CELL_KIND)


def _blocks(*specs: tuple[int, tuple[int, int, int], tuple[int, int, int]]) -> np.ndarray:
    """Label volume from (label, origin, size) blocks."""
    vol = np.zeros(SHAPE, dtype=np.int32)
    for label, origin, size in specs:
        vol[tuple(slice(o, o + s) for o, s in zip(origin, size))] = label
    return vol


def test_touching_instances_report_the_smallest_possible_gap():
    mito = _blocks((1, (2, 2, 2), (3, 3, 3)), (2, (2, 2, 5), (3, 3, 3)))  # share an X face
    row = ContactsProcessor().run_chunk(_cell(mito=(mito, "label")))

    assert row["contact_count"] == 1
    assert row["contact_entity_a"] == ["mito"]
    assert row["contact_label_a"] == [1]
    assert row["contact_label_b"] == [2]
    # One voxel step, not zero: the gap is a distance to the nearest voxel *of* the
    # other instance, and its own voxels are one step away. Matches analyze_cell.py.
    assert row["contact_gap_um"] == pytest.approx([VOXEL[2]], abs=1e-6)


def test_gap_grows_by_one_voxel_step_per_empty_voxel():
    # Two empty voxels along X (0.02 µm per voxel) → three steps.
    mito = _blocks((1, (2, 2, 2), (3, 3, 3)), (2, (2, 2, 7), (3, 3, 3)))
    row = ContactsProcessor().run_chunk(_cell(mito=(mito, "label")))

    assert row["contact_gap_um"] == pytest.approx([3 * VOXEL[2]], abs=1e-6)


def test_the_step_size_follows_the_axis_of_approach():
    """Anisotropy is respected: the same voxel distance along Z is 5× larger."""
    mito = _blocks((1, (2, 2, 2), (2, 3, 3)), (2, (5, 2, 2), (2, 3, 3)))  # one empty Z plane
    row = ContactsProcessor().run_chunk(_cell(mito=(mito, "label")))

    assert row["contact_gap_um"] == pytest.approx([2 * VOXEL[0]], abs=1e-6)


def test_pairs_beyond_the_threshold_are_not_recorded(monkeypatch):
    monkeypatch.setenv("CELLSKETCH_CONTACT_MAX_UM", "0.05")
    mito = _blocks((1, (2, 2, 2), (3, 3, 3)), (2, (2, 2, 8), (3, 3, 3)))  # 3 empty = 0.08 µm
    row = ContactsProcessor().run_chunk(_cell(mito=(mito, "label")))

    assert row["contact_count"] == 0
    # None, not []: an empty list would type the parquet column List(Null) and clash
    # with the lists written for cells that do have contacts.
    assert row.get("contact_gap_um") is None


def test_contacts_span_entities_and_mark_masks_with_a_null_label():
    mito = _blocks((1, (2, 2, 2), (3, 3, 3)))
    nucleus = _blocks((1, (2, 2, 5), (3, 3, 3)))
    row = ContactsProcessor().run_chunk(
        _cell(mito=(mito, "label"), nucleus=(nucleus, "mask"))
    )

    assert row["contact_count"] == 1
    assert row["contact_entity_a"] == ["mito"]
    assert row["contact_entity_b"] == ["nucleus"]
    assert row["contact_label_a"] == [1]
    assert row["contact_label_b"] == [None]   # a mask is one structure, with no instance id


def test_the_membrane_is_not_a_contact_partner():
    pm = np.ones(SHAPE, dtype=np.int32)          # encloses everything
    mito = _blocks((1, (2, 2, 2), (3, 3, 3)), (2, (2, 2, 5), (3, 3, 3)))
    row = ContactsProcessor().run_chunk(_cell(pm=(pm, "mask"), mito=(mito, "label")))

    # Only mito 1 ↔ mito 2. Membrane proximity is a distance, not a contact.
    assert row["contact_count"] == 1
    assert "pm" not in row["contact_entity_a"] + row["contact_entity_b"]


def test_a_cell_with_one_instance_has_no_pairs():
    row = ContactsProcessor().run_chunk(_cell(mito=(_blocks((1, (2, 2, 2), (3, 3, 3))), "label")))

    assert row["contact_count"] == 0


def test_partial_cell_is_refused():
    mito = _blocks((1, (2, 2, 2), (3, 3, 3)))
    record = _cell(mito=(mito, "label"))
    record.meta["cell_shape_zyx"] = [10, 40, 40]   # as if the volume had been split

    with pytest.raises(ValueError, match="mb-per-task"):
        ContactsProcessor().run_chunk(record)


def test_missing_channels_are_refused():
    """C-splitting is as damaging as spatial splitting: pairs span entities."""
    mito = _blocks((1, (2, 2, 2), (3, 3, 3)))
    record = _cell(mito=(mito, "label"))
    record.meta["channel_names"] = ["mito", "nucleus"]   # a channel that is not in the data

    with pytest.raises(ValueError, match="fragment"):
        ContactsProcessor().run_chunk(record)
