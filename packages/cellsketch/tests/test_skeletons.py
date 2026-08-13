"""Which entities get skeletonised, and computing each cell's set only once."""

import numpy as np
import pytest
from pixel_patrol_base.core.record import record_from

from pixel_patrol_cellsketch import skeletons
from pixel_patrol_cellsketch.mesh import MeshOptions, mesh_rows_for_cell
from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND
from pixel_patrol_cellsketch.plugins.processors.instances import InstanceProcessor
from pixel_patrol_cellsketch.skeletons import CACHE, parse_entity_filter, wants_skeletons

SHAPE = (10, 24, 24)
VOXEL = (0.1, 0.02, 0.02)


@pytest.fixture(autouse=True)
def clean_cache():
    CACHE.clear()
    yield
    CACHE.clear()


def _strand(label=1, y=6) -> np.ndarray:
    vol = np.zeros(SHAPE, dtype=np.int32)
    vol[4:6, y:y + 2, 4:20] = label          # a filament, worth a skeleton
    return vol


def _cell(**entities):
    names = list(entities)
    stack = np.stack([entities[n][0] for n in names], axis=0)
    meta = {
        "dim_order": "CZYX", "dim_names": ["C", "Z", "Y", "X"], "shape": list(stack.shape),
        "ndim": 4, "channel_names": names, "entity_kinds": [entities[n][1] for n in names],
        "cell_id": "cell_a", "membrane_name": "pm", "cell_shape_zyx": list(SHAPE),
        "pixel_size_Z": VOXEL[0], "pixel_size_Y": VOXEL[1], "pixel_size_X": VOXEL[2],
    }
    return record_from(stack, meta, kind=CELL_KIND)


# ── the filter ────────────────────────────────────────────────────────────────

def test_no_filter_means_every_entity():
    assert wants_skeletons("granules", None) is True


def test_names_match_however_they_were_capitalised():
    # --skeleton-entities ER has to match the entity discovered as 'er'.
    allowed = parse_entity_filter("mito, ER")
    assert wants_skeletons("er", allowed)
    assert wants_skeletons("mito", allowed)
    assert not wants_skeletons("granules", allowed)


def test_an_empty_filter_means_nothing_gets_skeletonised():
    assert parse_entity_filter(None, none=True) == frozenset()
    assert not wants_skeletons("mito", frozenset())


def test_excluded_entities_report_skeleton_metrics_as_not_measured(monkeypatch):
    monkeypatch.setenv("CELLSKETCH_SKELETON_ENTITIES", "mito")
    record = _cell(mito=(_strand(), "label"), granules=(_strand(1, 12), "label"))

    row = InstanceProcessor().run_chunk(record)

    by_entity = dict(zip(row["instance_entity"], row["instance_length_um"]))
    assert by_entity["mito"] > 0                    # asked for
    assert by_entity["granules"] is None            # not measured: null, not zero


def test_skeleton_metrics_are_measured_for_everything_by_default(monkeypatch):
    monkeypatch.delenv("CELLSKETCH_SKELETON_ENTITIES", raising=False)
    record = _cell(mito=(_strand(), "label"), granules=(_strand(1, 12), "label"))

    row = InstanceProcessor().run_chunk(record)

    assert all(length > 0 for length in row["instance_length_um"])


# ── the cache ─────────────────────────────────────────────────────────────────

def test_the_second_reader_of_a_cell_gets_the_cached_skeletons(monkeypatch):
    calls = []
    real = skeletons.compute_curve_skeletons

    def counting(labels, voxel, **kwargs):
        calls.append(labels.shape)
        return real(labels, voxel, **kwargs)

    monkeypatch.setattr(skeletons, "compute_curve_skeletons", counting)
    record = _cell(mito=(_strand(), "label"))
    volumes = {"mito": record.data[0]}

    InstanceProcessor().run_chunk(record)                       # metrics
    mesh_rows_for_cell(volumes, {"mito": "label"}, VOXEL, cell_id="cell_a",
                       options=MeshOptions(contact_max_um=None))  # geometry

    # Skeletonising is the most expensive step; --with-mesh must not pay for it twice.
    assert len(calls) == 1


def test_a_different_cell_is_not_served_from_the_cache(monkeypatch):
    calls = []
    real = skeletons.compute_curve_skeletons
    monkeypatch.setattr(
        skeletons, "compute_curve_skeletons",
        lambda labels, voxel, **kw: (calls.append(1), real(labels, voxel, **kw))[1],
    )

    InstanceProcessor().run_chunk(_cell(mito=(_strand(), "label")))
    InstanceProcessor().run_chunk(_cell(mito=(_strand(1, 12), "label")))

    # Same cell_id, different data: cell folder names are not unique across groups, so
    # answering on the name alone would hand one cell another cell's skeletons.
    assert len(calls) == 2


def test_geometry_matches_whether_it_was_cached_or_not():
    volumes, kinds = {"mito": _strand()}, {"mito": "label"}
    fresh = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                               options=MeshOptions(contact_max_um=None))
    CACHE.clear()
    InstanceProcessor().run_chunk(_cell(mito=(volumes["mito"], "label")))
    cached = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                                options=MeshOptions(contact_max_um=None))

    assert [r["skeleton_b64"] for r in cached] == [r["skeleton_b64"] for r in fresh]
    assert all(r["skeleton_b64"] for r in cached)


def test_contacts_are_found_once_however_many_readers_ask(monkeypatch):
    import pixel_patrol_cellsketch.contacts as contacts_module
    from pixel_patrol_cellsketch.plugins.processors.contacts import ContactsProcessor

    calls = []
    real = contacts_module.pairwise_instance_gaps
    monkeypatch.setattr(
        contacts_module, "pairwise_instance_gaps",
        lambda *a, **kw: (calls.append(1), real(*a, **kw))[1],
    )
    record = _cell(mito=(_strand(), "label"), granules=(_strand(1, 12), "label"))
    volumes = {"mito": record.data[0], "granules": record.data[1]}
    kinds = {"mito": "label", "granules": "label"}

    ContactsProcessor().run_chunk(record)                        # the table's edge list
    mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                       options=MeshOptions(contact_max_um=0.5))  # the 3D viewer's copy

    # Finding pairs is seconds to minutes depending on instance count; --with-mesh must
    # not pay for it twice.
    assert len(calls) == 1


def test_a_different_gap_threshold_is_not_served_from_the_cache():
    volumes, kinds = {"mito": _strand(), "granules": _strand(1, 12)}, {"mito": "label", "granules": "label"}
    wide = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                              options=MeshOptions(contact_max_um=0.5))
    tight = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                               options=MeshOptions(contact_max_um=0.001))

    n_wide = sum(1 for r in wide if r["row_type"] == "contact")
    n_tight = sum(1 for r in tight if r["row_type"] == "contact")
    assert n_wide > n_tight


def test_the_mesh_overlay_follows_the_same_filter():
    volumes = {"mito": _strand(), "granules": _strand(1, 12)}
    kinds = {"mito": "label", "granules": "label"}
    rows = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                              options=MeshOptions(skeleton_entities=frozenset({"mito"}),
                                                  contact_max_um=None))

    overlay = {r["entity_name"]: bool(r["skeleton_b64"]) for r in rows}
    assert overlay == {"mito": True, "granules": False}
