"""A 2D object is measured as a plane, not as a volume one voxel deep.

These cover the split: a plane carries area, perimeter and circularity and no volume,
sphericity or elevation; the 3D case is unchanged; and what means the same in both
(distances, nearest neighbour, skeleton length) keeps its name.
"""

import numpy as np
import pytest
from pixel_patrol_base.core.record import record_from

from pixel_patrol_anatomy.geometry import region_metrics
from pixel_patrol_anatomy.mesh import MeshOptions, mesh_rows_for_object, payload_counts
from pixel_patrol_anatomy.plugins.loaders.object_loader import OBJECT_KIND, ObjectLoader
from pixel_patrol_anatomy.plugins.processors.contacts import ContactsProcessor
from pixel_patrol_anatomy.plugins.processors.instances import InstanceProcessor
from pixel_patrol_anatomy.plugins.processors.morphology import MorphologyProcessor
from pixel_patrol_anatomy.viewer_extensions import get_viewer_extension_dir
from synthetic import PIXEL_SIZE_UM, make_object_2d

PIXEL = PIXEL_SIZE_UM


@pytest.fixture
def object_2d(tmp_path):
    make_object_2d(tmp_path / "flat_a", prefix="sample_a")
    return tmp_path / "flat_a"


def _record(**entities: tuple[np.ndarray, str]):
    """A 2D object record whose channels are the given name → (image, kind) entities."""
    names = list(entities)
    stack = np.stack([entities[n][0] for n in names], axis=0)
    meta = {
        "dim_order": "CYX",
        "channel_names": names,
        "entity_kinds": [entities[n][1] for n in names],
        "object_id": "flat_a",
        "object_mask_name": "pm",
        "object_shape": list(stack.shape[1:]),
        "spatial_dims": 2,
        "pixel_size_Y": PIXEL[0],
        "pixel_size_X": PIXEL[1],
    }
    return record_from(stack, meta, kind=OBJECT_KIND)


def _leaf(image: np.ndarray, *, name: str, kind: str):
    """One entity of one 2D object, as a LEAF processor sees it."""
    record = _record(**{name: (image, kind)})
    record.meta["dim_c"] = 0
    return record


def _blocks(*specs) -> np.ndarray:
    vol = np.zeros((40, 40), dtype=np.int32)
    for label, origin, size in specs:
        vol[tuple(slice(o, o + s) for o, s in zip(origin, size))] = label
    return vol


# ── loading ───────────────────────────────────────────────────────────────────

def test_a_2d_folder_loads_as_a_cyx_record(object_2d):
    record = ObjectLoader().load(object_2d)

    # Not CZYX with Z=1: the axes are what tell every processor which geometry to use.
    assert record.dim_order == "CYX"
    assert record.data.ndim == 3
    assert record.meta["spatial_dims"] == 2
    assert record.meta["object_mask_name"] == "pm"


def test_a_2d_object_has_no_z_pixel_size(object_2d):
    record = ObjectLoader().load(object_2d)

    # A made-up depth would silently turn every area into a volume.
    assert "pixel_size_Z" not in record.meta
    assert record.meta["pixel_size_Y"] == pytest.approx(PIXEL[0])
    assert "object_center_z_um" not in record.meta


def test_a_voxel_size_with_three_values_is_refused_for_2d(object_2d, monkeypatch):
    monkeypatch.setenv("PP_ANATOMY_VOXEL_SIZE_UM", "0.1,0.02,0.02")

    with pytest.raises(ValueError, match="images are 2D"):
        ObjectLoader().load(object_2d)


def test_a_2d_pixel_size_can_be_given_as_two_values(object_2d, monkeypatch):
    monkeypatch.setenv("PP_ANATOMY_VOXEL_SIZE_UM", "0.05,0.05")

    record = ObjectLoader().load(object_2d)

    assert record.meta["voxel_size_source"] == "config"
    assert (record.meta["pixel_size_Y"], record.meta["pixel_size_X"]) == (0.05, 0.05)


# ── which metrics exist ───────────────────────────────────────────────────────

def test_a_square_measures_like_a_square():
    square = np.zeros((20, 20), bool)
    square[5:15, 5:15] = True                    # 10 x 10 pixels at 0.5 µm = 5 x 5 µm

    metrics = region_metrics(square, (0.5, 0.5))

    assert metrics["area_um2"] == pytest.approx(25.0)
    # The perimeter estimator is built for curved boundaries and reads a little short on
    # axis-aligned ones; see test_reference_agreement for the same effect on a cube.
    assert 0.9 < metrics["perimeter_um"] / 20.0 < 1.0
    assert set(metrics) == {"area_um2", "perimeter_um", "circularity",
                            "aspect_ratio_major_minor"}


def test_a_3d_region_still_reports_volume_and_sphericity():
    cube = np.zeros((10, 10, 10), bool)
    cube[2:8, 2:8, 2:8] = True

    metrics = region_metrics(cube, (0.5, 0.5, 0.5))

    # The 2D columns exist in the schema but are never filled for a volume.
    assert set(metrics) == {"volume_um3", "surface_area_um2", "sphericity",
                            "aspect_ratio_major_minor"}


def test_the_entity_row_of_a_2d_mask_carries_area_not_volume():
    disc = np.zeros((40, 40), np.int32)
    yy, xx = np.mgrid[:40, :40]
    disc[(yy - 20) ** 2 + (xx - 20) ** 2 <= 100] = 1

    row = MorphologyProcessor().run_chunk(_leaf(disc, name="pm", kind="mask"))

    assert row["area_um2"] > 0 and row["total_area_um2"] == row["area_um2"]
    assert row["perimeter_um"] > 0 and 0 < row["circularity"] <= 1
    assert "volume_um3" not in row and "sphericity" not in row


def test_the_entity_row_of_a_2d_label_totals_area():
    labels = _blocks((1, (2, 2), (4, 4)), (2, (20, 20), (4, 4)))

    row = MorphologyProcessor().run_chunk(_leaf(labels, name="mito", kind="label"))

    assert row["instance_count"] == 2
    assert row["total_area_um2"] == pytest.approx(32 * PIXEL[0] * PIXEL[1])
    assert "total_volume_um3" not in row


def test_a_2d_object_polarity_is_one_angle_not_two():
    pm = np.zeros((40, 40), np.int32)
    pm[10:30, 10:30] = 1
    mito = _blocks((1, (12, 24), (3, 3)))

    row = InstanceProcessor().run_chunk(_record(pm=(pm, "mask"), mito=(mito, "label")))

    assert np.isfinite(row["instance_polar_angle_deg"][0])
    assert np.isfinite(row["instance_polar_dist_um"][0])
    # A plane has no elevation and no third component. The column comes back as None, not a
    # list of nulls, which is what makes PixelPatrol drop it from a 2D-only report.
    for absent in ("instance_polar_el_deg", "instance_polar_az_deg", "instance_polar_nz"):
        assert row[absent] is None, absent


def test_per_instance_2d_shape_metrics_are_lists_of_the_2d_kind():
    mito = _blocks((1, (2, 2), (4, 4)), (2, (20, 20), (6, 6)))

    row = InstanceProcessor().run_chunk(_record(mito=(mito, "label")))

    assert len(row["instance_area_um2"]) == 2
    assert all(v > 0 for v in row["instance_area_um2"])
    assert all(v > 0 for v in row["instance_perimeter_um"])
    assert row["instance_volume_um3"] is None
    assert row["instance_sphericity"] is None


def test_the_object_row_of_a_2d_object_reports_enclosed_area():
    pm = np.zeros((40, 40), np.int32)
    pm[10:30, 10:30] = 1
    mito = _blocks((1, (12, 12), (3, 3)))

    row = InstanceProcessor().run_chunk(_record(pm=(pm, "mask"), mito=(mito, "label")))

    assert row["object_area_um2"] == pytest.approx(400 * PIXEL[0] * PIXEL[1])
    assert "object_volume_um3" not in row


# ── things that mean the same in both ─────────────────────────────────────────

def test_distances_and_contacts_work_in_a_plane():
    mito = _blocks((1, (10, 10), (4, 4)), (2, (10, 17), (4, 4)))
    pm = np.zeros((40, 40), np.int32)
    pm[5:35, 5:35] = 1

    inst = InstanceProcessor().run_chunk(_record(pm=(pm, "mask"), mito=(mito, "label")))
    contacts = ContactsProcessor().run_chunk(_record(pm=(pm, "mask"), mito=(mito, "label")))

    # Two blocks with three empty pixels between them, measured in the plane: four steps,
    # the same voxel-step convention the 3D case uses.
    assert inst["instance_distance_to_closest_same_type_um"][0] > 0
    assert contacts["contact_count"] == 1
    assert contacts["contact_gap_um"][0] == pytest.approx(4 * PIXEL[1])


def test_a_2d_filament_gets_skeleton_metrics():
    filament = np.zeros((40, 40), np.int32)
    filament[20, 5:25] = 1

    row = InstanceProcessor().run_chunk(_record(mito=(filament, "label")))

    # 20 pixels → 19 steps of one pixel each.
    assert row["instance_branches"][0] == 1
    assert row["instance_length_um"][0] == pytest.approx(19 * PIXEL[1])
    assert row["instance_tortuosity"][0] == pytest.approx(1.0)


# ── geometry ──────────────────────────────────────────────────────────────────

def test_2d_geometry_is_outlines_and_never_meshes():
    disc = np.zeros((40, 40), np.int32)
    yy, xx = np.mgrid[:40, :40]
    disc[(yy - 20) ** 2 + (xx - 20) ** 2 <= 64] = 1

    rows = mesh_rows_for_object({"mito": disc}, {"mito": "label"}, PIXEL,
                                object_id="flat_a", options=MeshOptions(contact_max_um=None))

    assert len(rows) == 1
    assert rows[0]["mesh"] == b""
    assert rows[0]["spatial_dims"] == 2
    n_vertices, n_edges = payload_counts(rows[0]["outline"])
    # A closed loop: one edge per vertex, so a canvas can fill it and not only stroke it.
    assert n_vertices > 8 and n_edges == n_vertices


def test_3d_geometry_is_still_meshes_and_no_outlines():
    ball = np.zeros((20, 20, 20), np.int32)
    zz, yy, xx = np.mgrid[:20, :20, :20]
    ball[(zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2 <= 36] = 1

    rows = mesh_rows_for_object({"mito": ball}, {"mito": "label"}, (0.1, 0.1, 0.1),
                                object_id="ball", options=MeshOptions(contact_max_um=None))

    assert rows[0]["outline"] == b""
    assert rows[0]["mesh"] != b""
    assert rows[0]["spatial_dims"] == 3


# ── the whole pipeline ────────────────────────────────────────────────────────

def test_a_2d_batch_produces_the_same_row_layout(table_2d):
    import polars as pl

    objects = table_2d.filter(pl.col("obs_level") == 0)
    entities = table_2d.filter(pl.col("obs_level") == 1)

    # One row per object, one per entity: the layout the widgets read is the same in 2D.
    assert sorted(objects["object_id"].to_list()) == ["object_a", "object_b",
                                                      "object_c", "object_d"]
    assert entities.height == 4 * 3
    assert objects["spatial_dims"].to_list() == [2, 2, 2, 2]
    # No dim_order column: spatial_dims says which geometry an object has, and the axis order
    # never varies.
    assert "dim_order" not in table_2d.columns


def test_a_2d_report_has_the_2d_columns_and_not_the_3d_ones(table_2d):
    columns = set(table_2d.columns)

    assert {"area_um2", "perimeter_um", "circularity", "total_area_um2",
            "object_area_um2", "instance_area_um2", "instance_circularity",
            "instance_polar_angle_deg"} <= columns
    # PixelPatrol drops a column no row filled, so a 2D-only report simply has no volume.
    assert not ({"volume_um3", "sphericity", "surface_area_um2", "object_volume_um3",
                 "instance_volume_um3", "instance_polar_el_deg"} & columns)


def test_a_3d_report_still_has_the_3d_columns_and_not_the_2d_ones(table):
    columns = set(table.columns)

    assert {"volume_um3", "sphericity", "surface_area_um2", "instance_volume_um3"} <= columns
    assert not ({"area_um2", "circularity", "perimeter_um", "instance_area_um2",
                 "instance_polar_angle_deg"} & columns)


# ── the widgets' own queries, against a real 2D geometry file ─────────────────

def test_the_widget_queries_find_2d_geometry(tmp_path):
    """The SQL the 3D widgets run must find outlines where a 3D batch has meshes."""
    import duckdb

    from pixel_patrol_anatomy.mesh import GEOMETRY_FILENAME, write_geometry

    disc = np.zeros((40, 40), np.int32)
    yy, xx = np.mgrid[:40, :40]
    disc[(yy - 20) ** 2 + (xx - 20) ** 2 <= 144] = 1
    mito = _blocks((1, (10, 10), (4, 4)), (2, (24, 24), (4, 4)))
    rows = mesh_rows_for_object({"pm": disc, "mito": mito}, {"pm": "mask", "mito": "label"},
                                PIXEL, object_id="flat_a",
                                options=MeshOptions(contact_max_um=0.5))
    path = write_geometry(tmp_path / "flat_a", rows)
    assert path.name == GEOMETRY_FILENAME

    con = duckdb.connect()
    source = f"read_parquet(['{path}'])"
    # Mirrors HAS_GEOMETRY and GEOMETRY_SIZE in plugin_anatomy_3d.js.
    has_geometry = '("mesh" IS NOT NULL OR "outline" IS NOT NULL)'
    size = 'COALESCE("volume_um3", "area_um2")'

    drawable = con.execute(
        f'''SELECT "row_type", "entity_name", "label_id", "mesh" IS NULL AS flat
            FROM {source} WHERE {has_geometry} ORDER BY {size} DESC NULLS LAST'''
    ).fetchall()
    assert [r[0] for r in drawable] == ["file", "instance", "instance"]
    # Every drawable row in a 2D object is flat, so the widgets build lines, not surfaces.
    assert all(r[3] for r in drawable)

    # And the gallery's filter still finds instances to rank by a 2D metric.
    gallery = con.execute(
        f'''SELECT count(*) FROM {source}
            WHERE "row_type" = 'instance' AND {has_geometry} AND isfinite("area_um2")'''
    ).fetchone()
    assert gallery == (2,)


def test_the_widgets_offer_only_the_metrics_a_2d_report_has(table_2d, table):
    """metricsFor() in the widgets drops the other dimensionality's metrics."""
    source = (get_viewer_extension_dir() / "plugin_anatomy_3d.js").read_text()

    # The lists the widgets filter by have to name every column the processors can write,
    # or a metric silently disappears from the picker.
    for metric in ("volume_um3", "surface_area_um2", "sphericity", "polar_az_deg",
                   "polar_el_deg"):
        assert f"'{metric}'" in source
    for metric in ("area_um2", "perimeter_um", "circularity", "polar_angle_deg"):
        assert f"'{metric}'" in source

    # And the report columns the switch reads are the ones the pipeline actually writes.
    assert "total_area_um2" in table_2d.columns
    assert "total_volume_um3" in table.columns


def test_a_report_can_hold_both_dimensionalities(tmp_path):
    """One report, one 2D object and one 3D object: each fills only its own columns."""
    import polars as pl
    from pixel_patrol_base import api

    from pixel_patrol_anatomy import pipeline
    from pixel_patrol_anatomy.cli import FLAVOR, find_object_dirs
    from synthetic import make_object, make_object_2d

    root = tmp_path / "mixed"
    make_object(root / "volumes" / "object_a", prefix="sample_a")
    make_object_2d(root / "planes" / "object_p", prefix="sample_p")
    out = tmp_path / "mixed.parquet"
    paths = ["volumes", "planes"]
    report = pipeline.analyse(find_object_dirs(root), root, paths, workers=1)
    assert not report.failures, report.failures
    table, _ = api.load(pipeline.write(report, out, root=root, paths=paths, flavor=FLAVOR))

    objects = table.filter(pl.col("obs_level") == 0).sort("object_id")
    assert objects["spatial_dims"].to_list() == [3, 2]
    # The volume carries a volume and no area; the plane the reverse. Neither borrows the
    # other's formula, which is the whole point of keeping the two sets apart.
    assert objects["object_volume_um3"].to_list()[0] is not None
    assert objects["object_volume_um3"].to_list()[1] is None
    assert objects["object_area_um2"].to_list()[0] is None
    assert objects["object_area_um2"].to_list()[1] is not None


def test_the_3d_widgets_list_structures_in_a_2d_report(tmp_path):
    """Their "has something to draw" test must accept an outline, or 2D looks empty."""
    import duckdb

    from pixel_patrol_anatomy.mesh import write_geometry

    mito = _blocks((1, (10, 10), (4, 4)), (2, (24, 24), (4, 4)))
    pm = np.zeros((40, 40), np.int32)
    pm[5:35, 5:35] = 1
    rows = mesh_rows_for_object({"pm": pm, "mito": mito}, {"pm": "mask", "mito": "label"},
                                PIXEL, object_id="flat_a", options=MeshOptions(contact_max_um=None))
    path = write_geometry(tmp_path / "flat_a", rows)

    source = f"read_parquet(['{path}'])"
    has_geometry = '("mesh" IS NOT NULL OR "outline" IS NOT NULL)'
    names = duckdb.connect().execute(
        f'''SELECT DISTINCT "entity_name" FROM {source} WHERE {has_geometry} ORDER BY 1'''
    ).fetchall()

    assert names == [("mito",), ("pm",)]
