"""Mesh and skeleton geometry, and the two ways of producing it.

The payload format is a contract with the viewer's 3D widgets and geometry_to_blender.py,
so the tests decode it the way those do rather than trusting the encoder.
"""

import struct
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner
from pixel_patrol_base.core.record import record_from

from pixel_patrol_anatomy.cli import cli
from pixel_patrol_anatomy.mesh import (
    GEOMETRY_COLUMNS,
    MeshOptions,
    generate_mesh,
    mesh_rows_for_object,
    payload_counts,
    sigma_for_shape,
    write_geometry,
)
from pixel_patrol_anatomy.plugins.loaders.object_loader import OBJECT_KIND
from pixel_patrol_anatomy.plugins.processors.mesh import MeshProcessor
from synthetic import make_object, make_dataset

SHAPE = (12, 24, 24)
VOXEL = (0.1, 0.02, 0.02)


def decode_payload(raw: bytes, per_index: int = 3):
    """Mirror of decodePayload in plugin_anatomy_3d.js and geometry_to_blender.py."""
    n_verts, n_indices = struct.unpack_from("<II", raw, 0)
    params = np.frombuffer(raw, dtype="<f4", count=6, offset=8)
    verts_q = np.frombuffer(raw, dtype="<u2", count=n_verts * 3, offset=32).reshape(n_verts, 3)
    indices = np.frombuffer(
        raw, dtype="<u4", count=n_indices * per_index, offset=32 + n_verts * 6
    ).reshape(n_indices, per_index)
    assert 32 + n_verts * 6 + n_indices * per_index * 4 == len(raw), "trailing bytes"
    return verts_q / 65535.0 * params[3:] + params[:3], indices


def _ball(radius=4, centre=(6, 12, 12)) -> np.ndarray:
    zz, yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    return (
        ((zz - centre[0]) / radius) ** 2 + ((yy - centre[1]) / radius) ** 2
        + ((xx - centre[2]) / radius) ** 2
    ) <= 1.0


# ── payloads ──────────────────────────────────────────────────────────────────

def test_mesh_payload_decodes_to_vertices_in_micrometres():
    verts, faces = decode_payload(generate_mesh(_ball(), (0, 0, 0), VOXEL))

    assert len(verts) > 0 and len(faces) > 0
    assert faces.max() < len(verts)              # every face indexes a real vertex
    # X/Y span at most 24 voxels × 0.02 µm, Z at most 12 × 0.1 µm.
    assert verts[:, 0].max() <= 24 * VOXEL[2]
    assert verts[:, 2].max() <= 12 * VOXEL[0]


def test_mesh_vertices_are_offset_by_the_bounding_box_origin():
    at_origin, _ = decode_payload(generate_mesh(_ball(), (0, 0, 0), VOXEL))
    offset, _ = decode_payload(generate_mesh(_ball(), (2, 3, 4), VOXEL))

    # The origin is in voxels, the vertices in µm: an instance meshed from its own bbox
    # still lands in whole-volume coordinates, which is what aligns it with its skeleton.
    assert offset[:, 0].min() - at_origin[:, 0].min() == pytest.approx(4 * VOXEL[2], abs=1e-4)
    assert offset[:, 2].min() - at_origin[:, 2].min() == pytest.approx(2 * VOXEL[0], abs=1e-4)


def test_a_structure_too_small_to_mesh_yields_no_geometry():
    tiny = np.zeros(SHAPE, dtype=bool)
    tiny[0, 0, 0] = True

    assert generate_mesh(tiny, (0, 0, 0), VOXEL) == b""


def test_smoothing_sigma_follows_shape():
    # A blob gets more smoothing than a sparse, thin structure.
    blob = sigma_for_shape(0.95, 0.5, sigma_min=0.3, sigma_max=1.5)
    strand = sigma_for_shape(0.2, 0.05, sigma_min=0.3, sigma_max=1.5)

    assert 0.3 <= strand < blob <= 1.5
    # No metric to go on: the midpoint, not an extreme.
    assert sigma_for_shape(float("nan"), 0.5) == pytest.approx(0.9)


# ── rows ──────────────────────────────────────────────────────────────────────

def _object_volumes():
    mito = np.zeros(SHAPE, dtype=np.int32)
    mito[_ball(3, (6, 6, 6))] = 1
    mito[_ball(3, (6, 18, 18))] = 2
    return {"pm": _ball(10).astype(np.int32), "mito": mito}, {"pm": "mask", "mito": "label"}


def test_rows_cover_label_instances_and_whole_masks():
    volumes, kinds = _object_volumes()
    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a",
                              options=MeshOptions(contact_max_um=None))

    by_type = {(r["row_type"], r["entity_name"]) for r in rows}
    # The widgets read instance rows for labels and file rows for whole masks.
    assert ("instance", "mito") in by_type
    assert ("file", "pm") in by_type
    assert all(r["mesh"] for r in rows)
    assert all(set(r) <= set(GEOMETRY_COLUMNS) for r in rows)


def test_instance_rows_get_their_polarity_without_a_report_to_carry_from():
    """`mesh` after the fact has no per-instance metrics to be handed, and still has to
    produce geometry the viewer can explode: the direction is measurable from the volumes.
    """
    volumes, kinds = _object_volumes()

    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a",
                                object_mask_name="pm",
                                options=MeshOptions(contact_max_um=None))

    instances = [r for r in rows if r["row_type"] == "instance"]
    assert instances and all(r["polar_dist_um"] > 0 for r in instances)
    # And the unit vector points somewhere: not every instance sits on the centre.
    assert any(abs(r["polar_nx"]) + abs(r["polar_ny"]) + abs(r["polar_nz"]) > 0
               for r in instances)


def test_instance_rows_carry_the_metrics_they_are_given():
    """The 3D widgets colour and explode from these, and never touch the report.

    A mask entity is measured on its own way through the same loop, and rebinding the
    parameter there left every label row without a single carried metric: the explode slider
    had nothing to push along and silently did nothing.
    """
    volumes, kinds = _object_volumes()
    carried = {("mito", 1): {"polar_dist_um": 1.25, "polar_nz": 0.0, "polar_ny": 0.6,
                             "polar_nx": 0.8, "length_um": 3.5}}

    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a",
                                options=MeshOptions(contact_max_um=None), metrics=carried)

    instance = next(r for r in rows if r["row_type"] == "instance" and r["label_id"] == 1)
    assert instance["polar_dist_um"] == 1.25
    assert (instance["polar_nz"], instance["polar_ny"], instance["polar_nx"]) == (0.0, 0.6, 0.8)
    assert instance["length_um"] == 3.5
    # A whole-structure mask has no per-instance metrics to carry, and measures its own shape.
    mask = next(r for r in rows if r["row_type"] == "file")
    assert mask["volume_um3"] > 0 and "polar_dist_um" not in mask


def test_smoothing_is_the_same_blur_in_every_direction():
    """Sigma is in the finest sample, so an anisotropic stack is not blurred along z.

    The field and the kernel were both in samples, which on a stack with 5x coarser z
    smoothed five times as far in z as in x and flattened everything.
    """
    ball = _ball(radius=4)
    # step_size=1 in both: the stride is in samples, and a coarse axis limits it on its own.
    isotropic = decode_payload(
        generate_mesh(ball, (0, 0, 0), (0.02, 0.02, 0.02), step_size=1))[0]
    anisotropic = decode_payload(generate_mesh(ball, (0, 0, 0), VOXEL, step_size=1))[0]

    # The ball spans 5 times as much in z at VOXEL, and its z extent has to follow the
    # sampling rather than the blur.
    span = lambda verts, axis: float(verts[:, axis].max() - verts[:, axis].min())
    z_ratio = span(anisotropic, 2) / span(isotropic, 2)
    assert 4.5 < z_ratio < 5.5, z_ratio
    # And x is unaffected: same sampling, same extent.
    assert span(anisotropic, 0) == pytest.approx(span(isotropic, 0), rel=0.05)


def test_skeletons_ride_with_the_instances_they_belong_to():
    volumes, kinds = _object_volumes()
    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a",
                              options=MeshOptions(contact_max_um=None))

    instances = [r for r in rows if r["row_type"] == "instance"]
    assert instances and all(r["skeleton"] for r in instances)
    for row in instances:
        mesh_verts, _ = decode_payload(row["mesh"])
        skel_verts, edges = decode_payload(row["skeleton"], per_index=2)
        assert edges.max() < len(skel_verts)
        # Same coordinate frame: centroids agree to well under a voxel, even though the
        # smoothed surface pulls in slightly from the skeleton's extreme voxel centres.
        for axis in range(3):
            assert abs(skel_verts[:, axis].mean() - mesh_verts[:, axis].mean()) < 0.05


def test_skeletons_can_be_left_out():
    volumes, kinds = _object_volumes()
    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a",
                              options=MeshOptions(with_skeletons=False, contact_max_um=None))

    assert not any(r["skeleton"] for r in rows)


def test_contact_rows_ride_in_the_same_file():
    volumes, kinds = _object_volumes()
    mito = volumes["mito"]
    mito[_ball(3, (6, 12, 12))] = 3            # between the other two, touching neither
    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a",
                              options=MeshOptions(contact_max_um=0.5))

    contacts = [r for r in rows if r["row_type"] == "contact"]
    # The 3D viewer colours instances by contact group, which needs these rows present.
    assert contacts
    assert all(r["entity_a"] and r["entity_b"] and r["gap_um"] >= 0 for r in contacts)
    assert all(r["mesh"] == b"" for r in contacts)


# ── the file the widgets query ────────────────────────────────────────────────

def test_geometry_parquet_carries_the_payloads_as_blobs(tmp_path):
    volumes, kinds = _object_volumes()
    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a")
    path = write_geometry(tmp_path / "object_a", rows)

    table = pl.read_parquet(path)
    assert path.name == "geometry.parquet"
    assert table.columns == GEOMETRY_COLUMNS
    assert table.height == len(rows)
    assert table["mesh"].dtype == pl.Binary
    # Round-trips: a blob read back out of the file decodes to the same mesh.
    meshed = table.filter(pl.col("mesh").is_not_null())
    verts, faces = decode_payload(meshed["mesh"][0])
    assert len(verts) and faces.max() < len(verts)
    # Nothing to draw is NULL rather than an empty blob, so "has geometry" is one predicate.
    assert table.filter(pl.col("row_type") == "contact")["mesh"].null_count() > 0


def test_geometry_parquet_counts_the_payloads_it_stores(tmp_path):
    volumes, kinds = _object_volumes()
    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a",
                              options=MeshOptions(contact_max_um=None))
    table = pl.read_parquet(write_geometry(tmp_path / "object_a", rows))

    # A widget budgets its draw calls from these without reading any geometry.
    for row in table.filter(pl.col("mesh").is_not_null()).iter_rows(named=True):
        verts, faces = decode_payload(row["mesh"])
        assert (row["mesh_vertices"], row["mesh_faces"]) == (len(verts), len(faces))


def test_a_mask_has_no_label_id_to_join_on(tmp_path):
    volumes, kinds = _object_volumes()
    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a",
                              options=MeshOptions(contact_max_um=None))
    table = pl.read_parquet(write_geometry(tmp_path / "object_a", rows))

    # An instance is identified by object + entity + label; a whole mask has no label, and
    # writing 0 or "" there would make it collide with a real instance in a join.
    assert table.filter(pl.col("row_type") == "file")["label_id"].null_count() == 1
    assert table.filter(pl.col("row_type") == "instance")["label_id"].to_list() == [1, 2]


# ── the two ways of producing it ───────────────────────────────────────────────

def _record(volumes, kinds):
    names = list(volumes)
    stack = np.stack([volumes[n] for n in names], axis=0)
    meta = {
        "dim_order": "CZYX", "dim_names": ["C", "Z", "Y", "X"], "shape": list(stack.shape),
        "ndim": 4, "channel_names": names, "entity_kinds": [kinds[n] for n in names],
        "object_id": "object_a", "object_mask_name": "pm", "object_shape": list(SHAPE),
        "pixel_size_Z": VOXEL[0], "pixel_size_Y": VOXEL[1], "pixel_size_X": VOXEL[2],
    }
    return record_from(stack, meta, kind=OBJECT_KIND)


def test_the_processor_does_nothing_until_a_destination_is_configured(monkeypatch):
    monkeypatch.delenv("PP_ANATOMY_MESH_DIR", raising=False)
    volumes, kinds = _object_volumes()

    # The column is still declared, so a report written with the processor enabled but no
    # destination has it as null rather than missing.
    assert MeshProcessor().run_chunk(_record(volumes, kinds)) == {"mesh_geometry_file": None}


def test_the_processor_writes_one_file_and_says_where(tmp_path, monkeypatch):
    monkeypatch.setenv("PP_ANATOMY_MESH_DIR", str(tmp_path / "meshes"))
    volumes, kinds = _object_volumes()

    row = MeshProcessor().run_chunk(_record(volumes, kinds))

    # Geometry belongs beside the report; the one column is the path the widgets follow.
    written = tmp_path / "meshes" / "object_a" / "geometry.parquet"
    assert written.is_file()
    assert row == {"mesh_geometry_file": str(written.resolve())}


def test_process_with_mesh_writes_geometry_beside_a_clean_report(tmp_path):
    root = make_dataset(tmp_path / "experiment")
    out = tmp_path / "report.parquet"
    result = CliRunner().invoke(cli, ["process", str(root), "-o", str(out), "--object-mask", "pm", "--with-mesh"])

    assert result.exit_code == 0, result.output
    report = pl.read_parquet(out)
    # No payload columns: the report carries the path to the geometry, never the geometry.
    assert [c for c in report.columns if "mesh" in c or "skel" in c] == ["mesh_geometry_file"]
    written = sorted(p.parent.name for p in (tmp_path / "report_meshes").rglob("*.parquet"))
    assert written == ["object_a", "object_b", "object_c", "object_d"]
    assert all(Path(p).is_file() for p in report["mesh_geometry_file"].drop_nulls())


def test_the_mesh_command_produces_the_same_files_after_the_fact(tmp_path):
    make_object(tmp_path / "object_a", prefix="sample_a")
    result = CliRunner().invoke(
        cli, ["mesh", str(tmp_path / "object_a"), "-o", str(tmp_path / "geometry"),
              "--object-mask", "pm"]
    )

    assert result.exit_code == 0, result.output
    path = tmp_path / "geometry" / "object_a" / "geometry.parquet"
    assert path.is_file()
    assert set(pl.read_parquet(path)["row_type"]) >= {"instance", "file"}


def test_a_small_instance_is_not_smoothed_away():
    """Smoothing is capped by the instance's own thickness.

    Roundness now reaches 1 for a smooth blob where the old sphericity read 0.67, so the
    shape-driven sigma is larger. Without the cap a blob a few samples across disappears from
    the 3D view: marching cubes finds no surface in a field the kernel has flattened.
    """
    blob = np.zeros((9, 9, 9), dtype=bool)
    blob[3:6, 3:6, 3:6] = True                # 3 samples per side

    payload = generate_mesh(blob, (0, 0, 0), VOXEL, smooth_sigma=2.0)

    assert payload, "a 3-sample blob still has a surface"
    n_verts, n_faces = payload_counts(payload)
    assert n_verts > 3 and n_faces > 3


# ── large instances are meshed in the parent, not farmed out ─────────────────
#
# An instance's meshing cost follows its padded bounding box, and one organelle threading
# through an object has a box spanning most of it. Eight of those, one per pool worker, is
# how a 22-process pool nearly took a machine down. They are meshed inline instead, which
# means the results come back from two places and have to be put back in order.

def test_oversized_and_pooled_instances_come_back_in_order(monkeypatch):
    import numpy as np
    from pixel_patrol_anatomy import mesh as mesh_mod

    # Low enough that the sprawling instance is routed inline and the cubes are not.
    monkeypatch.setattr(mesh_mod, "_INLINE_INSTANCE_SAMPLES", 50_000)

    labels = np.zeros((40, 120, 120), dtype=np.uint16)
    for i in range(12):
        z, y, x = 2 + (i % 4) * 9, 2 + (i // 4) * 38, 4
        labels[z:z + 7, y:y + 30, x:x + 30] = i + 1
    labels[1:39, 1:119, 60:64] = 99          # bbox spans the volume -> inline

    args = ({"m": labels}, {"m": "label"}, (0.024, 0.016, 0.016))
    opts = dict(with_skeletons=False, contact_max_um=None)
    serial = mesh_mod.mesh_rows_for_object(*args, "o",
                                           options=mesh_mod.MeshOptions(mesh_workers=1, **opts))
    split = mesh_mod.mesh_rows_for_object(*args, "o",
                                          options=mesh_mod.MeshOptions(mesh_workers=4, **opts))

    assert [r["label_id"] for r in serial] == [r["label_id"] for r in split]
    assert all(a["mesh"] == b["mesh"] for a, b in zip(serial, split))
    sprawling = [r for r in split if r["label_id"] == 99][0]
    assert sprawling["mesh"], "the inline instance lost its geometry"


def test_the_mesh_pool_is_bounded_by_memory_not_just_cores():
    from pixel_patrol_anatomy.parallel import mesh_worker_budget, _WORKER_CAP
    assert mesh_worker_budget(1000) <= _WORKER_CAP
    assert mesh_worker_budget(1) == 1
