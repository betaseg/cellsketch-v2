"""Mesh and skeleton geometry, and the two ways of producing it.

The payload formats are a contract with mesh_viewer.html and csv_to_blender.py, so the
tests decode them the way those do rather than trusting the encoder.
"""

import base64
import csv
import gzip
import struct

import numpy as np
import pytest
from click.testing import CliRunner
from pixel_patrol_base.core.record import record_from

from pixel_patrol_cellsketch.cli import cli
from pixel_patrol_cellsketch.mesh import (
    MESH_CSV_COLUMNS,
    MeshOptions,
    generate_mesh_b64,
    mesh_rows_for_cell,
    sigma_for_shape,
    write_mesh_csv,
)
from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND
from pixel_patrol_cellsketch.plugins.processors.mesh import MeshProcessor
from synthetic import make_cell, make_dataset

SHAPE = (12, 24, 24)
VOXEL = (0.1, 0.02, 0.02)


def decode_payload(payload_b64: str, per_index: int = 3):
    """Mirror of parseMeshB64 / parseSkeletonB64 in mesh_viewer.html."""
    raw = gzip.decompress(base64.b64decode(payload_b64))
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
    verts, faces = decode_payload(generate_mesh_b64(_ball(), (0, 0, 0), VOXEL))

    assert len(verts) > 0 and len(faces) > 0
    assert faces.max() < len(verts)              # every face indexes a real vertex
    # X/Y span at most 24 voxels × 0.02 µm, Z at most 12 × 0.1 µm.
    assert verts[:, 0].max() <= 24 * VOXEL[2]
    assert verts[:, 2].max() <= 12 * VOXEL[0]


def test_mesh_vertices_are_offset_by_the_bounding_box_origin():
    at_origin, _ = decode_payload(generate_mesh_b64(_ball(), (0, 0, 0), VOXEL))
    offset, _ = decode_payload(generate_mesh_b64(_ball(), (2, 3, 4), VOXEL))

    # The origin is in voxels, the vertices in µm: an instance meshed from its own bbox
    # still lands in whole-volume coordinates, which is what aligns it with its skeleton.
    assert offset[:, 0].min() - at_origin[:, 0].min() == pytest.approx(4 * VOXEL[2], abs=1e-4)
    assert offset[:, 2].min() - at_origin[:, 2].min() == pytest.approx(2 * VOXEL[0], abs=1e-4)


def test_a_structure_too_small_to_mesh_yields_no_geometry():
    tiny = np.zeros(SHAPE, dtype=bool)
    tiny[0, 0, 0] = True

    assert generate_mesh_b64(tiny, (0, 0, 0), VOXEL) == ""


def test_smoothing_sigma_follows_shape():
    # A blob gets more smoothing than a sparse, thin structure.
    blob = sigma_for_shape(0.95, 0.5, sigma_min=0.3, sigma_max=1.5)
    strand = sigma_for_shape(0.2, 0.05, sigma_min=0.3, sigma_max=1.5)

    assert 0.3 <= strand < blob <= 1.5
    # No metric to go on: the midpoint, not an extreme.
    assert sigma_for_shape(float("nan"), 0.5) == pytest.approx(0.9)


# ── rows ──────────────────────────────────────────────────────────────────────

def _cell_volumes():
    mito = np.zeros(SHAPE, dtype=np.int32)
    mito[_ball(3, (6, 6, 6))] = 1
    mito[_ball(3, (6, 18, 18))] = 2
    return {"pm": _ball(10).astype(np.int32), "mito": mito}, {"pm": "mask", "mito": "label"}


def test_rows_cover_label_instances_and_whole_masks():
    volumes, kinds = _cell_volumes()
    rows = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                              options=MeshOptions(contact_max_um=None))

    by_type = {(r["row_type"], r["entity_name"]) for r in rows}
    # mesh_viewer.html reads instance rows for labels and file rows for masks.
    assert ("instance", "mito") in by_type
    assert ("file", "pm") in by_type
    assert all(r["mesh_b64"] for r in rows)
    assert all(set(r) <= set(MESH_CSV_COLUMNS) for r in rows)


def test_skeletons_ride_with_the_instances_they_belong_to():
    volumes, kinds = _cell_volumes()
    rows = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                              options=MeshOptions(contact_max_um=None))

    instances = [r for r in rows if r["row_type"] == "instance"]
    assert instances and all(r["skeleton_b64"] for r in instances)
    for row in instances:
        mesh_verts, _ = decode_payload(row["mesh_b64"])
        skel_verts, edges = decode_payload(row["skeleton_b64"], per_index=2)
        assert edges.max() < len(skel_verts)
        # Same coordinate frame: centroids agree to well under a voxel, even though the
        # smoothed surface pulls in slightly from the skeleton's extreme voxel centres.
        for axis in range(3):
            assert abs(skel_verts[:, axis].mean() - mesh_verts[:, axis].mean()) < 0.05


def test_skeletons_can_be_left_out():
    volumes, kinds = _cell_volumes()
    rows = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                              options=MeshOptions(with_skeletons=False, contact_max_um=None))

    assert not any(r["skeleton_b64"] for r in rows)


def test_contact_rows_ride_in_the_same_file():
    volumes, kinds = _cell_volumes()
    mito = volumes["mito"]
    mito[_ball(3, (6, 12, 12))] = 3            # between the other two, touching neither
    rows = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a",
                              options=MeshOptions(contact_max_um=0.5))

    contacts = [r for r in rows if r["row_type"] == "contact"]
    # The 3D viewer colours instances by contact group, which needs these rows present.
    assert contacts
    assert all(r["entity_a"] and r["entity_b"] and r["gap_um"] >= 0 for r in contacts)
    assert all(r["mesh_b64"] == "" for r in contacts)


def test_written_csv_has_the_columns_the_viewers_read(tmp_path):
    volumes, kinds = _cell_volumes()
    rows = mesh_rows_for_cell(volumes, kinds, VOXEL, cell_id="cell_a")
    path = write_mesh_csv(tmp_path / "cell_a" / "report_meshes.csv", rows)

    with path.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == MESH_CSV_COLUMNS
        assert len(list(reader)) == len(rows)


# ── the two ways of producing it ───────────────────────────────────────────────

def _record(volumes, kinds):
    names = list(volumes)
    stack = np.stack([volumes[n] for n in names], axis=0)
    meta = {
        "dim_order": "CZYX", "dim_names": ["C", "Z", "Y", "X"], "shape": list(stack.shape),
        "ndim": 4, "channel_names": names, "entity_kinds": [kinds[n] for n in names],
        "cell_id": "cell_a", "membrane_name": "pm", "cell_shape_zyx": list(SHAPE),
        "pixel_size_Z": VOXEL[0], "pixel_size_Y": VOXEL[1], "pixel_size_X": VOXEL[2],
    }
    return record_from(stack, meta, kind=CELL_KIND)


def test_the_processor_does_nothing_until_a_destination_is_configured(monkeypatch):
    monkeypatch.delenv("CELLSKETCH_MESH_DIR", raising=False)
    volumes, kinds = _cell_volumes()

    assert MeshProcessor().run_chunk(_record(volumes, kinds)) == {}


def test_the_processor_writes_a_csv_and_no_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("CELLSKETCH_MESH_DIR", str(tmp_path / "meshes"))
    volumes, kinds = _cell_volumes()

    row = MeshProcessor().run_chunk(_record(volumes, kinds))

    # Geometry belongs beside the report, not inside it: no columns, one file.
    assert row == {}
    assert (tmp_path / "meshes" / "cell_a" / "report_meshes.csv").is_file()


def test_process_with_mesh_writes_geometry_beside_a_clean_report(tmp_path):
    import polars as pl

    root = make_dataset(tmp_path / "experiment")
    out = tmp_path / "report.parquet"
    result = CliRunner().invoke(cli, ["process", str(root), "-o", str(out), "--with-mesh"])

    assert result.exit_code == 0, result.output
    assert not [c for c in pl.read_parquet(out).columns if "mesh" in c or "skel" in c]
    written = sorted(p.parent.name for p in (tmp_path / "report_meshes").rglob("*.csv"))
    assert written == ["cell_a", "cell_b", "cell_c", "cell_d"]


def test_the_mesh_command_produces_the_same_files_after_the_fact(tmp_path):
    make_cell(tmp_path / "cell_a", prefix="sample_a")
    result = CliRunner().invoke(
        cli, ["mesh", str(tmp_path / "cell_a"), "-o", str(tmp_path / "geometry")]
    )

    assert result.exit_code == 0, result.output
    path = tmp_path / "geometry" / "cell_a" / "report_meshes.csv"
    assert path.is_file()
    rows = list(csv.DictReader(path.open()))
    assert {r["row_type"] for r in rows} >= {"instance", "file"}
