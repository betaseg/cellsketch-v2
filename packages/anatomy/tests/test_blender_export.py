"""The Blender export reads the same payload the widgets do.

`geometry_to_blender.py` is the geometry format's other consumer, and the only one that
cannot be run here — it imports `bpy`, which exists only inside Blender. What can be run
is the part that matters to the format: the decoder. It is loaded with a stub `bpy` in
place and checked against the encoder, and against the decode the viewer tests use, so a
change to the container cannot quietly break the Blender path while the widgets keep
working.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from pixel_patrol_anatomy.mesh import generate_mesh
from test_mesh import decode_payload

SCRIPT = Path(__file__).resolve().parents[3] / "geometry_to_blender.py"
VOXEL = (0.1, 0.02, 0.02)
SHAPE = (12, 24, 24)


def _ball(radius=4, centre=(6, 12, 12)) -> np.ndarray:
    zz, yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    return (
        ((zz - centre[0]) / radius) ** 2 + ((yy - centre[1]) / radius) ** 2
        + ((xx - centre[2]) / radius) ** 2
    ) <= 1.0


@pytest.fixture(scope="module")
def blender_script():
    """The script, importable outside Blender: bpy is stubbed, the decoder never uses it."""
    assert SCRIPT.is_file(), f"{SCRIPT} is the Blender export; it should sit beside the packages"
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))
    sys.modules.setdefault("mathutils", types.ModuleType("mathutils"))
    spec = importlib.util.spec_from_file_location("geometry_to_blender", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Importing a script that lives outside any package would otherwise drop a
    # __pycache__ beside it, in the repository root.
    written = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = written
    return module


def test_it_decodes_a_mesh_blob_to_the_vertices_that_were_written(blender_script):
    payload = generate_mesh(_ball(), (0, 0, 0), VOXEL)

    verts, faces = blender_script.decode_mesh(payload)
    expected_verts, expected_faces = decode_payload(payload)

    # Byte for byte the same reading as the viewer's, from the same blob: one container,
    # two consumers, no drift between them.
    assert np.allclose(verts, expected_verts, atol=1e-6)
    assert np.array_equal(faces, expected_faces)
    assert faces.max() < len(verts)


def test_it_reads_a_blob_however_parquet_hands_it_over(blender_script):
    payload = generate_mesh(_ball(), (0, 0, 0), VOXEL)

    # pandas gives a BLOB column as bytes; pyarrow can give a memoryview of the buffer.
    from_bytes, _ = blender_script.decode_mesh(payload)
    from_view, _ = blender_script.decode_mesh(memoryview(payload))
    from_array, _ = blender_script.decode_mesh(bytearray(payload))

    assert np.array_equal(from_bytes, from_view)
    assert np.array_equal(from_bytes, from_array)


def test_a_row_with_no_geometry_imports_nothing(blender_script):
    # Contact rows and unmeshed instances carry NULL, which pandas hands over as None.
    assert blender_script.decode_mesh(None) == (None, None)
    assert blender_script.decode_mesh(b"") == (None, None)
    assert blender_script.decode_mesh(b"\x00" * 8) == (None, None)


# ── --reuse-geometry ─────────────────────────────────────────────────────────
#
# Meshing dominates a run, so a batch that died partway is worth finishing in minutes. The
# hazard is reusing a file the killed run left half-written, which would lose an object's
# geometry with no error raised anywhere.

from pixel_patrol_anatomy.plugins.processors.mesh import _usable_geometry


def test_missing_geometry_is_written_again(tmp_path):
    assert _usable_geometry(tmp_path / "nothing-here.parquet") is None


def test_complete_geometry_is_reused(tmp_path):
    import polars as pl
    path = tmp_path / "geometry.parquet"
    pl.DataFrame({"object_id": ["a", "b"], "mesh": [b"x", b"y"]}).write_parquet(path)
    assert _usable_geometry(path) == 2


def test_geometry_with_no_rows_is_written_again(tmp_path):
    import polars as pl
    path = tmp_path / "geometry.parquet"
    pl.DataFrame({"object_id": [], "mesh": []}).write_parquet(path)
    assert _usable_geometry(path) is None


def test_a_truncated_file_is_written_again_rather_than_reused(tmp_path):
    # What a killed run actually leaves behind.
    import polars as pl
    path = tmp_path / "geometry.parquet"
    pl.DataFrame({"object_id": ["a"], "mesh": [b"x"]}).write_parquet(path)
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) // 2])
    assert _usable_geometry(path) is None
