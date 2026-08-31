"""The 3D widgets: the geometry they read, and the payloads they decode.

Two contracts are worth pinning here. The first is binary — the widgets decode the blob
mesh.py writes, so the decoder is run through node against a payload this suite generated,
at an unaligned offset, the way it arrives from Arrow. The second is the SQL: geometry
lives in a file beside the report, and every query the widgets build has to work whether
it is aimed at one object or at the whole cohort at once.
"""

import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pytest

from pixel_patrol_anatomy.mesh import (
    GEOMETRY_FILENAME,
    MeshOptions,
    generate_mesh,
    mesh_rows_for_object,
    write_geometry,
)
from pixel_patrol_anatomy.viewer_extensions import get_viewer_extension_dir

PLUGIN = get_viewer_extension_dir() / "plugin_anatomy_3d.js"
CHECKER = Path(__file__).parent / "geometry_decode_check.mjs"

SHAPE = (12, 24, 24)
VOXEL = (0.1, 0.02, 0.02)


def _ball(radius=4, centre=(6, 12, 12)) -> np.ndarray:
    zz, yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    return (
        ((zz - centre[0]) / radius) ** 2 + ((yy - centre[1]) / radius) ** 2
        + ((xx - centre[2]) / radius) ** 2
    ) <= 1.0


def _volumes():
    mito = np.zeros(SHAPE, dtype=np.int32)
    mito[_ball(3, (6, 6, 6))] = 1
    mito[_ball(3, (6, 18, 18))] = 2
    mito[_ball(2, (6, 7, 7))] = 3            # overlapping 1, so there is a contact to find
    return {"pm": _ball(10).astype(np.int32), "mito": mito}, {"pm": "mask", "mito": "label"}


@pytest.fixture(scope="module")
def geometry_dir(tmp_path_factory) -> Path:
    """Two objects' geometry, written exactly as a run with --with-mesh would."""
    root = tmp_path_factory.mktemp("geometry")
    volumes, kinds = _volumes()
    for object_id in ("object_a", "object_b"):
        rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id=object_id,
                                  options=MeshOptions(contact_max_um=0.5))
        write_geometry(root / object_id, rows)
    return root


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def _source(geometry_dir: Path, *objects: str) -> str:
    """Mirrors sourceOf() in plugin_anatomy_3d.js: one read over the chosen objects."""
    paths = ", ".join(f"'{geometry_dir / object / GEOMETRY_FILENAME}'" for object in objects)
    return f"read_parquet([{paths}])"


# ── the decoder, in the runtime the widgets actually run in ───────────────────

def _decode_with_node(payload: bytes, tmp_path: Path, per_index: int = 3,
                      merge_with: bytes = None, offset=None, explode=None) -> dict:
    node = shutil.which("node")
    assert node, "node is required: the viewer widgets are JavaScript"
    blob = tmp_path / "payload.bin"
    blob.write_bytes(payload)
    extra = ["--per-index", str(per_index)]
    if merge_with is not None:
        second = tmp_path / "payload_b.bin"
        second.write_bytes(merge_with)
        extra += ["--merge", str(second)]
    if offset is not None:
        extra += ["--offset", ",".join(str(v) for v in offset)]
    if explode is not None:
        extra += ["--explode", explode]
    out = subprocess.run(
        [node, str(CHECKER), str(PLUGIN), str(blob)] + extra,
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_the_widget_decodes_a_mesh_to_the_vertices_that_were_written(tmp_path):
    payload = generate_mesh(_ball(), (0, 0, 0), VOXEL)
    n_verts, n_faces = struct.unpack_from("<II", payload, 0)

    decoded = _decode_with_node(payload, tmp_path)

    assert (decoded["vertices"], decoded["indices"]) == (n_verts, n_faces)
    assert decoded["maxIndex"] < n_verts
    # Vertices are µm in XYZ: the ball spans at most the volume it was meshed from,
    # 24 voxels of 0.02 µm across X and Y, 12 of 0.1 µm through Z.
    assert 0 <= decoded["bbox"]["x"]["min"] and decoded["bbox"]["x"]["max"] <= 24 * VOXEL[2]
    assert decoded["bbox"]["z"]["max"] <= 12 * VOXEL[0]


def test_a_skeleton_decodes_as_line_segments(tmp_path):
    volumes, kinds = _volumes()
    rows = mesh_rows_for_object(volumes, kinds, VOXEL, object_id="object_a",
                              options=MeshOptions(contact_max_um=None))
    skeleton = next(r["skeleton"] for r in rows if r.get("skeleton"))

    decoded = _decode_with_node(skeleton, tmp_path, per_index=2)

    # Two indices per segment rather than three per face, and every one addresses a vertex.
    assert decoded["indices"] > 0
    assert decoded["maxIndex"] < decoded["vertices"]


def test_merging_two_instances_keeps_every_face_on_its_own_vertices(tmp_path):
    first = generate_mesh(_ball(4, (6, 6, 6)), (0, 0, 0), VOXEL)
    second = generate_mesh(_ball(3, (6, 18, 18)), (0, 0, 0), VOXEL)
    apart = _decode_with_node(first, tmp_path), _decode_with_node(second, tmp_path)

    merged = _decode_with_node(first, tmp_path, merge_with=second)

    # One merged geometry per object, so the second instance's indices have to be shifted
    # past the first's vertices; unshifted, its faces point back into the first mesh.
    assert merged["vertices"] == apart[0]["vertices"] + apart[1]["vertices"]
    assert merged["indices"] == apart[0]["indices"] + apart[1]["indices"]
    assert merged["maxIndex"] == merged["vertices"] - 1
    # Colour rides on the vertices, which is what lets one geometry hold a whole palette.
    assert merged["colours"] == {"first": [1, 0, 0], "last": [0, 0, 1]}


def test_exploding_moves_an_instance_without_reshaping_it(tmp_path):
    payload = generate_mesh(_ball(4, (6, 6, 6)), (0, 0, 0), VOXEL)
    alone = _decode_with_node(payload, tmp_path)["bbox"]["z"]

    put = _decode_with_node(payload, tmp_path, merge_with=payload, offset=(0, 0, 5))["bbox"]

    # The explode slider offsets each instance along its own direction from the object
    # centre. Here the copy is pushed 5 µm in Z: the pair spans its own extent plus the
    # offset, and neither copy is stretched to get there.
    assert put["z"]["min"] == pytest.approx(alone["min"], abs=1e-5)
    assert put["z"]["max"] == pytest.approx(alone["max"] + 5, abs=1e-5)
    assert put["x"] == pytest.approx(_decode_with_node(payload, tmp_path)["bbox"]["x"])


def test_exploding_pushes_an_instance_out_the_way_it_actually_lies(tmp_path):
    """The offset is in mesh order (x, y, z), the same order polar_nx/ny/nz are read in.

    Reordering it to ZYX mirrored x against z, so instances flew off in directions they had
    never been in - and with the carried metrics missing from the geometry as well, the slider
    moved nothing at all.
    """
    payload = generate_mesh(_ball(4, (6, 6, 6)), (0, 0, 0), VOXEL)

    def offset_for(polarity, factor=2.0):
        return _decode_with_node(payload, tmp_path, explode=f"{polarity}:{factor}")["explode"]

    # Straight out along x, twice its own distance from the centre.
    assert offset_for("3:1,0,0") == pytest.approx([6, 0, 0])
    # And along z, which is the component the swap used to send along x.
    assert offset_for("3:0,0,1") == pytest.approx([0, 0, 6])
    # A slider at zero, or an instance sitting on the centre, moves nothing.
    assert offset_for("3:1,0,0", factor=0) == [0, 0, 0]
    assert offset_for("0:1,0,0") == [0, 0, 0]


# ── the SQL the widgets build ─────────────────────────────────────────────────

def test_a_run_writes_geometry_the_explode_slider_can_use(tmp_path):
    """The whole path: the processor writes the polarity, the widget reads it and moves.

    The unit test above passes the metrics in by hand. This one goes through the processor,
    which is where they were lost: masks are measured in the same loop as labels and the mask
    branch overwrote them, so the slider had nothing to work with and quietly did nothing.
    """
    from pixel_patrol_anatomy.plugins.loaders.object_loader import ObjectLoader
    from pixel_patrol_anatomy.plugins.processors.instances import InstanceProcessor
    from pixel_patrol_anatomy.plugins.processors.mesh import MeshProcessor
    from synthetic import make_object

    os.environ["PP_ANATOMY_MESH_DIR"] = str(tmp_path)
    folder = tmp_path / "src" / "object_a"
    make_object(folder, prefix="s", n_mito=3, mito_radii=(2.0, 3.0, 3.0))
    record = ObjectLoader().load(folder)
    InstanceProcessor().run_chunk(record)             # publishes the per-instance metrics
    written = MeshProcessor().run_chunk(record)["mesh_geometry_file"]

    rows = duckdb.connect().execute(
        f"""SELECT "polar_dist_um", "polar_nx", "polar_ny", "polar_nz"
            FROM read_parquet('{written}') WHERE "row_type" = 'instance'"""
    ).fetchall()

    assert rows, "the object has instances"
    # A distance to be pushed along, a direction to push in, and not the same one for all.
    assert all(distance is not None and distance > 0 for distance, *_ in rows)
    assert len({tuple(vector) for _, *vector in rows}) > 1


def test_the_palette_comes_from_the_report_not_the_geometry(report_path):
    """Mirrors entityPalette() in plugin_anatomy_3d.js.

    These widgets list structures from the geometry files, which carry no colour, and take the
    colour from the report's entity rows: one settings file, one colour per structure, the same
    in the table, the bars and the meshes.
    """
    from conftest import MITO_COLOUR

    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW pp_all AS SELECT * FROM read_parquet('{report_path}')")

    rows = connection.execute(
        'SELECT DISTINCT "entity_name", "entity_colour" FROM pp_all '
        'WHERE "entity_colour" IS NOT NULL'
    ).fetchall()

    assert rows == [("mito", MITO_COLOUR)]


def test_the_object_view_summarises_an_object_without_reading_geometry(con, geometry_dir):
    rows = con.execute(
        f"""SELECT "entity_name", "row_type", COUNT(*) AS n, SUM("mesh_faces") AS faces
            FROM {_source(geometry_dir, 'object_a')}
            WHERE "mesh" IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2"""
    ).fetchall()

    by_key = {(name, row_type): (n, faces) for name, row_type, n, faces in rows}
    # Three labelled instances and the membrane mask, with the face counts that let the
    # widget budget its draw calls before a byte of geometry is transferred.
    assert by_key[("mito", "instance")][0] == 3
    assert by_key[("pm", "file")][0] == 1
    assert all(faces > 0 for _n, faces in by_key.values())


def test_the_object_view_reads_only_the_structures_it_draws(con, geometry_dir):
    rows = con.execute(
        f"""SELECT "entity_name", "label_id", "mesh", "polar_dist_um"
            FROM {_source(geometry_dir, 'object_a')}
            WHERE "mesh" IS NOT NULL AND "entity_name" IN ('mito')
            ORDER BY "volume_um3" DESC NULLS LAST LIMIT 2"""
    ).fetchall()

    # The point of the sidecar: two meshes come back, not the object's worth.
    assert len(rows) == 2
    assert all(isinstance(mesh, (bytes, bytearray)) and len(mesh) > 32 for *_, mesh, _ in rows)
    volumes = con.execute(
        f"""SELECT "volume_um3" FROM {_source(geometry_dir, 'object_a')}
            WHERE "entity_name" = 'mito' AND "mesh" IS NOT NULL
            ORDER BY "volume_um3" DESC"""
    ).fetchall()
    assert [v[0] for v in volumes[:2]] == sorted([v[0] for v in volumes], reverse=True)[:2]


def test_the_gallery_ranks_instances_across_every_object_in_scope(con, geometry_dir):
    rows = con.execute(
        f"""SELECT "object_id", "label_id", "sphericity"
            FROM {_source(geometry_dir, 'object_a', 'object_b')}
            WHERE "row_type" = 'instance' AND "mesh" IS NOT NULL AND isfinite("sphericity")
            ORDER BY "sphericity" DESC LIMIT 4"""
    ).fetchall()

    # One query over both objects - a cohort's tail, not an object's, which is the whole
    # reason the gallery reads a list of files rather than one.
    assert len(rows) == 4
    assert {r[0] for r in rows} == {"object_a", "object_b"}
    assert [r[2] for r in rows] == sorted([r[2] for r in rows], reverse=True)


def test_the_gallery_can_ask_for_the_other_tail(con, geometry_dir):
    source = _source(geometry_dir, 'object_a', 'object_b')
    where = """WHERE "row_type" = 'instance' AND "mesh" IS NOT NULL
               AND "entity_name" = 'mito' AND isfinite("volume_um3")"""
    highest = con.execute(
        f'SELECT "volume_um3" FROM {source} {where} ORDER BY "volume_um3" DESC LIMIT 1'
    ).fetchone()[0]
    lowest = con.execute(
        f'SELECT "volume_um3" FROM {source} {where} ORDER BY "volume_um3" ASC LIMIT 1'
    ).fetchone()[0]

    assert lowest < highest


def test_the_contact_edges_come_from_the_same_file(con, geometry_dir):
    widest = con.execute(
        f"""SELECT MAX("gap_um") FROM {_source(geometry_dir, 'object_a')}
            WHERE "row_type" = 'contact'"""
    ).fetchone()[0]
    edges = con.execute(
        f"""SELECT "entity_a", "label_a", "entity_b", "label_b"
            FROM {_source(geometry_dir, 'object_a')}
            WHERE "row_type" = 'contact' AND "gap_um" <= {widest}"""
    ).fetchall()

    # The object view colours by contact group without going back to the report: the edge
    # list rides in the geometry file, keyed the way the widget keys an instance.
    assert widest is not None
    assert edges and all(a and b for a, _la, b, _lb in edges)


def test_a_mesh_is_never_an_empty_blob(con, geometry_dir):
    empty = con.execute(
        f"""SELECT COUNT(*) FROM {_source(geometry_dir, 'object_a')}
            WHERE "mesh" IS NOT NULL AND octet_length("mesh") = 0"""
    ).fetchone()[0]

    # "Has geometry" is one IS NOT NULL for the widgets; an empty blob would pass it and
    # then decode to nothing.
    assert empty == 0


# ── wiring ────────────────────────────────────────────────────────────────────

def test_the_3d_widgets_are_gated_on_the_column_that_locates_the_geometry():
    source = PLUGIN.read_text()

    # Both widgets hide themselves unless the report says where its geometry is.
    assert source.count("schema.allCols.includes('mesh_geometry_file')") == 2


# ── the gallery's sample ──────────────────────────────────────────────────────

def _gallery_rows(con, geometry_dir, *, pick: str, count: int = 4, entity: str = "mito"):
    """Mirrors the gallery's query: pick the sample, then sort it by the metric."""
    source = _source(geometry_dir, "object_a", "object_b")
    has_geometry = '("mesh" IS NOT NULL OR "outline" IS NOT NULL)'
    order = {"highest": 'ORDER BY "volume_um3" DESC',
             "lowest": 'ORDER BY "volume_um3" ASC',
             "random": "ORDER BY random()"}[pick]
    return con.execute(
        f'''SELECT * FROM (
              SELECT "object_id", "label_id", "volume_um3" AS value
              FROM {source}
              WHERE "row_type" = 'instance' AND {has_geometry}
                AND "entity_name" = '{entity}' AND isfinite("volume_um3")
              {order} LIMIT {count}
            ) ORDER BY value DESC'''
    ).fetchall()


def test_the_gallery_asks_for_whole_rows_at_any_width():
    """The grid is auto-fill, so its column count belongs to the card, not to the widget.

    A round number of thumbnails left the last row part empty at every width that did not happen
    to divide it, so the widget asks for rows and works out the count from the columns it got.
    """
    source = (get_viewer_extension_dir() / "plugin_anatomy_3d.js").read_text()
    cap = int(re.search(r"MAX_THUMBS = (\d+)", source).group(1))
    rows_offered = [int(n) for n in
                    re.search(r"GALLERY_ROWS = \[([^\]]+)\]", source).group(1).split(",")]

    def wanted(columns, rows):        # mirrors wanted() in the widget
        return min(cap - cap % columns, columns * rows)

    for columns in range(1, 20):
        for rows in rows_offered:
            asked = wanted(columns, rows)
            assert asked % columns == 0, f"{columns} columns x {rows} rows leaves a part row"
            assert 0 < asked <= cap


def test_the_gallery_shows_its_sample_in_metric_order(con, geometry_dir):
    for pick in ("highest", "lowest", "random"):
        values = [r[2] for r in _gallery_rows(con, geometry_dir, pick=pick)]

        assert values == sorted(values, reverse=True), pick


def test_a_random_sample_varies_between_draws(con, geometry_dir):
    draws = {tuple((r[0], r[1]) for r in _gallery_rows(con, geometry_dir, pick="random", count=2))
             for _ in range(40)}

    # Not the same two instances every time, which is the whole difference from the tails.
    assert len(draws) > 1


def test_the_gallery_can_be_restricted_to_one_object(con, geometry_dir):
    # The group filter narrows which geometry files are read at all, so the same query over
    # one object is what a single-group gallery runs.
    source = _source(geometry_dir, "object_a")
    rows = con.execute(
        f'''SELECT DISTINCT "object_id" FROM {source} WHERE "row_type" = 'instance' '''
    ).fetchall()

    assert rows == [("object_a",)]
