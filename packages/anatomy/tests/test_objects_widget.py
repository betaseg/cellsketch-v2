"""Objects & Structures: what the overview says was segmented, and where it reads it from.

The point of the widget is to make an uneven batch obvious before anything is pooled
across it, so the cases worth pinning are the uneven ones: an object missing a structure, a
mask that has no instances to count, and a structure that only one object has at all.
"""

import json
import shutil
import subprocess
from pathlib import Path

import duckdb
import pytest

from pixel_patrol_anatomy.viewer_extensions import get_viewer_extension_dir

PLUGIN = get_viewer_extension_dir() / "plugin_anatomy_objects.js"
CHECKER = Path(__file__).parent / "object_overview_check.mjs"

# Mirrors ENTITY_ROW in plugin_anatomy_objects.js.
ENTITY_ROW = '"obs_level" = 1 AND "entity_name" IS NOT NULL'


def overview(objects, entity_rows, bigint: bool = False) -> dict:
    node = shutil.which("node")
    assert node, "node is required: the viewer widgets are JavaScript"
    fixture = json.dumps({"objects": objects, "entityRows": entity_rows})
    out = subprocess.run(
        [node, str(CHECKER), str(PLUGIN), "/dev/stdin"] + (["--bigint"] if bigint else []),
        input=fixture, capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _object(object_id, object_mask="pm", dims=3):
    return {"id": object_id, "objectMask": object_mask, "group": "control", "dims": dims,
            "hasInstances": True, "hasContacts": True}


def _entity(object_id, name, kind="label", instances=3, volume=1.5, colour=None):
    return {"object_id": object_id, "name": name, "kind": kind,
            "instances": instances, "volume": volume, "colour": colour}


# ── colour ────────────────────────────────────────────────────────────────────

def test_a_structure_is_drawn_in_the_colour_the_report_gives_it():
    """A study says what its structures look like, and the report carries the answer.

    The swatch here, the composition bar next to it and the meshes in the 3D widgets all read
    the same column, so a structure is one colour everywhere.
    """
    objects = [_object("object_a")]
    rows = [_entity("object_a", "mito", colour="#d62728"),
            _entity("object_a", "nucleus")]

    colours = overview(objects, rows)["colours"]

    assert colours["mito"] == "#d62728"
    # Not named in the settings file: its place in the built-in palette, so one file can name
    # the structures a study cares about and leave the rest alone.
    assert colours["nucleus"].startswith("#") and colours["nucleus"] != "#d62728"


# ── what the matrix says ──────────────────────────────────────────────────────

def test_a_structure_one_object_lacks_is_called_missing():
    objects = [_object("object_a"), _object("object_b")]
    rows = [
        _entity("object_a", "mito"), _entity("object_a", "nucleus"),
        _entity("object_b", "mito"),
    ]

    result = overview(objects, rows)

    assert result["matrix"]["object_b"]["nucleus"] == "missing"
    assert result["matrix"]["object_a"]["nucleus"] == 3
    assert "nucleus" in result["gaps"] and "1 of 2" in result["gaps"]


def test_the_preview_counts_what_the_batch_holds():
    objects = [_object("object_a"), _object("object_b")]
    rows = [
        _entity("object_a", "pm", kind="mask", instances=None),
        _entity("object_a", "mito", instances=40), _entity("object_a", "granules", instances=1200),
        _entity("object_b", "pm", kind="mask", instances=None),
        _entity("object_b", "mito", instances=14), _entity("object_b", "granules", instances=8000),
    ]

    counts = overview(objects, rows)["counts"]

    # "with instances" counts label entities only, and the instance total sums over those,
    # so the two always refer to the same rows.
    assert counts == {"objects": 2, "structures": 3, "labelled": 2,
                      "instances": 9254, "uneven": 0}


def test_the_preview_flags_how_many_structures_are_uneven():
    objects = [_object("object_a"), _object("object_b"), _object("object_c")]
    rows = [_entity(c, "mito") for c in ("object_a", "object_b", "object_c")]
    rows += [_entity("object_a", "golgi"), _entity("object_a", "nucleus", kind="mask",
                                                 instances=None)]

    counts = overview(objects, rows)["counts"]

    # Both golgi and nucleus are missing from two of the three objects, and the tile has to
    # say so: the counts above are pooled over objects that were not segmented alike.
    assert counts["uneven"] == 2
    assert counts["structures"] == 3


def test_an_even_batch_reports_no_gaps():
    objects = [_object("object_a"), _object("object_b")]
    rows = [_entity(c, "mito") for c in ("object_a", "object_b")]

    assert overview(objects, rows)["gaps"] is None


def test_a_mask_has_no_instances_to_count():
    objects = [_object("object_a")]
    rows = [_entity("object_a", "pm", kind="mask", instances=None), _entity("object_a", "mito")]

    result = overview(objects, rows)

    # null, not 0: a whole-structure mask is one thing, present or absent, and a 0 would
    # read as "segmented, and empty".
    assert result["matrix"]["object_a"]["pm"] is None
    assert result["matrix"]["object_a"]["mito"] == 3


def test_structures_are_ordered_masks_first_and_the_object_mask_before_them():
    objects = [_object("object_a", object_mask="pm")]
    rows = [
        _entity("object_a", "mito"), _entity("object_a", "granules"),
        _entity("object_a", "nucleus", kind="mask", instances=None),
        _entity("object_a", "pm", kind="mask", instances=None),
    ]

    result = overview(objects, rows)

    # The object's own boundary first, then the other masks, then the labelled structures -
    # the order the old report's overview used, and the order they nest in.
    assert result["entities"] == ["pm:mask", "nucleus:mask", "granules:label", "mito:label"]


def test_a_structure_only_one_object_has_still_gets_a_column():
    objects = [_object("object_a"), _object("object_b")]
    rows = [_entity("object_a", "mito"), _entity("object_b", "mito"), _entity("object_a", "golgi")]

    result = overview(objects, rows)

    assert "golgi:label" in result["entities"]
    assert result["matrix"]["object_b"]["golgi"] == "missing"


def test_counts_arrive_as_bigints_from_duckdb():
    objects = [_object("object_a")]
    rows = [_entity("object_a", "mito", instances=8263)]

    result = overview(objects, rows, bigint=True)

    # instance_count is an integer column, so DuckDB hands it over as BigInt - which
    # neither renders nor sums without being made a Number first.
    assert result["matrix"]["object_a"]["mito"] == 8263


def test_long_object_ids_are_shortened_to_what_differs():
    objects = [
        _object("a1_2026-03-16_alpha_object_1_segmentations"),
        _object("a1_2026-03-17_alpha_object_1_segmentations"),
    ]

    short = overview(objects, [_entity(c["id"], "mito") for c in objects])["short"]

    # Every id shares the prefix and the suffix; what is left is the part that identifies
    # the object, and the full id stays available as the object's tooltip.
    assert set(short.values()) == {"2026-03-16", "2026-03-17"}


# ── where the numbers come from ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def con(report_path):
    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW pp_all AS SELECT * FROM read_parquet('{report_path}')")
    connection.execute("CREATE VIEW pp_data AS SELECT * FROM pp_all WHERE obs_level = 0")
    return connection


def test_the_overview_reads_one_row_per_object_and_structure(con):
    rows = con.execute(
        f"""SELECT "object_id", "entity_name", "entity_kind", "instance_count",
                   "total_volume_um3"
            FROM pp_all WHERE {ENTITY_ROW} ORDER BY 1, 2"""
    ).fetchall()

    objects = {r[0] for r in rows}
    per_object = {c: sorted(r[1] for r in rows if r[0] == c) for c in objects}
    assert len(objects) == 4
    assert all(names == ["mito", "nucleus", "pm"] for names in per_object.values())
    # Labels carry a count, masks do not - the two kinds of table object.
    assert all(count is not None for *_, kind, count, _v in rows if kind == "label")


def test_those_rows_sit_a_level_below_the_object_rows_the_other_widgets_use(con):
    cell_rows = con.execute("SELECT COUNT(*) FROM pp_data").fetchone()[0]
    entity_rows = con.execute(f"SELECT COUNT(*) FROM pp_all WHERE {ENTITY_ROW}").fetchone()[0]

    # pp_data is the object rows alone, which is why the overview reads pp_all: the
    # per-structure rows it needs are not in the view every other widget queries.
    assert cell_rows == 4
    assert entity_rows == 12


def test_the_widget_finds_the_columns_it_requires(con):
    columns = {d[0] for d in con.execute("SELECT * FROM pp_data LIMIT 0").description}

    assert {"object_id", "entity_name"} <= columns


# ── a batch that mixes planes and volumes ─────────────────────────────────────

def test_one_dimensionality_is_not_a_warning():
    objects = [_object("object_a"), _object("object_b")]
    rows = [_entity(o["id"], "mito") for o in objects]

    assert overview(objects, rows)["mixedDims"] is None


def test_mixing_2d_and_3d_objects_is_called_out():
    objects = [_object("object_a"), _object("object_b"), _object("flat_c", dims=2)]
    rows = [_entity(o["id"], "mito") for o in objects]

    mixed = overview(objects, rows)["mixedDims"]

    # An area and a volume in one distribution is meaningless, and the widget that exists to
    # catch an unpoolable batch has to say so rather than charting one and dropping the other.
    assert mixed is not None
    assert "<b>2</b> in 3D" in mixed and "<b>1</b> in 2D" in mixed


# ── what the object-level processors did not measure ──────────────────────────

def _object_with(object_id, *, instances=True, contacts=True):
    return {**_object(object_id), "hasInstances": instances, "hasContacts": contacts}


def test_full_coverage_is_reported_as_full():
    objects = [_object_with("object_a"), _object_with("object_b")]

    coverage = overview(objects, [_entity(o["id"], "mito") for o in objects])["coverage"]

    assert {c["label"]: c["present"] for c in coverage} == {
        "per-instance measurements": 2, "contacts": 2}


def test_an_object_without_contacts_is_counted():
    objects = [_object_with("object_a"), _object_with("object_b", contacts=False)]

    coverage = overview(objects, [_entity(o["id"], "mito") for o in objects])["coverage"]

    # The viewer's own dataAvailabilityWarning turns this into "1 of 2 objects", the same
    # way the built-in widgets report a column only some rows have.
    assert {c["label"]: c["present"] for c in coverage} == {
        "per-instance measurements": 2, "contacts": 1}
