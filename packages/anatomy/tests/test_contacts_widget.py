"""The Contacts & Groups widget: its clustering, and the SQL it runs.

The clustering is JavaScript, so it is exercised through node (a hard requirement here -
it is the runtime the whole viewer is written for). The SQL side is checked the way the
other widgets' is: by running what the widget builds against a real report.
"""

import json
import shutil
import subprocess
from pathlib import Path

import duckdb
import pytest

from pixel_patrol_anatomy.viewer_extensions import get_viewer_extension_dir

PLUGIN = get_viewer_extension_dir() / "plugin_anatomy.js"
CHECKER = Path(__file__).parent / "contact_groups_check.mjs"


def cluster(instances, edges, bigint: bool = False) -> dict:
    """Run the widget's contactGroups/clustersByObject over a fixture.

    bigint=True passes label ids as BigInt, which is what DuckDB actually hands the widget
    for an int64 column.
    """
    node = shutil.which("node")
    assert node, "node is required: the viewer widgets are JavaScript"
    fixture = json.dumps({"instances": instances, "edges": edges})
    out = subprocess.run(
        [node, str(CHECKER), str(PLUGIN), "/dev/stdin"] + (["--bigint"] if bigint else []),
        input=fixture, capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _instance(object, entity, label):
    return {"object_id": object, "entity": entity, "label": label}


def _edge(object, a, la, b, lb):
    return {"object_id": object, "entity_a": a, "label_a": la, "entity_b": b, "label_b": lb}


# ── clustering ────────────────────────────────────────────────────────────────

def test_instances_that_touch_nothing_are_still_counted():
    result = cluster(
        [_instance("object_a", "mito", 1), _instance("object_a", "mito", 2)], [])

    # Every instance is seeded, so an instance with no contact is a group of one rather than
    # missing. That is what lets "share of instances in contact" be a real fraction.
    assert result["total"] == 2
    assert result["singletons"] == 2
    # A lone instance is not a cluster, so the object has none.
    assert result["objects"] == {}


def test_a_chain_of_contacts_becomes_one_cluster():
    instances = [_instance("object_a", "mito", i) for i in (1, 2, 3, 4)]
    edges = [_edge("object_a", "mito", 1, "mito", 2), _edge("object_a", "mito", 2, "mito", 3)]

    result = cluster(instances, edges)

    # 1-2-3 chain into one cluster of three; 4 stays alone.
    assert result["objects"]["object_a"] == {
        "clusters": 1, "instances": 3, "largest": 3, "mixed": 0}
    assert result["singletons"] == 1


def test_clusters_span_entities():
    instances = [_instance("object_a", "mito", 1), _instance("object_a", "granules", 7)]
    edges = [_edge("object_a", "mito", 1, "granules", 7)]

    result = cluster(instances, edges)

    # Two structures in one cluster, which is what "mixed" counts: not a chain of one.
    assert result["objects"]["object_a"] == {
        "clusters": 1, "instances": 2, "largest": 2, "mixed": 1}


def test_the_same_label_id_in_two_objects_is_two_instances():
    instances = [_instance("object_a", "mito", 1), _instance("object_b", "mito", 1)]

    result = cluster(instances, [])

    # Identity is object + entity + label. Keyed on the label alone, these two would merge
    # into one instance.
    assert result["total"] == 2


def test_label_ids_arrive_as_bigints_from_duckdb():
    instances = [_instance("object_a", "mito", 1), _instance("object_a", "mito", 2),
                 _instance("object_a", "mito", 3)]
    edges = [_edge("object_a", "mito", 1, "mito", 2)]

    result = cluster(instances, edges, bigint=True)

    # int64 columns come back as BigInt, which JSON.stringify refuses outright - the
    # widget threw "Do not know how to serialize a BigInt" until the key was stringified.
    assert result["total"] == 3
    assert result["objects"]["object_a"]["largest"] == 2


def test_object_names_with_awkward_characters_survive():
    instances = [_instance('a "1" b', "mito", 1), _instance('a "1" b', "mito", 2)]
    edges = [_edge('a "1" b', "mito", 1, "mito", 2)]

    result = cluster(instances, edges)

    # Keys are built with JSON.stringify and the object is tracked separately, so quotes,
    # spaces and separators in a folder name cannot split or merge anything.
    assert result["objects"]['a "1" b']["clusters"] == 1


def test_each_object_is_summarised_on_its_own():
    instances = [_instance("object_a", "mito", 1), _instance("object_a", "mito", 2),
                 _instance("object_b", "mito", 1), _instance("object_b", "mito", 2)]
    edges = [_edge("object_a", "mito", 1, "mito", 2)]

    result = cluster(instances, edges)

    # One point per object is what the charts plot, so a cluster has to land on the object it
    # belongs to and nowhere else.
    assert result["objects"] == {
        "object_a": {"clusters": 1, "instances": 2, "largest": 2, "mixed": 0}}


# ── the SQL the widget issues ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def con(report_path):
    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW pp_data AS SELECT * FROM read_parquet('{report_path}')")
    return connection


def _edge_source() -> str:
    columns = ["contact_entity_a", "contact_label_a", "contact_entity_b", "contact_label_b",
               "contact_gap_um"]
    selects = ", ".join(f'unnest("{c}") AS "{c}"' for c in columns)
    return (f'(SELECT "object_id", {selects} FROM pp_data WHERE "obs_level" = 0)')


def test_the_edge_query_returns_one_row_per_pair(con):
    total = con.execute(f'SELECT COUNT(*) FROM {_edge_source()}').fetchone()[0]
    on_object_rows = con.execute(
        "SELECT SUM(contact_count) FROM pp_data WHERE obs_level = 0"
    ).fetchone()[0]

    assert total == on_object_rows


def test_the_threshold_filter_narrows_the_edges(con):
    source = _edge_source()
    widest = con.execute(f'SELECT MAX("contact_gap_um") FROM {source}').fetchone()[0]
    wide = con.execute(f'SELECT COUNT(*) FROM {source} WHERE "contact_gap_um" <= {widest}').fetchone()[0]
    tight = con.execute(f'SELECT COUNT(*) FROM {source} WHERE "contact_gap_um" <= {widest / 4}').fetchone()[0]

    assert tight < wide


def test_same_type_contacts_can_be_selected(con):
    source = _edge_source()
    same = con.execute(
        f'SELECT COUNT(*) FROM {source} WHERE "contact_entity_a" = "contact_entity_b"'
    ).fetchone()[0]
    total = con.execute(f'SELECT COUNT(*) FROM {source}').fetchone()[0]

    assert 0 < same <= total


def test_partner_counts_cover_both_sides_of_a_contact(con):
    """Mirrors partnerCountSql() in plugin_anatomy.js.

    An instance can be in either contact column, so counting one column would report half the
    instances as having no partner. Both sides together have to reach every instance the edge
    list mentions.
    """
    source = _edge_source()
    both = f"""(SELECT "object_id", "contact_entity_a" AS entity, "contact_label_a" AS label
                FROM {source}
                UNION
                SELECT "object_id", "contact_entity_b", "contact_label_b" FROM {source})"""
    one_side = f'(SELECT DISTINCT "object_id", "contact_entity_a", "contact_label_a" FROM {source})'

    with_partner = con.execute(f"SELECT COUNT(*) FROM {both}").fetchone()[0]
    from_one_side = con.execute(f"SELECT COUNT(*) FROM {one_side}").fetchone()[0]

    assert from_one_side < with_partner


def test_the_pair_matrix_counts_each_pair_once(con):
    """Mirrors pairCountsSql() in plugin_anatomy.js.

    LEAST/GREATEST fold mito-granule and granule-mito into one cell, so a pair is not counted
    twice depending on which instance the processor happened to see first.
    """
    source = ("(SELECT unnest(contact_entity_a) AS a, unnest(contact_entity_b) AS b, "
              'unnest(contact_gap_um) AS gap FROM pp_data WHERE "obs_level" = 0)')

    rows = con.execute(
        f"SELECT LEAST(a, b) s1, GREATEST(a, b) s2, COUNT(*) n FROM {source} "
        f"WHERE gap <= 0.5 GROUP BY 1, 2"
    ).fetchall()

    pairs = {(s1, s2) for s1, s2, _ in rows}
    assert pairs, "the synthetic objects have contacts"
    # Never both orders of the same pair.
    assert not {(a, b) for a, b in pairs if (b, a) in pairs and a != b}
    # And every pair is between structures the report knows.
    known = {name for (name,) in con.execute(
        'SELECT DISTINCT "entity_name" FROM pp_data WHERE "entity_name" IS NOT NULL').fetchall()}
    assert {s for pair in pairs for s in pair} <= known


def test_every_distance_finds_its_instance(con):
    """Mirrors the distance join in instanceProfileSql().

    Instances and distances live in different list columns on the same row and are joined on
    object + structure + label, so every distance has to find its instance. A missed join
    would silently drop instances from the distance panel.
    """
    distances = ("(SELECT \"object_id\", unnest(distance_entity) AS entity, "
                 "unnest(distance_label) AS label, unnest(distance_target) AS target, "
                 'unnest(distance_um) AS distance FROM pp_data WHERE "obs_level" = 0)')
    instances = ("(SELECT \"object_id\", unnest(instance_entity) AS entity, "
                 'unnest(instance_label) AS label FROM pp_data WHERE "obs_level" = 0)')

    unmatched = con.execute(
        f"""SELECT COUNT(*) FROM {distances} d
            WHERE NOT EXISTS (SELECT 1 FROM {instances} i
                              WHERE i."object_id" = d."object_id"
                                AND i.entity = d.entity AND i.label = d.label)"""
    ).fetchone()[0]

    assert unmatched == 0


def test_the_object_mask_is_a_distance_target(con):
    """The default target of that panel: "does this group sit against the boundary"."""
    targets = {t for (t,) in con.execute(
        'SELECT DISTINCT unnest(distance_target) FROM pp_data WHERE "obs_level" = 0').fetchall()}
    mask = con.execute(
        'SELECT ANY_VALUE("object_mask_name") FROM pp_data WHERE "obs_level" = 0').fetchone()[0]

    assert mask in targets
