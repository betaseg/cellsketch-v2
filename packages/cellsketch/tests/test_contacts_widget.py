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

from pixel_patrol_cellsketch.viewer_extensions import get_viewer_extension_dir

PLUGIN = get_viewer_extension_dir() / "plugin_cellsketch.js"
CHECKER = Path(__file__).parent / "contact_groups_check.mjs"


def cluster(instances, edges, facets) -> dict:
    """Run the widget's contactGroups/summariseByFacet over a fixture."""
    node = shutil.which("node")
    assert node, "node is required: the viewer widgets are JavaScript"
    fixture = json.dumps({"instances": instances, "edges": edges, "facets": facets})
    out = subprocess.run(
        [node, str(CHECKER), str(PLUGIN), "/dev/stdin"],
        input=fixture, capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _instance(cell, entity, label):
    return {"cell_id": cell, "entity": entity, "label": label}


def _edge(cell, a, la, b, lb):
    return {"cell_id": cell, "entity_a": a, "label_a": la, "entity_b": b, "label_b": lb}


# ── clustering ────────────────────────────────────────────────────────────────

def test_instances_that_touch_nothing_are_still_counted():
    result = cluster(
        [_instance("cell_a", "mito", 1), _instance("cell_a", "mito", 2)], [],
        {"cell_a": "control"},
    )

    # The denominator is every instance, not only those with a contact - otherwise
    # "share of instances touching" would always be 100%.
    assert result["total"] == 2
    assert result["singletons"] == 2
    assert result["facets"]["control"] == {
        "instances": 2, "groups": 0, "largest": 0, "touching": 0, "sizes": [],
    }


def test_a_chain_of_contacts_becomes_one_group():
    instances = [_instance("cell_a", "mito", i) for i in (1, 2, 3, 4)]
    edges = [_edge("cell_a", "mito", 1, "mito", 2), _edge("cell_a", "mito", 2, "mito", 3)]

    result = cluster(instances, edges, {"cell_a": "control"})

    # 1-2-3 chain into one group of three; 4 stays alone.
    assert result["facets"]["control"]["sizes"] == [3]
    assert result["facets"]["control"]["touching"] == 3
    assert result["singletons"] == 1


def test_groups_span_entities():
    instances = [_instance("cell_a", "mito", 1), _instance("cell_a", "granules", 7)]
    edges = [_edge("cell_a", "mito", 1, "granules", 7)]

    result = cluster(instances, edges, {"cell_a": "control"})

    assert result["facets"]["control"]["sizes"] == [2]


def test_the_same_label_id_in_two_cells_is_two_instances():
    instances = [_instance("cell_a", "mito", 1), _instance("cell_b", "mito", 1)]
    edges = []

    result = cluster(instances, edges, {"cell_a": "control", "cell_b": "treated"})

    # Identity is cell + entity + label. Keyed on the label alone, these two would merge
    # and one facet would lose an instance.
    assert result["total"] == 2
    assert result["facets"]["control"]["instances"] == 1
    assert result["facets"]["treated"]["instances"] == 1


def test_cell_names_with_awkward_characters_survive():
    instances = [_instance('a "1" b', "mito", 1), _instance('a "1" b', "mito", 2)]
    edges = [_edge('a "1" b', "mito", 1, "mito", 2)]

    result = cluster(instances, edges, {'a "1" b': "control"})

    # Keys are built with JSON.stringify and the cell is tracked separately, so quotes,
    # spaces and separators in a folder name cannot split or merge anything.
    assert result["facets"]["control"]["sizes"] == [2]


def test_each_facet_is_summarised_on_its_own():
    instances = [_instance("cell_a", "mito", 1), _instance("cell_a", "mito", 2),
                 _instance("cell_b", "mito", 1), _instance("cell_b", "mito", 2)]
    edges = [_edge("cell_a", "mito", 1, "mito", 2)]

    result = cluster(instances, edges, {"cell_a": "control", "cell_b": "treated"})

    assert result["facets"]["control"] == {
        "instances": 2, "groups": 1, "largest": 2, "touching": 2, "sizes": [2],
    }
    assert result["facets"]["treated"]["groups"] == 0


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
    return (f'(SELECT "cell_id", {selects} FROM pp_data WHERE "obs_level" = 0)')


def test_the_edge_query_returns_one_row_per_pair(con):
    total = con.execute(f'SELECT COUNT(*) FROM {_edge_source()}').fetchone()[0]
    on_cell_rows = con.execute(
        "SELECT SUM(contact_count) FROM pp_data WHERE obs_level = 0"
    ).fetchone()[0]

    assert total == on_cell_rows


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
