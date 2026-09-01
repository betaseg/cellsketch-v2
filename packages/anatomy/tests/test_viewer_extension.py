"""The viewer extension: is it wired in, and does the SQL its widgets build still work?

The widgets are JavaScript, so what is testable from here is the contract between them
and the table: the entry point PixelPatrol discovers them through, and the queries they
issue. Those queries are the fragile part - the data they plot lives in list columns,
so every widget builds its own unnesting subquery.
"""

import json
import re
from importlib.metadata import entry_points

import duckdb
import pytest

from pixel_patrol_anatomy.viewer_extensions import get_viewer_extension_dir

# Mirrors unnestedSource() in plugin_anatomy.js.
GROUP_ALIAS = '"__cs_group__"'


def _unnested_source(group_col: str, columns: list[str]) -> str:
    selects = ", ".join(f'unnest("{c}") AS "{c}"' for c in columns)
    return (
        f'(SELECT "object_id", "{group_col}" AS {GROUP_ALIAS}, {selects} '
        'FROM pp_data WHERE "obs_level" = 0)'
    )


def _reach_source(entity: str) -> str:
    """Mirrors reachSource() in plugin_anatomy.js."""
    long = _unnested_source(
        "imported_path_short",
        ["distance_entity", "distance_label", "distance_target", "distance_um"],
    )
    return f"""(
      SELECT a.{GROUP_ALIAS} AS grp, a."distance_target" AS target_a,
             b."distance_target" AS target_b,
             GREATEST(a."distance_um", b."distance_um") AS reach
      FROM {long} a JOIN {long} b
        ON a."object_id" = b."object_id" AND a."distance_entity" = b."distance_entity"
       AND a."distance_label" = b."distance_label"
       AND a."distance_target" < b."distance_target"
      WHERE a."distance_entity" = '{entity}')"""


@pytest.fixture(scope="module")
def con(report_path):
    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW pp_data AS SELECT * FROM read_parquet('{report_path}')")
    return connection


# ── wiring ────────────────────────────────────────────────────────────────────

def test_pixel_patrol_discovers_the_extension_through_its_entry_point():
    eps = [ep for ep in entry_points(group="pixel_patrol.viewer_extensions")
           if ep.name == "anatomy_viewer"]

    assert len(eps) == 1
    assert eps[0].load()() == get_viewer_extension_dir()


def test_every_plugin_the_manifest_lists_exists():
    ext_dir = get_viewer_extension_dir()
    manifest = json.loads((ext_dir / "extension.json").read_text())

    assert manifest["name"] == "Anatomy"
    assert manifest["plugins"]
    for rel in manifest["plugins"]:
        assert (ext_dir / rel).is_file(), rel


def test_the_declared_widget_ids_are_unique_and_namespaced():
    ext_dir = get_viewer_extension_dir()
    manifest = json.loads((ext_dir / "extension.json").read_text())
    ids = [
        line.split("'")[1]
        for rel in manifest["plugins"]
        for line in (ext_dir / rel).read_text().splitlines()
        if line.strip().startswith("id: '")
    ]

    # A duplicate id would not be a second widget: the registry replaces in place.
    assert sorted(ids) == sorted(set(ids))
    assert set(ids) == {
        "anatomy-objects",
        "anatomy-instance-morphology",
        "anatomy-distances",
        "anatomy-reach",
        "anatomy-contacts",
        "anatomy-object-3d",
        "anatomy-gallery",
    }


def test_every_widget_explains_itself_the_way_the_viewer_renders_it():
    """`info` is the viewer's own widget description: an ⓘ in the card header, markdown-lite.

    Rolling our own paragraph into the widget body meant it could not be collapsed, ignored the
    sidebar's info switch and looked nothing like the built-in widgets.
    """
    for source in sorted(get_viewer_extension_dir().glob("plugin_*.js")):
        text = source.read_text()
        widgets = text.count("\n  group: '")
        assert text.count("\n  info: [") == widgets, f"{source.name}: a widget with no info"
        # Markdown-lite, not HTML: renderInfoHtml escapes tags before it emphasises anything,
        # so a <b> in here would reach the reader as literal text.
        blocks = re.findall(r"\n  info: \[(.*?)\n  \]", text, re.S)
        assert len(blocks) == widgets
        for block in blocks:
            assert "<" not in block, f"{source.name}: HTML in an info panel"


def test_every_widget_declares_a_group_the_viewer_lays_out():
    ext_dir = get_viewer_extension_dir()
    manifest = json.loads((ext_dir / "extension.json").read_text())
    groups = [
        line.split("'")[1]
        for rel in manifest["plugins"]
        for line in (ext_dir / rel).read_text().splitlines()
        if line.strip().startswith("group: '")
    ]

    # plugin-groups.js orders these; anything else is appended under its own heading.
    assert set(groups) <= {"Summary", "File Stats", "Metadata", "Dataset Stats", "Visualization"}


# ── the SQL those widgets issue ───────────────────────────────────────────────

def test_the_instance_widget_source_yields_one_row_per_instance(con):
    source = _unnested_source("imported_path_short", ["instance_entity", "instance_volume_um3"])
    rows = con.execute(
        f"""SELECT {GROUP_ALIAS} AS grp, COUNT(*) AS n FROM {source}
            WHERE "instance_entity" = 'mito' GROUP BY 1 ORDER BY 1"""
    ).fetchall()

    assert rows == [("control", 8), ("treated", 6)]


def test_the_distribution_engines_stats_query_runs_on_that_source(con):
    """plot-engine.js STAT_SELECT: what the violin/box summary is built from."""
    source = _unnested_source("imported_path_short", ["instance_entity", "instance_volume_um3"])
    rows = con.execute(
        f"""SELECT {GROUP_ALIAS} AS __cat__, COUNT("instance_volume_um3") AS n,
                   MIN("instance_volume_um3") AS mn, MAX("instance_volume_um3") AS mx,
                   approx_quantile("instance_volume_um3", 0.5) AS med
            FROM {source} WHERE "instance_entity" = 'mito'
              AND "instance_volume_um3" IS NOT NULL
            GROUP BY 1 ORDER BY 1"""
    ).fetchall()

    groups = {r[0]: r[1] for r in rows}
    assert groups == {"control": 8, "treated": 6}
    assert all(r[2] > 0 and r[3] >= r[2] for r in rows)


def test_significance_brackets_can_be_computed_from_that_source(con):
    """fetchCategoryRankSums in plot-engine.js, verbatim - the query the brackets need.

    Reproducing its Mann-Whitney maths here shows the whole path works on unnested
    instance data: this is the comparison the report exists to make.
    """
    source = _unnested_source("imported_path_short", ["instance_entity", "instance_volume_um3"])
    rows = con.execute(
        f"""WITH base AS (
              SELECT CAST({GROUP_ALIAS} AS VARCHAR) AS cat, "instance_volume_um3"::DOUBLE AS v
              FROM {source} WHERE "instance_entity" = 'mito' AND "instance_volume_um3" IS NOT NULL
            ), ranked AS (SELECT cat, v, ROW_NUMBER() OVER (ORDER BY v) AS rn FROM base),
               avg_ranked AS (SELECT cat, AVG(rn) OVER (PARTITION BY v) AS rnk FROM ranked)
            SELECT cat, COUNT(*) AS n, SUM(rnk) AS rank_sum FROM avg_ranked GROUP BY cat"""
    ).fetchall()

    by_group = {cat: (n, rank_sum) for cat, n, rank_sum in rows}
    n1, rank_sum1 = by_group["control"]
    n2, _ = by_group["treated"]
    u1 = rank_sum1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5

    # Treated objects were built with larger mitochondria, and control ranks below them.
    assert u1 == 0
    assert abs(u1 - mu) / sigma > 3      # ≈ p < 0.01, drawn as "**"


def test_the_distance_widget_source_yields_one_row_per_instance_and_target(con):
    source = _unnested_source(
        "imported_path_short", ["distance_entity", "distance_target", "distance_um"]
    )
    rows = con.execute(
        f"""SELECT "distance_target", COUNT(*) AS n, MIN("distance_um") AS mn FROM {source}
            WHERE "distance_entity" = 'mito' GROUP BY 1 ORDER BY 1"""
    ).fetchall()

    assert [(r[0], r[1]) for r in rows] == [("nucleus", 14), ("pm", 14)]


def test_reach_pairs_stay_within_their_own_object(con):
    pairs = con.execute(
        f"""SELECT target_a, target_b, grp, COUNT(*) AS n FROM {_reach_source('mito')}
            GROUP BY 1, 2, 3 ORDER BY 3"""
    ).fetchall()

    # One row per instance per unordered pair of the other structures - 14 instances,
    # one pair (nucleus & pm). A join that matched across objects would multiply this.
    assert pairs == [("nucleus", "pm", "control", 8), ("nucleus", "pm", "treated", 6)]


def test_the_reach_curve_is_drawn_from_quantiles(con):
    probs = ", ".join(str(i / 50) for i in range(51))
    rows = con.execute(
        f"""SELECT grp, COUNT(*) AS n, quantile_cont(reach, [{probs}]) AS quantiles
            FROM {_reach_source('mito')} WHERE reach IS NOT NULL GROUP BY 1 ORDER BY 1"""
    ).fetchall()

    for _grp, _n, quantiles in rows:
        # Fixed vertex count whatever the instance count, and monotone by construction.
        assert len(quantiles) == 51
        assert quantiles == sorted(quantiles)

    # Treated mitochondria were built larger and closer in, so they reach both
    # structures at shorter distances than control - the separation the curve shows.
    by_group = {grp: quantiles for grp, _n, quantiles in rows}
    assert max(by_group["treated"]) < max(by_group["control"])


def test_the_widgets_find_the_columns_they_require(con):
    """requires(schema) gates each widget on one column; both must be in the table."""
    columns = {d[0] for d in con.execute("SELECT * FROM pp_data LIMIT 0").description}

    assert "instance_entity" in columns    # anatomy-instance-morphology
    assert "distance_um" in columns        # anatomy-distances


def test_a_metric_with_no_measurements_is_left_out_of_the_grid(con):
    """Mirrors measuredMetrics() in plugin_anatomy.js.

    A metric can be null for every instance of one structure and fine for the next — a
    skeleton is skipped above --max-skeleton-voxels, and the nearest same-structure instance
    needs a second one in the same object. Those get no panel, so the grid has no holes.
    """
    metrics = ["instance_volume_um3", "instance_branches",
               "instance_distance_to_closest_same_type_um"]
    unnested = ", ".join(f"unnest({m}) AS {m}" for m in metrics)
    source = (f"(SELECT unnest(instance_entity) AS instance_entity, {unnested} "
              f"FROM pp_data)")
    counts = ", ".join(f'COUNT("{m}") AS "{m}"' for m in metrics)

    row = con.execute(
        f"SELECT {counts} FROM {source} WHERE instance_entity = 'mito'"
    ).fetchone()

    # The synthetic mitochondria are small enough to skeletonise and there are several per
    # object, so every metric here is measured; the query is what decides.
    assert all(count > 0 for count in row)


def test_one_value_per_group_is_detected(con):
    """Mirrors isOnePerGroup() in plugin_anatomy.js.

    A structure with one instance per object gives every group a single point: a violin of
    one, and a ladder of brackets that can only read "ns". The widget draws bars instead and
    runs no test, which this query is what decides.
    """
    source = ("(SELECT object_id AS g, unnest(instance_entity) AS e, "
              "unnest(instance_volume_um3) AS v FROM pp_data)")

    def most_per_group(entity):
        return con.execute(
            f"SELECT MAX(n) FROM (SELECT g, COUNT(v) n FROM {source} "
            f"WHERE e = '{entity}' GROUP BY 1)"
        ).fetchone()[0]

    # The synthetic objects have several mitochondria each, so this one is a distribution.
    assert most_per_group("mito") > 1
