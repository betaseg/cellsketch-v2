"""The viewer extension: is it wired in, and does the SQL its widgets build still work?

The widgets are JavaScript, so what is testable from here is the contract between them
and the table: the entry point PixelPatrol discovers them through, and the queries they
issue. Those queries are the fragile part - the data they plot lives in list columns,
so every widget builds its own unnesting subquery.
"""

import json
from importlib.metadata import entry_points

import duckdb
import pytest

from pixel_patrol_cellsketch.viewer_extensions import get_viewer_extension_dir

# Mirrors unnestedSource() in plugin_cellsketch.js.
GROUP_ALIAS = '"__cs_group__"'


def _unnested_source(group_col: str, columns: list[str]) -> str:
    selects = ", ".join(f'unnest("{c}") AS "{c}"' for c in columns)
    return f'(SELECT "{group_col}" AS {GROUP_ALIAS}, {selects} FROM pp_data WHERE "obs_level" = 0)'


@pytest.fixture(scope="module")
def con(report_path):
    connection = duckdb.connect()
    connection.execute(f"CREATE VIEW pp_data AS SELECT * FROM read_parquet('{report_path}')")
    return connection


# ── wiring ────────────────────────────────────────────────────────────────────

def test_pixel_patrol_discovers_the_extension_through_its_entry_point():
    eps = [ep for ep in entry_points(group="pixel_patrol.viewer_extensions")
           if ep.name == "cellsketch_viewer"]

    assert len(eps) == 1
    assert eps[0].load()() == get_viewer_extension_dir()


def test_every_plugin_the_manifest_lists_exists():
    ext_dir = get_viewer_extension_dir()
    manifest = json.loads((ext_dir / "extension.json").read_text())

    assert manifest["name"] == "CellSketch"
    assert manifest["plugins"]
    for rel in manifest["plugins"]:
        assert (ext_dir / rel).is_file(), rel


def test_the_declared_widget_ids_are_unique_and_namespaced():
    source = (get_viewer_extension_dir() / "plugin_cellsketch.js").read_text()
    ids = [line.split("'")[1] for line in source.splitlines() if line.strip().startswith("id: '")]

    assert ids == ["cellsketch-instance-morphology", "cellsketch-distances"]


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

    # Treated cells were built with larger mitochondria, and control ranks below them.
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


def test_the_widgets_find_the_columns_they_require(con):
    """requires(schema) gates each widget on one column; both must be in the table."""
    columns = {d[0] for d in con.execute("SELECT * FROM pp_data LIMIT 0").description}

    assert "instance_entity" in columns    # cellsketch-instance-morphology
    assert "distance_um" in columns        # cellsketch-distances
