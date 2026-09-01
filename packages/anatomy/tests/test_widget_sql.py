"""Run the SQL the widgets build against a real report.

The widget tests elsewhere mirror these queries by hand, which is how a query that names a
column its own source does not expose still reached the viewer: the mirror was right and the
widget was wrong. Here the statements come out of the plugin itself and go through DuckDB, so
a binder error is a failing test rather than a message in the browser console.
"""

import json
import shutil
import subprocess
from pathlib import Path

import duckdb
import pytest

from pixel_patrol_anatomy.viewer_extensions import get_viewer_extension_dir

CHECKER = Path(__file__).parent / "widget_sql_check.mjs"
GROUP_COL = "imported_path_short"


def widget_output(plugin: str, structure: str, target: str) -> dict:
    node = shutil.which("node")
    assert node, "node is required: the viewer widgets are JavaScript"
    out = subprocess.run(
        [node, str(CHECKER), str(get_viewer_extension_dir() / plugin),
         GROUP_COL, structure, target],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def con(report_path):
    connection = duckdb.connect()
    connection.execute(
        f"CREATE VIEW pp_all AS SELECT * FROM read_parquet('{report_path}')")
    connection.execute("CREATE VIEW pp_data AS SELECT * FROM pp_all WHERE obs_level = 0")
    return connection


@pytest.mark.parametrize("plugin,expected", [("plugin_anatomy.js", 13),
                                             ("plugin_anatomy_objects.js", 2)])
def test_every_statement_the_widgets_build_runs(con, plugin, expected):
    statements = widget_output(plugin, structure="mito", target="pm")["statements"]

    assert len(statements) >= expected, "a builder stopped being exported"
    for sql in statements:
        try:
            con.execute(sql).fetchall()
        except duckdb.Error as error:
            pytest.fail(f"widget SQL does not run:\n{sql}\n\n{error}")


def test_every_chart_the_widget_exports_draws():
    """Runs the plotting code, which is where a missing helper shows up.

    `num is not defined` reached the viewer because nothing ever called the function that used
    it: the SQL check builds queries without drawing anything.
    """
    drawn = widget_output("plugin_anatomy.js", structure="mito", target="pm")["drawn"]

    assert drawn == ["partnerMix", "shares", "reach", "chance", "ownKindLeftOut",
                     "comparison", "clusters"]
