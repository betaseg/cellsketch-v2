"""One real pipeline run, shared by the tests that inspect its output."""

from pathlib import Path

import polars as pl
import pytest
from pixel_patrol_base import api

from synthetic import make_dataset

# One leaf block = one whole entity volume. Z must be pinned to full extent
# explicitly: PixelPatrol's default leaf shape steps every non-XY dim by 1.
SLICE_SIZE = {"C": 1, "Z": -1}


@pytest.fixture(scope="session")
def report_path(tmp_path_factory) -> Path:
    root = make_dataset(tmp_path_factory.mktemp("cells"))
    out = root.parent / "report.parquet"
    project = api.create_project("cellsketch-test", root, loader="cellsketch", output_path=out)
    api.add_paths(project, ["control", "treated"])
    api.process_files(project, slice_size=SLICE_SIZE, max_workers=1, mb_per_task=4096)
    return out


@pytest.fixture(scope="session")
def table(report_path) -> pl.DataFrame:
    df, _ = api.load(report_path)
    return df
