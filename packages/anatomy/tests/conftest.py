"""One real pipeline run, shared by the tests that inspect its output."""

import os
from pathlib import Path

import polars as pl
import pytest
from pixel_patrol_base import api

from pixel_patrol_anatomy import pipeline
from pixel_patrol_anatomy.cli import FLAVOR, find_object_dirs
from pixel_patrol_anatomy.skeletons import CACHE
from synthetic import make_dataset, make_dataset_2d


# The synthetic objects are bounded by a mask called "pm". Nothing is guessed, so every
# test that loads one names it, the same way a run does.
OBJECT_MASK = "pm"


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    """No PP_ANATOMY_* setting and no cached per-object work crosses a test boundary.

    Plugin options travel through the environment and the per-object cache is module-level,
    so without this the suite's result depends on the order it happens to run in. The object
    mask is then set back, because it is not a tuning knob: without it nothing loads at all.
    """
    for key in [k for k in os.environ if k.startswith("PP_ANATOMY_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PP_ANATOMY_OBJECT_MASK", OBJECT_MASK)
    CACHE.clear()
    yield
    CACHE.clear()


# One structure gets a colour from a settings file and the others do not, so both halves of
# every widget's colour lookup are exercised by the shared report.
MITO_COLOUR = "#d62728"


def _run(root: Path, out: Path) -> Path:
    """One batch through the real pipeline, exactly as `process` runs it."""
    os.environ["PP_ANATOMY_OBJECT_MASK"] = OBJECT_MASK
    paths = ["control", "treated"]
    report = pipeline.analyse(find_object_dirs(root), root, paths, workers=1)
    assert not report.failures, report.failures
    written = pipeline.write(report, out, root=root, paths=paths, flavor=FLAVOR)
    # `process --colours` does exactly this, and so does the `colours` command afterwards.
    pipeline.recolour(written, {"mito": MITO_COLOUR})
    return written


@pytest.fixture(scope="session")
def report_path(tmp_path_factory) -> Path:
    root = make_dataset(tmp_path_factory.mktemp("objects"))
    return _run(root, root.parent / "report.parquet")


@pytest.fixture(scope="session")
def table(report_path) -> pl.DataFrame:
    df, _ = api.load(report_path)
    return df


@pytest.fixture(scope="session")
def report_path_2d(tmp_path_factory) -> Path:
    """The same batch shape as report_path, in a plane rather than a volume."""
    root = make_dataset_2d(tmp_path_factory.mktemp("flat_objects"))
    return _run(root, root.parent / "report_2d.parquet")


@pytest.fixture(scope="session")
def table_2d(report_path_2d) -> pl.DataFrame:
    df, _ = api.load(report_path_2d)
    return df
