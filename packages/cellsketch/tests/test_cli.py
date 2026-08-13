import os

import polars as pl
import pytest
from click.testing import CliRunner

from pixel_patrol_cellsketch.cli import (
    SLICE_SIZE,
    _apply_analysis_env,
    auto_mb_per_task,
    cli,
    estimate_peak_gb,
    find_cell_dirs,
    suggest_max_workers,
)
from synthetic import make_cell, make_dataset


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / "experiment")


def test_finds_every_cell_folder_but_not_the_folders_holding_them(dataset):
    found = {p.name for p in find_cell_dirs(dataset)}

    assert found == {"cell_a", "cell_b", "cell_c", "cell_d"}


def test_a_single_cell_folder_is_its_own_dataset(tmp_path):
    make_cell(tmp_path / "cell_a", prefix="sample_a")

    assert find_cell_dirs(tmp_path / "cell_a") == [tmp_path / "cell_a"]


def test_leaf_blocks_are_configured_to_be_whole_entity_volumes():
    # The reason this CLI exists: these are requirements of the loader, not preferences.
    assert SLICE_SIZE["C"] == 1
    assert all(SLICE_SIZE[axis] == -1 for axis in "ZYX")


def test_task_budget_is_sized_so_no_cell_gets_split(dataset):
    # Synthetic cells are tiny, so the floor applies; the point is that it is a floor.
    assert auto_mb_per_task(find_cell_dirs(dataset)) >= 512


def test_peak_memory_estimate_scales_with_the_cell(tmp_path):
    small = make_cell(tmp_path / "small", prefix="s")   # 3 entities, 20×40×40
    big = tmp_path / "big"
    make_cell(big, prefix="b")

    # Same shape here, so the estimate is about entity count and voxels, not file size.
    assert estimate_peak_gb(tmp_path / "small") == pytest.approx(estimate_peak_gb(big))
    assert 0 < estimate_peak_gb(tmp_path / "small") < 0.01


def test_worker_count_is_bounded_by_memory_and_cpus(dataset):
    cells = find_cell_dirs(dataset)

    workers = suggest_max_workers(cells)
    assert 1 <= workers <= (os.cpu_count() or 1)


def test_a_folder_with_no_source_estimates_nothing(tmp_path):
    (tmp_path / "empty").mkdir()

    assert estimate_peak_gb(tmp_path / "empty") == 0.0
    assert suggest_max_workers([tmp_path / "empty"]) == 1


def test_analysis_flags_travel_as_environment_variables(monkeypatch):
    # Writes land in a throwaway copy: a leaked CELLSKETCH_* here would silently
    # reconfigure every later test, since that is exactly how plugins read their options.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CELLSKETCH_")}
    monkeypatch.setattr(os, "environ", env)

    _apply_analysis_env("0.5,0.1,0.1", True, False, 0.25, None, None)

    assert env["CELLSKETCH_VOXEL_SIZE_UM"] == "0.5,0.1,0.1"
    assert env["CELLSKETCH_AUTO_CLIP_TO_PM"] == "1"
    assert env["CELLSKETCH_CONTACT_MAX_UM"] == "0.25"
    # Flags left alone must not be forced to a default here - config.py owns those.
    assert "CELLSKETCH_AUTO_LABEL_MASKS" not in env
    assert "CELLSKETCH_MAX_SKELETON_VOXELS" not in env


def test_dry_run_lists_the_cells_and_their_entities(dataset):
    result = CliRunner().invoke(cli, ["dry-run", str(dataset)])

    assert result.exit_code == 0
    assert "control/cell_a" in result.output
    assert "labels  mito" in result.output
    assert "masks   nucleus, pm*" in result.output      # * marks the plasma membrane
    assert "label:mito               4/4" in result.output


def test_dry_run_reports_a_folder_it_cannot_analyse(dataset):
    (dataset / "control" / "cell_a" / "sample_a_pm_mask.tif").unlink()

    result = CliRunner().invoke(cli, ["dry-run", str(dataset)])

    # No membrane means no cell: it defines the volume everything else is measured in.
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_dry_run_says_so_when_nothing_looks_like_a_cell(tmp_path):
    result = CliRunner().invoke(cli, ["dry-run", str(tmp_path)])

    assert result.exit_code == 1
    assert "No cell folders found" in result.output


def test_process_writes_a_report_without_being_told_how_to_slice(dataset, tmp_path):
    out = tmp_path / "report.parquet"
    result = CliRunner().invoke(
        cli, ["process", str(dataset), "-o", str(out), "-p", "control", "-p", "treated"]
    )

    assert result.exit_code == 0, result.output
    table = pl.read_parquet(out)
    # The entity rows only exist if slice_size was set for us.
    assert table.filter(pl.col("obs_level") == 1).height == 12
    assert table.filter(pl.col("obs_level") == 0)["instance_count"].sum() == 14


def test_process_can_skip_the_expensive_processors(dataset, tmp_path):
    out = tmp_path / "lean.parquet"
    result = CliRunner().invoke(
        cli, ["process", str(dataset), "-o", str(out), "--no-instances", "--no-contacts"]
    )

    assert result.exit_code == 0, result.output
    columns = pl.read_parquet(out).columns
    assert "instance_entity" not in columns
    assert "contact_count" not in columns
    assert "entity_name" in columns          # per-entity morphology still runs


def test_process_refuses_a_directory_with_no_cells(tmp_path):
    result = CliRunner().invoke(
        cli, ["process", str(tmp_path), "-o", str(tmp_path / "x.parquet")]
    )

    assert result.exit_code != 0
    assert "No cell folders found" in result.output
    assert "dry-run" in result.output        # points at the command that explains why
