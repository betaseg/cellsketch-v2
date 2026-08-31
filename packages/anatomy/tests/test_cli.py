import json
import os
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from pixel_patrol_anatomy.cli import (
    FLAVOR,
    _apply_analysis_env,
    cli,
    estimate_peak_gb,
    find_object_dirs,
)
from pixel_patrol_anatomy.config import AnatomyConfig
from synthetic import make_object, make_dataset


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / "experiment")


def test_finds_every_object_folder_but_not_the_folders_holding_them(dataset):
    found = {p.name for p in find_object_dirs(dataset)}

    assert found == {"object_a", "object_b", "object_c", "object_d"}


def test_a_single_object_folder_is_its_own_dataset(tmp_path):
    make_object(tmp_path / "object_a", prefix="sample_a")

    assert find_object_dirs(tmp_path / "object_a") == [tmp_path / "object_a"]


def test_peak_memory_estimate_scales_with_the_object(tmp_path):
    make_object(tmp_path / "small", prefix="s")           # 3 entities, 20×40×40
    big = tmp_path / "big"
    make_object(big, prefix="b")

    # Same shape here, so the estimate is about entity count and voxels, not file size.
    assert estimate_peak_gb(tmp_path / "small") == pytest.approx(estimate_peak_gb(big))
    assert 0 < estimate_peak_gb(tmp_path / "small") < 0.01


def test_a_folder_with_no_source_estimates_nothing(tmp_path):
    (tmp_path / "empty").mkdir()

    assert estimate_peak_gb(tmp_path / "empty") == 0.0


def test_analysis_flags_travel_as_environment_variables(monkeypatch):
    # Writes land in a throwaway copy: a leaked PP_ANATOMY_* here would silently
    # reconfigure every later test, since that is exactly how plugins read their options.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PP_ANATOMY_")}
    monkeypatch.setattr(os, "environ", env)

    _apply_analysis_env("pm", "0.5,0.1,0.1", True, False, 0.25, None, None)

    assert env["PP_ANATOMY_OBJECT_MASK"] == "pm"
    assert env["PP_ANATOMY_VOXEL_SIZE_UM"] == "0.5,0.1,0.1"
    assert env["PP_ANATOMY_AUTO_CLIP"] == "1"
    assert env["PP_ANATOMY_CONTACT_MAX_UM"] == "0.25"
    # Flags left alone must not be forced to a default here - config.py owns those.
    assert "PP_ANATOMY_AUTO_LABEL_MASKS" not in env
    assert "PP_ANATOMY_MAX_SKELETON_VOXELS" not in env


def test_a_colour_settings_file_is_read_and_expanded(tmp_path, monkeypatch):
    """One file per study, hand-edited: short hex and any case have to work."""
    settings = tmp_path / "colours.json"
    settings.write_text(json.dumps({"mito": "#D62728", "er": "#2c3"}))
    monkeypatch.setenv("PP_ANATOMY_ENTITY_COLOURS", str(settings))

    assert AnatomyConfig.from_env().entity_colours == {"mito": "#d62728", "er": "#22cc33"}


@pytest.mark.parametrize("contents,complaint", [
    ('{"mito": "red"}', "hex colour"),
    ('{"mito": 16711680}', "hex colour"),
    ('["mito"]', "structure: colour pairs"),
    ('{"mito": ', "not valid JSON"),
])
def test_a_broken_colour_file_says_what_is_wrong(tmp_path, monkeypatch, contents, complaint):
    settings = tmp_path / "colours.json"
    settings.write_text(contents)
    monkeypatch.setenv("PP_ANATOMY_ENTITY_COLOURS", str(settings))

    with pytest.raises(ValueError, match=complaint):
        AnatomyConfig.from_env()


def test_a_missing_colour_file_is_an_error_not_a_default(tmp_path, monkeypatch):
    """Silently ignoring it would produce a report coloured nothing like the study asked."""
    monkeypatch.setenv("PP_ANATOMY_ENTITY_COLOURS", str(tmp_path / "nope.json"))

    with pytest.raises(ValueError, match="no such file"):
        AnatomyConfig.from_env()


def test_colouring_a_report_keeps_everything_else_about_it(tmp_path, report_path):
    """Colours are presentation, and a run takes minutes: changing them re-reads no pixels.

    The report is rewritten, so the footer the viewer's provenance strip reads has to survive,
    and so do the rows.
    """
    before = pq.read_table(report_path)
    palette = tmp_path / "palette.json"
    palette.write_text(json.dumps({"nucleus": "#9467bd"}))
    coloured = tmp_path / "coloured.parquet"
    coloured.write_bytes(Path(report_path).read_bytes())

    result = CliRunner().invoke(cli, ["colours", str(coloured), str(palette)])

    assert result.exit_code == 0, result.output
    after = pq.read_table(coloured)
    assert after.num_rows == before.num_rows
    assert after.schema.metadata[b"pp_flavor"] == before.schema.metadata[b"pp_flavor"]
    assert after.schema.metadata[b"pp_paths"] == before.schema.metadata[b"pp_paths"]
    entities = pl.from_arrow(after).filter(pl.col("obs_level") == 1)
    by_name = dict(zip(entities["entity_name"], entities["entity_colour"]))
    # The new palette replaces the old one outright: it is the whole answer, not an addition.
    assert by_name["nucleus"] == "#9467bd"
    assert by_name["mito"] is None


def test_colouring_a_report_from_a_broken_palette_fails_before_writing(tmp_path, report_path):
    palette = tmp_path / "palette.json"
    palette.write_text('{"mito": "crimson"}')
    coloured = tmp_path / "untouched.parquet"
    coloured.write_bytes(Path(report_path).read_bytes())
    before = coloured.read_bytes()

    result = CliRunner().invoke(cli, ["colours", str(coloured), str(palette)])

    assert result.exit_code != 0 and "hex colour" in result.output
    assert coloured.read_bytes() == before


def test_dry_run_lists_the_objects_and_their_entities(dataset):
    result = CliRunner().invoke(cli, ["dry-run", str(dataset)])

    assert result.exit_code == 0
    assert "control/object_a" in result.output
    assert "labels  mito" in result.output
    assert "masks   nucleus, pm" in result.output
    assert "label:mito               4/4" in result.output


def test_dry_run_without_a_mask_says_which_masks_to_choose_from(dataset):
    """This is the survey you run first: nothing is guessed, so it has to offer the names."""
    result = CliRunner().invoke(cli, ["dry-run", str(dataset)])

    assert result.exit_code == 0
    assert "pass it as --object-mask: nucleus, pm" in result.output
    # And no mask is marked as the boundary, because none was chosen.
    assert "pm*" not in result.output


def test_dry_run_marks_and_checks_the_mask_it_is_given(dataset):
    result = CliRunner().invoke(cli, ["dry-run", str(dataset), "--object-mask", "pm"])

    assert result.exit_code == 0
    assert "masks   nucleus, pm*" in result.output      # * marks the object mask
    assert "--object-mask" not in result.output.split("=====")[-1]


def test_dry_run_reports_a_folder_missing_the_mask_it_was_given(dataset):
    (dataset / "control" / "object_a" / "sample_a_pm_mask.tif").unlink()

    result = CliRunner().invoke(cli, ["dry-run", str(dataset), "--object-mask", "pm"])

    # Naming a mask a folder does not have is the error: measuring against a different
    # boundary would put every distance in that object on a different origin.
    assert result.exit_code == 1
    assert "No mask named 'pm'" in result.output


def test_dry_run_says_so_when_nothing_looks_like_an_object(tmp_path):
    result = CliRunner().invoke(cli, ["dry-run", str(tmp_path)])

    assert result.exit_code == 1
    assert "No object folders found" in result.output


def test_process_writes_a_report_without_being_told_how_to_slice(dataset, tmp_path):
    out = tmp_path / "report.parquet"
    result = CliRunner().invoke(
        cli, ["process", str(dataset), "-o", str(out), "--object-mask", "pm",
              "-p", "control", "-p", "treated"]
    )

    assert result.exit_code == 0, result.output
    table = pl.read_parquet(out)
    # The entity rows only exist if slice_size was set for us.
    assert table.filter(pl.col("obs_level") == 1).height == 12
    assert table.filter(pl.col("obs_level") == 0)["instance_count"].sum() == 14


def test_the_report_says_what_kind_of_analysis_it_is(dataset, tmp_path):
    out = tmp_path / "report.parquet"
    result = CliRunner().invoke(cli, ["process", str(dataset), "-o", str(out), "--object-mask", "pm"])

    assert result.exit_code == 0, result.output
    # The viewer shows the flavour as a chip beside the title, so a report is recognisable
    # as this analysis before a widget is read.
    metadata = pq.read_metadata(out).metadata
    assert metadata[b"pp_flavor"].decode() == FLAVOR == "object anatomy"


def test_process_can_skip_the_expensive_processors(dataset, tmp_path):
    out = tmp_path / "lean.parquet"
    result = CliRunner().invoke(
        cli, ["process", str(dataset), "-o", str(out), "--object-mask", "pm",
              "--no-instances", "--no-contacts"]
    )

    assert result.exit_code == 0, result.output
    columns = pl.read_parquet(out).columns
    assert "instance_entity" not in columns
    assert "contact_count" not in columns
    assert "entity_name" in columns          # per-entity morphology still runs


def test_process_refuses_a_directory_with_no_objects(tmp_path):
    result = CliRunner().invoke(
        cli, ["process", str(tmp_path), "-o", str(tmp_path / "x.parquet"),
              "--object-mask", "pm"]
    )

    assert result.exit_code != 0
    assert "No object folders found" in result.output
    assert "dry-run" in result.output        # points at the command that explains why
