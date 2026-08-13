"""End-to-end: run the real PixelPatrol pipeline over a synthetic grouped batch.

This is the phase-1 validation — it asserts the row layout the whole design rests on:
one row per cell at obs_level 0, one row per entity at obs_level 1 keyed by dim_c.
"""

import polars as pl
import pytest

def test_one_row_per_cell_at_obs_level_0(table):
    cells = table.filter(pl.col("obs_level") == 0)

    assert sorted(cells["cell_id"].to_list()) == ["cell_a", "cell_b", "cell_c", "cell_d"]
    assert cells["n_entities"].to_list() == [3, 3, 3, 3]
    # instance_count rolls up across the cell's label entities (mito only here); the
    # mask entities contribute null, not 1.
    assert cells.sort("cell_id")["instance_count"].to_list() == [4, 4, 3, 3]


def test_cell_row_carries_no_per_entity_measurements(table):
    cells = table.filter(pl.col("obs_level") == 0)

    # Per-entity values must not leak onto the cell row even when only one entity
    # produced them — otherwise a one-label-entity cell looks different in kind from
    # a two-label-entity one.
    for col in ("entity_name", "entity_kind", "volume_um3", "sphericity"):
        assert cells[col].null_count() == cells.height


def test_one_row_per_entity_at_obs_level_1(table):
    entities = table.filter(pl.col("obs_level") == 1)

    assert entities.height == 4 * 3
    per_cell = entities.group_by("cell_id").agg(pl.col("entity_name").sort())
    for names in per_cell["entity_name"].to_list():
        assert names == ["mito", "nucleus", "pm"]


def test_entity_rows_are_keyed_by_dim_c(table):
    row = table.filter((pl.col("obs_level") == 1) & (pl.col("cell_id") == "cell_a")).sort("dim_c")

    # dim_c indexes channel_names, which is how a widget maps a row back to its entity.
    assert row["dim_c"].to_list() == [0, 1, 2]
    assert row["entity_name"].to_list() == ["pm", "mito", "nucleus"]
    assert row["entity_kind"].to_list() == ["mask", "label", "mask"]


def test_the_discovered_unit_is_the_cell_folder(table):
    # 4 cells × (1 cell row + 3 entity rows). The TIFFs inside a claimed folder are
    # never records of their own, so nothing has to be skipped or declined.
    assert table.height == 4 * 4
    assert table["name"].unique().sort().to_list() == ["cell_a", "cell_b", "cell_c", "cell_d"]
    assert table["parent"].unique().sort().to_list() == ["control", "treated"]
    # And its size describes the volumes it was built from, not the 4 KB inode of the
    # directory holding them.
    assert table["size_bytes"].min() > 100_000


def test_group_comes_from_the_import_path(table):
    groups = dict(
        table.filter(pl.col("obs_level") == 0)
        .select("cell_id", "imported_path_short")
        .iter_rows()
    )

    assert groups == {
        "cell_a": "control", "cell_b": "control",
        "cell_c": "treated", "cell_d": "treated",
    }


def _instances(table) -> pl.DataFrame:
    """The instance table, as a widget gets it: one unnest of the cell row's lists."""
    instance_cols = [c for c in table.columns if c.startswith("instance_") and c != "instance_count"]
    return (
        table.filter(pl.col("obs_level") == 0)
        .select("cell_id", "imported_path_short", *instance_cols)
        .explode(instance_cols, empty_as_null=True)
    )


def test_instance_lists_unnest_to_one_row_per_instance(table):
    instances = _instances(table)

    # 4 + 4 + 3 + 3 mitochondria, and every instance_* column stays aligned with them.
    assert instances.height == 14
    assert instances["instance_entity"].unique().to_list() == ["mito"]
    assert instances["instance_volume_um3"].min() > 0
    per_cell = instances.group_by("cell_id").len().sort("cell_id")
    assert per_cell["len"].to_list() == [4, 4, 3, 3]


def test_instance_count_on_the_cell_row_matches_the_instance_table(table):
    cells = table.filter(pl.col("obs_level") == 0).sort("cell_id")
    counted = _instances(table).group_by("cell_id").len().sort("cell_id")

    # instance_count comes from the per-entity rows, the table from the cell row: two
    # processors, one answer.
    assert cells["instance_count"].to_list() == counted["len"].to_list()


def test_treated_cells_have_larger_mitochondria(table):
    """The comparison the viewer's significance brackets will be drawn on."""
    by_group = (
        _instances(table)
        .group_by("imported_path_short")
        .agg(pl.col("instance_volume_um3").mean())
    )
    means = dict(by_group.iter_rows())

    assert means["treated"] > means["control"]


def test_instances_carry_distances_to_the_other_entities(table):
    distances = (
        table.filter(pl.col("obs_level") == 0)
        .select("cell_id", "distance_entity", "distance_label", "distance_target", "distance_um")
        .explode("distance_entity", "distance_label", "distance_target", "distance_um",
                 empty_as_null=True)
    )

    # Every mitochondrion is measured to the two entities that are not its own.
    assert sorted(distances["distance_target"].unique().to_list()) == ["nucleus", "pm"]
    assert distances.height == 14 * 2
    # 0 is a real reading, not a missing one: the synthetic mitochondria ring overlaps
    # the nucleus, and an instance inside its target is zero away from it.
    assert distances["distance_um"].min() == 0.0
    assert distances["distance_um"].max() > 0
    assert distances["distance_um"].is_finite().all()


def test_contacts_ride_on_the_cell_row_as_an_edge_list(table):
    cells = table.filter(pl.col("obs_level") == 0)

    # The synthetic mitochondria sit on a ring around the nucleus, so every cell has
    # neighbouring pairs; the columns are parallel, one element per pair.
    assert cells["contact_count"].min() > 0
    for row in cells.iter_rows(named=True):
        n = row["contact_count"]
        assert len(row["contact_entity_a"]) == n
        assert len(row["contact_gap_um"]) == n
        assert all(gap <= 0.5 for gap in row["contact_gap_um"])
        # A pair always names two entities, and never the enclosing membrane.
        assert "pm" not in set(row["contact_entity_a"]) | set(row["contact_entity_b"])


def test_contacts_are_absent_from_entity_rows(table):
    entities = table.filter(pl.col("obs_level") == 1)

    # A pair belongs to no single entity, so it stays on the cell row.
    assert entities["contact_count"].null_count() == entities.height


def test_voxel_size_lands_in_the_standard_pixel_size_columns(table):
    cells = table.filter(pl.col("obs_level") == 0)

    assert cells["pixel_size_Z"].to_list() == pytest.approx([0.1] * 4)
    assert cells["pixel_size_X"].to_list() == pytest.approx([0.02] * 4)
    assert cells["voxel_size_source"].unique().to_list() == ["tiff-metadata"]
