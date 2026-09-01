"""End-to-end: run the real PixelPatrol pipeline over a synthetic grouped batch.

This is the phase-1 validation — it asserts the row layout the whole design rests on:
one row per object at obs_level 0, one row per entity at obs_level 1 keyed by dim_c.
"""

from pathlib import Path

import polars as pl
import pytest

from pixel_patrol_anatomy import pipeline

from conftest import MITO_COLOUR

def test_one_row_per_object_at_obs_level_0(table):
    objects = table.filter(pl.col("obs_level") == 0)

    assert sorted(objects["object_id"].to_list()) == ["object_a", "object_b", "object_c", "object_d"]
    assert objects["n_entities"].to_list() == [3, 3, 3, 3]
    # instance_count rolls up across the object's label entities (mito only here); the
    # mask entities contribute null, not 1.
    assert objects.sort("object_id")["instance_count"].to_list() == [4, 4, 3, 3]


def test_object_row_carries_no_per_entity_measurements(table):
    objects = table.filter(pl.col("obs_level") == 0)

    # Per-entity values must not leak onto the object row even when only one entity
    # produced them — otherwise a one-label-entity object looks different in kind from
    # a two-label-entity one.
    for col in ("entity_name", "entity_kind", "volume_um3", "sphericity"):
        assert objects[col].null_count() == objects.height


def test_one_row_per_entity_at_obs_level_1(table):
    entities = table.filter(pl.col("obs_level") == 1)

    assert entities.height == 4 * 3
    per_object = entities.group_by("object_id").agg(pl.col("entity_name").sort())
    for names in per_object["entity_name"].to_list():
        assert names == ["mito", "nucleus", "pm"]


def test_entity_rows_are_keyed_by_dim_c(table):
    row = table.filter((pl.col("obs_level") == 1) & (pl.col("object_id") == "object_a")).sort("dim_c")

    # dim_c indexes channel_names, which is how a widget maps a row back to its entity.
    assert row["dim_c"].to_list() == [0, 1, 2]
    assert row["entity_name"].to_list() == ["pm", "mito", "nucleus"]
    assert row["entity_kind"].to_list() == ["mask", "label", "mask"]


def test_the_discovered_unit_is_the_object_folder(table):
    # 4 objects × (1 object row + 3 entity rows). The TIFFs inside a claimed folder are
    # never records of their own, so nothing has to be skipped or declined.
    assert table.height == 4 * 4
    assert table["object_id"].unique().sort().to_list() == [
        "object_a", "object_b", "object_c", "object_d"]
    assert table["imported_path_short"].unique().sort().to_list() == ["control", "treated"]
    # And its size describes the volumes it was built from, not the 4 KB inode of the
    # directory holding them.
    assert table["size_bytes"].min() > 100_000


def test_group_comes_from_the_import_path(table):
    groups = dict(
        table.filter(pl.col("obs_level") == 0)
        .select("object_id", "imported_path_short")
        .iter_rows()
    )

    assert groups == {
        "object_a": "control", "object_b": "control",
        "object_c": "treated", "object_d": "treated",
    }


def _instances(table) -> pl.DataFrame:
    """The instance table, as a widget gets it: one unnest of the object row's lists."""
    instance_cols = [c for c in table.columns if c.startswith("instance_") and c != "instance_count"]
    return (
        table.filter(pl.col("obs_level") == 0)
        .select("object_id", "imported_path_short", *instance_cols)
        .explode(instance_cols, empty_as_null=True)
    )


def test_instance_lists_unnest_to_one_row_per_instance(table):
    instances = _instances(table)

    # 4 + 4 + 3 + 3 mitochondria, and every instance_* column stays aligned with them.
    assert instances.height == 14
    assert instances["instance_entity"].unique().to_list() == ["mito"]
    assert instances["instance_volume_um3"].min() > 0
    per_object = instances.group_by("object_id").len().sort("object_id")
    assert per_object["len"].to_list() == [4, 4, 3, 3]


def test_instance_count_on_the_object_row_matches_the_instance_table(table):
    objects = table.filter(pl.col("obs_level") == 0).sort("object_id")
    counted = _instances(table).group_by("object_id").len().sort("object_id")

    # instance_count comes from the per-entity rows, the table from the object row: two
    # processors, one answer.
    assert objects["instance_count"].to_list() == counted["len"].to_list()


def test_treated_objects_have_larger_mitochondria(table):
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
        .select("object_id", "distance_entity", "distance_label", "distance_target", "distance_um")
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


def test_contacts_ride_on_the_object_row_as_an_edge_list(table):
    objects = table.filter(pl.col("obs_level") == 0)

    # The synthetic mitochondria sit on a ring around the nucleus, so every object has
    # neighbouring pairs; the columns are parallel, one element per pair.
    assert objects["contact_count"].min() > 0
    for row in objects.iter_rows(named=True):
        n = row["contact_count"]
        assert len(row["contact_entity_a"]) == n
        assert len(row["contact_gap_um"]) == n
        assert all(gap <= 0.5 for gap in row["contact_gap_um"])
        # A pair always names two entities, and never the enclosing membrane.
        assert "pm" not in set(row["contact_entity_a"]) | set(row["contact_entity_b"])


def test_contacts_are_absent_from_entity_rows(table):
    entities = table.filter(pl.col("obs_level") == 1)

    # A pair belongs to no single entity, so it stays on the object row.
    assert entities["contact_count"].null_count() == entities.height


def test_a_structure_carries_the_colour_the_settings_file_gave_it(table):
    """The palette rides in the report, so a shared report arrives already coloured.

    Named structures carry their colour; the rest are null and every widget falls back to its
    own palette, which is what lets one settings file cover a whole project.
    """
    entities = table.filter(pl.col("obs_level") == 1)
    by_name = dict(zip(entities["entity_name"], entities["entity_colour"]))

    assert by_name["mito"] == MITO_COLOUR
    assert by_name["pm"] is None and by_name["nucleus"] is None


def test_voxel_size_lands_in_the_standard_pixel_size_columns(table):
    objects = table.filter(pl.col("obs_level") == 0)

    assert objects["pixel_size_Z"].to_list() == pytest.approx([0.1] * 4)
    assert objects["pixel_size_X"].to_list() == pytest.approx([0.02] * 4)
    assert objects["voxel_size_source"].unique().to_list() == ["tiff-metadata"]


# ── surviving a worker that is killed rather than raising ────────────────────
#
# The OOM killer sends SIGKILL, so the worker reports nothing and the pool breaks. What must
# not happen is what did happen on a real 7-object run: every finished object thrown away
# because one of them died after two hours.

def _fake_dispatch(killed_by_worker_count):
    """A dispatch that kills the object named 'bad' unless few enough workers are running."""
    def dispatch(work, excluded, n_workers):
        finished, unrun = [], []
        for folder, group in work:
            if folder.name == "bad" and n_workers >= killed_by_worker_count:
                unrun.append((folder, group))
            else:
                finished.append(pipeline.ObjectResult(folder.name, object_row={"n": folder.name}))
        return finished, unrun
    return dispatch


def _work(*names):
    return [(Path(f"/nowhere/{n}"), "") for n in names]


def test_a_killed_worker_does_not_lose_the_objects_that_finished():
    results = pipeline._measure_all(_work("a", "bad", "b"), (), 4,
                                    dispatch=_fake_dispatch(killed_by_worker_count=2))
    measured = {r.object_id for r in results if r.error is None}
    assert {"a", "b"} <= measured, "objects that finished were thrown away"


def test_the_killed_object_is_retried_at_lower_concurrency():
    # 'bad' survives once the pool is down to a single worker, which is the usual cause:
    # too many large objects resident at once.
    results = pipeline._measure_all(_work("a", "bad", "b"), (), 4,
                                    dispatch=_fake_dispatch(killed_by_worker_count=2))
    assert {r.object_id for r in results} == {"a", "bad", "b"}
    assert all(r.error is None for r in results)


def test_an_object_that_always_kills_its_worker_is_reported_not_retried_forever():
    results = pipeline._measure_all(_work("a", "bad", "b"), (), 4,
                                    dispatch=_fake_dispatch(killed_by_worker_count=1))
    by_id = {r.object_id: r for r in results}
    assert set(by_id) == {"a", "bad", "b"}
    assert by_id["bad"].error and "out of memory" in by_id["bad"].error
    assert by_id["a"].error is None and by_id["b"].error is None
