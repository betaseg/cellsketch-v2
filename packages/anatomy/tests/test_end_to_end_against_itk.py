"""Run the whole pipeline over a batch, then re-measure the same TIFFs with ITK.

The other reference tests check one function against one shape. This checks the path a user
actually takes — folders in, report out — against an implementation that shares none of our
code: read the label volumes back off disk, measure every instance with ITK, and compare.

Both dimensionalities, because they are measured by different formulas.

The comparison is fair only because every structure in the synthetic batch lies inside the
object mask: we measure the region cropped to that mask, ITK measures the whole file.
"""

import numpy as np
import polars as pl
import pytest
import tifffile

from pixel_patrol_anatomy import pipeline
from pixel_patrol_anatomy.cli import FLAVOR, find_object_dirs
from synthetic import (
    PIXEL_SIZE_UM,
    VOXEL_SIZE_UM,
    make_dataset,
    make_dataset_2d,
)

sitk = pytest.importorskip("SimpleITK")


def itk_instance_extents(label_path, sample_size) -> dict[int, float]:
    """{label id: extent in µm} straight from the TIFF, measured by ITK."""
    image = sitk.GetImageFromArray(tifffile.imread(label_path).astype(np.uint16))
    image.SetSpacing(tuple(float(v) for v in reversed(sample_size)))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(image)
    return {int(label): stats.GetPhysicalSize(int(label)) for label in stats.GetLabels()}


def report_instance_extents(table: pl.DataFrame, object_id: str, entity: str,
                            column: str) -> dict[int, float]:
    """{label id: extent} as the report has it, one unnest of the object row."""
    row = (table.filter((pl.col("obs_level") == 0) & (pl.col("object_id") == object_id))
           .select("instance_entity", "instance_label", column)
           .explode("instance_entity", "instance_label", column, empty_as_null=True))
    mine = row.filter(pl.col("instance_entity") == entity)
    return dict(zip(mine["instance_label"].to_list(), mine[column].to_list()))


def _run(root, out):
    report = pipeline.analyse(find_object_dirs(root), root, [], workers=1)
    assert not report.failures, report.failures
    pipeline.write(report, out, root=root, paths=[], flavor=FLAVOR)
    return pl.read_parquet(out)


def test_every_3d_instance_volume_matches_itk(tmp_path):
    root = make_dataset(tmp_path / "volumes")
    table = _run(root, tmp_path / "report.parquet")

    for folder in find_object_dirs(root):
        label_file = next(folder.glob("*_mito_label.tif"))
        reference = itk_instance_extents(label_file, VOXEL_SIZE_UM)
        ours = report_instance_extents(table, folder.name, "mito", "instance_volume_um3")

        assert ours.keys() == reference.keys()
        for label, volume in ours.items():
            assert volume == pytest.approx(reference[label], rel=1e-4), (folder.name, label)


def test_every_2d_instance_area_matches_itk(tmp_path):
    root = make_dataset_2d(tmp_path / "planes")
    table = _run(root, tmp_path / "report_2d.parquet")

    for folder in find_object_dirs(root):
        label_file = next(folder.glob("*_mito_label.tif"))
        reference = itk_instance_extents(label_file, PIXEL_SIZE_UM)
        ours = report_instance_extents(table, folder.name, "mito", "instance_area_um2")

        assert ours.keys() == reference.keys()
        for label, area in ours.items():
            assert area == pytest.approx(reference[label], rel=1e-4), (folder.name, label)


def test_the_object_extent_matches_itk_on_the_object_mask(tmp_path):
    root = make_dataset(tmp_path / "volumes")
    table = _run(root, tmp_path / "report.parquet")
    objects = table.filter(pl.col("obs_level") == 0)

    for folder in find_object_dirs(root):
        reference = itk_instance_extents(next(folder.glob("*_pm_mask.tif")), VOXEL_SIZE_UM)
        ours = objects.filter(pl.col("object_id") == folder.name)["object_volume_um3"][0]

        # One mask, one label: the object's own volume, cropped to its bounding box by the
        # loader — which changes nothing about how much of it is filled.
        assert ours == pytest.approx(sum(reference.values()), rel=1e-4)
