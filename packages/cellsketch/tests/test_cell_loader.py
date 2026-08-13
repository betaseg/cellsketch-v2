import pytest

from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND, CellLoader
from synthetic import VOXEL_SIZE_UM, make_cell, make_dataset


@pytest.fixture
def cell_dir(tmp_path):
    make_cell(tmp_path / "cell_a", prefix="sample_a")
    return tmp_path / "cell_a"


def test_loads_cell_folder_as_one_czyx_record(cell_dir):
    record = CellLoader().load(cell_dir)

    assert record.kind == CELL_KIND
    assert record.dim_order == "CZYX"
    assert record.data.shape[0] == 3          # pm, nucleus, mito
    assert record.meta["cell_id"] == "cell_a"
    assert record.meta["membrane_name"] == "pm"
    assert record.meta["n_entities"] == 3


def test_membrane_is_first_channel_then_alphabetical(cell_dir):
    record = CellLoader().load(cell_dir)

    assert record.meta["channel_names"] == ["pm", "mito", "nucleus"]
    assert record.meta["entity_kinds"] == ["mask", "label", "mask"]


def test_voxel_size_read_from_tiff_metadata(cell_dir):
    record = CellLoader().load(cell_dir)

    assert record.meta["voxel_size_source"] == "tiff-metadata"
    assert record.meta["pixel_size_Z"] == pytest.approx(VOXEL_SIZE_UM[0])
    assert record.meta["pixel_size_Y"] == pytest.approx(VOXEL_SIZE_UM[1])
    assert record.meta["pixel_size_X"] == pytest.approx(VOXEL_SIZE_UM[2])


def test_voxel_size_override_from_env(cell_dir, monkeypatch):
    monkeypatch.setenv("CELLSKETCH_VOXEL_SIZE_UM", "0.5,0.25,0.25")
    record = CellLoader().load(cell_dir)

    assert record.meta["voxel_size_source"] == "config"
    assert record.meta["pixel_size_Z"] == pytest.approx(0.5)


def test_volume_is_cropped_to_membrane_bbox(cell_dir):
    record = CellLoader().load(cell_dir)

    # The membrane ellipsoid is inset from the image border, so the analysed volume
    # is smaller than the source image and cell_shape_zyx reports what was measured.
    assert list(record.data.shape[1:]) == record.meta["cell_shape_zyx"]
    assert record.data.shape[1:] < (20, 40, 40)


def test_read_header_reports_the_stacked_shape(cell_dir):
    info = CellLoader().read_header(cell_dir)

    assert info.dim_order == "CZYX"
    assert info.shape == (3, 20, 40, 40)
    assert info.n_images == 1


# ── Folder discovery ──────────────────────────────────────────────────────────

def test_cell_folder_is_claimed_as_one_dataset(cell_dir):
    # Claiming the folder is what stops PixelPatrol from walking into it and offering
    # each label/mask TIFF as a record of its own.
    assert CellLoader().is_folder_supported(cell_dir) is True


def test_folders_that_only_contain_cells_are_not_claimed(tmp_path):
    root = make_dataset(tmp_path / "experiment")
    loader = CellLoader()

    assert loader.is_folder_supported(root) is False
    assert loader.is_folder_supported(root / "control") is False
    assert loader.is_folder_supported(root / "control" / "cell_a") is True


def test_folder_without_entity_volumes_is_not_claimed(tmp_path, cell_dir):
    lonely = tmp_path / "just_an_image"
    lonely.mkdir()
    (cell_dir / "sample_a.tif").replace(lonely / "sample_a.tif")

    assert CellLoader().is_folder_supported(lonely) is False
