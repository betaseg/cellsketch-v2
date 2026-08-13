"""Loader that turns one cell folder into one PixelPatrol record.

The unit of analysis is a *folder*: a source image plus its entity volumes
(``<prefix>_<name>_label.tif`` / ``_mask.tif``). ``is_folder_supported`` claims such
a folder, so PixelPatrol hands the directory itself to ``read_header`` / ``load``
and never descends into it — the TIFFs inside are not records of their own.

``load`` returns every entity volume stacked along a C axis (``CZYX``), with
``channel_names`` naming the entities. Two consequences of that shape, both
deliberate:

* one row per cell at ``obs_level=0`` and one row per entity at ``obs_level=1``
  (``dim_c``) — process with ``--slice-size C=1 --slice-size Z=-1`` so a leaf block
  is one whole entity volume;
* every entity of a cell is in one record, so cross-entity metrics (distances,
  contacts) can be computed by a single processor.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import numpy as np
import tifffile
from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import Record, record_from

from pixel_patrol_cellsketch.config import CellSketchConfig
from pixel_patrol_cellsketch.distances import cell_center_um
from pixel_patrol_cellsketch.discovery import (
    Dataset,
    clip_to_membrane,
    crop_to_membrane_bbox,
    discover_dataset,
    infer_voxel_size_um_from_source,
    inspect_cell_dir,
    load_volume,
    promote_multicomponent_masks,
)

logger = logging.getLogger(__name__)

# Record kind. Deliberately not "intensity": PixelPatrol's own raster processors
# (raster-basic, raster-histogram, thumbnail) declare INPUT kinds={"intensity"} and
# so skip these records, which are label maps where pixel statistics are meaningless.
CELL_KIND = "cell/segmentation"

@lru_cache(maxsize=256)
def _cached_inspect(cell_dir: str):
    """Cache discovery per folder: discovery, read_header and load all ask for it."""
    return inspect_cell_dir(Path(cell_dir))


def _source_header(source_path: Path) -> Tuple[Tuple[int, int, int], str]:
    with tifffile.TiffFile(source_path) as tf:
        series = tf.series[0]
        shape = tuple(int(s) for s in series.shape)
        dtype = str(np.dtype(series.dtype))
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D source volume in {source_path.name}, got shape {shape}")
    return shape, dtype  # type: ignore[return-value]


def _label_dtype(volumes: Dict[str, np.ndarray], cell_id: str) -> np.dtype:
    """The narrowest integer type that holds every label id in this cell.

    Segmentations are often stored as float32 or int32 whatever their ids need, and the
    stack is the single largest allocation in a run: a 371×1257×1176 cell with five
    entities is 10.5 GB as int32 and 5.2 GB as uint16. Nothing here rounds ids - a
    non-integral value would mean the volume is not a segmentation, and says so.
    """
    highest = 0
    for name, vol in volumes.items():
        if vol.size == 0:
            continue
        if np.issubdtype(vol.dtype, np.floating):
            finite = vol[np.isfinite(vol)]
            if finite.size and not np.all(np.equal(np.mod(finite, 1), 0)):
                raise ValueError(
                    f"{cell_id}: entity '{name}' has non-integer values, so it is not a "
                    "label or mask volume"
                )
        lo, hi = float(vol.min()), float(vol.max())
        if lo < 0:
            raise ValueError(f"{cell_id}: entity '{name}' has negative label ids ({lo})")
        highest = max(highest, hi)
    for dtype in (np.uint8, np.uint16, np.uint32):
        if highest <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    return np.dtype(np.uint64)


def _stack_narrowest(volumes: Dict[str, np.ndarray], keys: List[str], cell_id: str) -> np.ndarray:
    """Stack the entities along C, converting one at a time and freeing as we go.

    Written channel by channel into a preallocated array rather than via np.stack, so the
    source volumes and the stack are never both fully in memory.
    """
    dtype = _label_dtype(volumes, cell_id)
    first = volumes[keys[0]]
    stack = np.empty((len(keys), *first.shape), dtype=dtype)
    for i, key in enumerate(keys):
        np.copyto(stack[i], volumes[key], casting="unsafe")
        volumes[key] = np.empty((0, 0, 0), dtype=dtype)  # release the source volume
    return stack


def _channel_order(dataset: Dataset) -> List[str]:
    """Entity keys in C order: membrane first, then the rest by name.

    Ordered by *name*, never by kind, so the C axis of a cell does not shift when
    auto-label promotion turns a mask into a label.
    """
    membrane_key = f"mask:{dataset.membrane_name}"
    others = sorted(k for k in dataset.entities if k != membrane_key)
    return ([membrane_key] if membrane_key in dataset.entities else []) + others


class CellLoader:
    """Load a cell folder (source + label/mask entities) as one CZYX record."""

    NAME = "cellsketch"
    DESCRIPTION = (
        "Loads a cell folder — a source image plus its <prefix>_<name>_label/_mask volumes — "
        "as one record with the entity volumes stacked along C."
    )

    # A cell is a folder, never a file: nothing is loaded by extension, and a cell
    # folder carries no suffix to declare in FOLDER_EXTENSIONS — is_folder_supported
    # recognises one by what is inside it.
    SUPPORTED_EXTENSIONS: Set[str] = set()
    FOLDER_EXTENSIONS: Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = set()

    OUTPUT_SCHEMA: Dict[str, Any] = {
        **RASTER_IMAGE_LOADER_SCHEMA,
        "cell_id": str,
        "membrane_name": str,
        "entity_kinds": list,
        "n_entities": int,
        "cell_shape_zyx": list,
        "voxel_size_source": str,
        "cell_center_z_um": float,
        "cell_center_y_um": float,
        "cell_center_x_um": float,
    }
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {
        "cell_id": "Name of the cell folder the entity volumes were read from.",
        "membrane_name": "Entity name of the plasma-membrane mask that defines the cell.",
        "entity_kinds": "Kind ('label' or 'mask') of each entity, in channel_names order.",
        "n_entities": "Number of entity volumes stacked along C for this cell.",
        "cell_shape_zyx": "Voxel extent (Z, Y, X) of the analysed volume after cropping to the membrane bounding box.",
        "voxel_size_source": "Where the voxel size came from: 'tiff-metadata' or 'config'.",
        "cell_center_z_um": "Z coordinate in µm of the plasma-membrane centroid, the origin every polarity metric is measured from.",
        "cell_center_y_um": "Y coordinate in µm of the plasma-membrane centroid.",
        "cell_center_x_um": "X coordinate in µm of the plasma-membrane centroid.",
    }
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    def __init__(self) -> None:
        self._config = CellSketchConfig.from_env()

    def is_folder_supported(self, path: Path) -> bool:
        """True for a folder holding a source image and at least one label/mask volume.

        Called for every directory while scanning, so it must stay cheap: one glob of
        the folder's TIFF names, no pixel data. Folders that hold cells rather than
        being one (a group folder, the batch root) have no entity files of their own
        and are walked into as usual.
        """
        if not path.is_dir():
            return False
        d = _cached_inspect(str(path))
        return d.source is not None and bool(d.entities)

    def read_header(self, cell_dir: Path) -> FileInfo:
        """Shape/dtype of the stacked cell record, for task routing only.

        The reported extent is the *uncropped* source extent times the entity count —
        an over-estimate once load() crops to the membrane bounding box, which is the
        safe direction for PixelPatrol's memory budget.
        """
        d = _cached_inspect(str(cell_dir))
        if d.source is None:
            raise ValueError(f"{cell_dir}: no source image found")
        (nz, ny, nx), _ = _source_header(d.source)
        n_entities = max(1, len(d.entities))
        return FileInfo(
            shape=(n_entities, nz, ny, nx),
            dtype=np.dtype("int32"),
            dim_order="CZYX",
            n_images=1,
        )

    def load(self, cell_dir: Path) -> Record:
        """Return one record holding every entity volume of this cell folder."""
        dataset = discover_dataset(cell_dir)
        cfg = self._config

        if cfg.voxel_size_um is not None:
            voxel_size_zyx = cfg.voxel_size_um
            voxel_size_source = "config"
        else:
            voxel_size_zyx = infer_voxel_size_um_from_source(dataset.source)
            voxel_size_source = "tiff-metadata"

        membrane_key = f"mask:{dataset.membrane_name}"
        volumes = {key: load_volume(entity.path) for key, entity in dataset.entities.items()}

        source_shape, source_dtype = _source_header(dataset.source)
        for key, vol in volumes.items():
            if vol.shape != source_shape:
                raise ValueError(
                    f"{cell_dir.name}: entity '{dataset.entities[key].name}' has shape {vol.shape}, "
                    f"source image has {source_shape}"
                )

        if cfg.auto_clip_to_pm:
            membrane = volumes[membrane_key] > 0
            for key in volumes:
                if key != membrane_key:
                    volumes[key] = clip_to_membrane(volumes[key], membrane)

        volumes = crop_to_membrane_bbox(volumes, membrane_key)
        if cfg.auto_label_masks:
            promote_multicomponent_masks(volumes, dataset.entities, membrane_key)

        keys = _channel_order(dataset)
        stack = _stack_narrowest(volumes, keys, cell_dir.name)

        meta: Dict[str, Any] = {
            "dim_order": "CZYX",
            "dim_names": ["C", "Z", "Y", "X"],
            "shape": list(stack.shape),
            "ndim": 4,
            "dtype": str(stack.dtype),
            "n_images": 1,
            "channel_names": [dataset.entities[k].name for k in keys],
            "entity_kinds": [dataset.entities[k].kind for k in keys],
            "n_entities": len(keys),
            "cell_id": cell_dir.name,
            "membrane_name": dataset.membrane_name,
            "cell_shape_zyx": list(stack.shape[1:]),
            "voxel_size_source": voxel_size_source,
            "pixel_size_Z": float(voxel_size_zyx[0]),
            "pixel_size_Y": float(voxel_size_zyx[1]),
            "pixel_size_X": float(voxel_size_zyx[2]),
        }
        # The membrane centroid is the origin for every polarity metric. Computed here,
        # once, so a processor that sees only one entity can still measure against it.
        # Read from the stack: _stack_narrowest has released the source volumes by now.
        center = cell_center_um(stack[keys.index(membrane_key)], voxel_size_zyx)
        if center is not None:
            meta["cell_center_z_um"], meta["cell_center_y_um"], meta["cell_center_x_um"] = center
        logger.info(
            "cellsketch: %s — %d entities, %s voxels as %s (%.1f GB), "
            "voxel size (z,y,x) µm: %.4g, %.4g, %.4g",
            cell_dir.name, len(keys), "×".join(str(s) for s in stack.shape[1:]),
            stack.dtype, stack.nbytes / 1024**3, *voxel_size_zyx,
        )
        return record_from(stack, meta, kind=CELL_KIND)

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Not used: a cell folder is a single record, never a container."""
        raise NotImplementedError("CellLoader does not support container files")
