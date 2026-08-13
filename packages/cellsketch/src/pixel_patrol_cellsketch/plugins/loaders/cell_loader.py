"""Loader that turns one cell folder into one PixelPatrol record.

A cell folder holds a source image plus its entity volumes
(``<prefix>_<name>_label.tif`` / ``_mask.tif``). PixelPatrol discovers *files*, so
this loader is driven by the source TIFF and picks up its siblings: the record it
returns is all entity volumes stacked along a C axis (``CZYX``), with
``channel_names`` naming the entities.

Two consequences of that shape, both deliberate:

* one row per cell at ``obs_level=0`` and one row per entity at ``obs_level=1``
  (``dim_c``) — process with ``--slice-size C=1 --slice-size Z=-1`` so a leaf block
  is one whole entity volume;
* every entity of a cell is in one record, so cross-entity metrics (distances,
  contacts) can be computed by a single processor.

The entity TIFFs are discovered as files too, and are declined by returning
``None`` from ``load()``. That is the only batch-safe way to opt out: raising
inside ``load()`` fails the *whole* task, taking the sibling source image with it,
whereas ``None`` is skipped per file (``processing._execute_batch_task``).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import tifffile
from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import Record, record_from

from pixel_patrol_cellsketch.config import CellSketchConfig
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

# Shape a declined (entity/ignored) file reports from read_header. It is a routing
# hint only — load() returns None for these files, so the number is never used for
# anything but keeping them on the cheap batch path.
_DECLINED_INFO = FileInfo(shape=(1, 1, 1), dtype=np.dtype("uint8"), dim_order="ZYX", n_images=1)


@lru_cache(maxsize=256)
def _cached_inspect(cell_dir: str):
    """Cache discovery per folder: every TIFF in a cell folder asks the same question."""
    return inspect_cell_dir(Path(cell_dir))


def _is_source(file_path: Path) -> bool:
    """True when this file is the source image of its cell folder (the record anchor)."""
    d = _cached_inspect(str(file_path.parent))
    return d.source is not None and d.source == file_path


def _source_header(source_path: Path) -> Tuple[Tuple[int, int, int], str]:
    with tifffile.TiffFile(source_path) as tf:
        series = tf.series[0]
        shape = tuple(int(s) for s in series.shape)
        dtype = str(np.dtype(series.dtype))
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D source volume in {source_path.name}, got shape {shape}")
    return shape, dtype  # type: ignore[return-value]


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

    SUPPORTED_EXTENSIONS: Set[str] = {"tif", "tiff"}
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
    }
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {
        "cell_id": "Name of the cell folder the entity volumes were read from.",
        "membrane_name": "Entity name of the plasma-membrane mask that defines the cell.",
        "entity_kinds": "Kind ('label' or 'mask') of each entity, in channel_names order.",
        "n_entities": "Number of entity volumes stacked along C for this cell.",
        "cell_shape_zyx": "Voxel extent (Z, Y, X) of the analysed volume after cropping to the membrane bounding box.",
        "voxel_size_source": "Where the voxel size came from: 'tiff-metadata' or 'config'.",
    }
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    def __init__(self) -> None:
        self._config = CellSketchConfig.from_env()

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        """Shape/dtype of the stacked cell record, for task routing only.

        The reported extent is the *uncropped* source extent times the entity count —
        an over-estimate once load() crops to the membrane bounding box, which is the
        safe direction for PixelPatrol's memory budget.
        """
        if not _is_source(file_path):
            return _DECLINED_INFO
        d = _cached_inspect(str(file_path.parent))
        (nz, ny, nx), _ = _source_header(file_path)
        n_entities = max(1, len(d.entities))
        return FileInfo(
            shape=(n_entities, nz, ny, nx),
            dtype=np.dtype("int32"),
            dim_order="CZYX",
            n_images=1,
        )

    def load(self, file_path: Path) -> Optional[Record]:
        """Return the cell record for a source image, or None for any other TIFF."""
        if not _is_source(file_path):
            return None

        cell_dir = file_path.parent
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
        stack = np.stack([volumes[k].astype(np.int32, copy=False) for k in keys], axis=0)

        meta: Dict[str, Any] = {
            "dim_order": "CZYX",
            "dim_names": ["C", "Z", "Y", "X"],
            "shape": list(stack.shape),
            "ndim": 4,
            "dtype": "int32",
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
        logger.info(
            "cellsketch: %s — %d entities, %s voxels, voxel size (z,y,x) µm: %.4g, %.4g, %.4g "
            "(source dtype %s)",
            cell_dir.name, len(keys), "×".join(str(s) for s in stack.shape[1:]),
            *voxel_size_zyx, source_dtype,
        )
        return record_from(stack, meta, kind=CELL_KIND)

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Not used: a cell folder is a single record, never a container."""
        raise NotImplementedError("CellLoader does not support container files")
