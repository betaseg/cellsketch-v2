"""Loader that turns one object folder into one PixelPatrol record.

The unit of analysis is a *folder*: a source image plus its entity volumes
(``<prefix>_<name>_label.tif`` / ``_mask.tif``). ``is_folder_supported`` claims such
a folder, so PixelPatrol hands the directory itself to ``read_header`` / ``load``
and never descends into it, so the TIFFs inside are not records of their own.

``load`` stacks every entity along a C axis, ``CZYX`` for a volume and ``CYX`` for a plane,
with ``channel_names`` naming them. That shape gives:

* one row per object at ``obs_level=0``, one row per entity at ``obs_level=1`` (``dim_c``);
  process with ``--slice-size C=1 --slice-size Z=-1`` so a leaf block is one entity volume;
* every entity of an object in one record, so cross-entity metrics (distances, contacts)
  can be computed by a single processor.
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

from pixel_patrol_anatomy.config import AnatomyConfig
from pixel_patrol_anatomy.distances import object_center_um
from pixel_patrol_anatomy.discovery import (
    Dataset,
    clip_to_object_mask,
    crop_to_object_bbox,
    discover_dataset,
    infer_voxel_size_um_from_source,
    inspect_object_dir,
    load_volume,
    promote_multicomponent_masks,
)

logger = logging.getLogger(__name__)

# Not "intensity": PixelPatrol's raster processors declare kinds={"intensity"} and so skip
# these records, which are label maps where pixel statistics mean nothing.
OBJECT_KIND = "object/segmentation"


@lru_cache(maxsize=256)
def _cached_inspect(object_dir: str, object_mask: str | None = None):
    """Cache discovery per folder: discovery, read_header and load all ask for it.

    Keyed on the object mask too, so naming a different boundary is not served a stale
    answer from a warm process.
    """
    return inspect_object_dir(Path(object_dir), object_mask)


def _source_header(source_path: Path) -> Tuple[Tuple[int, ...], str]:
    """Spatial shape and dtype of the source image: (Z, Y, X) or (Y, X)."""
    with tifffile.TiffFile(source_path) as tf:
        series = tf.series[0]
        shape = tuple(int(s) for s in series.shape)
        dtype = str(np.dtype(series.dtype))
    if len(shape) not in (2, 3):
        raise ValueError(
            f"Expected a 2D or 3D source image in {source_path.name}, got shape {shape}"
        )
    return shape, dtype


def _label_dtype(volumes: Dict[str, np.ndarray], object_id: str) -> np.dtype:
    """The narrowest integer type that holds every label id in this object.

    Segmentations are often stored as float32 or int32 whatever their ids need, and the
    stack is the single largest allocation in a run: a 371×1257×1176 object with five
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
                    f"{object_id}: entity '{name}' has non-integer values, so it is not a "
                    "label or mask volume"
                )
        lo, hi = float(vol.min()), float(vol.max())
        if lo < 0:
            raise ValueError(f"{object_id}: entity '{name}' has negative label ids ({lo})")
        highest = max(highest, hi)
    for dtype in (np.uint8, np.uint16, np.uint32):
        if highest <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    return np.dtype(np.uint64)


def _stack_narrowest(volumes: Dict[str, np.ndarray], keys: List[str], object_id: str) -> np.ndarray:
    """Stack the entities along C, converting one at a time and freeing as we go.

    Written channel by channel into a preallocated array rather than via np.stack, so the
    source volumes and the stack are never both fully in memory.
    """
    dtype = _label_dtype(volumes, object_id)
    first = volumes[keys[0]]
    stack = np.empty((len(keys), *first.shape), dtype=dtype)
    for i, key in enumerate(keys):
        np.copyto(stack[i], volumes[key], casting="unsafe")
        volumes[key] = np.empty((0,) * first.ndim, dtype=dtype)  # release the source volume
    return stack


SPATIAL_AXES_3D = "ZYX"
SPATIAL_AXES_2D = "YX"


def _config_voxel_size(configured: Tuple[float, ...], ndim: int, object_dir: Path) -> Tuple[float, ...]:
    """The configured sample size, checked against the data's dimensionality.

    ``--voxel-size-um`` takes ``z,y,x`` for volumes, ``y,x`` for planes. A mismatch is
    refused rather than padded or truncated, which would rescale every measurement.
    """
    if len(configured) == ndim:
        return tuple(float(v) for v in configured)
    wanted = "z,y,x" if ndim == 3 else "y,x"
    raise ValueError(
        f"{object_dir.name}: images are {ndim}D, so --voxel-size-um needs '{wanted}', "
        f"got {len(configured)} value(s): {','.join(str(v) for v in configured)}"
    )


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _channel_order(dataset: Dataset) -> List[str]:
    """Entity keys in C order: the object mask first, then the rest by name.

    Ordered by *name*, never by kind, so the C axis of an object does not shift when
    auto-label promotion turns a mask into a label.
    """
    object_mask_key = f"mask:{dataset.object_mask_name}"
    others = sorted(k for k in dataset.entities if k != object_mask_key)
    return ([object_mask_key] if object_mask_key in dataset.entities else []) + others


class ObjectLoader:
    """Load an object folder (source + label/mask entities) as one CZYX record."""

    NAME = "anatomy"
    DESCRIPTION = (
        "Loads an object folder: a source image plus its <prefix>_<name>_label/_mask volumes, "
        "as one record with the entity volumes stacked along C."
    )

    # An object is a folder, never a file, and folders have no extension to declare:
    # is_folder_supported recognises one by what is inside it.
    SUPPORTED_EXTENSIONS: Set[str] = set()
    FOLDER_EXTENSIONS: Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = set()

    OUTPUT_SCHEMA: Dict[str, Any] = {
        **RASTER_IMAGE_LOADER_SCHEMA,
        "object_id": str,
        "object_mask_name": str,
        "entity_kinds": list,
        "entity_files": list,
        "entity_file_bytes": list,
        "n_entities": int,
        "object_shape": list,
        "spatial_dims": int,
        "voxel_size_source": str,
        "object_center_z_um": float,
        "object_center_y_um": float,
        "object_center_x_um": float,
    }
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {
        "object_id": "Name of the object folder the entity volumes were read from.",
        "object_mask_name": "Entity name of the mask that bounds the object; every measurement is relative to it.",
        "entity_kinds": "Kind ('label' or 'mask') of each entity, in channel_names order.",
        "entity_files": "File each entity was read from, in channel_names order.",
        "entity_file_bytes": "Size on disk of each entity's file, in channel_names order.",
        "n_entities": "Number of entity volumes stacked along C for this object.",
        "object_shape": "Extent of the analysed region after cropping to the object mask's bounding box, in the spatial axes of dim_order: (Z, Y, X) for a volume, (Y, X) for a plane.",
        "spatial_dims": "3 for a volume, 2 for a plane. Which size and shape metrics the object's rows carry follows from it.",
        "voxel_size_source": "Where the voxel size came from: 'tiff-metadata' or 'config'.",
        "object_center_z_um": "Z coordinate in µm of the object mask's centroid, the origin every polarity metric is measured from. Null for a 2D object.",
        "object_center_y_um": "Y coordinate in µm of the object mask's centroid.",
        "object_center_x_um": "X coordinate in µm of the object mask's centroid.",
    }
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    def __init__(self) -> None:
        self._config = AnatomyConfig.from_env()

    def is_folder_supported(self, path: Path) -> bool:
        """True for a folder holding a source image and at least one label/mask volume.

        Called for every directory while scanning, so it stays cheap: one glob of the
        folder's TIFF names, no pixel data. A folder that holds objects rather than being
        one has no entity files of its own and is walked into as usual.
        """
        if not path.is_dir():
            return False
        # Deliberately not passing the configured mask: whether a folder is an object does
        # not depend on which of its masks bounds it, and claiming it here is what makes the
        # missing-mask error reachable at load time instead of skipping the folder.
        d = _cached_inspect(str(path))
        return d.source is not None and bool(d.entities)

    def read_header(self, object_dir: Path) -> FileInfo:
        """Shape/dtype of the record, for task routing only.

        The uncropped source extent times the entity count, as int32. That is 2-5x what load()
        actually returns, since it crops to the object mask's bounding box and narrows the
        dtype. Over-estimating is the *unsafe* direction here: PixelPatrol splits a record
        that exceeds its budget, and the object-level processors cannot use a fragment. The
        CLI sizes its worker count from the same over-estimate, so the two agree.
        """
        d = _cached_inspect(str(object_dir), self._config.object_mask)
        if d.source is None:
            raise ValueError(f"{object_dir}: no source image found")
        spatial, _ = _source_header(d.source)
        n_entities = max(1, len(d.entities))
        return FileInfo(
            shape=(n_entities, *spatial),
            dtype=np.dtype("int32"),
            # A plane is CYX, a volume CZYX: the processors read the record's own axes, so
            # a 2D object is measured as one rather than as a volume one voxel deep.
            dim_order="CZYX" if len(spatial) == 3 else "CYX",
            n_images=1,
        )

    def load(self, object_dir: Path) -> Record:
        """Return one record holding every entity volume of this object folder.

        A 2D folder becomes a CYX record, a 3D one CZYX. Nothing else differs between them.
        """
        cfg = self._config
        dataset = discover_dataset(object_dir, cfg.object_mask)

        source_shape, source_dtype = _source_header(dataset.source)
        ndim = len(source_shape)

        if cfg.voxel_size_um is not None:
            voxel_size = _config_voxel_size(cfg.voxel_size_um, ndim, object_dir)
            voxel_size_source = "config"
        else:
            voxel_size = infer_voxel_size_um_from_source(dataset.source, ndim)
            voxel_size_source = "tiff-metadata"

        object_mask_key = f"mask:{dataset.object_mask_name}"
        volumes = {key: load_volume(entity.path) for key, entity in dataset.entities.items()}

        for key, vol in volumes.items():
            if vol.shape != source_shape:
                raise ValueError(
                    f"{object_dir.name}: entity '{dataset.entities[key].name}' has shape {vol.shape}, "
                    f"source image has {source_shape}"
                )

        if cfg.auto_clip:
            inside = volumes[object_mask_key] > 0
            for key in volumes:
                if key != object_mask_key:
                    volumes[key] = clip_to_object_mask(volumes[key], inside)

        volumes = crop_to_object_bbox(volumes, object_mask_key)
        if cfg.auto_label_masks:
            promote_multicomponent_masks(volumes, dataset.entities, object_mask_key)

        keys = _channel_order(dataset)
        stack = _stack_narrowest(volumes, keys, object_dir.name)

        # Only what the record cannot derive for itself: PixelPatrol stamps dim_order,
        # dim_names, ndim, dtype and size_<axis> from the array it is handed.
        axes = SPATIAL_AXES_3D if ndim == 3 else SPATIAL_AXES_2D
        meta: Dict[str, Any] = {
            "dim_order": "C" + axes,
            "channel_names": [dataset.entities[k].name for k in keys],
            "entity_kinds": [dataset.entities[k].kind for k in keys],
            "entity_files": [dataset.entities[k].path.name for k in keys],
            "entity_file_bytes": [_file_size(dataset.entities[k].path) for k in keys],
            "n_entities": len(keys),
            "object_id": object_dir.name,
            "object_mask_name": dataset.object_mask_name,
            # A plane's extent is (Y, X); which axes these are is in dim_order.
            "object_shape": list(stack.shape[1:]),
            "spatial_dims": ndim,
            "voxel_size_source": voxel_size_source,
        }
        meta.update({f"pixel_size_{ax}": float(size) for ax, size in zip(axes, voxel_size)})
        # The origin for every polarity metric, computed once here so a leaf processor that
        # sees one entity can still measure against it. From the stack: _stack_narrowest has
        # released the source volumes.
        center = object_center_um(stack[keys.index(object_mask_key)], voxel_size)
        if center is not None:
            meta.update({f"object_center_{ax.lower()}_um": value
                         for ax, value in zip(axes, center)})
        logger.info(
            "anatomy: %s: %d entities, %s %s as %s (%.1f GB), sample size (%s) µm: %s",
            object_dir.name, len(keys), "×".join(str(s) for s in stack.shape[1:]),
            "voxels" if ndim == 3 else "pixels",
            stack.dtype, stack.nbytes / 1024**3, ",".join(axes.lower()),
            ", ".join(f"{v:.4g}" for v in voxel_size),
        )
        return record_from(stack, meta, kind=OBJECT_KIND)

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Not used: an object folder is a single record, never a container."""
        raise NotImplementedError("ObjectLoader does not support container files")
