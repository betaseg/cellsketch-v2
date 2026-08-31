"""Per-entity morphology processor.

One leaf block is one entity volume of one object (process with
``--slice-size C=1 --slice-size Z=-1``), so this processor emits one row per entity per
object at ``obs_level=1``, keyed by ``dim_c`` → ``channel_names``. Those rows are what
makes ``entity_name`` a groupable column in the viewer, with scalar metrics PixelPatrol's
own widgets can plot.

A leaf sees only its own entity, which bounds what belongs here: whole-structure
morphology for masks, and counts and totals for labels. Everything that needs the rest
of the object - per-instance measurements, distances, contacts - is measured by the
object-level (MEMORY) processors instead.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec

from pixel_patrol_anatomy.distances import polarity_from_offset
from pixel_patrol_anatomy.geometry import size_key, total_size_key
from pixel_patrol_anatomy.skeletons import region_metrics_for
from pixel_patrol_anatomy.plugins.loaders.object_loader import OBJECT_KIND
from pixel_patrol_anatomy.spatial import (
    object_center,
    spatial_axes,
    unit_of_measure,
    voxel_size,
)
from pixel_patrol_anatomy.plugins.processors.instances import null_if_not_finite

logger = logging.getLogger(__name__)

# Both dimensionalities are declared so one report can hold objects of either kind; each
# object fills its own set and leaves the other null.
_COLUMNS: Dict[str, Any] = {
    "entity_name":              str,
    "entity_kind":              str,
    "entity_colour":            str,
    "instance_count":           np.int64,
    # 3D
    "total_volume_um3":         np.float64,
    "volume_um3":               np.float64,
    "surface_area_um2":         np.float64,
    "sphericity":               np.float64,
    # 2D
    "total_area_um2":           np.float64,
    "area_um2":                 np.float64,
    "perimeter_um":             np.float64,
    "circularity":              np.float64,
    # both
    "aspect_ratio_major_minor": np.float64,
    "polar_dist_um":            np.float64,
    "polar_ny":                 np.float64,
    "polar_nx":                 np.float64,
    # 3D polarity
    "polar_az_deg":             np.float64,
    "polar_el_deg":             np.float64,
    "polar_nz":                 np.float64,
    # 2D polarity
    "polar_angle_deg":          np.float64,
    "file_name":                str,
    "file_size_bytes":          np.int64,
}

_DESCRIPTIONS: Dict[str, str] = {
    "entity_name":              "Name of the entity (organelle / structure) this row measures.",
    "entity_colour":            "Colour this structure is drawn in, as #rrggbb, from the settings file given to --colours (or `anatomy colours` afterwards). Null where the file did not name it, and the widgets fall back to their own palette.",
    "entity_kind":              "'mask' for a single whole structure, 'label' for an instance-segmented entity.",
    "instance_count":           "Number of labelled instances in this entity; null for whole-structure masks, and summed across the object's label entities on the object row.",
    "total_volume_um3":         "Total segmented volume of this entity in µm³ (3D objects).",
    "volume_um3":               "Volume of the structure in µm³ (3D mask entities).",
    "surface_area_um2":         "Surface area of the structure in µm², counted over voxel faces (3D mask entities). A staircase estimate: ~1.5× the area of the smooth surface it approximates.",
    "sphericity":               "Sphericity of the structure from its voxel-face surface area (3D mask entities). A voxelised sphere reads ~0.67, not 1, so compare values rather than reading 1 as round.",
    "total_area_um2":           "Total segmented area of this entity in µm² (2D objects).",
    "area_um2":                 "Area of the structure in µm² (2D mask entities).",
    "perimeter_um":             "Perimeter of the structure in µm, counted over pixel edges (2D mask entities). A staircase estimate: ~1.3× the smooth boundary.",
    "circularity":              "Circularity of the structure, 4πA/P², from its pixel-edge perimeter (2D mask entities). A voxelised disc reads ~0.58, not 1.",
    "aspect_ratio_major_minor": "Ratio of largest to smallest PCA axis length (mask entities).",
    "polar_dist_um":            "Distance in µm from the object centre to this structure's centroid (mask entities).",
    "polar_az_deg":             "Azimuth in degrees of this structure's centroid as seen from the object centre (3D mask entities).",
    "polar_el_deg":             "Elevation in degrees of this structure's centroid as seen from the object centre (3D mask entities).",
    "polar_angle_deg":          "Angle in degrees of this structure's centroid as seen from the object centre (2D mask entities); a plane has one angle, not an azimuth and an elevation.",
    "polar_nz":                 "Z component of the unit vector from the object centre to this structure's centroid (3D).",
    "polar_ny":                 "Y component of the unit vector from the object centre to this structure's centroid.",
    "polar_nx":                 "X component of the unit vector from the object centre to this structure's centroid.",
    "file_name":                "Name of the TIFF this entity was read from.",
    "file_size_bytes":          "Size on disk of that TIFF, in bytes.",
}

# Only the instance count adds up across entities. Extents do not: an object's structures
# all sit inside the object mask, so summing them means nothing.
_SUMMED = {"instance_count"}


def _make_passthrough(col: str) -> Callable[[List[Dict], Dict[str, Any]], Any]:
    """The single row's value, or None when the group spans several entities.

    Per-entity morphology has no meaningful object-level aggregate: an object's "sphericity"
    is not the sphericity of its entities.
    """
    def agg(rows: List[Dict], _g_dims: Dict[str, Any]) -> Any:
        return rows[0].get(col) if len(rows) == 1 else None
    return agg


def _make_sum(col: str) -> Callable[[List[Dict], Dict[str, Any]], Any]:
    def agg(rows: List[Dict], _g_dims: Dict[str, Any]) -> Any:
        vals = [r[col] for r in rows if r.get(col) is not None]
        return sum(vals) if vals else None
    return agg


def _foreground_centroid(binary: np.ndarray) -> Optional[np.ndarray]:
    """Mean position of the foreground, in samples, in array order. None if there is none.

    One weighted reduction per axis, not ``np.argwhere``: a whole-object mask holds tens of
    millions of samples, and listing their coordinates to take one mean costs gigabytes.
    Measured on a 53-Mvoxel mask, the coordinate array alone peaked at 2.5 GB.
    """
    total = int(binary.sum())
    if not total:
        return None
    return np.array([
        float((binary.sum(axis=tuple(j for j in range(binary.ndim) if j != axis))
               * np.arange(binary.shape[axis])).sum()) / total
        for axis in range(binary.ndim)
    ])


class MorphologyProcessor:
    """Whole-structure morphology for a mask entity; counts and totals for a label entity."""

    NAME = "anatomy-morphology"
    DESCRIPTION = (
        "Computes per-entity morphology of an object: volume, surface area, sphericity and PCA "
        "aspect ratio for whole-structure masks, and instance count and total volume for "
        "instance-segmented label entities."
    )

    CHUNK_KIND = ChunkKind.LEAF
    INPUT = RecordSpec(axes={"Y", "X"}, kinds={OBJECT_KIND})
    OUTPUT = "features"

    OUTPUT_SCHEMA: Dict[str, Any] = dict(_COLUMNS)
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = dict(_DESCRIPTIONS)

    def run_chunk(self, record: Record) -> Dict[str, Any]:
        meta = record.meta
        arr = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)

        c_axis = record.dim_order.index("C")
        c_index = int(meta.get("dim_c") or 0)
        names = list(meta.get("channel_names") or [])
        kinds = list(meta.get("entity_kinds") or [])
        if c_index >= len(names) or c_index >= len(kinds):
            raise ValueError(
                f"channel {c_index} has no entity metadata (channel_names={names}, entity_kinds={kinds})"
            )

        volume = np.squeeze(arr, axis=c_axis)
        axes = spatial_axes(record.dim_order)
        expected = [int(v) for v in (meta.get("object_shape") or [])]
        if expected and list(volume.shape) != expected:
            # A fragment gives wrong answers, not partial ones. An axis sliced to 1 is a
            # leaf-configuration mistake; anything else is the memory budget.
            sliced = [
                ax for ax, got, want in zip(axes, volume.shape, expected)
                if got == 1 and want > 1
            ]
            cause = (
                f"pass --slice-size {' '.join(f'{ax}=-1' for ax in sliced)} so a leaf block "
                "is one whole entity volume"
                if sliced else
                "an object is measured whole, so a fragment means the caller split it"
            )
            raise ValueError(
                f"entity '{names[c_index]}' arrived as a {volume.shape} fragment of a "
                f"{tuple(expected)} {unit_of_measure(len(expected))}: {cause}"
            )

        sample_size = voxel_size(meta, record.dim_order)
        entity_name, entity_kind = names[c_index], kinds[c_index]
        sample_extent = float(np.prod(sample_size))

        row: Dict[str, Any] = {"entity_name": entity_name, "entity_kind": entity_kind}
        # Which file this entity came from: the record is a folder, so PixelPatrol's own
        # name/size_bytes describe the whole object, not this entity.
        files = list(meta.get("entity_files") or [])
        sizes = list(meta.get("entity_file_bytes") or [])
        if c_index < len(files):
            row["file_name"] = files[c_index]
        if c_index < len(sizes):
            row["file_size_bytes"] = int(sizes[c_index])
        if entity_kind == "mask":
            binary = volume > 0
            metrics = region_metrics_for(
                str(meta.get("object_id") or "object"), entity_name, volume, sample_size)
            centroid = _foreground_centroid(binary)
            # instance_count stays null: a mask is one structure, not one instance, which
            # keeps the object-row sum a count of label instances.
            row.update(metrics)
            row[total_size_key(volume.ndim)] = metrics[size_key(volume.ndim)]
            # The loader recorded the object mask's centroid, so a leaf that sees one
            # entity can still measure where it sits.
            center = object_center(meta, record.dim_order)
            if centroid is not None and center is not None:
                centroid_um = centroid * np.array(sample_size)
                row.update(polarity_from_offset(centroid_um - np.array(center, dtype=float)))
        else:
            labels = volume.astype(np.int32, copy=False)
            row.update({
                "instance_count": int(np.unique(labels[labels > 0]).size),
                total_size_key(volume.ndim): float((labels > 0).sum() * sample_extent),
            })
        # NaN would break the very widgets these scalars exist for: see null_if_not_finite.
        return {key: null_if_not_finite(value) for key, value in row.items()}

    def get_aggregation(self, name: str):
        if name in _SUMMED:
            return _make_sum(name)
        if name in _COLUMNS:
            return _make_passthrough(name)
        return None
