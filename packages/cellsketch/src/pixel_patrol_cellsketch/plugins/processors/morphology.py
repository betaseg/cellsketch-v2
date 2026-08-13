"""Per-entity morphology processor.

One leaf block is one entity volume of one cell (process with
``--slice-size C=1 --slice-size Z=-1``), so this processor emits one row per entity per
cell at ``obs_level=1``, keyed by ``dim_c`` → ``channel_names``. Those rows are what
makes ``entity_name`` a groupable column in the viewer, with scalar metrics PixelPatrol's
own widgets can plot.

A leaf sees only its own entity, which bounds what belongs here: whole-structure
morphology for masks, and counts and totals for labels. Everything that needs the rest
of the cell - per-instance measurements, distances, contacts - is measured by the
cell-level (MEMORY) processors instead.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import numpy as np
from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec

from pixel_patrol_cellsketch.distances import polarity_from_offset
from pixel_patrol_cellsketch.geometry import (
    aspect_ratio_from_coords,
    estimate_surface_area_um2,
    sphericity,
)
from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND
from pixel_patrol_cellsketch.plugins.processors.instances import null_if_not_finite

logger = logging.getLogger(__name__)

_COLUMNS: Dict[str, Any] = {
    "entity_name":              str,
    "entity_kind":              str,
    "instance_count":           np.int64,
    "total_volume_um3":         np.float64,
    "volume_um3":               np.float64,
    "surface_area_um2":         np.float64,
    "sphericity":               np.float64,
    "aspect_ratio_major_minor": np.float64,
    "polar_dist_um":            np.float64,
    "polar_az_deg":             np.float64,
    "polar_el_deg":             np.float64,
    "polar_nz":                 np.float64,
    "polar_ny":                 np.float64,
    "polar_nx":                 np.float64,
    "file_name":                str,
    "file_size_bytes":          np.int64,
}

_DESCRIPTIONS: Dict[str, str] = {
    "entity_name":              "Name of the entity (organelle / structure) this row measures.",
    "entity_kind":              "'mask' for a single whole structure, 'label' for an instance-segmented entity.",
    "instance_count":           "Number of labelled instances in this entity; null for whole-structure masks, and summed across the cell's label entities on the cell row.",
    "total_volume_um3":         "Total segmented volume of this entity in µm³.",
    "volume_um3":               "Volume of the structure in µm³ (mask entities).",
    "surface_area_um2":         "Voxel-face surface area of the structure in µm² (mask entities).",
    "sphericity":               "Sphericity of the structure, 1 = perfect sphere (mask entities).",
    "aspect_ratio_major_minor": "Ratio of largest to smallest PCA axis length (mask entities).",
    "polar_dist_um":            "Distance in µm from the cell centre to this structure's centroid (mask entities).",
    "polar_az_deg":             "Azimuth in degrees of this structure's centroid as seen from the cell centre (mask entities).",
    "polar_el_deg":             "Elevation in degrees of this structure's centroid as seen from the cell centre (mask entities).",
    "polar_nz":                 "Z component of the unit vector from the cell centre to this structure's centroid.",
    "polar_ny":                 "Y component of the unit vector from the cell centre to this structure's centroid.",
    "polar_nx":                 "X component of the unit vector from the cell centre to this structure's centroid.",
    "file_name":                "Name of the TIFF this entity was read from.",
    "file_size_bytes":          "Size on disk of that TIFF, in bytes.",
}

# Only the instance count adds up across entities: it counts labelled objects, and mask
# entities leave it null. Volumes deliberately do not - a cell's entity volumes overlap
# (every organelle sits inside the membrane), so their sum means nothing.
_SUMMED = {"instance_count"}


def _make_passthrough(col: str) -> Callable[[List[Dict], Dict[str, Any]], Any]:
    """The single row's value, or None when the group spans several entities.

    Per-entity morphology has no meaningful cell-level aggregate: a cell's "sphericity"
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


class MorphologyProcessor:
    """Whole-structure morphology for a mask entity; counts and totals for a label entity."""

    NAME = "cellsketch-morphology"
    DESCRIPTION = (
        "Computes per-entity morphology of a cell: volume, surface area, sphericity and PCA "
        "aspect ratio for whole-structure masks, and instance count and total volume for "
        "instance-segmented label entities."
    )

    CHUNK_KIND = ChunkKind.LEAF
    INPUT = RecordSpec(axes={"Z", "Y", "X"}, kinds={CELL_KIND})
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
        expected_zyx = [int(v) for v in (meta.get("cell_shape_zyx") or [])]
        if expected_zyx and list(volume.shape) != expected_zyx:
            # A fragment gives wrong answers, not partial ones, so refuse it - and name
            # the cause: an axis sliced down to 1 is the leaf configuration, anything
            # else is PixelPatrol splitting the volume to fit its memory budget.
            sliced = [
                ax for ax, got, want in zip("ZYX", volume.shape, expected_zyx)
                if got == 1 and want > 1
            ]
            cause = (
                f"pass --slice-size {' '.join(f'{ax}=-1' for ax in sliced)} so a leaf block "
                "is one whole entity volume"
                if sliced else
                "raise --mb-per-task above the size of one cell"
            )
            raise ValueError(
                f"entity '{names[c_index]}' arrived as a {volume.shape} fragment of a "
                f"{tuple(expected_zyx)} volume — {cause}"
            )

        voxel_size_zyx = (
            float(meta["pixel_size_Z"]),
            float(meta["pixel_size_Y"]),
            float(meta["pixel_size_X"]),
        )
        entity_name, entity_kind = names[c_index], kinds[c_index]
        voxel_um3 = float(np.prod(voxel_size_zyx))

        row: Dict[str, Any] = {"entity_name": entity_name, "entity_kind": entity_kind}
        # Provenance per entity: which file this row came from, and how big it was. The
        # record is a folder, so PixelPatrol's own name/size_bytes describe the whole cell.
        files = list(meta.get("entity_files") or [])
        sizes = list(meta.get("entity_file_bytes") or [])
        if c_index < len(files):
            row["file_name"] = files[c_index]
        if c_index < len(sizes):
            row["file_size_bytes"] = int(sizes[c_index])
        if entity_kind == "mask":
            binary = volume > 0
            vol_um3 = float(binary.sum() * voxel_um3)
            area_um2 = estimate_surface_area_um2(binary, voxel_size_zyx)
            coords = np.argwhere(binary)
            row.update({
                # instance_count stays absent: a mask is one structure, not one instance,
                # and leaving it null keeps the cell-row sum a count of label instances.
                "total_volume_um3": vol_um3,
                "volume_um3": vol_um3,
                "surface_area_um2": area_um2,
                "sphericity": sphericity(vol_um3, area_um2),
                "aspect_ratio_major_minor": (
                    aspect_ratio_from_coords(coords, voxel_size_zyx) if coords.size else float("nan")
                ),
            })
            # Where this structure sits relative to the cell: the loader put the membrane
            # centroid in the record, so even a leaf that sees one entity can measure it.
            center = [meta.get(f"cell_center_{ax}_um") for ax in "zyx"]
            if coords.size and all(c is not None for c in center):
                centroid_um = coords.mean(axis=0) * np.array(voxel_size_zyx)
                row.update(polarity_from_offset(*(centroid_um - np.array(center, dtype=float))))
        else:
            labels = volume.astype(np.int32, copy=False)
            row.update({
                "instance_count": int(np.unique(labels[labels > 0]).size),
                "total_volume_um3": float((labels > 0).sum() * voxel_um3),
            })
        # NaN would break the very widgets these scalars exist for: see null_if_not_finite.
        return {key: null_if_not_finite(value) for key, value in row.items()}

    def get_aggregation(self, name: str):
        if name in _SUMMED:
            return _make_sum(name)
        if name in _COLUMNS:
            return _make_passthrough(name)
        return None
