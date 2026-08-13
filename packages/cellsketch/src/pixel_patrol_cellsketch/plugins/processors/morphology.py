"""Per-entity morphology processor.

One leaf block is one entity volume of one cell (process with
``--slice-size C=1 --slice-size Z=-1``), so this processor emits one row per entity
per cell at ``obs_level=1``, keyed by ``dim_c`` → ``channel_names``.

Mask entities are single structures: their morphology lands in the scalar columns
(``volume_um3``, ``surface_area_um2``, ``sphericity``, ``aspect_ratio_major_minor``).
Label entities are many instances: they report ``instance_count`` /
``total_volume_um3`` as scalars and every per-instance measurement as a same-length
list column (``instance_*``), because PixelPatrol's table has no row granularity
below a dimension slice. Widgets unnest those lists in SQL:

    SELECT cell_id, entity_name, unnest(instance_volume_um3) AS volume_um3
    FROM pp_data WHERE obs_level = 1 AND entity_kind = 'label'
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from skimage.measure import regionprops

from pixel_patrol_cellsketch.config import CellSketchConfig
from pixel_patrol_cellsketch.geometry import (
    aspect_ratio_from_coords,
    compute_curve_skeletons,
    estimate_surface_area_um2,
    skeleton_graph_metrics,
    sphericity,
)
from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND

logger = logging.getLogger(__name__)


# ── Aggregators ───────────────────────────────────────────────────────────────
# Called by processing._rollup for every declared column, at obs_level=1 (one leaf
# row per group — one entity) and at obs_level=0 (all entities of the cell).

def _make_passthrough(col: str) -> Callable[[List[Dict], Dict[str, Any]], Any]:
    """The single row's value, or None when the group spans several entities.

    Per-entity morphology has no meaningful cell-level aggregate: a cell's
    "sphericity" is not the sphericity of its entities, and concatenating instance
    lists across entities would silently mix mitochondria with ER.
    """
    def agg(rows: List[Dict], _g_dims: Dict[str, Any]) -> Any:
        # Keyed on the group size, not on how many rows happen to carry the column:
        # a cell whose only label entity is 'mito' must not have that entity's
        # instance list promoted onto the cell row just because it is the only one.
        return rows[0].get(col) if len(rows) == 1 else None
    return agg


def _make_sum(col: str) -> Callable[[List[Dict], Dict[str, Any]], Any]:
    def agg(rows: List[Dict], _g_dims: Dict[str, Any]) -> Any:
        vals = [r[col] for r in rows if r.get(col) is not None]
        return sum(vals) if vals else None
    return agg


# Only the instance count adds up across entities: it counts labelled objects, and
# mask entities leave it null. Volumes deliberately do not — a cell's entity volumes
# overlap (every organelle sits inside the membrane), so their sum means nothing.
_SUMMED = {"instance_count"}

_SCALAR_COLUMNS: Dict[str, Any] = {
    "entity_name":              str,
    "entity_kind":              str,
    "instance_count":           np.int64,
    "total_volume_um3":         np.float64,
    "volume_um3":               np.float64,
    "surface_area_um2":         np.float64,
    "sphericity":               np.float64,
    "aspect_ratio_major_minor": np.float64,
}

_INSTANCE_COLUMNS: Dict[str, Any] = {
    "instance_label":                    list,
    "instance_volume_um3":               list,
    "instance_surface_area_um2":         list,
    "instance_sphericity":               list,
    "instance_aspect_ratio_major_minor": list,
    "instance_branches":                 list,
    "instance_length_um":                list,
    "instance_tortuosity":               list,
}

_DESCRIPTIONS: Dict[str, str] = {
    "entity_name":              "Name of the entity (organelle / structure) this row measures.",
    "entity_kind":              "'mask' for a single whole structure, 'label' for an instance-segmented entity.",
    "instance_count":           "Number of labelled instances in this entity; null for whole-structure masks, and summed across the cell's label entities on the cell row.",
    "total_volume_um3":         "Total segmented volume of this entity in µm³.",
    "volume_um3":               "Volume of the structure in µm³ (mask entities).",
    "surface_area_um2":         "Voxel-face surface area of the structure in µm² (mask entities).",
    "sphericity":              "Sphericity of the structure, 1 = perfect sphere (mask entities).",
    "aspect_ratio_major_minor": "Ratio of largest to smallest PCA axis length (mask entities).",
    "instance_label":                    "Label id of each instance, in list order shared by all instance_* columns.",
    "instance_volume_um3":               "Per-instance volume in µm³.",
    "instance_surface_area_um2":         "Per-instance voxel-face surface area in µm².",
    "instance_sphericity":               "Per-instance sphericity, 1 = perfect sphere.",
    "instance_aspect_ratio_major_minor": "Per-instance ratio of largest to smallest PCA axis length.",
    "instance_branches":                 "Per-instance number of curve-skeleton branches.",
    "instance_length_um":                "Per-instance curve-skeleton cable length in µm.",
    "instance_tortuosity":               "Per-instance length-weighted branch arc/chord ratio (≥1, 1 = straight).",
}


def _as_list(values: List[Any]) -> Optional[List[Any]]:
    """None rather than [] for empty: an empty list would type the parquet column
    List(Null) and clash with List(Double) written by another part file."""
    return values or None


class MorphologyProcessor:
    """Morphology of one entity volume: whole-structure for masks, per-instance for labels."""

    NAME = "cellsketch-morphology"
    DESCRIPTION = (
        "Computes per-entity morphology of a cell: volume, surface area, sphericity and PCA "
        "aspect ratio for whole-structure masks, and per-instance morphology plus curve-skeleton "
        "metrics (as list columns) for instance-segmented label entities."
    )

    CHUNK_KIND = ChunkKind.LEAF
    INPUT = RecordSpec(axes={"Z", "Y", "X"}, kinds={CELL_KIND})
    OUTPUT = "features"

    OUTPUT_SCHEMA: Dict[str, Any] = {**_SCALAR_COLUMNS, **_INSTANCE_COLUMNS}
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = dict(_DESCRIPTIONS)

    def __init__(self) -> None:
        self._config = CellSketchConfig.from_env()

    def run_chunk(self, record: Record) -> Dict[str, Any]:
        meta = record.meta
        dim_order = record.dim_order
        arr = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)

        c_axis = dim_order.index("C")
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
            # PixelPatrol split the cell spatially to stay inside its memory budget.
            # Instance labels, skeletons and distances are not chunk-aggregatable, so
            # refuse rather than report metrics measured on a fragment.
            raise ValueError(
                f"entity '{names[c_index]}' arrived as a {volume.shape} fragment of a "
                f"{tuple(expected_zyx)} volume — raise --mb-per-task above the size of one cell"
            )

        voxel_size_zyx = (
            float(meta["pixel_size_Z"]),
            float(meta["pixel_size_Y"]),
            float(meta["pixel_size_X"]),
        )
        entity_name, entity_kind = names[c_index], kinds[c_index]

        row: Dict[str, Any] = {"entity_name": entity_name, "entity_kind": entity_kind}
        if entity_kind == "mask":
            row.update(self._mask_metrics(volume, voxel_size_zyx))
        else:
            row.update(self._label_metrics(volume, voxel_size_zyx))
        return row

    def get_aggregation(self, name: str):
        if name in _SUMMED:
            return _make_sum(name)
        if name in self.OUTPUT_SCHEMA:
            return _make_passthrough(name)
        return None

    # ── Metric computation ────────────────────────────────────────────────────

    @staticmethod
    def _mask_metrics(volume: np.ndarray, voxel_size_zyx) -> Dict[str, Any]:
        voxel_um3 = float(np.prod(voxel_size_zyx))
        binary = volume > 0
        vol_um3 = float(binary.sum() * voxel_um3)
        area_um2 = estimate_surface_area_um2(binary, voxel_size_zyx)
        coords = np.argwhere(binary)
        return {
            # instance_count stays absent: a mask is one structure, not one instance,
            # and leaving it null keeps the cell-row sum a count of label instances.
            "total_volume_um3": vol_um3,
            "volume_um3": vol_um3,
            "surface_area_um2": area_um2,
            "sphericity": sphericity(vol_um3, area_um2),
            "aspect_ratio_major_minor": (
                aspect_ratio_from_coords(coords, voxel_size_zyx) if coords.size else float("nan")
            ),
        }

    def _label_metrics(self, volume: np.ndarray, voxel_size_zyx) -> Dict[str, Any]:
        cfg = self._config
        voxel_um3 = float(np.prod(voxel_size_zyx))
        labels = volume.astype(np.int32, copy=False)
        props = regionprops(labels)

        # One TEASAR pass over the whole label volume yields a clean curve skeleton per
        # instance, keyed by label id.
        skels = compute_curve_skeletons(
            labels,
            voxel_size_zyx,
            max_voxels=cfg.max_skeleton_voxels,
            num_threads=cfg.num_threads,
        )

        unmeasured = {"branches": float("nan"), "length_um": float("nan"), "tortuosity": float("nan")}

        ids, volumes_um3, areas, sphericities, aspects = [], [], [], [], []
        branches, lengths, tortuosities = [], [], []
        for rp in props:
            vol_um3 = float(rp.area * voxel_um3)
            area_um2 = estimate_surface_area_um2(rp.image, voxel_size_zyx)
            if cfg.max_skeleton_voxels is not None and rp.area > cfg.max_skeleton_voxels:
                # Over the size cap: report not-measured (NaN) rather than a
                # misleading zero-length skeleton.
                skel = unmeasured
            else:
                skel = skeleton_graph_metrics(skels.get(int(rp.label)))

            ids.append(int(rp.label))
            volumes_um3.append(vol_um3)
            areas.append(area_um2)
            sphericities.append(sphericity(vol_um3, area_um2))
            aspects.append(aspect_ratio_from_coords(rp.coords, voxel_size_zyx))
            branches.append(float(skel["branches"]))
            lengths.append(float(skel["length_um"]))
            tortuosities.append(float(skel["tortuosity"]))

        return {
            "instance_count": len(ids),
            "total_volume_um3": float((labels > 0).sum() * voxel_um3),
            "instance_label": _as_list(ids),
            "instance_volume_um3": _as_list(volumes_um3),
            "instance_surface_area_um2": _as_list(areas),
            "instance_sphericity": _as_list(sphericities),
            "instance_aspect_ratio_major_minor": _as_list(aspects),
            "instance_branches": _as_list(branches),
            "instance_length_um": _as_list(lengths),
            "instance_tortuosity": _as_list(tortuosities),
        }
