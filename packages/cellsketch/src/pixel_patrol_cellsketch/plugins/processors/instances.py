"""Instance processor: one flattened instance table per cell.

Per-instance measurements need the whole cell, not one entity: a distance is *to*
another entity, and polarity is relative to the membrane. So this is a MEMORY
processor - it sees every channel at once - and it writes the instance table onto the
cell's ``obs_level=0`` row as parallel list columns, the same shape the contact edge
list uses. One unnest gives back ``analyze_cell.py``'s ``row_type=instance`` table:

    SELECT cell_id, unnest(instance_entity) AS entity_name,
                    unnest(instance_label)  AS label,
                    unnest(instance_volume_um3) AS volume_um3
    FROM pp_data WHERE obs_level = 0

Distances are a second, longer list group (one element per instance × target) because
PixelPatrol drops columns a processor did not declare, and target names come from the
data. Unnesting that group yields one row per instance per target:

    SELECT cell_id, unnest(distance_entity) AS entity_name, unnest(distance_label) AS label,
                    unnest(distance_target) AS target, unnest(distance_um) AS distance_um
    FROM pp_data WHERE obs_level = 0

Instances come from label entities. A whole-structure mask has no instances; its
morphology is on its own entity row, from cellsketch-morphology.

Memory is the design constraint here - a real cell is ~550 megavoxels across five
entities - so the two passes are arranged to keep at most one whole-volume float32
array alive: morphology walks instances one at a time, and distances walk *targets* one
at a time, reducing each transform over every entity's labels before freeing it.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

from pixel_patrol_cellsketch.config import CellSketchConfig
from pixel_patrol_cellsketch.distances import (
    cell_center_um,
    distance_target,
    distance_transform_um,
    polarity_from_offset,
)
from pixel_patrol_cellsketch.geometry import (
    aspect_ratio_from_coords,
    estimate_surface_area_um2,
    skeleton_graph_metrics,
    sphericity,
)
from pixel_patrol_cellsketch.skeletons import skeletons_for, wants_skeletons
from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND

logger = logging.getLogger(__name__)

DISTANCE_HISTOGRAM_BINS = 20

# One element per instance.
_INSTANCE_COLUMNS: Dict[str, Any] = {
    "instance_entity": list,
    "instance_label": list,
    "instance_volume_um3": list,
    "instance_surface_area_um2": list,
    "instance_sphericity": list,
    "instance_aspect_ratio_major_minor": list,
    "instance_branches": list,
    "instance_length_um": list,
    "instance_tortuosity": list,
    "instance_distance_to_closest_same_type_um": list,
    "instance_polar_dist_um": list,
    "instance_polar_az_deg": list,
    "instance_polar_el_deg": list,
    "instance_polar_spread_deg": list,
}

# One element per instance × target entity.
_DISTANCE_COLUMNS: Dict[str, Any] = {
    "distance_entity": list,
    "distance_label": list,
    "distance_target": list,
    "distance_um": list,
    "distance_mean_um": list,
    "distance_hist_min_um": list,
    "distance_hist_max_um": list,
    "distance_hist_counts": list,
}

_CELL_COLUMNS: Dict[str, Any] = {
    "cell_volume_um3": np.float64,
}

_DESCRIPTIONS: Dict[str, str] = {
    "instance_entity": "Entity each instance belongs to, in list order shared by all instance_* columns.",
    "instance_label": "Label id of each instance within its entity.",
    "instance_volume_um3": "Per-instance volume in µm³.",
    "instance_surface_area_um2": "Per-instance voxel-face surface area in µm².",
    "instance_sphericity": "Per-instance sphericity, 1 = perfect sphere.",
    "instance_aspect_ratio_major_minor": "Per-instance ratio of largest to smallest PCA axis length.",
    "instance_branches": "Per-instance number of curve-skeleton branches.",
    "instance_length_um": "Per-instance curve-skeleton cable length in µm.",
    "instance_tortuosity": "Per-instance length-weighted branch arc/chord ratio (≥1, 1 = straight).",
    "instance_distance_to_closest_same_type_um": "Centroid distance in µm to the nearest other instance of the same entity.",
    "instance_polar_dist_um": "Distance in µm from the cell centre to the instance centroid.",
    "instance_polar_az_deg": "Azimuth in degrees of the instance centroid as seen from the cell centre.",
    "instance_polar_el_deg": "Elevation in degrees of the instance centroid as seen from the cell centre.",
    "instance_polar_spread_deg": "Angular spread in degrees of the instance's voxels on the polarity sphere: how much of a direction range it covers as seen from the cell centre. Null unless polarity spread is enabled.",
    "distance_entity": "Entity of the instance being measured, in list order shared by all distance_* columns.",
    "distance_label": "Label id of the instance being measured.",
    "distance_target": "Entity measured to; for the plasma membrane this is the distance to the cell boundary.",
    "distance_um": "Smallest distance in µm from the instance's voxels to the target entity.",
    "distance_mean_um": "Mean distance in µm over the instance's voxels. Null unless distance histograms are enabled.",
    "distance_hist_min_um": "Lower bound of the histogram range, shared by every instance of this entity/target pair.",
    "distance_hist_max_um": "Upper bound of the histogram range, shared by every instance of this entity/target pair.",
    "distance_hist_counts": "Per-instance voxel counts over the histogram range, as a JSON array of fixed-width bins.",
    "cell_volume_um3": "Volume in µm³ enclosed by the plasma-membrane mask.",
}


def channel_view(arr: np.ndarray, c_axis: int, index: int) -> np.ndarray:
    """One channel of a CZYX array, as a view - never a copy.

    np.take would copy, which on a 550-megavoxel channel is a gigabyte per entity.
    """
    key: List[Any] = [slice(None)] * arr.ndim
    key[c_axis] = index
    return arr[tuple(key)]


def null_if_not_finite(value: Any) -> Any:
    """NaN and infinity become NULL, because that is how a table says "not measured".

    Left as NaN they are not merely untidy: DuckDB's STDDEV raises "out of range" on a
    column that holds one, so a single unmeasured instance took a whole widget down, and
    min/max/quantiles came back nan.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_list(values: List[Any]) -> Optional[List[Any]]:
    """The list with non-finite elements nulled, or None if there is nothing to say.

    Both empty and all-null come back as None: a list of nothing but NULLs types the
    parquet column List(Null), which cannot be written alongside the List(Double) another
    cell wrote - the same reason an empty list is not emitted either.
    """
    nulled = [null_if_not_finite(v) for v in values]
    return nulled if any(v is not None for v in nulled) else None


class InstanceProcessor:
    """Per-instance morphology, distances and polarity for a whole cell."""

    NAME = "cellsketch-instances"
    DESCRIPTION = (
        "Measures every labelled instance in a cell: volume, surface area, sphericity, PCA aspect "
        "ratio, curve-skeleton metrics, distance to each other entity, distance to the closest "
        "instance of its own entity, and its direction from the cell centre."
    )

    CHUNK_KIND = ChunkKind.MEMORY
    INPUT = RecordSpec(axes={"C", "Z", "Y", "X"}, kinds={CELL_KIND})
    OUTPUT = "features"

    OUTPUT_SCHEMA: Dict[str, Any] = {**_INSTANCE_COLUMNS, **_DISTANCE_COLUMNS, **_CELL_COLUMNS}
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = dict(_DESCRIPTIONS)

    def __init__(self) -> None:
        self._config = CellSketchConfig.from_env()

    def run_chunk(self, record: Record) -> Dict[str, Any]:
        meta = record.meta
        arr = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        c_axis = record.dim_order.index("C")

        names = list(meta.get("channel_names") or [])
        kinds = list(meta.get("entity_kinds") or [])
        expected_zyx = [int(v) for v in (meta.get("cell_shape_zyx") or [])]
        spatial = [s for i, s in enumerate(arr.shape) if i != c_axis]
        if arr.shape[c_axis] != len(names) or (expected_zyx and spatial != expected_zyx):
            raise ValueError(
                f"cell arrived as a {arr.shape} fragment of {len(names)}×{tuple(expected_zyx)} — "
                "raise --mb-per-task above the size of one cell"
            )

        voxel_size_zyx = (
            float(meta["pixel_size_Z"]),
            float(meta["pixel_size_Y"]),
            float(meta["pixel_size_X"]),
        )
        views = {name: channel_view(arr, c_axis, i) for i, name in enumerate(names)}
        kinds_by_name = dict(zip(names, kinds))
        membrane_name = meta.get("membrane_name")

        out: Dict[str, Any] = {}
        if membrane_name in views:
            voxel_um3 = float(np.prod(voxel_size_zyx))
            out["cell_volume_um3"] = float((views[membrane_name] > 0).sum() * voxel_um3)

        # The loader computed the membrane centroid, so polarity here and on the entity
        # rows is measured from the same origin.
        from_meta = [meta.get(f"cell_center_{ax}_um") for ax in "zyx"]
        if all(c is not None for c in from_meta):
            center = tuple(float(c) for c in from_meta)
        elif membrane_name in views:
            center = cell_center_um(views[membrane_name], voxel_size_zyx)
        else:
            center = None

        label_names = [n for n in names if kinds_by_name[n] == "label"]
        inst: Dict[str, List[Any]] = {col: [] for col in _INSTANCE_COLUMNS}
        ids_by_entity: Dict[str, List[int]] = {}
        for name in label_names:
            ids_by_entity[name] = self._measure_morphology(
                name, views[name], voxel_size_zyx, center, inst,
                cell_id=str(meta.get("cell_id") or "cell"),
            )

        dist = self._measure_distances(views, kinds_by_name, label_names, ids_by_entity,
                                       voxel_size_zyx)

        logger.info(
            "cellsketch: %s — %d instances, %d instance-target distances",
            meta.get("cell_id"), len(inst["instance_label"]), len(dist["distance_um"]),
        )
        out.update({col: _as_list(vals) for col, vals in inst.items()})
        out.update({col: _as_list(vals) for col, vals in dist.items()})
        return out

    def get_aggregation(self, name: str):
        if name not in self.OUTPUT_SCHEMA:
            return None

        def agg(rows: List[Dict], _g_dims: Dict[str, Any]) -> Optional[Any]:
            # One chunk per cell by construction (run_chunk refuses fragments), so there
            # is nothing to merge; tables from partial cells must not be concatenated.
            return rows[0].get(name) if len(rows) == 1 else None

        return agg

    # ── morphology, one instance at a time ────────────────────────────────────

    def _measure_morphology(
        self,
        entity: str,
        labels: np.ndarray,
        voxel_size_zyx: Sequence[float],
        center: tuple[float, float, float] | None,
        inst: Dict[str, List[Any]],
        cell_id: str = "cell",
    ) -> List[int]:
        """Append one element per instance to every instance_* list; return the label ids."""
        cfg = self._config
        voxel_um3 = float(np.prod(voxel_size_zyx))
        props = regionprops(np.ascontiguousarray(labels))
        if not props:
            return []

        # One TEASAR pass over the whole entity yields a skeleton per instance - shared
        # with the mesh processor through the per-cell cache, and skipped entirely for
        # entities where a skeleton says nothing (blobs).
        skels = (
            skeletons_for(cell_id, entity, labels, voxel_size_zyx,
                          cfg.max_skeleton_voxels, cfg.num_threads)
            if wants_skeletons(entity, cfg.skeleton_entities) else {}
        )
        unmeasured = {"branches": float("nan"), "length_um": float("nan"), "tortuosity": float("nan")}

        ids: List[int] = []
        centroids = []
        for rp in props:
            vol_um3 = float(rp.area * voxel_um3)
            area_um2 = estimate_surface_area_um2(rp.image, voxel_size_zyx)
            if not skels or (cfg.max_skeleton_voxels is not None and rp.area > cfg.max_skeleton_voxels):
                # Not asked for, or over the size cap: not-measured (NaN) rather than a
                # misleading zero.
                skel = unmeasured
            else:
                skel = skeleton_graph_metrics(skels.get(int(rp.label)))

            ids.append(int(rp.label))
            inst["instance_entity"].append(entity)
            inst["instance_label"].append(int(rp.label))
            inst["instance_volume_um3"].append(vol_um3)
            inst["instance_surface_area_um2"].append(area_um2)
            inst["instance_sphericity"].append(sphericity(vol_um3, area_um2))
            inst["instance_aspect_ratio_major_minor"].append(
                aspect_ratio_from_coords(rp.coords, voxel_size_zyx)
            )
            inst["instance_branches"].append(float(skel["branches"]))
            inst["instance_length_um"].append(float(skel["length_um"]))
            inst["instance_tortuosity"].append(float(skel["tortuosity"]))

            centroid_um = np.array(rp.centroid) * np.array(voxel_size_zyx)
            centroids.append(centroid_um)
            polar = (
                polarity_from_offset(*(centroid_um - np.array(center))) if center is not None
                else {"polar_dist_um": float("nan"), "polar_az_deg": float("nan"),
                      "polar_el_deg": float("nan")}
            )
            inst["instance_polar_dist_um"].append(polar["polar_dist_um"])
            inst["instance_polar_az_deg"].append(polar["polar_az_deg"])
            inst["instance_polar_el_deg"].append(polar["polar_el_deg"])
            inst["instance_polar_spread_deg"].append(
                _polar_spread_deg(rp.coords, voxel_size_zyx, center)
                if (cfg.polarity_spread and center is not None) else float("nan")
            )

        # Nearest neighbour of the same entity, centroid to centroid. Appended in one go
        # after the loop, in the same order the instances were appended above.
        if len(centroids) > 1:
            d = cdist(np.vstack(centroids), np.vstack(centroids))
            np.fill_diagonal(d, np.inf)
            nearest = d.min(axis=1).tolist()
        else:
            nearest = [float("nan")] * len(props)
        inst["instance_distance_to_closest_same_type_um"].extend(nearest)
        return ids

    # ── distances, one target at a time ──────────────────────────────────────

    def _measure_distances(
        self,
        views: Dict[str, np.ndarray],
        kinds_by_name: Dict[str, str],
        label_names: List[str],
        ids_by_entity: Dict[str, List[int]],
        voxel_size_zyx: Sequence[float],
    ) -> Dict[str, List[Any]]:
        """One row per (instance, target), reducing each transform over every entity.

        Targets are the outer loop so only one distance transform exists at a time.

        Each entity's foreground is indexed once (voxel positions sorted by label id),
        after which measuring it against a transform is a gather plus a reduceat over the
        foreground alone. scipy's ndimage.minimum does the obvious thing instead - one
        labelled pass over the whole volume - and is pathologically slow at it: 54 s per
        call on a 197-megavoxel cell with 10k instances, against 0.01 s here, because its
        cost follows the volume and the label count rather than the foreground.
        """
        cfg = self._config
        # The histogram columns are only created when they are being filled: a column of
        # nothing but nulls cannot be written as a parquet list.
        columns = list(_DISTANCE_COLUMNS) if cfg.distance_histograms else [
            "distance_entity", "distance_label", "distance_target", "distance_um",
        ]
        dist: Dict[str, List[Any]] = {col: [] for col in columns}
        indexes = {
            name: _foreground_index(views[name])
            for name in label_names if ids_by_entity.get(name)
        }
        for target, target_view in views.items():
            measured = [name for name in indexes if name != target]
            if not measured:
                continue
            transform = distance_transform_um(
                distance_target(target_view, target, kinds_by_name[target]),
                voxel_size_zyx, cfg.edt_threads,
            )
            flat = transform.reshape(-1)
            for name in measured:
                index = indexes[name]
                values = flat[index.positions]
                mins = np.minimum.reduceat(values, index.starts)
                stats = _distance_stats(values, index) if cfg.distance_histograms else None
                for position, label_id in enumerate(index.ids):
                    dist["distance_entity"].append(name)
                    dist["distance_label"].append(int(label_id))
                    dist["distance_target"].append(target)
                    dist["distance_um"].append(float(mins[position]))
                    if stats is not None:
                        dist["distance_mean_um"].append(stats["mean"][position])
                        dist["distance_hist_min_um"].append(stats["lo"])
                        dist["distance_hist_max_um"].append(stats["hi"])
                        dist["distance_hist_counts"].append(stats["counts"][position])
            del transform, flat
        return dist


@dataclass(frozen=True)
class _ForegroundIndex:
    """An entity's labelled voxels, grouped by instance.

    positions: flat voxel indices, sorted by label id
    starts:    where each instance's run begins in positions
    ids:       the label ids, in the same order as starts
    """
    positions: np.ndarray
    starts: np.ndarray
    ids: np.ndarray


def _foreground_index(labels: np.ndarray) -> _ForegroundIndex:
    """Index an entity's foreground once, so each later measurement is a gather."""
    positions = np.flatnonzero(labels)
    # int32 halves this array where the volume allows, and it is the second largest
    # allocation after the distance transform itself.
    if labels.size <= np.iinfo(np.int32).max:
        positions = positions.astype(np.int32, copy=False)
    ids_at = labels.reshape(-1)[positions]
    order = np.argsort(ids_at, kind="stable")
    positions = positions[order]
    ids, starts = np.unique(ids_at[order], return_index=True)
    return _ForegroundIndex(positions=positions, starts=starts, ids=ids)


def _distance_stats(values: np.ndarray, index: _ForegroundIndex) -> Dict[str, Any]:
    """Mean and a binned distribution per instance, from the gathered values.

    The histogram range is shared by every instance of the entity/target pair, unlike
    ``analyze_cell.py``'s per-instance range: shared bins are what makes two instances'
    distributions comparable, and one pass computes them all.
    """
    counts_per_instance = np.diff(np.append(index.starts, len(values)))
    sums = np.add.reduceat(values.astype(np.float64), index.starts)
    means = sums / counts_per_instance
    lo, hi = float(values.min()), float(values.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    # One bincount over (instance, bin) pairs rather than a histogram per instance.
    bins = np.clip(
        ((values - lo) / (hi - lo) * DISTANCE_HISTOGRAM_BINS).astype(np.int64),
        0, DISTANCE_HISTOGRAM_BINS - 1,
    )
    instance_of = np.repeat(np.arange(len(index.ids)), counts_per_instance)
    flat_counts = np.bincount(
        instance_of * DISTANCE_HISTOGRAM_BINS + bins,
        minlength=len(index.ids) * DISTANCE_HISTOGRAM_BINS,
    ).reshape(len(index.ids), DISTANCE_HISTOGRAM_BINS)
    return {
        "mean": [float(m) for m in means],
        "lo": lo,
        "hi": hi,
        "counts": [json.dumps([int(c) for c in row]) for row in flat_counts],
    }


def _polar_spread_deg(
    coords: np.ndarray,
    voxel_size_zyx: Sequence[float],
    center: Tuple[float, float, float],
) -> float:
    """Angular spread of an instance's voxels on the polarity sphere, in degrees.

    How wide a range of directions the instance covers as seen from the cell centre: a
    compact granule reads near zero, a strand wrapping the cell reads large.
    """
    if coords.shape[0] < 3:
        return float("nan")
    offsets = coords * np.array(voxel_size_zyx) - np.array(center)
    radius = np.linalg.norm(offsets, axis=1)
    valid = radius > 0
    if valid.sum() < 3:
        return float("nan")
    unit = offsets[valid] / radius[valid, None]
    mean_dir = unit.mean(axis=0)
    length = float(np.linalg.norm(mean_dir))
    if length > 0:
        mean_dir = mean_dir / length
    dots = np.clip(unit @ mean_dir, -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).std())
