"""Instance processor: one flattened instance table per cell.

Per-instance measurements need the whole cell, not one entity: a distance is *to*
another entity, and polarity is relative to the membrane. So this is a MEMORY
processor - it sees every channel at once - and it writes the instance table onto the
cell's ``obs_level=0`` row as parallel list columns, the same shape the contact edge
list uses. One unnest gives back ``analyze_cell.py``'s ``row_type=instance`` table:

    SELECT cell_id, unnest(instance_entity) AS entity_name,
                    unnest(instance_label)  AS label,
                    unnest(instance_volume_um3) AS volume_um3
    FROM pp_data WHERE obs_level = 1 IS NOT TRUE

Distances are a second, longer list group (one element per instance × target) because
PixelPatrol drops columns a processor did not declare, and target names come from the
data. Unnesting that group yields one row per instance per target:

    SELECT cell_id, unnest(distance_entity) AS entity_name, unnest(distance_label) AS label,
                    unnest(distance_target) AS target, unnest(distance_um) AS distance_um
    FROM pp_data

Instances come from label entities. A whole-structure mask has no instances; its
morphology is on its own entity row, from cellsketch-morphology.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

from pixel_patrol_cellsketch.config import CellSketchConfig
from pixel_patrol_cellsketch.distances import (
    build_distance_targets,
    cell_center_um,
    distance_transform_um,
    polarity_from_offset,
)
from pixel_patrol_cellsketch.geometry import (
    aspect_ratio_from_coords,
    compute_curve_skeletons,
    estimate_surface_area_um2,
    skeleton_graph_metrics,
    sphericity,
)
from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND

logger = logging.getLogger(__name__)

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
}

# One element per instance × target entity.
_DISTANCE_COLUMNS: Dict[str, Any] = {
    "distance_entity": list,
    "distance_label": list,
    "distance_target": list,
    "distance_um": list,
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
    "distance_entity": "Entity of the instance being measured, in list order shared by all distance_* columns.",
    "distance_label": "Label id of the instance being measured.",
    "distance_target": "Entity measured to; for the plasma membrane this is the distance to the cell boundary.",
    "distance_um": "Smallest distance in µm from the instance's voxels to the target entity.",
    "cell_volume_um3": "Volume in µm³ enclosed by the plasma-membrane mask.",
}


def _as_list(values: List[Any]) -> Optional[List[Any]]:
    """None rather than [] for empty: an empty list would type the parquet column
    List(Null) and clash with the typed list written for cells that have instances."""
    return values or None


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
        volumes = {name: np.take(arr, i, axis=c_axis) for i, name in enumerate(names)}
        kinds_by_name = dict(zip(names, kinds))
        membrane_name = meta.get("membrane_name")

        out: Dict[str, Any] = {}
        if membrane_name in volumes:
            voxel_um3 = float(np.prod(voxel_size_zyx))
            out["cell_volume_um3"] = float((volumes[membrane_name] > 0).sum() * voxel_um3)
        center = (
            cell_center_um(volumes[membrane_name], voxel_size_zyx)
            if membrane_name in volumes else None
        )

        # One distance transform per target, reused by every instance measured against it.
        targets = build_distance_targets(volumes, kinds_by_name)
        transforms = {
            name: distance_transform_um(mask, voxel_size_zyx, self._config.num_threads)
            for name, mask in targets.items()
        }

        inst: Dict[str, List[Any]] = {col: [] for col in _INSTANCE_COLUMNS}
        dist: Dict[str, List[Any]] = {col: [] for col in _DISTANCE_COLUMNS}
        for name in names:
            if kinds_by_name[name] != "label":
                continue
            self._measure_entity(
                name, volumes[name], voxel_size_zyx, transforms, center, inst, dist
            )

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

    # ── per entity ────────────────────────────────────────────────────────────

    def _measure_entity(
        self,
        entity: str,
        labels: np.ndarray,
        voxel_size_zyx: Sequence[float],
        transforms: Dict[str, np.ndarray],
        center: tuple[float, float, float] | None,
        inst: Dict[str, List[Any]],
        dist: Dict[str, List[Any]],
    ) -> None:
        cfg = self._config
        voxel_um3 = float(np.prod(voxel_size_zyx))
        labels = labels.astype(np.int32, copy=False)
        props = regionprops(labels)
        if not props:
            return

        # One TEASAR pass over the whole entity yields a skeleton per instance.
        skels = compute_curve_skeletons(
            labels, tuple(voxel_size_zyx),
            max_voxels=cfg.max_skeleton_voxels, num_threads=cfg.num_threads,
        )
        unmeasured = {"branches": float("nan"), "length_um": float("nan"), "tortuosity": float("nan")}

        centroids = []
        for rp in props:
            vol_um3 = float(rp.area * voxel_um3)
            area_um2 = estimate_surface_area_um2(rp.image, voxel_size_zyx)
            if cfg.max_skeleton_voxels is not None and rp.area > cfg.max_skeleton_voxels:
                # Over the size cap: not-measured (NaN) rather than a misleading zero.
                skel = unmeasured
            else:
                skel = skeleton_graph_metrics(skels.get(int(rp.label)))

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
            if center is None:
                polar = {"polar_dist_um": float("nan"), "polar_az_deg": float("nan"),
                         "polar_el_deg": float("nan")}
            else:
                polar = polarity_from_offset(*(centroid_um - np.array(center)))
            inst["instance_polar_dist_um"].append(polar["polar_dist_um"])
            inst["instance_polar_az_deg"].append(polar["polar_az_deg"])
            inst["instance_polar_el_deg"].append(polar["polar_el_deg"])

            # Distance to every other entity. An instance's distance to its own entity
            # would be zero by construction, so its own target is skipped.
            for target, dt in transforms.items():
                if target == entity:
                    continue
                vals = dt[tuple(rp.coords.T)] if rp.coords.size else np.array([], dtype=np.float32)
                dist["distance_entity"].append(entity)
                dist["distance_label"].append(int(rp.label))
                dist["distance_target"].append(target)
                dist["distance_um"].append(float(vals.min()) if vals.size else float("nan"))

        # Nearest neighbour of the same entity, centroid to centroid. Appended in one go
        # after the loop, in the same order the instances were appended above.
        if len(centroids) > 1:
            d = cdist(np.vstack(centroids), np.vstack(centroids))
            np.fill_diagonal(d, np.inf)
            nearest = d.min(axis=1).tolist()
        else:
            nearest = [float("nan")] * len(props)
        inst["instance_distance_to_closest_same_type_um"].extend(nearest)
