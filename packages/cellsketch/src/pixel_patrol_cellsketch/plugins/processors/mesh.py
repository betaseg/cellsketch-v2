"""Mesh processor: geometry as a side file, never as table columns.

Base64 meshes would multiply the size of the report that every stats query loads, so
this processor contributes *no columns at all*. It writes one ``report_meshes.csv`` per
cell - the file ``mesh_viewer.html`` and ``csv_to_blender.py`` read - and returns an
empty result, which PixelPatrol merges as nothing.

It only runs when a destination is configured (``CELLSKETCH_MESH_DIR``, set by
``cellsketch process --with-mesh``), so an ordinary run pays nothing for it. That is the
same opt-in ``analyze_cell.py --with-mesh`` was, and for the same reason: meshing every
instance, and skeletonising it again for the overlay, is the most expensive thing here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec

import numpy as np

from pixel_patrol_cellsketch.config import CellSketchConfig
from pixel_patrol_cellsketch.mesh import MeshOptions, mesh_rows_for_cell, write_mesh_csv
from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CELL_KIND
from pixel_patrol_cellsketch.plugins.processors.instances import channel_view
from pixel_patrol_cellsketch.skeletons import CACHE

logger = logging.getLogger(__name__)


class MeshProcessor:
    """Write per-instance meshes and skeletons for one cell to its own CSV."""

    NAME = "cellsketch-mesh"
    DESCRIPTION = (
        "Writes one report_meshes.csv per cell - per-instance marching-cubes meshes and curve "
        "skeletons for the 3D viewer and the Blender export. Adds no columns to the table: "
        "geometry payloads belong beside the report, not inside it. Enabled by "
        "CELLSKETCH_MESH_DIR (cellsketch process --with-mesh)."
    )

    CHUNK_KIND = ChunkKind.MEMORY
    INPUT = RecordSpec(axes={"C", "Z", "Y", "X"}, kinds={CELL_KIND})
    OUTPUT = "features"

    # Deliberately empty: nothing this processor produces belongs in the parquet.
    OUTPUT_SCHEMA: Dict[str, Any] = {}
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {}

    def __init__(self) -> None:
        self._config = CellSketchConfig.from_env()

    def run_chunk(self, record: Record) -> Dict[str, Any]:
        cfg = self._config
        if not cfg.mesh_dir:
            return {}

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

        cell_id = str(meta.get("cell_id") or "cell")
        # Whatever cellsketch-instances already measured for this cell, so the CSV carries
        # the same metrics without measuring them twice. Empty when that processor is off.
        metrics = CACHE.get_or_compute(cell_id, ("instance_metrics",), arr, dict)
        rows = mesh_rows_for_cell(
            {name: channel_view(arr, c_axis, i) for i, name in enumerate(names)},
            dict(zip(names, kinds)),
            (float(meta["pixel_size_Z"]), float(meta["pixel_size_Y"]), float(meta["pixel_size_X"])),
            cell_id=cell_id,
            options=MeshOptions(
                smooth_sigma=cfg.mesh_smooth_sigma,
                step_size=cfg.mesh_step_size,
                target_reduction=cfg.mesh_target_reduction,
                level=cfg.mesh_level,
                skeleton_entities=cfg.skeleton_entities,
                max_skeleton_voxels=cfg.max_skeleton_voxels,
                num_threads=cfg.num_threads,
                contact_max_um=cfg.contact_max_um,
            ),
            metrics=metrics,
        )
        path = write_mesh_csv(Path(cfg.mesh_dir) / cell_id / "report_meshes.csv", rows)
        with_mesh = sum(1 for row in rows if row["mesh_b64"])
        logger.info(
            "cellsketch: %s — %d/%d rows meshed → %s (%.1f MB)",
            cell_id, with_mesh, len(rows), path, path.stat().st_size / 1024**2,
        )
        return {}

    def get_aggregation(self, name: str) -> Optional[Any]:
        return None
