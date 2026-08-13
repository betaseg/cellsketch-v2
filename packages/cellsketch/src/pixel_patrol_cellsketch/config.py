"""Run configuration for the CellSketch plugins.

PixelPatrol discovers loaders and processors as *classes* and instantiates them
with no arguments (``pixel_patrol_base.plugin_registry``), and its CLI has no way
to pass plugin options. Analysis knobs therefore come from the environment, read
once per worker process. Every value has the same default as the corresponding
``analyze_cell.py`` flag.

    CELLSKETCH_VOXEL_SIZE_UM      "z,y,x" — skip inference from TIFF metadata
    CELLSKETCH_AUTO_CLIP_TO_PM    1 → clip non-membrane entities to the membrane
    CELLSKETCH_AUTO_LABEL_MASKS   1 → promote multi-component masks to labels
    CELLSKETCH_MAX_SKELETON_VOXELS  skip skeletons above this instance size
    CELLSKETCH_SKELETON_ENTITIES  comma-separated entities to skeletonise (default: all)
    CELLSKETCH_NO_SKELETONS       1 -> skeletonise nothing
    CELLSKETCH_NUM_THREADS        kimimaro worker count; 1 by default because
                                  PixelPatrol already runs cells in parallel
                                  (0 = all cores)
    CELLSKETCH_EDT_THREADS        distance-transform threads (0 = all cores)
    CELLSKETCH_CONTACT_MAX_UM     largest instance-pair gap recorded, in µm
    CELLSKETCH_POLARITY_SPREAD    1 -> per-instance angular spread on the polarity sphere
    CELLSKETCH_DISTANCE_HISTOGRAMS  1 -> per-instance distance distributions, not just minima
    CELLSKETCH_MESH_DIR           where to write report_meshes.csv per cell; unset = no meshing
    CELLSKETCH_MESH_SMOOTH_SIGMA / _STEP_SIZE / _TARGET_REDUCTION / _LEVEL
                                  mesh generation knobs, as analyze_cell.py's --mesh-* flags
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

from pixel_patrol_cellsketch.skeletons import parse_entity_filter

_TRUE = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUE


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_voxel_size(name: str) -> Optional[Tuple[float, float, float]]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    parts = [p for p in raw.replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise ValueError(f"{name} must be 'z,y,x' in µm, got {raw!r}")
    z, y, x = (float(p) for p in parts)
    return (z, y, x)


@dataclass(frozen=True)
class CellSketchConfig:
    voxel_size_um: Optional[Tuple[float, float, float]] = None
    auto_clip_to_pm: bool = False
    auto_label_masks: bool = False
    max_skeleton_voxels: int = 500_000
    # None = every entity; an empty set = none. Skeletonising dominates a run, and a
    # blob's skeleton says nothing, so restricting it to filaments is the big lever.
    skeleton_entities: Optional[FrozenSet[str]] = None
    # 1, not analyze_cell.py's 0: processors run inside Dask worker processes, which
    # cannot fork children of their own, so asking kimimaro for a process pool costs a
    # failed attempt per cell before it falls back. Cells already run in parallel.
    num_threads: int = 1
    # Separate from num_threads: edt is a C++ loop with no subprocesses, so it can use
    # all cores even inside a Dask worker, where kimimaro's process pool cannot.
    edt_threads: int = 0
    contact_max_um: float = 0.5
    # Both walk every voxel of every instance, so they are opt-in, as the equivalent
    # analyze_cell.py flags were.
    polarity_spread: bool = False
    distance_histograms: bool = False
    # Geometry is written beside the report, never into it; unset means no meshing at all.
    mesh_dir: Optional[str] = None
    mesh_smooth_sigma: float = 0.7
    mesh_step_size: int = 2
    mesh_target_reduction: float = 0.8
    mesh_level: Optional[float] = None

    @classmethod
    def from_env(cls) -> "CellSketchConfig":
        return cls(
            voxel_size_um=_env_voxel_size("CELLSKETCH_VOXEL_SIZE_UM"),
            auto_clip_to_pm=_env_flag("CELLSKETCH_AUTO_CLIP_TO_PM"),
            auto_label_masks=_env_flag("CELLSKETCH_AUTO_LABEL_MASKS"),
            max_skeleton_voxels=_env_int("CELLSKETCH_MAX_SKELETON_VOXELS", 500_000),
            skeleton_entities=parse_entity_filter(
                os.environ.get("CELLSKETCH_SKELETON_ENTITIES"),
                none=_env_flag("CELLSKETCH_NO_SKELETONS"),
            ),
            num_threads=_env_int("CELLSKETCH_NUM_THREADS", 1),
            edt_threads=_env_int("CELLSKETCH_EDT_THREADS", 0),
            contact_max_um=_env_float("CELLSKETCH_CONTACT_MAX_UM", 0.5),
            polarity_spread=_env_flag("CELLSKETCH_POLARITY_SPREAD"),
            distance_histograms=_env_flag("CELLSKETCH_DISTANCE_HISTOGRAMS"),
            mesh_dir=os.environ.get("CELLSKETCH_MESH_DIR") or None,
            mesh_smooth_sigma=_env_float("CELLSKETCH_MESH_SMOOTH_SIGMA", 0.7),
            mesh_step_size=_env_int("CELLSKETCH_MESH_STEP_SIZE", 2),
            mesh_target_reduction=_env_float("CELLSKETCH_MESH_TARGET_REDUCTION", 0.8),
            mesh_level=(
                _env_float("CELLSKETCH_MESH_LEVEL", 0.0)
                if os.environ.get("CELLSKETCH_MESH_LEVEL") else None
            ),
        )
