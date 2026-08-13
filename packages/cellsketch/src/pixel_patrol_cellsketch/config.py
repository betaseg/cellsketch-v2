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
    CELLSKETCH_NUM_THREADS        skeleton/DT thread count (0 = all cores)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

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
    num_threads: int = 0

    @classmethod
    def from_env(cls) -> "CellSketchConfig":
        return cls(
            voxel_size_um=_env_voxel_size("CELLSKETCH_VOXEL_SIZE_UM"),
            auto_clip_to_pm=_env_flag("CELLSKETCH_AUTO_CLIP_TO_PM"),
            auto_label_masks=_env_flag("CELLSKETCH_AUTO_LABEL_MASKS"),
            max_skeleton_voxels=_env_int("CELLSKETCH_MAX_SKELETON_VOXELS", 500_000),
            num_threads=_env_int("CELLSKETCH_NUM_THREADS", 0),
        )
