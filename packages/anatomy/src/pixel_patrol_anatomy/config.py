"""Run configuration for the Anatomy plugins.

A loader and a processor are constructed with no arguments, and each object is
measured in its own worker process. Analysis knobs therefore come from the
environment, read once per process, and the CLI sets them from its own flags.

    PP_ANATOMY_OBJECT_MASK        name of the mask that bounds each object. Required:
                                  nothing loads without it, and nothing is guessed
    PP_ANATOMY_VOXEL_SIZE_UM      "z,y,x" (3D) or "y,x" (2D). Skips inference from
                                  the TIFF metadata
    PP_ANATOMY_AUTO_CLIP          1 → clip the entities to the object mask
    PP_ANATOMY_AUTO_LABEL_MASKS   1 → promote multi-component masks to labels
    PP_ANATOMY_ENTITY_COLOURS     path to a JSON file of structure: hex colour pairs, which
                                  the report carries so every widget colours the same way
    PP_ANATOMY_MAX_SKELETON_VOXELS  skip skeletons above this instance size
    PP_ANATOMY_SKELETON_ENTITIES  comma-separated entities to skeletonise (default: all)
    PP_ANATOMY_NO_SKELETONS       1 -> skeletonise nothing
    PP_ANATOMY_NUM_THREADS        kimimaro worker count; 1 by default because
                                  PixelPatrol already runs objects in parallel
                                  (0 = all cores)
    PP_ANATOMY_EDT_THREADS        distance-transform threads (0 = all cores)
    PP_ANATOMY_CONTACT_MAX_UM     largest instance-pair gap recorded, in µm
    PP_ANATOMY_POLARITY_SPREAD    1 -> per-instance angular spread on the polarity sphere
    PP_ANATOMY_DISTANCE_HISTOGRAMS  1 -> per-instance distance distributions, not just minima
    PP_ANATOMY_MESH_DIR           where to write geometry.parquet per object; unset = no meshing
    PP_ANATOMY_MESH_SMOOTH_SIGMA / _STEP_SIZE / _TARGET_REDUCTION / _LEVEL
                                  mesh generation knobs, as the --mesh-* flags
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

from pixel_patrol_anatomy.skeletons import parse_entity_filter

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


_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _colours_from(path: Path, name: str) -> Dict[str, str]:
    """Structure colours from a JSON file: ``{"mito": "#d62728", "er": "#2ca02c"}``.

    A study decides what each structure should look like, and the report carries the answer so
    every widget agrees and a shared parquet arrives already coloured. Names the batch does not
    have are ignored, so one file can cover a whole project; structures the file does not name
    keep their place in the built-in palette.
    """
    try:
        loaded = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"{name}: no such file: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{name}: {path} is not valid JSON: {error}") from None
    if not isinstance(loaded, dict):
        raise ValueError(f"{name}: {path} must hold an object of structure: colour pairs")

    colours: Dict[str, str] = {}
    for entity, colour in loaded.items():
        if not isinstance(colour, str) or not _HEX.match(colour):
            raise ValueError(
                f"{name}: {path} gives {entity!r} the colour {colour!r}; it has to be a hex "
                "colour like '#d62728' or '#d62'"
            )
        # Expanded and lower-cased here so the report holds one form and the widgets can
        # compare and hand them to a canvas without minding which was written.
        short = len(colour) == 4
        colours[str(entity)] = (
            "#" + "".join(c * 2 for c in colour[1:]).lower() if short else colour.lower()
        )
    return colours


def _env_colours(name: str) -> Dict[str, str]:
    """The colours the settings file gives, read once per process."""
    raw = os.environ.get(name)
    return dict(_colours_from(Path(raw.strip()), name)) if raw and raw.strip() else {}


def _env_voxel_size(name: str) -> Optional[Tuple[float, ...]]:
    """Sample size in µm: 'z,y,x' for volumes, 'y,x' for planes.

    Which of the two is right depends on the images, which this cannot see - the loader
    checks the length against the data it read and refuses a mismatch.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    parts = [p for p in raw.replace(" ", "").split(",") if p]
    if len(parts) not in (2, 3):
        raise ValueError(f"{name} must be 'z,y,x' (3D) or 'y,x' (2D) in µm, got {raw!r}")
    return tuple(float(p) for p in parts)


@dataclass(frozen=True)
class AnatomyConfig:
    # Required, never guessed: the region is cropped to this mask and polarity is measured
    # from its centroid. None is an omission, not a default - the loader refuses it and
    # names the masks the folder has.
    object_mask: Optional[str] = None
    # 'z,y,x' for a volume, 'y,x' for a plane; the loader refuses one that does not match
    # the dimensionality of the images it just read.
    voxel_size_um: Optional[Tuple[float, ...]] = None
    auto_clip: bool = False
    auto_label_masks: bool = False
    # Structure name -> "#rrggbb", from a settings file. Empty means the built-in palette.
    entity_colours: Dict[str, str] = field(default_factory=dict)
    max_skeleton_voxels: int = 500_000
    # None = every entity; an empty set = none. Skeletonising dominates a run, and a
    # blob's skeleton says nothing, so restricting it to filaments is the big lever.
    skeleton_entities: Optional[FrozenSet[str]] = None
    # 1, not 0: processors run in Dask workers, which cannot fork children, so asking
    # kimimaro for a process pool costs a failed attempt per object. Objects already run in
    # parallel.
    num_threads: int = 1
    # Separate from num_threads: edt is a C++ loop with no subprocesses, so it can use
    # all cores even inside a Dask worker, where kimimaro's process pool cannot.
    edt_threads: int = 0
    contact_max_um: float = 0.5
    # Both walk every voxel of every instance, so they are opt-in.
    polarity_spread: bool = False
    distance_histograms: bool = False
    # Geometry is written beside the report, never into it; unset means no meshing at all.
    mesh_dir: Optional[str] = None
    mesh_smooth_sigma: float = 0.7
    mesh_step_size: int = 2
    mesh_target_reduction: float = 0.8
    mesh_level: Optional[float] = None

    @classmethod
    def from_env(cls) -> "AnatomyConfig":
        return cls(
            object_mask=os.environ.get("PP_ANATOMY_OBJECT_MASK") or None,
            voxel_size_um=_env_voxel_size("PP_ANATOMY_VOXEL_SIZE_UM"),
            auto_clip=_env_flag("PP_ANATOMY_AUTO_CLIP"),
            auto_label_masks=_env_flag("PP_ANATOMY_AUTO_LABEL_MASKS"),
            entity_colours=_env_colours("PP_ANATOMY_ENTITY_COLOURS"),
            max_skeleton_voxels=_env_int("PP_ANATOMY_MAX_SKELETON_VOXELS", 500_000),
            skeleton_entities=parse_entity_filter(
                os.environ.get("PP_ANATOMY_SKELETON_ENTITIES"),
                none=_env_flag("PP_ANATOMY_NO_SKELETONS"),
            ),
            num_threads=_env_int("PP_ANATOMY_NUM_THREADS", 1),
            edt_threads=_env_int("PP_ANATOMY_EDT_THREADS", 0),
            contact_max_um=_env_float("PP_ANATOMY_CONTACT_MAX_UM", 0.5),
            polarity_spread=_env_flag("PP_ANATOMY_POLARITY_SPREAD"),
            distance_histograms=_env_flag("PP_ANATOMY_DISTANCE_HISTOGRAMS"),
            mesh_dir=os.environ.get("PP_ANATOMY_MESH_DIR") or None,
            mesh_smooth_sigma=_env_float("PP_ANATOMY_MESH_SMOOTH_SIGMA", 0.7),
            mesh_step_size=_env_int("PP_ANATOMY_MESH_STEP_SIZE", 2),
            mesh_target_reduction=_env_float("PP_ANATOMY_MESH_TARGET_REDUCTION", 0.8),
            mesh_level=(
                _env_float("PP_ANATOMY_MESH_LEVEL", 0.0)
                if os.environ.get("PP_ANATOMY_MESH_LEVEL") else None
            ),
        )
