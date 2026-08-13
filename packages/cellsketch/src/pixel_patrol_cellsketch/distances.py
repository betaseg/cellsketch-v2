"""Distance transforms, per-instance distances, and polarity.

Moved from ``analyze_cell.py``. The distance transform of every target is computed once
per cell, then sampled at each instance's voxels — so this needs the whole cell, which
is why it runs in the cell-level (MEMORY) processor rather than the per-entity one.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, Mapping, Sequence, Tuple

import edt
import numpy as np
from scipy.ndimage import distance_transform_edt

from pixel_patrol_cellsketch.discovery import is_membrane_name

logger = logging.getLogger(__name__)


def distance_transform_um(
    target_mask: np.ndarray,
    voxel_size_zyx: Sequence[float],
    num_threads: int = 0,
) -> np.ndarray:
    """Distance in µm from every voxel to the nearest voxel of ``target_mask``."""
    inverted = np.ascontiguousarray(~target_mask)
    threads = num_threads if num_threads and num_threads > 0 else (os.cpu_count() or 1)
    anisotropy = tuple(float(v) for v in voxel_size_zyx)
    try:
        return edt.edt(inverted, anisotropy=anisotropy, parallel=threads)
    except Exception:
        return distance_transform_edt(inverted, sampling=anisotropy)


def distance_target(volume: np.ndarray, name: str, kind: str) -> np.ndarray:
    """Binary target for one entity: what "distance to this entity" measures.

    A label entity's target is the union of its instances. The plasma membrane is a
    *filled* volume, so its target is inverted - distance to the membrane means distance
    to the cell boundary, measured from inside.

    One target at a time, on purpose: a whole-cell distance transform is float32 over
    every voxel, so holding one per entity is the difference between ~2 GB and ~11 GB on
    a 550-megavoxel cell.
    """
    mask = volume > 0
    if kind != "label" and is_membrane_name(name):
        return ~mask
    return mask


def cell_center_um(
    membrane_mask: np.ndarray,
    voxel_size_zyx: Sequence[float],
) -> Tuple[float, float, float] | None:
    coords = np.argwhere(membrane_mask > 0)
    if coords.size == 0:
        return None
    centroid = coords.mean(axis=0)
    return tuple(float(centroid[i] * voxel_size_zyx[i]) for i in range(3))  # type: ignore[return-value]


def polarity_from_offset(dz: float, dy: float, dx: float) -> Dict[str, float]:
    """Direction (and distance) from the cell center to a point, on a unit sphere."""
    dist_um = math.sqrt(dz**2 + dy**2 + dx**2)
    row: Dict[str, float] = {"polar_dist_um": dist_um}
    if dist_um > 0:
        nz, ny, nx_ = dz / dist_um, dy / dist_um, dx / dist_um
        row["polar_az_deg"] = math.degrees(math.atan2(ny, nx_))
        row["polar_el_deg"] = math.degrees(math.asin(max(-1.0, min(1.0, nz))))
    else:
        row["polar_az_deg"] = float("nan")
        row["polar_el_deg"] = float("nan")
    return row
