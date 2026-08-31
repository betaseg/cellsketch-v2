"""Distance transforms, per-instance distances, and polarity.

The distance transform of every target is computed once
per object, then sampled at each instance's voxels, so this needs the whole object, which
is why it runs in the object-level (MEMORY) processor rather than the per-entity one.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, Sequence, Tuple

import edt
import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


def distance_transform_um(
    target_mask: np.ndarray,
    sample_size: Sequence[float],
    num_threads: int = 0,
) -> np.ndarray:
    """Distance in µm from every sample to the nearest sample of ``target_mask``.

    2D and 3D alike: both edt and the scipy fallback take one anisotropy value per axis.
    """
    inverted = np.ascontiguousarray(~target_mask)
    threads = num_threads if num_threads and num_threads > 0 else (os.cpu_count() or 1)
    anisotropy = tuple(float(v) for v in sample_size)
    try:
        return edt.edt(inverted, anisotropy=anisotropy, parallel=threads)
    except Exception:
        return distance_transform_edt(inverted, sampling=anisotropy)


def distance_target(
    volume: np.ndarray,
    name: str,
    kind: str,
    object_mask_name: str | None = None,
) -> np.ndarray:
    """Binary target for one entity: what "distance to this entity" measures.

    A label entity's target is the union of its instances. The object mask is a *filled*
    volume, so its target is inverted - distance to it means distance to the object
    boundary, measured from inside.

    One target at a time, on purpose: a whole-object distance transform is float32 over
    every voxel, so holding one per entity is the difference between ~2 GB and ~11 GB on
    a 550-megavoxel object.
    """
    mask = volume > 0
    if kind != "label" and name == object_mask_name:
        return ~mask
    return mask


def object_center_um(
    object_mask: np.ndarray,
    voxel_size: Sequence[float],
) -> Tuple[float, ...] | None:
    """Centroid of the mask that bounds the object, in µm - the origin for polarity.

    Dimension-agnostic: three coordinates for a volume, two for a plane, in array order.
    """
    coords = np.argwhere(object_mask > 0)
    if coords.size == 0:
        return None
    centroid = coords.mean(axis=0)
    return tuple(float(centroid[i] * float(voxel_size[i])) for i in range(len(voxel_size)))


# The polarity columns of each dimensionality: a plane has one angle, not an azimuth and
# an elevation.
POLARITY_3D = ("polar_dist_um", "polar_nz", "polar_ny", "polar_nx",
               "polar_az_deg", "polar_el_deg")
POLARITY_2D = ("polar_dist_um", "polar_ny", "polar_nx", "polar_angle_deg")


def polarity_from_offset(offset: Sequence[float]) -> Dict[str, float]:
    """Direction (and distance) from the object centre to a point.

    3D: the unit vector (polar_nz/ny/nx) plus azimuth and elevation. 2D: the unit vector
    (polar_ny/nx) plus polar_angle_deg. The other set is left out rather than zeroed: an
    elevation of 0° would read as "equatorial" rather than "no third axis".
    """
    offsets = [float(v) for v in offset]
    dist_um = math.sqrt(sum(v * v for v in offsets))
    row: Dict[str, float] = {"polar_dist_um": dist_um}
    planar = len(offsets) == 2
    keys = POLARITY_2D if planar else POLARITY_3D
    if dist_um <= 0:
        for key in keys[1:]:
            row[key] = float("nan")
        return row
    units = [v / dist_um for v in offsets]
    if planar:
        ny, nx_ = units
        # The vector, not only the angle: the viewer explodes instances along it.
        row["polar_ny"], row["polar_nx"] = ny, nx_
        row["polar_angle_deg"] = math.degrees(math.atan2(ny, nx_))
        return row
    nz, ny, nx_ = units
    row["polar_nz"], row["polar_ny"], row["polar_nx"] = nz, ny, nx_
    row["polar_az_deg"] = math.degrees(math.atan2(ny, nx_))
    row["polar_el_deg"] = math.degrees(math.asin(max(-1.0, min(1.0, nz))))
    return row
