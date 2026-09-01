"""The spatial shape of a record: which axes are spatial, and how big one sample is.

Read from the record's own ``dim_order`` (``CZYX`` for a volume, ``CYX`` for a plane), so a
coordinate array and a sample size are always in the same order.
"""

from __future__ import annotations

from typing import Mapping, Tuple

_SPATIAL = ("Z", "Y", "X")


def spatial_axes(dim_order: str) -> str:
    """The spatial axes in array order: ``"ZYX"`` or ``"YX"``."""
    return "".join(ax for ax in dim_order if ax in _SPATIAL)


def voxel_size(meta: Mapping[str, object], dim_order: str) -> Tuple[float, ...]:
    """Size of one sample along each spatial axis, µm, in array order.

    ``pixel_size_Z`` is absent for a plane, never 1.0: a made-up depth would turn an area
    into a volume.
    """
    return tuple(float(meta[f"pixel_size_{ax}"]) for ax in spatial_axes(dim_order))


def object_center(meta: Mapping[str, object], dim_order: str) -> Tuple[float, ...] | None:
    """The object mask's centroid in µm, in array order, or None if it was not recorded."""
    center = [meta.get(f"object_center_{ax.lower()}_um") for ax in spatial_axes(dim_order)]
    if any(value is None for value in center):
        return None
    return tuple(float(value) for value in center)  # type: ignore[arg-type]


def unit_of_measure(ndim: int) -> str:
    """"area" or "volume", for error messages and log lines."""
    return "area" if ndim == 2 else "volume"
