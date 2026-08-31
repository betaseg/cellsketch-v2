"""Pairwise surface-to-surface gaps between the instances of an object.

Volumes are passed in by entity name so this can be driven from a PixelPatrol record's
channels, and ``object_mask_name`` says which of them is the boundary rather than
something inside it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import edt
import numpy as np
from scipy.ndimage import find_objects

# One contact: (entity_a, label_a, entity_b, label_b, gap_um). label is None for a
# whole-structure mask, which has no instance id.
Contact = Tuple[str, int | None, str, int | None, float]


def pairwise_instance_gaps(
    volumes: Mapping[str, np.ndarray],
    kinds: Mapping[str, str],
    sample_size: Sequence[float],
    max_gap_um: float,
    object_mask_name: str | None = None,
) -> List[Contact]:
    """Surface-to-surface gaps between instances, up to ``max_gap_um`` µm.

    Every instance (each label id, and each mask as a whole) gets a global id in a
    combined volume; the gap between two instances is the smallest distance from one's
    voxels to the nearest voxel of the other.

    That distance is measured in whole voxel steps, so instances sharing a face read one
    voxel step rather than zero, and n empty voxels between them read n+1 steps. With
    anisotropic voxels the smallest reportable gap therefore depends on direction (in
    0.1×0.02×0.02 µm data, touching along Z reads 0.1 µm and along X reads 0.02 µm).

    Returned sorted, so a report is reproducible.
    """
    names = list(volumes)
    if not names:
        return []
    shape = volumes[names[0]].shape
    label_names = [n for n in names if kinds[n] == "label"]
    mask_names = [n for n in names if kinds[n] != "label"]

    # A global id per instance, and what each maps back to. Labels go first and are never
    # overwritten; other masks fill only unclaimed samples. The object mask is excluded: it
    # is the whole region, so its proximity is a distance to the boundary, not a contact.
    #
    # Remapped one entity at a time (label id + running offset), so the combined volume takes
    # a few vectorised passes rather than one full-volume scan per instance.
    L = np.zeros(shape, dtype=np.int32)
    owner: Dict[int, Tuple[str, int | None]] = {}
    offset = 0
    for name in label_names:
        lab = volumes[name]
        mask = lab > 0
        if not mask.any():
            continue
        ids = np.unique(lab[mask])
        L[mask] = lab[mask].astype(np.int32) + offset
        for lid in ids.tolist():
            owner[offset + int(lid)] = (name, int(lid))
        offset += int(ids.max())
    for name in mask_names:
        if name == object_mask_name:
            continue
        m = (volumes[name] > 0) & (L == 0)
        if not m.any():
            continue
        offset += 1
        L[m] = offset
        owner[offset] = (name, None)

    gid = int(L.max())
    if gid < 2 or len(owner) < 2:
        return []

    # Local EDT per instance: within its own bounding box padded by max_gap, measure every
    # other instance's samples to this one; the minimum is the surface-to-surface gap.
    # Bounded regions, so this stays fast with thousands of instances.
    slices = find_objects(L)
    anisotropy = tuple(float(v) for v in sample_size)
    # Bounding-box padding per axis, so everything within max_gap is inside the window.
    # One entry per spatial axis, which is the only difference between 2D and 3D here.
    reach = [int(math.ceil(max_gap_um / v)) for v in anisotropy]
    pair_min: Dict[Tuple[int, int], float] = {}
    for a_id in range(1, gid + 1):
        sl = slices[a_id - 1]
        if sl is None:
            continue
        window = tuple(
            slice(max(0, axis_slice.start - pad), min(extent, axis_slice.stop + pad))
            for axis_slice, pad, extent in zip(sl, reach, shape)
        )
        sub = L[window]
        other = (sub > 0) & (sub != a_id)
        if not other.any():
            continue
        # distance from every sample to the nearest a_id sample (a_id samples are the zeros)
        dt_a = edt.edt(np.ascontiguousarray((sub != a_id).astype(np.uint8)),
                       anisotropy=anisotropy, parallel=1)
        o_ids = sub[other]
        o_dist = dt_a[other]
        order = np.argsort(o_ids, kind="stable")
        o_ids = o_ids[order]
        o_dist = o_dist[order]
        uniq, first = np.unique(o_ids, return_index=True)
        mins = np.minimum.reduceat(o_dist, first)
        for b_id, gap in zip(uniq.tolist(), mins.tolist()):
            if gap > max_gap_um:
                continue
            key = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            if key not in pair_min or gap < pair_min[key]:
                pair_min[key] = float(gap)

    contacts: List[Contact] = []
    for (lo_id, hi_id), gap in pair_min.items():
        ea, la = owner[lo_id]
        eb, lb = owner[hi_id]
        contacts.append((ea, la, eb, lb, gap))
    contacts.sort(key=lambda c: (c[0], c[1] if c[1] is not None else -1,
                                 c[2], c[3] if c[3] is not None else -1))
    return contacts
