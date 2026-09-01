"""Size and shape of labelled regions, in 2D or 3D, plus the skeleton metrics.

The measuring is ITK's, through SimpleITK's LabelShapeStatisticsImageFilter: it is spacing
aware, it does 2D and 3D, and its boundary estimate is the one a smooth surface deserves
(counting voxel faces reads about 1.5x high in 3D). Each dimensionality gets the columns that
mean something for it: volume, surface area and sphericity for a volume, area, perimeter and
circularity for a plane.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence, Tuple

import kimimaro
import numpy as np

logger = logging.getLogger(__name__)


# Column names minus any prefix: a caller adds "instance_" or nothing.
METRICS_3D = ("volume_um3", "surface_area_um2", "sphericity", "aspect_ratio_major_minor")
METRICS_2D = ("area_um2", "perimeter_um", "circularity", "aspect_ratio_major_minor")


def size_key(ndim: int) -> str:
    """volume_um3 for a volume, area_um2 for a plane."""
    return "area_um2" if ndim == 2 else "volume_um3"


def total_size_key(ndim: int) -> str:
    """Where an entity's summed extent goes: total_volume_um3 or total_area_um2."""
    return "total_area_um2" if ndim == 2 else "total_volume_um3"


def boundary_key(ndim: int) -> str:
    """surface_area_um2 for a volume, perimeter_um for a plane."""
    return "perimeter_um" if ndim == 2 else "surface_area_um2"


def roundness_key(ndim: int) -> str:
    """sphericity for a volume, circularity for a plane."""
    return "circularity" if ndim == 2 else "sphericity"


def label_metrics(labels: np.ndarray, sample_size: Sequence[float]) -> Dict[int, Dict[str, Any]]:
    """Size and shape of every labelled region, from ITK, in one pass over the volume.

    One call for the whole volume, not one per instance: 600 instances take 0.04 s this way
    against 0.9 s one at a time.

    Keys are the label ids. Each holds the extent, the boundary, the roundness,
    aspect_ratio_major_minor, the centroid in µm in array order, and the sample count.
    """
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.ascontiguousarray(labels.astype(np.uint32)))
    image.SetSpacing(tuple(float(v) for v in reversed(sample_size)))   # ITK orders (x, y, z)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.SetComputePerimeter(True)
    stats.Execute(image)

    planar = labels.ndim == 2
    measured: Dict[int, Dict[str, Any]] = {}
    for label in stats.GetLabels():
        # Elongation is the ratio of the two largest principal axes, flatness of the two
        # smallest, so their product is the largest over the smallest.
        aspect = stats.GetElongation(label) * (1.0 if planar else stats.GetFlatness(label))
        measured[int(label)] = {
            size_key(labels.ndim): stats.GetPhysicalSize(label),
            boundary_key(labels.ndim): stats.GetPerimeter(label),
            roundness_key(labels.ndim): stats.GetRoundness(label),
            "aspect_ratio_major_minor": aspect,
            "centroid_um": tuple(reversed(stats.GetCentroid(label))),
            "n_samples": int(stats.GetNumberOfPixels(label)),
        }
    return measured


def region_metrics(binary: np.ndarray, sample_size: Sequence[float]) -> Dict[str, float]:
    """Size and shape of one binary region, in the columns its dimensionality has."""
    keys = METRICS_2D if binary.ndim == 2 else METRICS_3D
    measured = label_metrics((binary > 0).astype(np.uint8), sample_size).get(1)
    if measured is None:
        return {key: float("nan") for key in keys}
    return {key: measured[key] for key in keys}


# Minimum component size (voxels) for kimimaro to attempt a skeleton.
SKELETON_DUST_VOXELS = 2

# TEASAR parameters for curve-skeleton extraction. Distances are in µm (matching the
# anisotropy we pass). scale/const set the branch-pruning aggressiveness; soma handling
# is disabled (it targets neurons, not organelles).
_TEASAR_PARAMS = {
    "scale": 1.5,
    "const": 0.05,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_detection_threshold": 1e9,
    "soma_acceptance_threshold": 1e9,
}


class _Polyline:
    """A 2D skeleton in the shape kimimaro's Skeleton is read in: vertices, edges, length.

    Lets ``skeleton_graph_metrics`` and the geometry writer treat both the same.
    """

    __slots__ = ("vertices", "edges")

    def __init__(self, vertices: np.ndarray, edges: np.ndarray) -> None:
        self.vertices = vertices
        self.edges = edges

    def cable_length(self) -> float:
        if len(self.edges) == 0:
            return 0.0
        a = self.vertices[self.edges[:, 0]]
        b = self.vertices[self.edges[:, 1]]
        return float(np.linalg.norm(a - b, axis=1).sum())


# 8-connected neighbour offsets, each pair counted once: the edges of a 2D skeleton.
_PLANAR_NEIGHBOURS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _planar_skeleton(binary: np.ndarray, pixel_size_yx: Sequence[float]) -> "_Polyline":
    """One instance's medial axis in a plane, as a graph in µm.

    ``skeletonize`` gives a one-pixel-wide set; edges join every 8-connected pair, so a
    diagonal step costs √2 pixels rather than 2.
    """
    from skimage.morphology import skeletonize

    thin = skeletonize(binary)
    coords = np.argwhere(thin)
    if not len(coords):
        return _Polyline(np.zeros((0, 2)), np.zeros((0, 2), dtype=int))
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(coords)}
    edges = []
    for i, (y, x) in enumerate(coords):
        for dy, dx in _PLANAR_NEIGHBOURS:
            j = index.get((int(y) + dy, int(x) + dx))
            if j is not None:
                edges.append((i, j))
    vertices = coords.astype(np.float64) * np.asarray(pixel_size_yx, dtype=float)
    return _Polyline(vertices, np.asarray(edges, dtype=int).reshape(-1, 2))


def compute_planar_skeletons(
    labels: np.ndarray,
    pixel_size_yx: Sequence[float],
    max_voxels: int | None = None,
) -> dict:
    """``{label_id: _Polyline}`` for every instance in a 2D label image.

    TEASAR is 3D-only; in a plane the medial axis is a thinning.
    """
    from skimage.measure import regionprops

    out: dict = {}
    for rp in regionprops(np.ascontiguousarray(labels.astype(np.int32))):
        if max_voxels is not None and rp.area > max_voxels:
            continue
        if rp.area < SKELETON_DUST_VOXELS:
            continue
        skeleton = _planar_skeleton(rp.image, pixel_size_yx)
        # regionprops cropped to the bbox; put the vertices back in the object's frame.
        origin = np.asarray(rp.bbox[:2], dtype=float) * np.asarray(pixel_size_yx, dtype=float)
        skeleton.vertices = skeleton.vertices + origin
        out[int(rp.label)] = skeleton
    return out


def compute_skeletons(
    labels: np.ndarray,
    sample_size: Sequence[float],
    max_voxels: int | None = None,
    num_threads: int = 0,
) -> dict:
    """Skeletons for every instance, by whichever dimensionality the labels have."""
    if labels.ndim == 2:
        return compute_planar_skeletons(labels, sample_size, max_voxels)
    return compute_curve_skeletons(labels, tuple(float(v) for v in sample_size),
                                   max_voxels=max_voxels, num_threads=num_threads)


def compute_curve_skeletons(
    labels: np.ndarray,
    voxel_size_zyx: Tuple[float, float, float],
    max_voxels: int | None = None,
    num_threads: int = 0,
) -> dict:
    """TEASAR curve skeletons for every instance in a 3D label volume (kimimaro).

    Returns ``{label_id: cloudvolume.Skeleton}``, vertices in µm in the volume frame
    (axis order ZYX). Instances larger than ``max_voxels`` are skipped to bound
    runtime.
    """
    lab = np.ascontiguousarray(labels.astype(np.uint32))
    object_ids = None
    if max_voxels is not None:
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        object_ids = [int(i) for i, c in zip(ids, counts) if c <= max_voxels]
        if not object_ids:
            return {}

    def _run(parallel: int) -> dict:
        return kimimaro.skeletonize(
            lab,
            teasar_params=_TEASAR_PARAMS,
            anisotropy=tuple(float(v) for v in voxel_size_zyx),
            object_ids=object_ids,
            dust_threshold=SKELETON_DUST_VOXELS,
            fix_branching=True,
            fix_borders=True,
            progress=False,
            parallel=parallel,
        )

    # kimimaro's multi-process path needs posix_ipc/psutil for shared memory; if anything
    # about it fails, fall back to single-threaded rather than crashing the whole run.
    parallel = num_threads if num_threads and num_threads > 0 else 0
    if parallel == 1:
        return _run(1)
    try:
        return _run(parallel)
    except Exception as exc:
        logger.info("anatomy: parallel kimimaro failed (%s); retrying single-threaded", type(exc).__name__)
        return _run(1)


def _branch_segments(adj: list[list[int]], deg: np.ndarray) -> list[list[int]]:
    """Split a skeleton graph into maximal chains between nodes of degree ≠ 2."""
    def ek(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    seen: set[tuple[int, int]] = set()
    segments: list[list[int]] = []
    breakpoints = [v for v in range(len(adj)) if adj[v] and deg[v] != 2]
    # A component made entirely of degree-2 nodes (a closed loop) has no breakpoints;
    # seed from any of its vertices so it still yields one segment.
    if not breakpoints:
        nonempty = [v for v in range(len(adj)) if adj[v]]
        breakpoints = nonempty[:1]
    for s in breakpoints:
        for nb in adj[s]:
            if ek(s, nb) in seen:
                continue
            seen.add(ek(s, nb))
            path = [s, nb]
            prev, cur = s, nb
            while deg[cur] == 2:
                nxt = [x for x in adj[cur] if x != prev]
                if not nxt or ek(cur, nxt[0]) in seen:
                    break
                seen.add(ek(cur, nxt[0]))
                path.append(nxt[0])
                prev, cur = cur, nxt[0]
            segments.append(path)
    return segments


def _smooth_polyline(pts: np.ndarray, window: int = 2, iters: int = 2) -> np.ndarray:
    """Moving-average smooth a polyline (endpoints fixed).

    Removes the 1-voxel staircase that voxelised skeletons carry, which would otherwise
    inflate tortuosity. Applied to the metric computation only, never to stored geometry.
    """
    if len(pts) <= 2:
        return pts
    p = pts.astype(np.float64, copy=True)
    for _ in range(iters):
        q = p.copy()
        for i in range(1, len(p) - 1):
            lo = max(0, i - window)
            hi = min(len(p), i + window + 1)
            q[i] = p[lo:hi].mean(axis=0)
        p = q
    return p


def skeleton_graph_metrics(sk) -> Dict[str, float]:
    """Shape metrics from a kimimaro curve skeleton, robust to voxel staircasing.

    branches: number of anatomical branches (chains between endpoints/junctions).
    length_um: total cable length (µm; raw skeleton).
    tortuosity: length-weighted mean of per-branch arc/chord ratio (≥1; 1 = straight).
    """
    nan = float("nan")
    empty = {"branches": 0, "length_um": 0.0, "tortuosity": nan}
    if sk is None or len(sk.vertices) == 0 or len(sk.edges) == 0:
        return empty
    verts = np.asarray(sk.vertices, dtype=np.float64)
    edges = np.asarray(sk.edges)
    n = len(verts)
    deg = np.bincount(edges.reshape(-1), minlength=n)
    length_um = float(sk.cable_length())

    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    segments = _branch_segments(adj, deg)
    branches = len(segments)

    # Tortuosity: length-weighted mean of arc/chord over branches (skip zero-chord loops).
    tot_arc = 0.0
    acc = 0.0
    for path in segments:
        pts = _smooth_polyline(verts[path])
        arc = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        if arc > 0 and chord > 0:
            acc += arc * (arc / chord)
            tot_arc += arc
    tortuosity = acc / tot_arc if tot_arc > 0 else nan

    return {"branches": branches, "length_um": length_um, "tortuosity": tortuosity}
