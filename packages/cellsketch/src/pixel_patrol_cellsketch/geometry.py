"""Shape metrics for a 3D binary structure or label instance.

Moved unchanged from ``analyze_cell.py`` — surface area, sphericity, PCA aspect
ratio, and the kimimaro curve-skeleton metrics (branches / length / tortuosity).
Skeletons are optional: without kimimaro installed the metrics come back NaN
instead of failing the run.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def estimate_surface_area_um2(binary_mask: np.ndarray, voxel_size_zyx: Tuple[float, float, float]) -> float:
    if binary_mask.sum() == 0:
        return 0.0
    m = binary_mask.astype(bool)
    p = np.pad(m, 1, mode="constant", constant_values=False)
    c = p[1:-1, 1:-1, 1:-1]
    zneg = c & ~p[:-2, 1:-1, 1:-1]
    zpos = c & ~p[2:, 1:-1, 1:-1]
    yneg = c & ~p[1:-1, :-2, 1:-1]
    ypos = c & ~p[1:-1, 2:, 1:-1]
    xneg = c & ~p[1:-1, 1:-1, :-2]
    xpos = c & ~p[1:-1, 1:-1, 2:]
    az = voxel_size_zyx[1] * voxel_size_zyx[2]
    ay = voxel_size_zyx[0] * voxel_size_zyx[2]
    ax = voxel_size_zyx[0] * voxel_size_zyx[1]
    return float((zneg.sum() + zpos.sum()) * az + (yneg.sum() + ypos.sum()) * ay + (xneg.sum() + xpos.sum()) * ax)


def sphericity(volume_um3: float, area_um2: float) -> float:
    if not np.isfinite(area_um2) or area_um2 <= 0 or volume_um3 <= 0:
        return float("nan")
    return float((math.pi ** (1.0 / 3.0)) * ((6.0 * volume_um3) ** (2.0 / 3.0)) / area_um2)


def aspect_ratio_from_coords(coords_zyx: np.ndarray, voxel_size_zyx: Tuple[float, float, float]) -> float:
    if coords_zyx.shape[0] < 3:
        return float("nan")
    pts = coords_zyx * np.array(voxel_size_zyx)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 0, None)
    lengths = np.sqrt(evals)
    # NaN when the smallest axis is negligible relative to the largest — flat/planar
    # structures (e.g. ER) where one PCA eigenvalue is near-zero from floating point,
    # not a true physical dimension.
    if lengths.max() <= 0 or lengths.min() < lengths.max() * 1e-8:
        return float("nan")
    return float(lengths.max() / lengths.min())


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


def compute_curve_skeletons(
    labels: np.ndarray,
    voxel_size_zyx: Tuple[float, float, float],
    max_voxels: int | None = None,
    num_threads: int = 0,
) -> dict | None:
    """TEASAR curve skeletons for every instance in a label volume (kimimaro).

    Returns ``{label_id: cloudvolume.Skeleton}``, vertices in µm in the volume frame
    (axis order ZYX). Instances larger than ``max_voxels`` are skipped to bound
    runtime. Returns ``None`` — as opposed to an empty dict — when kimimaro is not
    installed, so callers can report "not measured" instead of a skeleton of zero
    length.
    """
    try:
        import kimimaro
    except ImportError:
        logger.info("cellsketch: kimimaro unavailable — skeleton metrics disabled")
        return None

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
        logger.info("cellsketch: parallel kimimaro failed (%s); retrying single-threaded", type(exc).__name__)
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
