#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.2.0",
#   "scipy>=1.13.0",
#   "scikit-image>=0.23.0",
#   "tifffile>=2024.5.0",
#   "edt>=2.4.0",
#   "fast-simplification>=0.1.6",
#   "imagecodecs"
# ]
# ///

from __future__ import annotations

import argparse
import base64
import gzip
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import edt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter, label as nd_label
import fast_simplification
from skimage.measure import marching_cubes, regionprops
from skimage.morphology import skeletonize


@dataclass
class Entity:
    name: str
    kind: str  # "label" or "mask"
    path: Path


@dataclass
class Dataset:
    source: Path
    membrane_name: str
    entities: Dict[str, Entity]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic 3D label/mask spatial analysis for one cell.")
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--voxel-size-um", default=None, help="Optional z,y,x um. If omitted, infer from source TIFF.")
    parser.add_argument("--auto-clip-to-pm", action="store_true")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--with-mesh", action="store_true", help="Generate a 3D mesh per instance (required for mesh_viewer.html and Blender export).")
    parser.add_argument("--mesh-smooth-sigma", type=float, default=0.7, metavar="SIGMA", help="Gaussian sigma for mesh smoothing before marching cubes (default: 0.7). Set to 0 to disable.")
    parser.add_argument("--mesh-step-size", type=int, default=2, metavar="N", help="Marching cubes step size — controls mesh resolution (default: 2). 1 = full resolution, higher = coarser.")
    parser.add_argument("--mesh-target-reduction", type=float, default=0.8, metavar="F", help="QEM decimation target reduction fraction (default: 0.8 = keep 20%% of faces). Set to 0 to disable.")
    parser.add_argument("--mesh-level", type=float, default=None, metavar="L", help="Marching cubes iso-surface level (default: 0.25 with smoothing, 0.5 without). Lower values capture finer structures.")
    parser.add_argument("--force-reprocess", action="store_true", help="Re-process cells even if report.csv already exists.")
    parser.add_argument(
        "--max-skeleton-voxels",
        type=int,
        default=500_000,
        help="Skip branch counting for label instances larger than this voxel count (default: 500000).",
    )
    parser.add_argument(
        "--dist-histogram-labels",
        default="",
        metavar="NAMES",
        help="Comma-separated label entity names for which per-pixel distance distributions are computed "
             "(adds mean/hist_min/hist_max/hist columns). Example: mito,er",
    )
    parser.add_argument(
        "--dist-histogram-bins",
        type=int,
        default=20,
        metavar="N",
        help="Number of histogram bins for --dist-histogram-labels (default: 20).",
    )
    parser.add_argument(
        "--polarity-spread-labels",
        default="",
        metavar="NAMES",
        help="Comma-separated label entity names for which per-pixel angular spread on the polarity sphere is computed "
             "(adds polar_angular_spread_deg). Example: mito,er",
    )
    return parser.parse_args()


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def shared_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def parse_entity_file(stem: str) -> tuple[str, str, str] | None:
    s = normalize_name(stem)
    m = re.match(r"^(.*)_([a-z0-9_]+)_(labels?|mask)$", s)
    if not m:
        return None
    prefix = m.group(1)
    name = normalize_name(m.group(2))
    suffix = m.group(3)
    kind = "label" if suffix.startswith("label") else "mask"
    if not prefix or not name:
        return None
    return prefix, name, kind


def is_membrane_name(name: str) -> bool:
    n = normalize_name(name)
    return ("pm" == n) or ("plasma" in n) or ("membrane" in n)


def discover_dataset(cell_dir: Path) -> Dataset:
    tiffs = sorted(cell_dir.glob("*.tif*"))
    if not tiffs:
        raise FileNotFoundError(f"No TIFF files found in {cell_dir}")

    parsed = {}
    for p in tiffs:
        entry = parse_entity_file(p.stem)
        if entry:
            parsed[p] = entry

    if not parsed:
        raise FileNotFoundError("No NAME_label(s) or NAME_mask files found.")

    source_candidates = [p for p in tiffs if p not in parsed]
    if not source_candidates:
        raise FileNotFoundError("No source TIFF found (expected a non label/mask TIFF).")

    # Choose source TIFF with strongest shared prefix with derived entity prefixes.
    derived_prefixes = [pref for pref, _, _ in parsed.values()]

    def source_score(p: Path) -> tuple[int, int]:
        s = normalize_name(p.stem)
        score = sum(shared_prefix_len(s, pref) for pref in derived_prefixes)
        return score, int(p.stat().st_size)

    source = max(source_candidates, key=source_score)
    source_norm = normalize_name(source.stem)

    # Adaptive prefix matching:
    # 1. Find the longest shared prefix any entity has with the source name.
    # 2. Accept only entities whose shared length equals that maximum.
    # 3. When names fully exhaust the shorter string (clean match), additionally
    #    require a word-boundary in the longer one to reject "c1" ↔ "c10" style
    #    false positives.  When names diverge mid-string (e.g. source has extra
    #    tokens the entity files don't), skip the boundary check — the max-shared
    #    filter is sufficient.
    shared_lens = {p: shared_prefix_len(source_norm, pref) for p, (pref, _, _) in parsed.items()}
    max_shared = max(shared_lens.values()) if shared_lens else 0

    entities: Dict[str, Entity] = {}
    for p, (pref, name, kind) in parsed.items():
        sl = shared_lens[p]
        if sl < max_shared:
            continue
        min_len = min(len(source_norm), len(pref))
        if sl >= min_len:
            # Names match all the way to the end of the shorter one: apply
            # word-boundary guard so "c1" does not match a "c10" prefix.
            longer = source_norm if len(source_norm) >= len(pref) else pref
            if sl < len(longer) and longer[sl] != "_":
                continue
        key = f"{kind}:{name}"
        entities[key] = Entity(name=name, kind=kind, path=p)

    if not entities:
        raise FileNotFoundError(
            "No NAME_label(s) or NAME_mask entities matching source basename were found."
        )

    membrane_candidates = [e.name for e in entities.values() if e.kind == "mask" and is_membrane_name(e.name)]
    if not membrane_candidates:
        raise FileNotFoundError("No membrane mask found (expected NAME with pm/plasma/membrane).")
    membrane_name = sorted(membrane_candidates)[0]

    return Dataset(source=source, membrane_name=membrane_name, entities=entities)


def infer_voxel_size_um_from_source(source_path: Path) -> Tuple[float, float, float]:
    with tifffile.TiffFile(source_path) as tf:
        page = tf.pages[0]
        ij = tf.imagej_metadata or {}

        z_um = float(ij["spacing"]) if "spacing" in ij else None

        x_um = None
        y_um = None
        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        unit_tag = page.tags.get("ResolutionUnit")
        unit_code = int(unit_tag.value) if unit_tag is not None else 1
        unit_to_um = {2: 25400.0, 3: 10000.0}.get(unit_code)
        if unit_to_um and xres_tag is not None and yres_tag is not None:
            x_num, x_den = xres_tag.value
            y_num, y_den = yres_tag.value
            if x_num:
                x_um = unit_to_um * float(x_den) / float(x_num)
            if y_num:
                y_um = unit_to_um * float(y_den) / float(y_num)
        if x_um is None or y_um is None:
            if xres_tag is not None and yres_tag is not None:
                x_num, x_den = xres_tag.value
                y_num, y_den = yres_tag.value
                if x_num:
                    x_um = float(x_den) / float(x_num)
                if y_num:
                    y_um = float(y_den) / float(y_num)

        if z_um is None or x_um is None or y_um is None:
            raise ValueError(f"Could not infer voxel size from source metadata: {source_path.name}")

    print(f"[meta] voxel_size_um from source {source_path.name}: z={z_um}, y={y_um}, x={x_um}", flush=True)
    return (z_um, y_um, x_um)


def load_volume(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume in {path}, got shape {arr.shape}")
    return arr


def clip_to_membrane(arr: np.ndarray, membrane: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[~membrane] = 0
    return out


def crop_to_membrane_bbox(volumes: Dict[str, np.ndarray], membrane_key: str) -> Dict[str, np.ndarray]:
    membrane = volumes[membrane_key] > 0
    coords = np.argwhere(membrane)
    if coords.size == 0:
        return volumes
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    sl = tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))
    return {k: v[sl] for k, v in volumes.items()}


def compute_distance_transform_exact(target_mask: np.ndarray, voxel_size_zyx: Tuple[float, float, float], num_threads: int) -> np.ndarray:
    inverted = np.ascontiguousarray(~target_mask)
    threads = num_threads if num_threads and num_threads > 0 else (os.cpu_count() or 1)
    try:
        return edt.edt(inverted, anisotropy=voxel_size_zyx, parallel=threads)
    except Exception:
        return distance_transform_edt(inverted, sampling=voxel_size_zyx)


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
    # Return NaN when the smallest axis is negligible relative to the largest —
    # happens for flat/planar structures (e.g. ER) where one PCA eigenvalue is
    # near-zero due to floating point, not a true physical dimension.
    if lengths.max() <= 0 or lengths.min() < lengths.max() * 1e-8:
        return float("nan")
    return float(lengths.max() / lengths.min())


def skeleton_metrics(
    binary_mask: np.ndarray,
    voxel_size_zyx: Tuple[float, float, float],
    max_voxels: int | None = None,
) -> dict:
    """Run skeletonise once and return branches, length_um, and tortuosity.

    Length: sum of physical edge lengths in the 26-connected skeleton graph.
    Tortuosity: length / straight-line end-to-end distance (only for simple
    filaments with exactly two endpoints; NaN for branching structures).
    """
    nan_row = {"branches": float("nan"), "length_um": float("nan"), "tortuosity": float("nan")}
    n = int(binary_mask.sum())
    if n == 0:
        return {"branches": 0, "length_um": 0.0, "tortuosity": float("nan")}
    if max_voxels is not None and n > max_voxels:
        return nan_row

    skel = skeletonize(binary_mask.astype(bool))
    if not skel.any():
        return {"branches": 0, "length_um": 0.0, "tortuosity": float("nan")}

    vz, vy, vx = voxel_size_zyx

    # ── Skeleton length ────────────────────────────────────────────────────
    # Precompute the 26 neighbour offsets and their physical distances.
    neighbour_steps: list[tuple[tuple[int, int, int], float]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                neighbour_steps.append(
                    ((dz, dy, dx), math.sqrt((dz * vz) ** 2 + (dy * vy) ** 2 + (dx * vx) ** 2))
                )

    coords = np.argwhere(skel)
    coord_set = set(map(tuple, coords.tolist()))
    length_um = 0.0
    for z, y, x in coords:
        for (dz, dy, dx), dist in neighbour_steps:
            if (z + dz, y + dy, x + dx) in coord_set:
                length_um += dist
    length_um /= 2.0  # each edge is traversed from both endpoints

    # ── Branch count ───────────────────────────────────────────────────────
    kernel = np.ones((3, 3, 3), dtype=np.int16)
    kernel[1, 1, 1] = 0
    degree = convolve(skel.astype(np.int16), kernel, mode="constant", cval=0)
    junction_mask = skel & (degree >= 3)
    connectivity = np.ones((3, 3, 3), dtype=np.uint8)
    segment_mask = skel & ~junction_mask
    _, n_segments = nd_label(segment_mask.astype(np.uint8), structure=connectivity)
    branches = int(n_segments)

    # ── Tortuosity ─────────────────────────────────────────────────────────
    # Defined only for simple filaments (exactly two endpoints), matching
    # tortuosity = length / straight_line_distance.
    endpoint_mask = skel & (degree == 1)
    ep_coords = np.argwhere(endpoint_mask)
    tortuosity = float("nan")
    if len(ep_coords) == 2:
        p1 = ep_coords[0].astype(float) * np.array([vz, vy, vx])
        p2 = ep_coords[1].astype(float) * np.array([vz, vy, vx])
        end_to_end = float(np.linalg.norm(p2 - p1))
        if end_to_end > 0:
            tortuosity = length_um / end_to_end

    return {"branches": branches, "length_um": length_um, "tortuosity": tortuosity}


def compute_cell_center_um(membrane_mask: np.ndarray, voxel_size_zyx: Tuple[float, float, float]) -> Tuple[float, float, float] | None:
    coords = np.argwhere(membrane_mask > 0)
    if coords.size == 0:
        return None
    centroid = coords.mean(axis=0)
    return (float(centroid[0] * voxel_size_zyx[0]),
            float(centroid[1] * voxel_size_zyx[1]),
            float(centroid[2] * voxel_size_zyx[2]))


def per_label_metrics(
    labels: np.ndarray,
    voxel_size_zyx: Tuple[float, float, float],
    distance_transforms: Dict[str, np.ndarray],
    max_skeleton_voxels: int | None = None,
    compute_dist_histogram: bool = False,
    dist_histogram_bins: int = 20,
    cell_center_zyx_um: Tuple[float, float, float] | None = None,
    compute_polarity_spread: bool = False,
) -> pd.DataFrame:
    import json
    voxel_um3 = float(np.prod(voxel_size_zyx))
    rows = []
    props = regionprops(labels)
    centroids = []
    for rp in props:
        skel = skeleton_metrics(rp.image, voxel_size_zyx, max_voxels=max_skeleton_voxels)
        row = {
            "label": int(rp.label),
            "volume_um3": float(rp.area * voxel_um3),
            "surface_area_um2": estimate_surface_area_um2(rp.image, voxel_size_zyx),
            "aspect_ratio_major_minor": aspect_ratio_from_coords(rp.coords, voxel_size_zyx),
            "branches": skel["branches"],
            "length_um": skel["length_um"],
            "tortuosity": skel["tortuosity"],
        }
        row["sphericity"] = sphericity(row["volume_um3"], row["surface_area_um2"])
        for key, dt in distance_transforms.items():
            vals = dt[tuple(rp.coords.T)] if rp.coords.size else np.array([], dtype=np.float32)
            row[f"distance_to_{key}_um"] = float(vals.min()) if vals.size else float("nan")
            if compute_dist_histogram:
                if vals.size:
                    hist_min = float(vals.min())
                    hist_max = float(vals.max())
                    mean_val = float(vals.mean())
                    if hist_min < hist_max:
                        counts, _ = np.histogram(vals, bins=dist_histogram_bins, range=(hist_min, hist_max))
                    else:
                        counts = np.array([int(vals.size)] + [0] * (dist_histogram_bins - 1), dtype=np.int64)
                    row[f"distance_to_{key}_mean_um"] = mean_val
                    row[f"distance_to_{key}_hist_min_um"] = hist_min
                    row[f"distance_to_{key}_hist_max_um"] = hist_max
                    row[f"distance_to_{key}_hist_um"] = json.dumps(counts.tolist())
                else:
                    row[f"distance_to_{key}_mean_um"] = float("nan")
                    row[f"distance_to_{key}_hist_min_um"] = float("nan")
                    row[f"distance_to_{key}_hist_max_um"] = float("nan")
                    row[f"distance_to_{key}_hist_um"] = None

        # Polarity: direction from cell center to instance centroid on a unit sphere
        if cell_center_zyx_um is not None:
            cz, cy, cx = cell_center_zyx_um
            iz = float(rp.centroid[0]) * voxel_size_zyx[0]
            iy = float(rp.centroid[1]) * voxel_size_zyx[1]
            ix = float(rp.centroid[2]) * voxel_size_zyx[2]
            dz, dy, dx = iz - cz, iy - cy, ix - cx
            dist_um = math.sqrt(dz**2 + dy**2 + dx**2)
            row["polar_dist_um"] = dist_um
            if dist_um > 0:
                nz, ny, nx_ = dz / dist_um, dy / dist_um, dx / dist_um
                row["polar_nz"] = nz
                row["polar_ny"] = ny
                row["polar_nx"] = nx_
                row["polar_az_deg"] = math.degrees(math.atan2(ny, nx_))
                row["polar_el_deg"] = math.degrees(math.asin(max(-1.0, min(1.0, nz))))
            else:
                for k in ("polar_nz", "polar_ny", "polar_nx", "polar_az_deg", "polar_el_deg"):
                    row[k] = float("nan")

            if compute_polarity_spread and rp.coords.size >= 3:
                pz = rp.coords[:, 0].astype(float) * voxel_size_zyx[0] - cz
                py = rp.coords[:, 1].astype(float) * voxel_size_zyx[1] - cy
                px = rp.coords[:, 2].astype(float) * voxel_size_zyx[2] - cx
                dists_px = np.sqrt(pz**2 + py**2 + px**2)
                valid = dists_px > 0
                if valid.sum() >= 3:
                    pnz = pz[valid] / dists_px[valid]
                    pny = py[valid] / dists_px[valid]
                    pnx = px[valid] / dists_px[valid]
                    mean_dir = np.array([pnz.mean(), pny.mean(), pnx.mean()])
                    mlen = float(np.linalg.norm(mean_dir))
                    if mlen > 0:
                        mean_dir /= mlen
                    dots = np.clip(pnz * mean_dir[0] + pny * mean_dir[1] + pnx * mean_dir[2], -1.0, 1.0)
                    row["polar_angular_spread_deg"] = float(np.degrees(np.arccos(dots)).std())
                else:
                    row["polar_angular_spread_deg"] = float("nan")

        rows.append(row)
        centroids.append(np.array(rp.centroid) * np.array(voxel_size_zyx))

    df = pd.DataFrame(rows)
    if not df.empty and len(centroids) > 1:
        c = np.vstack(centroids)
        nearest = np.full((c.shape[0],), np.nan, dtype=float)
        for i in range(c.shape[0]):
            d = np.sqrt(np.sum((c - c[i]) ** 2, axis=1))
            d[i] = np.inf
            nearest[i] = d.min()
        df["distance_to_closest_same_type_um"] = nearest
    return df




def _mesh_sigma_for_shape(sphericity: float, fill_ratio: float,
                          sigma_min: float = 0.3, sigma_max: float = 1.5) -> float:
    """Choose Gaussian sigma based on shape: blobs get more smoothing, thin structures less.

    Uses two complementary metrics:
    - sphericity [0,1]: how sphere-like the shape is (surface/volume ratio)
    - fill_ratio [0,1]: voxels / bbox_voxels — how densely the shape fills its
      bounding box. Captures curved filaments that fold back on themselves and
      therefore appear compact in PCA but are actually sparse.

    Falls back to the midpoint when metrics are NaN.
    """
    _SPHERE_FILL = math.pi / 6.0  # theoretical fill ratio of a perfect sphere (~0.524)
    if math.isnan(sphericity) or math.isnan(fill_ratio):
        return (sigma_min + sigma_max) / 2.0
    fill_score = min(1.0, fill_ratio / _SPHERE_FILL)
    blob_score = math.sqrt(sphericity * fill_score)  # geometric mean of both
    return sigma_min + (sigma_max - sigma_min) * blob_score


def _generate_mesh_b64(
    binary: np.ndarray,
    bbox_origin_zyx: Tuple[int, int, int],
    voxel_size_zyx: Tuple[float, float, float],
    step_size: int = 2,
    smooth_sigma: float = 0.7,
    target_reduction: float = 0.8,
    level: float | None = None,
) -> str:
    """Mesh a (Z,Y,X) binary mask via marching cubes.

    Binary payload (gzip-compressed, then base64):
      [uint32 nV][uint32 nF]
      [float32×3 min_xyz][float32×3 scale_xyz]   ← dequantisation params
      [uint16 × nV×3 quantised XYZ vertices]
      [uint32 × nF×3 face indices]
    """
    if binary.sum() < 8:
        return ""
    try:
        padded = np.pad(binary.astype(np.float32), pad_width=1)
        if smooth_sigma > 0:
            padded = gaussian_filter(padded, sigma=smooth_sigma)
            iso_level = level if level is not None else 0.25
        else:
            iso_level = level if level is not None else 0.5
        verts, faces, _, _ = marching_cubes(padded, level=iso_level, step_size=step_size)
        verts = verts - 1.0  # undo padding offset → local voxel coords
        oz, oy, ox = bbox_origin_zyx
        sz, sy, sx = voxel_size_zyx
        # Reorder ZYX → XYZ in µm (Three.js convention)
        verts_xyz = np.column_stack([
            (verts[:, 2] + ox) * sx,
            (verts[:, 1] + oy) * sy,
            (verts[:, 0] + oz) * sz,
        ]).astype(np.float32)
        if target_reduction > 0 and len(faces) > 100:
            verts_xyz, faces_s = fast_simplification.simplify(
                verts_xyz, faces.astype(int), target_reduction=target_reduction, verbose=False,
            )
            verts_xyz = verts_xyz.astype(np.float32)
            faces = faces_s
        # Quantise to uint16 to halve vertex storage
        min_xyz = verts_xyz.min(axis=0)
        scale_xyz = verts_xyz.max(axis=0) - min_xyz
        scale_xyz[scale_xyz == 0] = 1.0
        verts_q = np.clip(
            (verts_xyz - min_xyz) / scale_xyz * 65535 + 0.5, 0, 65535
        ).astype(np.uint16)
        faces32 = faces.astype(np.uint32)
        quant_params = np.concatenate([min_xyz, scale_xyz]).astype(np.float32)
        header = np.array([len(verts_xyz), len(faces32)], dtype=np.uint32)
        payload = header.tobytes() + quant_params.tobytes() + verts_q.tobytes() + faces32.tobytes()
        return base64.b64encode(gzip.compress(payload, compresslevel=9)).decode()
    except Exception:
        return ""


def resolve_voxel_size_zyx(args: argparse.Namespace, source_path: Path) -> Tuple[float, float, float]:
    if args.voxel_size_um:
        voxel_size_zyx = tuple(float(x) for x in args.voxel_size_um.split(","))
        if len(voxel_size_zyx) != 3:
            raise ValueError("Expected voxel size as z,y,x")
        print(f"Using provided voxel size (z,y,x): {voxel_size_zyx}", flush=True)
        return voxel_size_zyx
    return infer_voxel_size_um_from_source(source_path)


def load_entity_volumes(dataset: Dataset) -> Dict[str, np.ndarray]:
    print("Loading entity volumes...", flush=True)
    volumes = {key: load_volume(entity.path) for key, entity in dataset.entities.items()}
    print("Loaded all entity volumes.", flush=True)
    return volumes


def apply_membrane_clipping(volumes: Dict[str, np.ndarray], membrane_key: str, enabled: bool) -> None:
    if not enabled:
        return
    print("Clipping all entities to membrane...", flush=True)
    membrane = volumes[membrane_key] > 0
    for key in list(volumes.keys()):
        if key != membrane_key:
            volumes[key] = clip_to_membrane(volumes[key], membrane)
    print("Clipping done.", flush=True)


def load_or_generate_masks(
    dataset: Dataset,
    generated_masks_dir: Path,
    membrane_key: str,
    auto_clip: bool,
) -> Dict[str, np.ndarray]:
    """Load or cache per-entity volumes after membrane clipping.

    CCA-based promotion of multi-component masks to label entities happens
    separately, after cropping to the cell bounding box, via
    promote_multicomponent_masks().
    """
    generated_masks_dir.mkdir(parents=True, exist_ok=True)
    cached_paths = {key: generated_masks_dir / entity.path.name for key, entity in dataset.entities.items()}
    all_cached = all(p.exists() for p in cached_paths.values())

    if all_cached:
        print(f"Generated masks cache hit — loading from {generated_masks_dir.name}/", flush=True)
        return {key: load_volume(path) for key, path in cached_paths.items()}

    print("Generated masks cache miss — loading originals...", flush=True)
    volumes = load_entity_volumes(dataset)
    apply_membrane_clipping(volumes, membrane_key, auto_clip)

    for key, arr in volumes.items():
        entity = dataset.entities.get(key)
        if entity is None:
            continue
        path = cached_paths.get(key)
        if path and not path.exists():
            out = arr if entity.kind == "label" else (arr > 0).astype(np.uint8)
            tifffile.imwrite(path, out, compression="zlib")
    print("Generated masks written.", flush=True)

    return volumes


def promote_multicomponent_masks(
    volumes: Dict[str, np.ndarray],
    dataset: Dataset,
    membrane_key: str,
) -> None:
    """Promote masks with >1 connected component (in the cropped volume) to label entities.

    Must be called after crop_to_membrane_bbox so the component count reflects
    what is actually inside the cell, not the full image.
    """
    cc_struct = np.ones((3, 3, 3), dtype=np.uint8)
    for key in list(volumes.keys()):
        entity = dataset.entities[key]
        if key == membrane_key:
            continue
        # Skip label entities that already have multiple distinct labels — already segmented.
        if entity.kind == "label" and int(volumes[key].max()) > 1:
            continue

        binary = volumes[key] > 0
        if not binary.any():
            continue

        labeled, n = nd_label(binary, structure=cc_struct)
        if n <= 1:
            continue

        labeled = labeled.astype(np.int32)
        volumes[key] = labeled
        dataset.entities[key] = Entity(name=entity.name, kind="label", path=entity.path)
        print(f"  Auto-label: '{entity.name}' — {n} components → label entity.", flush=True)



def build_distance_targets(
    volumes: Dict[str, np.ndarray],
    entities: Dict[str, Entity],
    label_keys: list[str],
    mask_keys: list[str],
    membrane_key: str,
) -> Dict[str, np.ndarray]:
    target_binary: Dict[str, np.ndarray] = {}
    for mk in mask_keys:
        mask_name = entities[mk].name
        mask = volumes[mk] > 0
        target_binary[mask_name] = ~mask if mk == membrane_key else mask
    for lk in label_keys:
        label_name = entities[lk].name
        target_binary[f"label_{label_name}"] = volumes[lk] > 0
    return target_binary


def build_or_load_dt_cache(
    target_binary: Dict[str, np.ndarray],
    dt_cache_dir: Path,
    voxel_size_zyx: Tuple[float, float, float],
    num_threads: int,
) -> Dict[str, np.ndarray]:
    print("Building/reusing distance-transform cache...", flush=True)
    dt_cache_dir.mkdir(parents=True, exist_ok=True)
    all_dts: Dict[str, np.ndarray] = {}
    for target_name, target_mask in target_binary.items():
        cache_file = dt_cache_dir / f"{normalize_name(target_name)}_dt.npy"
        use_cached = False
        if cache_file.exists():
            try:
                dt_cached = np.load(cache_file, mmap_mode="r")
                if dt_cached.shape == target_mask.shape:
                    all_dts[target_name] = dt_cached
                    use_cached = True
            except Exception:
                use_cached = False
        if use_cached:
            print(f"DT cache hit: {target_name}", flush=True)
            continue

        print(f"Computing DT: {target_name}", flush=True)
        dt = compute_distance_transform_exact(target_mask, voxel_size_zyx, num_threads).astype(np.float32)
        np.save(cache_file, dt)
        all_dts[target_name] = np.load(cache_file, mmap_mode="r")
    print("Distance-transform cache ready.", flush=True)
    return all_dts


def distance_transforms_for_source(
    source_key: str,
    label_keys: list[str],
    mask_keys: list[str],
    entities: Dict[str, Entity],
    all_dts: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    dts_for_src = {entities[mk].name: all_dts[entities[mk].name] for mk in mask_keys}
    for lk in label_keys:
        if lk == source_key:
            continue
        lname = entities[lk].name
        dts_for_src[f"label_{lname}"] = all_dts[f"label_{lname}"]
    return dts_for_src


def analyze_label_entities(
    args: argparse.Namespace,
    volumes: Dict[str, np.ndarray],
    entities: Dict[str, Entity],
    label_keys: list[str],
    mask_keys: list[str],
    all_dts: Dict[str, np.ndarray],
    voxel_size_zyx: Tuple[float, float, float],
    cell_center_zyx_um: Tuple[float, float, float] | None = None,
) -> Dict[str, pd.DataFrame]:
    hist_names = {normalize_name(n) for n in args.dist_histogram_labels.split(",") if n.strip()} if args.dist_histogram_labels else set()
    spread_names = {normalize_name(n) for n in args.polarity_spread_labels.split(",") if n.strip()} if args.polarity_spread_labels else set()
    label_dfs: Dict[str, pd.DataFrame] = {}
    for src_key in label_keys:
        src_entity = entities[src_key]
        print(f"Analyzing label entity: {src_entity.name}", flush=True)
        src_labels = volumes[src_key].astype(np.int32)
        dts_for_src = distance_transforms_for_source(src_key, label_keys, mask_keys, entities, all_dts)
        df = per_label_metrics(
            src_labels,
            voxel_size_zyx,
            dts_for_src,
            max_skeleton_voxels=args.max_skeleton_voxels,
            compute_dist_histogram=src_entity.name in hist_names,
            dist_histogram_bins=args.dist_histogram_bins,
            cell_center_zyx_um=cell_center_zyx_um,
            compute_polarity_spread=src_entity.name in spread_names,
        )
        label_dfs[src_key] = df
        print(f"  {src_entity.name}: {len(df)} instances", flush=True)
    return label_dfs


def _sanitize(v: object) -> object:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def build_report_rows(
    cell_id: str,
    group_id: str,
    volumes: Dict[str, np.ndarray],
    entities: Dict[str, Entity],
    label_keys: list[str],
    mask_keys: list[str],
    label_dfs: Dict[str, pd.DataFrame],
    voxel_size_zyx: Tuple[float, float, float],
    voxel_um3: float,
    with_mesh: bool = False,
    mesh_smooth_sigma: float = 0.7,
    mesh_step_size: int = 2,
    mesh_target_reduction: float = 0.8,
    mesh_level: float | None = None,
    source_path: Path | None = None,
    generated_masks_dir: Path | None = None,
) -> list[dict]:
    def _file_meta(path: Path) -> dict:
        resolved = path
        if generated_masks_dir is not None:
            candidate = generated_masks_dir / path.name
            if candidate.exists():
                resolved = candidate
        try:
            st = resolved.stat()
            return {
                "file_size_bytes": st.st_size,
                "file_mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "file_name": resolved.name,
            }
        except OSError:
            return {"file_size_bytes": None, "file_mtime": None, "file_name": path.name}

    rows: list[dict] = []

    if source_path is not None:
        rows.append({
            "cell_id": cell_id, "group_id": group_id, "entity_name": "source",
            "entity_kind": "source", "row_type": "file",
            "label_id": None, "instance_count": None, "total_volume_um3": None,
            **_file_meta(source_path),
        })

    for mk in mask_keys:
        ent = entities[mk]
        binary = (volumes[mk] > 0)
        vol_um3 = float(binary.sum() * voxel_um3)
        area_um2 = estimate_surface_area_um2(binary, voxel_size_zyx)
        sph = sphericity(vol_um3, area_um2)
        rps = regionprops(binary.astype(np.uint8))
        if rps:
            rp0 = rps[0]
            coords = np.argwhere(binary)
            ar = aspect_ratio_from_coords(coords, voxel_size_zyx)
            bb = rp0.bbox
            bb_vol = (bb[3]-bb[0]) * (bb[4]-bb[1]) * (bb[5]-bb[2])
            fill_ratio = rp0.area / bb_vol if bb_vol > 0 else float("nan")
        else:
            ar = float("nan")
            fill_ratio = float("nan")

        mask_mesh_b64 = ""
        if with_mesh:
            print(f"  Generating mesh for mask '{ent.name}'...", flush=True)
            mask_sigma = _mesh_sigma_for_shape(sph, fill_ratio,
                                               sigma_min=0.3,
                                               sigma_max=mesh_smooth_sigma * 2)
            mask_mesh_b64 = _generate_mesh_b64(
                binary.astype(bool),
                bbox_origin_zyx=(0, 0, 0),
                voxel_size_zyx=voxel_size_zyx,
                smooth_sigma=mask_sigma,
                step_size=mesh_step_size,
                target_reduction=mesh_target_reduction,
                level=mesh_level,
            )
        rows.append({
            "cell_id": cell_id, "group_id": group_id, "entity_name": ent.name, "entity_kind": ent.kind,
            "row_type": "file", "label_id": None, "instance_count": None,
            "total_volume_um3": vol_um3,
            "volume_um3": vol_um3,
            "surface_area_um2": area_um2,
            "sphericity": sph,
            "aspect_ratio_major_minor": ar,
            "mesh_b64": mask_mesh_b64,
            **_file_meta(ent.path),
        })

    for lk in label_keys:
        ent = entities[lk]
        labels = volumes[lk].astype(np.int32)
        df = label_dfs.get(lk, pd.DataFrame())
        n_inst = int(len(np.unique(labels[labels > 0]))) if (labels > 0).any() else 0
        rows.append({
            "cell_id": cell_id, "group_id": group_id, "entity_name": ent.name, "entity_kind": ent.kind,
            "row_type": "file", "label_id": None, "instance_count": n_inst,
            "total_volume_um3": float((labels > 0).sum() * voxel_um3),
            **_file_meta(ent.path),
        })
        if df.empty:
            continue

        props_map = {rp.label: rp for rp in regionprops(labels)}

        if with_mesh:
            mesh_ids = set(df["label"].astype(int).tolist())
            print(f"  Generating {len(mesh_ids)} meshes for {ent.name}...", flush=True)
        else:
            mesh_ids: set[int] = set()

        for _, irow in df.iterrows():
            lbl = int(irow["label"])
            rp = props_map.get(lbl)
            mesh_b64 = ""
            if lbl in mesh_ids and rp is not None:
                bbox_vol = (rp.bbox[3]-rp.bbox[0]) * (rp.bbox[4]-rp.bbox[1]) * (rp.bbox[5]-rp.bbox[2])
                fill_ratio = rp.area / bbox_vol if bbox_vol > 0 else float("nan")
                sph = irow.get("sphericity", float("nan"))
                sph = float(sph) if sph is not None and not math.isnan(float(sph)) else float("nan")
                sigma = _mesh_sigma_for_shape(sph, fill_ratio,
                                              sigma_min=0.3,
                                              sigma_max=mesh_smooth_sigma * 2)
                mesh_b64 = _generate_mesh_b64(
                    rp.image.astype(bool),
                    bbox_origin_zyx=(rp.bbox[0], rp.bbox[1], rp.bbox[2]),
                    voxel_size_zyx=voxel_size_zyx,
                    smooth_sigma=sigma,
                    step_size=mesh_step_size,
                    target_reduction=mesh_target_reduction,
                    level=mesh_level,
                )
            row_dict: dict = {
                "cell_id": cell_id, "group_id": group_id, "entity_name": ent.name, "entity_kind": ent.kind,
                "row_type": "instance", "label_id": lbl,
                "instance_count": None, "total_volume_um3": None,
                "file_size_bytes": None, "file_mtime": None, "file_name": ent.path.name,
                "mesh_b64": mesh_b64,
            }
            for col in df.columns:
                if col != "label":
                    row_dict[col] = _sanitize(irow[col])
            rows.append(row_dict)

    return rows


def write_report_csv(out_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame([{k: _sanitize(v) for k, v in r.items()} for r in rows])
    _MESH = "mesh_b64"
    cols_stats = [c for c in df.columns if c != _MESH]
    cols_full  = cols_stats + ([_MESH] if _MESH in df.columns else [])

    # Stats-only CSV — no mesh data, small enough to load in browser for multi-cell reports
    df[cols_stats].to_csv(out_path, index=False)
    print(f"Wrote {out_path.name} ({len(df)} rows, {out_path.stat().st_size // 1024} KB)", flush=True)

    # Full CSV with mesh_b64 — for the 3D mesh viewer (per-cell only)
    if _MESH in df.columns:
        mesh_path = out_path.parent / (out_path.stem + "_meshes" + out_path.suffix)
        df[cols_full].to_csv(mesh_path, index=False)
        print(f"Wrote {mesh_path.name} ({len(df)} rows, {mesh_path.stat().st_size // 1024} KB)", flush=True)


def collect_cell_dirs(root: Path) -> list[tuple[Path, str]]:
    """Return (cell_dir, group_name) pairs, supporting up to two levels of nesting.

    root/               → single cell, group=""
    root/cell/          → flat batch, group=""
    root/group/cell/    → grouped batch, group=group-folder-name
    """
    if not root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root}")
    if any(root.glob("*.tif*")):
        return [(root, "")]

    entries: list[tuple[Path, str]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if any(sub.glob("*.tif*")):
            entries.append((sub, ""))          # flat: root/cell/
        else:
            for subsub in sorted(sub.iterdir()):
                if subsub.is_dir() and any(subsub.glob("*.tif*")):
                    entries.append((subsub, sub.name))  # grouped: root/group/cell/

    if not entries:
        raise FileNotFoundError(
            f"No TIFF files found in {root} or its immediate subdirectories."
        )
    return entries


def run_single_cell(args: argparse.Namespace, cell_dir: Path, out_dir: Path, group_id: str = "") -> None:
    report_csv = out_dir / "report.csv"
    if report_csv.exists() and not args.force_reprocess:
        print(f"  Skipping {cell_dir.name} — report.csv exists (--force-reprocess to re-run)", flush=True)
        return

    print("Discovering dataset entities...", flush=True)
    dataset = discover_dataset(cell_dir)
    print(f"Source image: {dataset.source.name}", flush=True)
    print(f"Detected entities: {len(dataset.entities)}", flush=True)

    voxel_size_zyx = resolve_voxel_size_zyx(args, dataset.source)
    membrane_key = f"mask:{dataset.membrane_name}"
    volumes = load_or_generate_masks(
        dataset=dataset,
        generated_masks_dir=out_dir / "masks_for_analysis",
        membrane_key=membrane_key,
        auto_clip=args.auto_clip_to_pm,
    )

    volumes = crop_to_membrane_bbox(volumes, membrane_key)
    promote_multicomponent_masks(volumes, dataset, membrane_key)
    voxel_um3 = float(np.prod(voxel_size_zyx))
    out_dir.mkdir(parents=True, exist_ok=True)

    label_keys = [k for k, e in dataset.entities.items() if e.kind == "label"]
    mask_keys = [k for k, e in dataset.entities.items() if e.kind == "mask"]

    cell_center_zyx_um = compute_cell_center_um(volumes[membrane_key], voxel_size_zyx)
    if cell_center_zyx_um is not None:
        print(f"[polarity] cell center (z,y,x) μm: ({cell_center_zyx_um[0]:.2f}, {cell_center_zyx_um[1]:.2f}, {cell_center_zyx_um[2]:.2f})", flush=True)

    target_binary = build_distance_targets(volumes, dataset.entities, label_keys, mask_keys, membrane_key)
    all_dts = build_or_load_dt_cache(
        target_binary=target_binary,
        dt_cache_dir=out_dir / ".dt_cache",
        voxel_size_zyx=voxel_size_zyx,
        num_threads=args.num_threads,
    )

    label_dfs = analyze_label_entities(
        args=args,
        volumes=volumes,
        entities=dataset.entities,
        label_keys=label_keys,
        mask_keys=mask_keys,
        all_dts=all_dts,
        voxel_size_zyx=voxel_size_zyx,
        cell_center_zyx_um=cell_center_zyx_um,
    )

    print("Building report...", flush=True)
    rows = build_report_rows(
        cell_id=cell_dir.name,
        group_id=group_id,
        volumes=volumes,
        entities=dataset.entities,
        label_keys=label_keys,
        mask_keys=mask_keys,
        label_dfs=label_dfs,
        voxel_size_zyx=voxel_size_zyx,
        voxel_um3=voxel_um3,
        with_mesh=args.with_mesh,
        mesh_smooth_sigma=args.mesh_smooth_sigma,
        mesh_step_size=args.mesh_step_size,
        mesh_target_reduction=args.mesh_target_reduction,
        mesh_level=args.mesh_level,
        source_path=dataset.source,
        generated_masks_dir=out_dir / "masks_for_analysis",
    )
    write_report_csv(report_csv, rows)
    print("Cell analysis complete.", flush=True)


def main() -> None:
    args = parse_args()
    cell_entries = collect_cell_dirs(args.cell_dir)
    batch = len(cell_entries) > 1
    if batch:
        print(f"Batch mode: {len(cell_entries)} cell folders.", flush=True)

    out_dirs: list[Path] = []
    for idx, (cell_dir, group_id) in enumerate(cell_entries, start=1):
        if not batch:
            out_dir = args.out_dir
        elif group_id:
            out_dir = args.out_dir / group_id / cell_dir.name
        else:
            out_dir = args.out_dir / cell_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.append(out_dir)
        label = f"{group_id}/{cell_dir.name}" if group_id else cell_dir.name
        print(f"===== [{idx}/{len(cell_entries)}] {label} =====", flush=True)
        run_single_cell(args, cell_dir, out_dir, group_id=group_id)

    if batch:
        # Joint stats CSV — concatenate per-cell report.csv (no b64, safe to load in browser)
        dfs = [pd.read_csv(od / "report.csv") for od in out_dirs if (od / "report.csv").exists()]
        if dfs:
            joint = pd.concat(dfs, ignore_index=True)
            joint_path = args.out_dir / "report.csv"
            joint.to_csv(joint_path, index=False)
            kb = joint_path.stat().st_size // 1024
            print(f"Joint report.csv: {len(joint)} rows, {kb} KB → {joint_path}", flush=True)

    print("Done.", flush=True)
    print("  Stats viewer  → open stats_viewer.html, load report.csv", flush=True)
    print("  3D mesh viewer → open mesh_viewer.html, load a per-cell report_meshes.csv", flush=True)


if __name__ == "__main__":
    main()

