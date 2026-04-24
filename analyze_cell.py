#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.2.0",
#   "scipy>=1.13.0",
#   "scikit-image>=0.23.0",
#   "tifffile>=2024.5.0",
#   "matplotlib>=3.9.0",
#   "edt>=2.4.0",
# ]
# ///

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import edt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter, label as nd_label
from skimage.measure import regionprops
from skimage.morphology import skeletonize

matplotlib.use("Agg")


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
    parser.add_argument("--generated-masks-dirname", default="masks_for_analysis")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--max-skeleton-voxels",
        type=int,
        default=500_000,
        help="Skip branch counting for label instances larger than this voxel count (default: 500000).",
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

    entities: Dict[str, Entity] = {}
    for p, (pref, name, kind) in parsed.items():
        if shared_prefix_len(source_norm, pref) < 12:
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
    if lengths.min() <= 0:
        return float("nan")
    return float(lengths.max() / lengths.min())


def count_branch_points(binary_mask: np.ndarray, max_voxels: int | None = None) -> int | float:
    n = int(binary_mask.sum())
    if n == 0:
        return 0
    if max_voxels is not None and n > max_voxels:
        return float("nan")
    skel = skeletonize(binary_mask.astype(bool))
    if skel.sum() == 0:
        return 0
    kernel = np.ones((3, 3, 3), dtype=np.int16)
    kernel[1, 1, 1] = 0
    degree = convolve(skel.astype(np.int16), kernel, mode="constant", cval=0)

    endpoint_mask = skel & (degree == 1)
    junction_mask = skel & (degree >= 3)

    # Merge adjacent junction voxels into single graph nodes.
    connectivity = np.ones((3, 3, 3), dtype=np.uint8)
    junction_cc, n_junction_cc = nd_label(junction_mask.astype(np.uint8), structure=connectivity)

    # Branch segments are skeleton components with junction voxels removed.
    segment_mask = skel & ~junction_mask
    segment_cc, n_segments = nd_label(segment_mask.astype(np.uint8), structure=connectivity)
    if n_segments == 0:
        return 0

    branch_count = 0
    for seg_id in range(1, n_segments + 1):
        seg = segment_cc == seg_id
        seg_coords = np.argwhere(seg)
        attached_nodes: set[tuple[str, int]] = set()

        for z, y, x in seg_coords:
            z0, z1 = max(0, z - 1), min(skel.shape[0], z + 2)
            y0, y1 = max(0, y - 1), min(skel.shape[1], y + 2)
            x0, x1 = max(0, x - 1), min(skel.shape[2], x + 2)

            j_patch = junction_cc[z0:z1, y0:y1, x0:x1]
            for j in np.unique(j_patch):
                if j > 0:
                    attached_nodes.add(("junction", int(j)))

            e_patch = endpoint_mask[z0:z1, y0:y1, x0:x1]
            if np.any(e_patch):
                endpoint_coords = np.argwhere(e_patch)
                for ez, ey, ex in endpoint_coords:
                    # Encode local endpoint coordinates into stable global IDs.
                    attached_nodes.add(("endpoint", int((z0 + ez) * 10**12 + (y0 + ey) * 10**6 + (x0 + ex))))

        # Closed loops can have no junctions/endpoints but still represent one branch.
        if len(attached_nodes) == 0:
            branch_count += 1
        else:
            branch_count += 1

    return int(branch_count)


def per_label_metrics(
    labels: np.ndarray,
    voxel_size_zyx: Tuple[float, float, float],
    distance_transforms: Dict[str, np.ndarray],
    max_skeleton_voxels: int | None = None,
) -> pd.DataFrame:
    voxel_um3 = float(np.prod(voxel_size_zyx))
    rows = []
    props = regionprops(labels)
    centroids = []
    for rp in props:
        row = {
            "label": int(rp.label),
            "volume_um3": float(rp.area * voxel_um3),
            "surface_area_um2": estimate_surface_area_um2(rp.image, voxel_size_zyx),
            "aspect_ratio_major_minor": aspect_ratio_from_coords(rp.coords, voxel_size_zyx),
            "branches": count_branch_points(rp.image, max_voxels=max_skeleton_voxels),
        }
        row["sphericity"] = sphericity(row["volume_um3"], row["surface_area_um2"])
        for key, dt in distance_transforms.items():
            row[f"distance_to_{key}_um"] = float(dt[tuple(rp.coords.T)].min()) if rp.coords.size else float("nan")
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


def center_slice_index(mask_3d: np.ndarray) -> int:
    coords = np.argwhere(mask_3d)
    if coords.size == 0:
        return int(mask_3d.shape[0] // 2)
    return int(np.median(coords[:, 0]))


def save_qc_plots_for_label_entity(out_dir: Path, membrane: np.ndarray, labels: np.ndarray, df: pd.DataFrame, entity_name: str) -> None:
    if df.empty:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    z = center_slice_index(membrane)
    labels_2d = labels[z]

    for value_col in [c for c in df.columns if c in ("volume_um3", "distance_to_membrane_um")]:
        values = np.full(labels_2d.shape, np.nan, dtype=np.float32)
        vmap = dict(zip(df["label"].astype(int), df[value_col].astype(float)))
        for lab in np.unique(labels_2d):
            if lab == 0:
                continue
            v = vmap.get(int(lab))
            if v is not None and np.isfinite(v):
                values[labels_2d == lab] = v

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(membrane[z], cmap="gray", alpha=0.2)
        im = ax.imshow(np.ma.masked_invalid(values), cmap="viridis")
        ax.set_axis_off()
        ax.set_title(f"{entity_name}: {value_col} (z={z})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{entity_name}_center_slice_{value_col}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = df["volume_um3"].replace([np.inf, -np.inf], np.nan).dropna()
    if not vals.empty:
        ax.hist(vals, bins=30)
    ax.set_title(f"{entity_name}: volume distribution")
    ax.set_xlabel("volume_um3")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{entity_name}_volume_hist.png", dpi=180)
    plt.close(fig)


def _render_depth_shaded(binary: np.ndarray, rgb: Tuple[float, float, float]) -> np.ndarray:
    """
    Project a (Z,Y,X) binary volume along Z and shade by surface depth.
    Returns an (Y, X, 4) uint8 RGBA image.
    """
    occupied = binary.any(axis=0)  # (Y, X)
    if not occupied.any():
        return np.zeros((binary.shape[1], binary.shape[2], 4), dtype=np.uint8)

    depth = binary.argmax(axis=0).astype(float)
    depth[~occupied] = float(np.nanmean(depth[occupied]))

    if binary.shape[1] < 2 or binary.shape[2] < 2:
        shading = np.where(occupied, 0.85, 0.0)
        r, g, b = rgb
        img = np.zeros((binary.shape[1], binary.shape[2], 4), dtype=np.float32)
        img[occupied, 0] = np.clip(r * shading[occupied], 0.0, 1.0)
        img[occupied, 1] = np.clip(g * shading[occupied], 0.0, 1.0)
        img[occupied, 2] = np.clip(b * shading[occupied], 0.0, 1.0)
        img[occupied, 3] = 1.0
        return (img * 255).astype(np.uint8)

    sigma = float(np.clip(min(binary.shape[1], binary.shape[2]) * 0.04, 0.5, 3.0))
    depth_s = gaussian_filter(depth, sigma=sigma)

    gy, gx = np.gradient(depth_s)
    norm_len = np.sqrt(gx**2 + gy**2 + 1.0)
    # light from upper-left, elevated ~45°
    lx, ly, lz = -0.5, -0.5, 1.0
    llen = math.sqrt(lx**2 + ly**2 + lz**2)
    diffuse = np.clip((-gx * lx - gy * ly + lz) / (norm_len * llen), 0.0, 1.0)
    shading = 0.25 + 0.75 * diffuse

    r, g, b = rgb
    img = np.zeros((binary.shape[1], binary.shape[2], 4), dtype=np.float32)
    img[occupied, 0] = np.clip(r * shading[occupied], 0.0, 1.0)
    img[occupied, 1] = np.clip(g * shading[occupied], 0.0, 1.0)
    img[occupied, 2] = np.clip(b * shading[occupied], 0.0, 1.0)
    img[occupied, 3] = 1.0
    return (img * 255).astype(np.uint8)


def save_mosaic_3d_labels(
    out_dir: Path,
    labels: np.ndarray,
    df: pd.DataFrame,
    entity_name: str,
    voxel_size_zyx: Tuple[float, float, float],
) -> None:
    if df.empty:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_sorted = df.copy()
    df_sorted["_sort"] = df_sorted["branches"].fillna(np.inf)
    df_sorted = df_sorted.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    finite_branches = df_sorted["branches"].dropna()
    if finite_branches.empty:
        return

    props_map = {rp.label: rp for rp in regionprops(labels)}
    n = len(df_sorted)
    if n == 0:
        return

    ncols = max(1, int(np.ceil(np.sqrt(n))))
    nrows = max(1, int(np.ceil(n / ncols)))

    b_min = float(finite_branches.min())
    b_max = float(finite_branches.max())
    if b_min == b_max:
        b_max = b_min + 1.0
    cmap = matplotlib.cm.plasma
    norm = matplotlib.colors.Normalize(vmin=b_min, vmax=b_max)

    cell_size = 2.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size, nrows * cell_size + 0.7), squeeze=False)

    for pos, (_, row) in enumerate(df_sorted.iterrows()):
        ax = axes[pos // ncols][pos % ncols]
        ax.set_axis_off()
        lbl = int(row["label"])
        branches = row["branches"]

        rp = props_map.get(lbl)
        if rp is not None:
            rgb = (0.65, 0.65, 0.65) if pd.isna(branches) else cmap(norm(float(branches)))[:3]
            thumb = _render_depth_shaded(rp.image.astype(bool), rgb)
            ax.imshow(thumb, interpolation="bilinear", aspect="equal")

        branch_str = "N/A" if pd.isna(branches) else str(int(branches))
        ax.set_title(f"#{lbl}  b={branch_str}", fontsize=7, pad=1)

    for pos in range(len(df_sorted), nrows * ncols):
        axes[pos // ncols][pos % ncols].set_visible(False)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.1, 0.03, 0.8, 0.022])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="branch count")

    fig.suptitle(f"{entity_name} — mosaic sorted by branch count", fontsize=10)
    out_path = plots_dir / f"{entity_name}_mosaic_3d.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote 3D mosaic: {out_path.name}", flush=True)


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


def write_generated_masks(
    volumes: Dict[str, np.ndarray],
    entities: Dict[str, Entity],
    generated_masks_dir: Path,
) -> None:
    generated_masks_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing generated masks to {generated_masks_dir} ...", flush=True)
    for key, arr in volumes.items():
        entity = entities[key]
        out = arr if entity.kind == "label" else (arr > 0).astype(np.uint8)
        tifffile.imwrite(generated_masks_dir / entity.path.name, out, compression="zlib")
    print("Generated masks written.", flush=True)


def init_overall_row(cell_id: str, volumes: Dict[str, np.ndarray], entities: Dict[str, Entity], membrane_key: str, voxel_um3: float) -> dict:
    membrane = volumes[membrane_key] > 0
    row = {"cell_id": cell_id, "cell_volume_um3": float(membrane.sum() * voxel_um3)}
    for key, entity in entities.items():
        if entity.kind == "mask":
            row[f"mask_{entity.name}_volume_um3"] = float((volumes[key] > 0).sum() * voxel_um3)
    return row


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


def update_overall_row_for_label(overall_row: dict, label_name: str, labels: np.ndarray, voxel_um3: float) -> None:
    n_labels = int((labels > 0).max() and len(np.unique(labels[labels > 0])) or 0)
    total_vol = float((labels > 0).sum() * voxel_um3)
    overall_row[f"label_{label_name}_count"] = n_labels
    overall_row[f"label_{label_name}_total_volume_um3"] = total_vol


def analyze_label_entities(
    args: argparse.Namespace,
    out_dir: Path,
    volumes: Dict[str, np.ndarray],
    entities: Dict[str, Entity],
    label_keys: list[str],
    mask_keys: list[str],
    all_dts: Dict[str, np.ndarray],
    voxel_size_zyx: Tuple[float, float, float],
    voxel_um3: float,
    membrane: np.ndarray,
    overall_row: dict,
) -> None:
    for src_key in label_keys:
        src_entity = entities[src_key]
        print(f"Analyzing label entity: {src_entity.name}", flush=True)
        src_labels = volumes[src_key].astype(np.int32)
        dts_for_src = distance_transforms_for_source(src_key, label_keys, mask_keys, entities, all_dts)
        df = per_label_metrics(src_labels, voxel_size_zyx, dts_for_src, max_skeleton_voxels=args.max_skeleton_voxels)
        out_name = f"individual_{src_entity.name}.csv"
        df.to_csv(out_dir / out_name, index=False)
        print(f"Wrote {out_name} ({len(df)} rows)", flush=True)
        update_overall_row_for_label(overall_row, src_entity.name, src_labels, voxel_um3)

        if not args.skip_plots:
            print(f"Generating QC plots for {src_entity.name}...", flush=True)
            save_qc_plots_for_label_entity(out_dir, membrane, src_labels, df, src_entity.name)
            save_mosaic_3d_labels(out_dir, src_labels, df, src_entity.name, voxel_size_zyx)
            print(f"QC plots for {src_entity.name} done.", flush=True)


def collect_cell_dirs(cell_dir: Path) -> list[Path]:
    has_tiffs = any(cell_dir.glob("*.tif*"))
    if has_tiffs:
        return [cell_dir]
    if not cell_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {cell_dir}")
    subdirs = [p for p in sorted(cell_dir.iterdir()) if p.is_dir()]
    cell_dirs = [d for d in subdirs if any(d.glob("*.tif*"))]
    if not cell_dirs:
        raise FileNotFoundError(
            f"No TIFF files in {cell_dir} and no TIFF-containing cell subdirectories found."
        )
    return cell_dirs


def run_single_cell(args: argparse.Namespace, cell_dir: Path, out_dir: Path) -> None:
    print("Discovering dataset entities...", flush=True)
    dataset = discover_dataset(cell_dir)
    print(f"Source image: {dataset.source.name}", flush=True)
    print(f"Detected entities: {len(dataset.entities)}", flush=True)

    voxel_size_zyx = resolve_voxel_size_zyx(args, dataset.source)
    volumes = load_entity_volumes(dataset)

    membrane_key = f"mask:{dataset.membrane_name}"
    apply_membrane_clipping(volumes, membrane_key, args.auto_clip_to_pm)
    write_generated_masks(volumes, dataset.entities, out_dir / args.generated_masks_dirname)

    volumes = crop_to_membrane_bbox(volumes, membrane_key)
    membrane = volumes[membrane_key] > 0
    voxel_um3 = float(np.prod(voxel_size_zyx))

    out_dir.mkdir(parents=True, exist_ok=True)
    overall_row = init_overall_row(cell_dir.name, volumes, dataset.entities, membrane_key, voxel_um3)

    label_keys = [k for k, e in dataset.entities.items() if e.kind == "label"]
    mask_keys = [k for k, e in dataset.entities.items() if e.kind == "mask"]

    target_binary = build_distance_targets(volumes, dataset.entities, label_keys, mask_keys, membrane_key)
    all_dts = build_or_load_dt_cache(
        target_binary=target_binary,
        dt_cache_dir=out_dir / ".dt_cache",
        voxel_size_zyx=voxel_size_zyx,
        num_threads=args.num_threads,
    )

    analyze_label_entities(
        args=args,
        out_dir=out_dir,
        volumes=volumes,
        entities=dataset.entities,
        label_keys=label_keys,
        mask_keys=mask_keys,
        all_dts=all_dts,
        voxel_size_zyx=voxel_size_zyx,
        voxel_um3=voxel_um3,
        membrane=membrane,
        overall_row=overall_row,
    )

    pd.DataFrame([overall_row]).to_csv(out_dir / "overall_cell.csv", index=False)
    print(f"Wrote overall_cell.csv to {out_dir}", flush=True)
    print("Analysis completed successfully.", flush=True)


def main() -> None:
    args = parse_args()
    cell_dirs = collect_cell_dirs(args.cell_dir)
    if len(cell_dirs) > 1:
        print(f"Batch mode: found {len(cell_dirs)} cell folders.", flush=True)

    for idx, cell_dir in enumerate(cell_dirs, start=1):
        out_dir = args.out_dir if len(cell_dirs) == 1 else (args.out_dir / cell_dir.name)
        print(f"===== Cell {idx}/{len(cell_dirs)}: {cell_dir.name} =====", flush=True)
        run_single_cell(args, cell_dir, out_dir)
    print("All done.", flush=True)


if __name__ == "__main__":
    main()

