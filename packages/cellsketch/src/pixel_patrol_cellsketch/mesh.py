"""Mesh and skeleton geometry for the 3D viewer and the Blender export.

Moved from ``analyze_cell.py`` (payload formats unchanged, so ``mesh_viewer.html`` and
``csv_to_blender.py`` read the output as-is). Geometry never enters the PixelPatrol
table: base64 meshes would multiply the size of the report every stats query loads, so
they go to one ``report_meshes.csv`` per cell, exactly as before.
"""

from __future__ import annotations

import base64
import csv
import gzip
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import fast_simplification
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes, regionprops

from pixel_patrol_cellsketch.geometry import estimate_surface_area_um2, sphericity
from pixel_patrol_cellsketch.skeletons import (
    EntityFilter,
    contacts_for,
    skeletons_for,
    wants_skeletons,
)

logger = logging.getLogger(__name__)

# The columns mesh_viewer.html and csv_to_blender.py read. Geometry last, so the file
# stays readable when truncated in a terminal.
# Per-instance metrics carried alongside the geometry when the instance processor has
# already measured them: mesh_viewer.html offers any numeric column as a sort key, and
# reads polar_n* as a direction to draw.
CARRIED_METRICS = [
    "aspect_ratio_major_minor", "branches", "length_um", "tortuosity",
    "distance_to_closest_same_type_um",
    "polar_dist_um", "polar_az_deg", "polar_el_deg", "polar_nz", "polar_ny", "polar_nx",
    "polar_spread_deg",
]

MESH_CSV_COLUMNS = [
    "cell_id", "group_id", "entity_name", "entity_kind", "row_type", "label_id",
    "volume_um3", "surface_area_um2", "sphericity", *CARRIED_METRICS,
    # row_type='contact' rows, which the viewer's contact-group colouring reads
    "entity_a", "label_a", "entity_b", "label_b", "gap_um",
    "mesh_b64", "skeleton_b64",
]


@dataclass(frozen=True)
class MeshOptions:
    """The knobs analyze_cell.py exposed as --mesh-* flags."""
    smooth_sigma: float = 0.7
    step_size: int = 2
    target_reduction: float = 0.8
    level: Optional[float] = None
    with_skeletons: bool = True
    # Which entities get a skeleton overlay; None = all. Same filter the metrics use, so
    # the two share one computation per cell rather than repeating the expensive one.
    skeleton_entities: EntityFilter = None
    max_skeleton_voxels: Optional[int] = 500_000
    num_threads: int = 1
    # Contacts ride in the same file so "Color by → Contact group" works in the 3D
    # viewer; None leaves them out. Cheap next to meshing (seconds against minutes).
    contact_max_um: Optional[float] = 0.5


def sigma_for_shape(sphericity_value: float, fill_ratio: float,
                    sigma_min: float = 0.3, sigma_max: float = 1.5) -> float:
    """Gaussian sigma from shape: blobs get more smoothing, thin structures less.

    Uses sphericity (how sphere-like) and fill_ratio (voxels / bbox voxels, which catches
    curved filaments that look compact in PCA but are sparse). Midpoint when either is NaN.
    """
    _SPHERE_FILL = math.pi / 6.0  # fill ratio of a perfect sphere (~0.524)
    if math.isnan(sphericity_value) or math.isnan(fill_ratio):
        return (sigma_min + sigma_max) / 2.0
    fill_score = min(1.0, fill_ratio / _SPHERE_FILL)
    blob_score = math.sqrt(sphericity_value * fill_score)  # geometric mean of both
    return sigma_min + (sigma_max - sigma_min) * blob_score


def generate_mesh_b64(
    binary: np.ndarray,
    bbox_origin_zyx: Tuple[int, int, int],
    voxel_size_zyx: Sequence[float],
    step_size: int = 2,
    smooth_sigma: float = 0.7,
    target_reduction: float = 0.8,
    level: Optional[float] = None,
) -> str:
    """Mesh a (Z,Y,X) binary mask via marching cubes on a signed distance field.

    The surface is the zero-level set of ``inside_EDT − outside_EDT`` (optionally
    smoothed), which sits at the true voxel boundary — this preserves thin structures and
    avoids the volume inflation of blurring the binary directly.

    Binary payload (gzip-compressed, then base64):
      [uint32 nV][uint32 nF]
      [float32×3 min_xyz][float32×3 scale_xyz]   ← dequantisation params
      [uint16 × nV×3 quantised XYZ vertices]
      [uint32 × nF×3 face indices]
    """
    if binary.sum() < 8:
        return ""
    try:
        # An instance's foreground reaches every face of its own bounding box, so pad with
        # background before meshing. With smoothing, pad by ~3σ so the smoothed field
        # settles to "outside" well before the border → closed watertight mesh.
        pad = 1 if smooth_sigma <= 0 else int(math.ceil(3.0 * smooth_sigma)) + 1
        b = np.pad(binary.astype(bool), pad_width=pad)
        # Signed distance field in voxel-index units: >0 inside, <0 outside, 0 at boundary.
        sdf = (distance_transform_edt(b) - distance_transform_edt(~b)).astype(np.float32)
        if smooth_sigma > 0:
            sdf = gaussian_filter(sdf, sigma=smooth_sigma)
        verts, faces, _, _ = marching_cubes(
            sdf, level=0.0 if level is None else level, step_size=step_size
        )
        verts = verts - pad  # undo padding offset → local voxel coords
        oz, oy, ox = bbox_origin_zyx
        sz, sy, sx = (float(v) for v in voxel_size_zyx)
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
        return _quantised_payload(verts_xyz, faces.astype(np.uint32))
    except Exception as exc:
        logger.debug("cellsketch: meshing failed (%s); emitting no geometry", exc)
        return ""


def skeleton_to_b64(skeleton) -> str:
    """Encode a kimimaro Skeleton as a line-segment payload for the 3D viewer.

    kimimaro vertices are µm in the cropped-volume frame, axis order ZYX; reordered to
    XYZ to match generate_mesh_b64 so the skeleton overlays exactly on its mesh.

    Binary payload (gzip-compressed, then base64):
      [uint32 nV][uint32 nE]
      [float32×3 min_xyz][float32×3 scale_xyz]   ← dequantisation params
      [uint16 × nV×3 quantised XYZ vertices]
      [uint32 × nE×2 edge index pairs]
    """
    if skeleton is None:
        return ""
    verts = np.asarray(skeleton.vertices, dtype=np.float32)
    edges = np.asarray(skeleton.edges, dtype=np.uint32)
    if len(verts) < 2 or len(edges) == 0:
        return ""
    verts_xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]]).astype(np.float32)
    return _quantised_payload(verts_xyz, edges)


def _quantised_payload(verts_xyz: np.ndarray, indices: np.ndarray) -> str:
    """Vertices quantised to uint16 plus an index array, gzipped and base64-encoded."""
    min_xyz = verts_xyz.min(axis=0)
    scale_xyz = verts_xyz.max(axis=0) - min_xyz
    scale_xyz[scale_xyz == 0] = 1.0
    verts_q = np.clip(
        (verts_xyz - min_xyz) / scale_xyz * 65535 + 0.5, 0, 65535
    ).astype(np.uint16)
    header = np.array([len(verts_xyz), len(indices)], dtype=np.uint32)
    quant_params = np.concatenate([min_xyz, scale_xyz]).astype(np.float32)
    payload = header.tobytes() + quant_params.tobytes() + verts_q.tobytes() + indices.tobytes()
    return base64.b64encode(gzip.compress(payload, compresslevel=9)).decode()


def mesh_rows_for_cell(
    volumes: Mapping[str, np.ndarray],
    kinds: Mapping[str, str],
    voxel_size_zyx: Sequence[float],
    cell_id: str,
    group_id: str = "",
    options: MeshOptions = MeshOptions(),
    metrics: Optional[Mapping[tuple, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """One row per label instance and per whole-structure mask, with its geometry.

    Shaped for mesh_viewer.html: it reads row_type='instance' rows for label entities and
    row_type='file' rows for masks, and offers every other column as a sort metric.
    """
    voxel_um3 = float(np.prod([float(v) for v in voxel_size_zyx]))
    rows: List[Dict[str, Any]] = []

    for name, volume in volumes.items():
        kind = kinds[name]
        if kind != "label":
            binary = volume > 0
            if not binary.any():
                continue
            vol_um3 = float(binary.sum() * voxel_um3)
            area_um2 = estimate_surface_area_um2(binary, voxel_size_zyx)
            sph = sphericity(vol_um3, area_um2)
            coords = np.argwhere(binary)
            extent = np.prod(coords.max(axis=0) - coords.min(axis=0) + 1)
            rows.append({
                "cell_id": cell_id, "group_id": group_id, "entity_name": name,
                "entity_kind": kind, "row_type": "file", "label_id": "",
                "volume_um3": vol_um3, "surface_area_um2": area_um2, "sphericity": sph,
                "mesh_b64": generate_mesh_b64(
                    binary, (0, 0, 0), voxel_size_zyx,
                    step_size=options.step_size,
                    smooth_sigma=sigma_for_shape(
                        sph, float(binary.sum() / extent) if extent else float("nan"),
                        sigma_min=0.3, sigma_max=options.smooth_sigma * 2),
                    target_reduction=options.target_reduction, level=options.level,
                ),
                "skeleton_b64": "",
            })
            continue

        labels = np.ascontiguousarray(volume)
        props = regionprops(labels)
        if not props:
            continue
        skeletons = (
            skeletons_for(cell_id, name, labels, voxel_size_zyx,
                          options.max_skeleton_voxels, options.num_threads)
            if options.with_skeletons and wants_skeletons(name, options.skeleton_entities)
            else {}
        )
        for rp in props:
            vol_um3 = float(rp.area * voxel_um3)
            area_um2 = estimate_surface_area_um2(rp.image, voxel_size_zyx)
            sph = sphericity(vol_um3, area_um2)
            bbox_extent = (
                (rp.bbox[3] - rp.bbox[0]) * (rp.bbox[4] - rp.bbox[1]) * (rp.bbox[5] - rp.bbox[2])
            )
            fill_ratio = rp.area / bbox_extent if bbox_extent > 0 else float("nan")
            carried = (metrics or {}).get((name, int(rp.label))) or {}
            rows.append({
                **{k: v for k, v in carried.items() if k in CARRIED_METRICS},
                "cell_id": cell_id, "group_id": group_id, "entity_name": name,
                "entity_kind": kind, "row_type": "instance", "label_id": int(rp.label),
                "volume_um3": vol_um3, "surface_area_um2": area_um2, "sphericity": sph,
                "mesh_b64": generate_mesh_b64(
                    rp.image.astype(bool), (rp.bbox[0], rp.bbox[1], rp.bbox[2]), voxel_size_zyx,
                    step_size=options.step_size,
                    smooth_sigma=sigma_for_shape(sph, fill_ratio, sigma_min=0.3,
                                                 sigma_max=options.smooth_sigma * 2),
                    target_reduction=options.target_reduction, level=options.level,
                ),
                "skeleton_b64": skeleton_to_b64(skeletons.get(int(rp.label))),
            })

    if options.contact_max_um is not None:
        rows.extend(_contact_rows(volumes, kinds, voxel_size_zyx, cell_id, group_id,
                                  options.contact_max_um))
    return rows


def _contact_rows(
    volumes: Mapping[str, np.ndarray],
    kinds: Mapping[str, str],
    voxel_size_zyx: Sequence[float],
    cell_id: str,
    group_id: str,
    max_gap_um: float,
) -> List[Dict[str, Any]]:
    """The pairwise edge list, in the row shape the 3D viewer splits out on load."""
    contacts = contacts_for(cell_id, volumes, kinds, voxel_size_zyx, max_gap_um)
    return [
        {
            "cell_id": cell_id, "group_id": group_id, "row_type": "contact",
            "entity_a": entity_a, "label_a": "" if label_a is None else label_a,
            "entity_b": entity_b, "label_b": "" if label_b is None else label_b,
            "gap_um": gap_um, "mesh_b64": "", "skeleton_b64": "",
        }
        for entity_a, label_a, entity_b, label_b, gap_um in contacts
    ]


def write_mesh_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write report_meshes.csv, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MESH_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
