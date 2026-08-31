"""Geometry for the 3D widgets and the Blender export: meshes, outlines, skeletons.

Geometry never enters the PixelPatrol table, because it would multiply the size of the report
every stats query loads. It goes beside it instead, one ``geometry.parquet`` per object, with
the payloads as BLOBs compressed by parquet's own zstd.

Parquet rather than a blob file makes it queryable: DuckDB filters by object, entity and
metric and returns only the rows about to be drawn, so opening one structure costs its own
few MB rather than the whole object's.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import fast_simplification
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes

from pixel_patrol_anatomy.geometry import (
    METRICS_2D,
    METRICS_3D,
)
from pixel_patrol_anatomy.distances import object_center_um, polarity_from_offset
from pixel_patrol_anatomy.geometry import skeleton_graph_metrics
from pixel_patrol_anatomy.skeletons import (
    EntityFilter,
    contacts_for,
    label_metrics_for,
    region_metrics_for,
    regions_for,
    skeletons_for,
    wants_skeletons,
)

logger = logging.getLogger(__name__)

# Metrics copied from the instance processor so the widgets can sort and colour without
# going back to the report; polar_n* is the direction to explode an instance along.
CARRIED_METRICS = [
    "aspect_ratio_major_minor", "branches", "length_um", "tortuosity",
    "distance_to_closest_same_type_um",
    "polar_dist_um", "polar_az_deg", "polar_el_deg", "polar_angle_deg",
    "polar_nz", "polar_ny", "polar_nx",
    "polar_spread_deg",
]

# Payload headers as columns of their own, so a widget can budget its draw calls without
# reading a byte of geometry.
GEOMETRY_COLUMNS = [
    "object_id", "group_id", "entity_name", "entity_kind", "row_type", "label_id",
    "spatial_dims",
    # 3D size and shape, then the 2D pair; a row carries whichever its object has.
    "volume_um3", "surface_area_um2", "sphericity",
    "area_um2", "perimeter_um", "circularity", *CARRIED_METRICS,
    "entity_a", "label_a", "entity_b", "label_b", "gap_um",
    "mesh_vertices", "mesh_faces", "skeleton_vertices", "skeleton_edges",
    "outline_vertices", "outline_edges",
    "mesh", "skeleton", "outline",
]

_TEXT_FIELDS = {"object_id", "group_id", "entity_name", "entity_kind", "row_type",
                "entity_a", "entity_b"}
# The carried metrics stay floats, branches included: "not measured" is NaN, which an
# integer column cannot hold.
_INT_FIELDS = {"label_id", "label_a", "label_b", "spatial_dims",
               "mesh_vertices", "mesh_faces", "skeleton_vertices", "skeleton_edges",
               "outline_vertices", "outline_edges"}
_BLOB_FIELDS = {"mesh", "skeleton", "outline"}

GEOMETRY_FILENAME = "geometry.parquet"


@dataclass(frozen=True)
class MeshOptions:
    """The knobs the --mesh-* flags set."""
    smooth_sigma: float = 0.7
    step_size: int = 2
    target_reduction: float = 0.8
    level: Optional[float] = None
    with_skeletons: bool = True
    # None = all. Same filter the metrics use, so the two share one computation per object.
    skeleton_entities: EntityFilter = None
    max_skeleton_voxels: Optional[int] = 500_000
    num_threads: int = 1
    # Contacts ride along so "Colour by → Contact group" works; None leaves them out. Cheap
    # next to meshing (seconds against minutes).
    contact_max_um: Optional[float] = 0.5
    # Processes to mesh instances with. 0 = work it out: the batch's share of the cores.
    mesh_workers: int = 0


def sigma_for_shape(sphericity_value: float, fill_ratio: float,
                    sigma_min: float = 0.3, sigma_max: float = 1.5) -> float:
    """Gaussian sigma from shape: blobs get more smoothing, thin structures less.

    Uses roundness (1 for a sphere, since ITK measures the boundary properly) and fill_ratio
    (samples / bbox samples, which catches curved filaments that look compact in PCA but are
    sparse). Midpoint when either is NaN. generate_mesh caps the result by the instance's own
    thickness, so a small instance is never smoothed away.
    """
    _SPHERE_FILL = math.pi / 6.0  # fill ratio of a perfect sphere (~0.524)
    if math.isnan(sphericity_value) or math.isnan(fill_ratio):
        return (sigma_min + sigma_max) / 2.0
    fill_score = min(1.0, fill_ratio / _SPHERE_FILL)
    blob_score = math.sqrt(sphericity_value * fill_score)  # geometric mean of both
    return sigma_min + (sigma_max - sigma_min) * blob_score


def generate_mesh(
    binary: np.ndarray,
    bbox_origin_zyx: Tuple[int, int, int],
    voxel_size_zyx: Sequence[float],
    step_size: int = 2,
    smooth_sigma: float = 0.7,
    target_reduction: float = 0.8,
    level: Optional[float] = None,
) -> bytes:
    """Mesh a (Z,Y,X) binary mask via marching cubes on a signed distance field.

    The surface is the zero-level set of ``inside_EDT − outside_EDT`` (optionally
    smoothed), which sits at the true voxel boundary. That preserves thin structures and
    avoids the volume inflation of blurring the binary directly.

    Binary payload:
      [uint32 nV][uint32 nF]
      [float32×3 min_xyz][float32×3 scale_xyz]   ← dequantisation params
      [uint16 × nV×3 quantised XYZ vertices]
      [uint32 × nF×3 face indices]

    Returned raw, as it goes into the parquet; ``payload_to_b64`` gzips and encodes it for
    the CSV, which is where the container format was first read.
    """
    if binary.sum() < 8:
        return b""
    try:
        spacing = tuple(float(v) for v in voxel_size_zyx)
        finest, coarsest = min(spacing), max(spacing)
        # σ is in the finest sample: the same blur in µm along every axis, which for
        # anisotropic data is a fraction of a sample along the coarse one.
        sigma_um = max(0.0, smooth_sigma) * finest
        # The foreground reaches every face of its own bbox, so pad with background first;
        # with smoothing, ~3σ per axis so the field settles to "outside" before the border.
        pads = tuple(1 if sigma_um <= 0 else int(math.ceil(3.0 * sigma_um / s)) + 1
                     for s in spacing)
        b = np.pad(binary.astype(bool), pad_width=[(p, p) for p in pads])
        # Signed distance field in µm: >0 inside, <0 outside, 0 at the boundary.
        inside = distance_transform_edt(b, sampling=spacing)
        sdf = (inside - distance_transform_edt(~b, sampling=spacing)).astype(np.float32)
        # Neither knob may exceed what the instance can carry. The largest value inside is its
        # inscribed radius: a kernel wider than half of that flattens the field to nothing, and
        # a step coarser than the radius walks straight over it. The step is in samples, so it
        # is the coarse axis that runs out first.
        radius_um = float(inside.max())
        sigma_um = min(sigma_um, radius_um / 2.0)
        step = max(1, min(step_size, int(radius_um / coarsest) - 1))
        if sigma_um > 0:
            sdf = gaussian_filter(sdf, sigma=tuple(sigma_um / s for s in spacing))
        verts, faces, _, _ = marching_cubes(
            sdf, level=0.0 if level is None else level, step_size=step
        )
        verts = verts - np.asarray(pads)  # undo padding offset → local voxel coords
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
        logger.debug("anatomy: meshing failed (%s); emitting no geometry", exc)
        return b""


def generate_outline(
    binary: np.ndarray,
    bbox_origin_yx: Tuple[int, int],
    pixel_size_yx: Sequence[float],
    simplify_tol_px: float = 0.0,
) -> bytes:
    """Trace a (Y,X) binary mask's boundary as closed line loops: the 2D counterpart of
    generate_mesh.

    A plane has no surface to triangulate, so an outline is what is stored, in the same
    quantised payload as the skeleton overlay, z fixed at 0, so one decoder reads both.
    Loops are closed, so a canvas can fill them and not only stroke them; holes come out as
    their own loops.

    Binary payload: the layout of _quantised_payload, index array holding
    [uint32 × nE×2] vertex pairs.
    """
    if binary.sum() < 1:
        return b""
    try:
        from skimage.measure import approximate_polygon, find_contours

        # Padded so an instance touching its own bbox still traces a closed loop.
        padded = np.pad(binary.astype(float), 1)
        loops = find_contours(padded, level=0.5)
        oy, ox = bbox_origin_yx
        sy, sx = (float(v) for v in pixel_size_yx)
        verts: List[np.ndarray] = []
        edges: List[np.ndarray] = []
        offset = 0
        for loop in loops:
            if simplify_tol_px > 0:
                loop = approximate_polygon(loop, tolerance=simplify_tol_px)
            loop = loop - 1.0
            if len(loop) < 3:
                continue
            # find_contours repeats the first point to close the ring; drop it and close
            # the loop through the index array instead.
            if np.allclose(loop[0], loop[-1]):
                loop = loop[:-1]
            n = len(loop)
            xy = np.column_stack([
                (loop[:, 1] + ox) * sx,
                (loop[:, 0] + oy) * sy,
                np.zeros(n),
            ]).astype(np.float32)
            verts.append(xy)
            ring = np.column_stack([np.arange(n), (np.arange(n) + 1) % n]) + offset
            edges.append(ring.astype(np.uint32))
            offset += n
        if not verts:
            return b""
        return _quantised_payload(np.vstack(verts), np.vstack(edges))
    except Exception as exc:
        logger.debug("anatomy: outlining failed (%s); emitting no geometry", exc)
        return b""


def skeleton_payload(skeleton) -> bytes:
    """Encode a skeleton as a line-segment payload for the viewer.

    Vertices are µm in the cropped frame, axis order ZYX from kimimaro or YX from the
    planar thinning; both are reordered to XYZ so a skeleton overlays exactly on the mesh
    or outline of the same instance.

    Binary payload:
      [uint32 nV][uint32 nE]
      [float32×3 min_xyz][float32×3 scale_xyz]   ← dequantisation params
      [uint16 × nV×3 quantised XYZ vertices]
      [uint32 × nE×2 edge index pairs]
    """
    if skeleton is None:
        return b""
    verts = np.asarray(skeleton.vertices, dtype=np.float32)
    edges = np.asarray(skeleton.edges, dtype=np.uint32)
    if len(verts) < 2 or len(edges) == 0:
        return b""
    if verts.shape[1] == 2:
        # A planar medial axis: (Y, X) in µm → XY0, the same frame the outlines use.
        verts_xyz = np.column_stack([
            verts[:, 1], verts[:, 0], np.zeros(len(verts))
        ]).astype(np.float32)
    else:
        verts_xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]]).astype(np.float32)
    return _quantised_payload(verts_xyz, edges)


def payload_counts(payload: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Vertex and index counts from a payload header, without decoding the arrays."""
    if len(payload) < 8:
        return None, None
    n_verts, n_indices = struct.unpack_from("<II", payload, 0)
    return int(n_verts), int(n_indices)


def _quantised_payload(verts_xyz: np.ndarray, indices: np.ndarray) -> bytes:
    """Vertices quantised to uint16 plus an index array."""
    min_xyz = verts_xyz.min(axis=0)
    scale_xyz = verts_xyz.max(axis=0) - min_xyz
    scale_xyz[scale_xyz == 0] = 1.0
    verts_q = np.clip(
        (verts_xyz - min_xyz) / scale_xyz * 65535 + 0.5, 0, 65535
    ).astype(np.uint16)
    header = np.array([len(verts_xyz), len(indices)], dtype=np.uint32)
    quant_params = np.concatenate([min_xyz, scale_xyz]).astype(np.float32)
    return header.tobytes() + quant_params.tobytes() + verts_q.tobytes() + indices.tobytes()


def _bbox_extent(binary: np.ndarray) -> float:
    """Number of samples in the tightest box around the foreground.

    One reduction per axis, not ``np.argwhere``: a whole-structure mask can hold fifty
    million samples, whose coordinates are 2.5 GB of int64, and the only thing wanted from
    them is where the foreground starts and stops along each axis. Measured on a 53-Mvoxel
    mask: 34 ms and nothing allocated, against 1.6 s and a 2.5 GB peak.
    """
    extent = 1
    for axis in range(binary.ndim):
        others = tuple(i for i in range(binary.ndim) if i != axis)
        present = np.flatnonzero(binary.any(axis=others))
        if not len(present):
            return 0.0
        extent *= int(present[-1] - present[0] + 1)
    return float(extent)


# Instances whose geometry is in flight at once. A pool consumes the whole iterable it is
# handed, so feeding it everything would hold every cropped mask in memory at the same time.
_GEOMETRY_BATCH = 256

# Instances in a batch below which a pool is not worth *opening*. Spawn (which is what this
# has to use) costs about a second to stand up eight workers importing numpy and skimage,
# and that is more than a small object spends meshing. Measured, 8 workers, 22 cores:
#   50 instances 0.39x   150 0.84x   400 1.72x   900 2.61x
# so the crossover sits near 200. Once a pool is open, using it again is free, so this gates
# opening one and not the trailing batch of an entity that already justified it.
_PARALLEL_MIN = 200


def _batched(items: Sequence[Any], size: int):
    """`items` in chunks of `size`. itertools.batched is 3.12+, and this supports 3.11."""
    it = iter(items)
    while chunk := list(islice(it, size)):
        yield chunk


def geometry_workers(requested: int = 0) -> int:
    """How many processes to mesh instances with.

    Objects already run in parallel, and that pool is sized by memory rather than by cores,
    so on a batch of big objects most of the machine sits idle while one instance at a time
    is meshed. This is the share of the cores left over, which the batch works out and
    passes down in PP_ANATOMY_MESH_WORKERS; meshing on its own gets the machine.
    """
    if requested:
        return max(1, int(requested))
    raw = os.environ.get("PP_ANATOMY_MESH_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("anatomy: PP_ANATOMY_MESH_WORKERS=%r is not a number; ignoring", raw)
    return max(1, os.cpu_count() or 1)


def _instance_geometry(task: Tuple[Any, ...]) -> Tuple[bytes, bytes]:
    """One instance's (mesh, outline).

    Module level and tuple-argued so a process pool can carry it. What crosses to the worker
    is one instance's cropped mask, which is why farming these out is worth it: the work is
    an EDT, a blur, marching cubes and a decimation, and the payload is a few kilobytes.
    """
    image, origin, sample_size, planar, step_size, sigma, target_reduction, level = task
    if planar:
        return b"", generate_outline(image, origin, sample_size)
    return generate_mesh(image, origin, sample_size, step_size=step_size,
                         smooth_sigma=sigma, target_reduction=target_reduction,
                         level=level), b""


class _GeometryPool:
    """Instance geometry, farmed out to processes once there is enough of it to be worth it.

    Started on the first batch big enough to pay for it, rather than up front: spawning
    costs a fraction of a second per worker, and an object with a handful of instances is
    finished serially before a pool would have opened.
    """

    def __init__(self, workers: int) -> None:
        self._workers = workers
        self._pool: Optional[ProcessPoolExecutor] = None

    def map(self, tasks: Sequence[Tuple[Any, ...]]) -> List[Tuple[bytes, bytes]]:
        if self._workers > 1 and (self._pool is not None
                                  or len(tasks) >= _PARALLEL_MIN):
            try:
                return list(self._open().map(_instance_geometry, tasks))
            except Exception as exc:  # noqa: BLE001 - a broken pool must not cost the geometry
                # Meshing one at a time is slow, not wrong, and losing an object's whole
                # geometry to a pool that died is much the worse outcome.
                logger.warning("anatomy: parallel meshing failed (%s); one at a time",
                               type(exc).__name__)
                self._workers = 1
        return [_instance_geometry(task) for task in tasks]

    def _open(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # spawn, not fork, for the reason the batch pool uses it: this process holds
            # native threads (ITK, BLAS, polars) and a forked child can deadlock on their
            # locks. Python 3.12 warns about exactly this.
            self._pool = ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()


def mesh_rows_for_object(
    volumes: Mapping[str, np.ndarray],
    kinds: Mapping[str, str],
    sample_size: Sequence[float],
    object_id: str,
    group_id: str = "",
    options: MeshOptions = MeshOptions(),
    metrics: Optional[Mapping[tuple, Mapping[str, Any]]] = None,
    object_mask_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """One row per label instance and per whole-structure mask, with its geometry.

    The shape the 3D widgets read: row_type='instance' rows for label entities,
    row_type='file' rows for whole-structure masks, and every other column offered as a
    sort key or a colour.
    """
    rows: List[Dict[str, Any]] = []
    ndim = len(list(sample_size))
    planar = ndim == 2
    # Where an instance sits relative to the object, which is what the viewer explodes along.
    # Measured here rather than only carried, so `mesh` on its own writes usable geometry too;
    # a carried value from the report wins, and is the same number either way.
    centre = (object_center_um(volumes[object_mask_name] > 0, sample_size)
              if object_mask_name and object_mask_name in volumes else None)

    pool = _GeometryPool(geometry_workers(options.mesh_workers))
    try:
        return _rows(volumes, kinds, sample_size, object_id, group_id, options, metrics,
                     object_mask_name, rows, ndim, planar, centre, pool)
    finally:
        pool.shutdown()


def _rows(volumes, kinds, sample_size, object_id, group_id, options, metrics,
          object_mask_name, rows, ndim, planar, centre, pool):
    """The body of mesh_rows_for_object, with a pool open for the instance geometry."""
    for name, volume in volumes.items():
        kind = kinds[name]
        if kind != "label":
            binary = volume > 0
            if not binary.any():
                continue
            # Not `metrics`: that parameter holds the per-instance values to carry, and
            # rebinding it here left every label row without them.
            shape = region_metrics_for(object_id, name, volume, sample_size)
            extent = _bbox_extent(binary)
            roundness = shape.get("sphericity", shape.get("circularity", float("nan")))
            rows.append({
                **shape,
                "object_id": object_id, "group_id": group_id, "entity_name": name,
                "entity_kind": kind, "row_type": "file", "label_id": None,
                "spatial_dims": ndim,
                "mesh": b"" if planar else generate_mesh(
                    binary, (0, 0, 0), sample_size,
                    step_size=options.step_size,
                    smooth_sigma=sigma_for_shape(
                        roundness, float(binary.sum() / extent) if extent else float("nan"),
                        sigma_min=0.3, sigma_max=options.smooth_sigma * 2),
                    target_reduction=options.target_reduction, level=options.level,
                ),
                "outline": generate_outline(binary, (0, 0), sample_size) if planar else b"",
                "skeleton": b"",
            })
            continue

        # The view itself, not a contiguous copy of it: label_metrics makes whatever
        # ITK needs, and passing the same array the instance processor passed is what
        # lets the per-object cache recognise the two calls as the same work.
        labels = volume
        props = regions_for(object_id, name, labels)
        if not props:
            continue
        measured = label_metrics_for(object_id, name, labels, sample_size)
        shape_keys = METRICS_2D if ndim == 2 else METRICS_3D
        skeletons = (
            skeletons_for(object_id, name, labels, sample_size,
                          options.max_skeleton_voxels, options.num_threads)
            if options.with_skeletons and wants_skeletons(name, options.skeleton_entities)
            else {}
        )
        # In batches, so the pool always has work but the cropped masks in flight never
        # add up to another copy of the object.
        for batch in _batched(props, _GEOMETRY_BATCH):
            pending: List[Dict[str, Any]] = []
            tasks: List[Tuple[Any, ...]] = []
            for rp in batch:
                stats = measured.get(int(rp.label), {})
                shape_metrics = {key: stats.get(key, float("nan")) for key in shape_keys}
                origin = tuple(int(v) for v in rp.bbox[:ndim])
                bbox_extent = float(np.prod([hi - lo for lo, hi
                                             in zip(rp.bbox[:ndim], rp.bbox[ndim:])]))
                fill_ratio = rp.area / bbox_extent if bbox_extent > 0 else float("nan")
                roundness = shape_metrics.get("sphericity",
                                              shape_metrics.get("circularity", float("nan")))
                carried = (metrics or {}).get((name, int(rp.label))) or {}
                pending.append({
                    **_polarity(stats.get("centroid_um"), centre),
                    **_skeleton_metrics(skeletons.get(int(rp.label))),
                    **{k: v for k, v in carried.items() if k in CARRIED_METRICS},
                    **shape_metrics,
                    "object_id": object_id, "group_id": group_id, "entity_name": name,
                    "entity_kind": kind, "row_type": "instance", "label_id": int(rp.label),
                    "spatial_dims": ndim,
                    "skeleton": skeleton_payload(skeletons.get(int(rp.label))),
                })
                tasks.append((
                    rp.image.astype(bool), origin, tuple(sample_size), planar,
                    options.step_size,
                    sigma_for_shape(roundness, fill_ratio, sigma_min=0.3,
                                    sigma_max=options.smooth_sigma * 2),
                    options.target_reduction, options.level,
                ))
            for row, (mesh_payload, outline_payload) in zip(pending, pool.map(tasks)):
                row["mesh"] = mesh_payload
                row["outline"] = outline_payload
                rows.append(row)

    if options.contact_max_um is not None:
        rows.extend(_contact_rows(volumes, kinds, sample_size, object_id, group_id,
                                  options.contact_max_um, object_mask_name))
    return rows


def _polarity(centroid_um: Optional[Sequence[float]],
              centre: Optional[Sequence[float]]) -> Dict[str, Any]:
    """Direction and distance from the object centre to this instance, or nothing."""
    if centroid_um is None or centre is None or len(centroid_um) != len(centre):
        return {}
    return polarity_from_offset([c - o for c, o in zip(centroid_um, centre)])


def _skeleton_metrics(skeleton) -> Dict[str, Any]:
    """Branches, length and tortuosity of a skeleton this run already computed."""
    if skeleton is None:
        return {}
    return skeleton_graph_metrics(skeleton)


def _contact_rows(
    volumes: Mapping[str, np.ndarray],
    kinds: Mapping[str, str],
    sample_size: Sequence[float],
    object_id: str,
    group_id: str,
    max_gap_um: float,
    object_mask_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """The pairwise edge list, in the row shape the 3D viewer splits out on load."""
    contacts = contacts_for(object_id, volumes, kinds, sample_size, max_gap_um,
                            object_mask_name)
    return [
        {
            "object_id": object_id, "group_id": group_id, "row_type": "contact",
            "entity_a": entity_a, "label_a": label_a,
            "entity_b": entity_b, "label_b": label_b,
            "gap_um": gap_um, "mesh": b"", "skeleton": b"",
        }
        for entity_a, label_a, entity_b, label_b, gap_um in contacts
    ]


def _maybe_int(value: Any) -> Optional[int]:
    """An id as an int, or None for the rows that have none (masks, contacts)."""
    if value is None or value == "":
        return None
    number = float(value)
    return None if math.isnan(number) else int(number)


def write_geometry_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write geometry.parquet: the payloads as BLOBs, zstd-compressed by parquet itself.

    An explicit schema, because an object whose instances all failed to mesh would otherwise
    give polars a column of nothing to guess a type from, and the widgets query these
    columns by name whether or not this particular object has anything in them.
    """
    import polars as pl

    types = {
        name: (pl.Utf8 if name in _TEXT_FIELDS else
               pl.Int64 if name in _INT_FIELDS else
               pl.Binary if name in _BLOB_FIELDS else pl.Float64)
        for name in GEOMETRY_COLUMNS
    }
    mesh_counts = [payload_counts(r.get("mesh") or b"") for r in rows]
    skeleton_counts = [payload_counts(r.get("skeleton") or b"") for r in rows]
    outline_counts = [payload_counts(r.get("outline") or b"") for r in rows]
    header_counts = {
        "mesh_vertices": [c[0] for c in mesh_counts],
        "mesh_faces": [c[1] for c in mesh_counts],
        "skeleton_vertices": [c[0] for c in skeleton_counts],
        "skeleton_edges": [c[1] for c in skeleton_counts],
        "outline_vertices": [c[0] for c in outline_counts],
        "outline_edges": [c[1] for c in outline_counts],
    }
    columns: Dict[str, List[Any]] = {}
    for name in GEOMETRY_COLUMNS:
        if name in header_counts:
            columns[name] = header_counts[name]
        elif name in _BLOB_FIELDS:
            # NULL rather than an empty blob: "has geometry" is then a plain IS NOT NULL.
            columns[name] = [(r.get(name) or None) for r in rows]
        elif name in _INT_FIELDS:
            columns[name] = [_maybe_int(r.get(name)) for r in rows]
        else:
            columns[name] = [r.get(name) for r in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(columns, schema=types).write_parquet(path, compression="zstd")
    return path


def write_geometry(object_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write one object's geometry into its own folder, and say where it went."""
    return write_geometry_parquet(object_dir / GEOMETRY_FILENAME, rows)
