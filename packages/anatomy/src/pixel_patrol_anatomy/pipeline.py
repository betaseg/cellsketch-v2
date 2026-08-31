"""Run the processors over a batch of object folders and write one report.

PixelPatrol's own pipeline splits a record that exceeds a memory budget. An object cannot
survive that: its measurements are cross-entity, so a chunk holding two of five entities
can answer almost nothing. This runs the processors directly, one whole object at a time in a
pool sized by what an object actually costs, and writes the parquet the viewer reads.

Row layout, unchanged from what the widgets expect:
  obs_level = 0   one row per object, with the instance/distance/contact list columns
  obs_level = 1   one row per entity, keyed by dim_c → channel_names
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import polars as pl
from pixel_patrol_base.core.record import Record, record_from

from pixel_patrol_anatomy.plugins.loaders.object_loader import ObjectLoader
from pixel_patrol_anatomy.plugins.processors.contacts import ContactsProcessor
from pixel_patrol_anatomy.plugins.processors.instances import InstanceProcessor
from pixel_patrol_anatomy.plugins.processors.mesh import MeshProcessor
from pixel_patrol_anatomy.plugins.processors.morphology import MorphologyProcessor
from pixel_patrol_anatomy.skeletons import CACHE

logger = logging.getLogger(__name__)

# Loader metadata that describes how the record was assembled rather than what was measured.
# dim_order goes too: spatial_dims says 2 or 3, and the axis order never varies.
_INTERNAL_META = {"dim_names", "ndim", "dtype", "shape", "n_images", "dim_c", "dim_order"}

# Columns the object row carries but an entity row does not: they describe the whole object.
_OBJECT_ONLY = ("object_shape", "object_center_z_um", "object_center_y_um",
                "object_center_x_um")


@dataclass
class ObjectResult:
    """One object's rows, or the error that stopped it."""
    object_id: str
    object_row: Optional[Dict[str, Any]] = None
    entity_rows: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    seconds: float = 0.0


def _image_meta(record: Record) -> Dict[str, Any]:
    """The loader's metadata, plus the extent of each axis.

    Columns that only describe the stacking (dtype, ndim, dim_names, dim_order) are left out:
    they are the same for every object and say nothing about the data. Which axes those are is
    in spatial_dims and the size_<axis> columns.
    """
    meta = {k: v for k, v in record.meta.items() if k not in _INTERNAL_META}
    for axis, size in zip(record.dim_order, record.data.shape):
        meta[f"size_{axis}"] = int(size)
    meta["num_pixels"] = int(np.prod(record.data.shape))
    return meta


def _entity_record(record: Record, c_index: int) -> Record:
    """One channel of an object, in the shape a per-entity processor expects."""
    c_axis = record.dim_order.index("C")
    data = np.take(record.data, [c_index], axis=c_axis)
    meta = {k: v for k, v in record.meta.items() if k != "shape"}
    return record_from(data, {**meta, "dim_c": c_index}, kind=record.kind)


def _roll_up(processor, entity_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-entity values as they appear on the object row (summed, or dropped if they vary)."""
    rolled: Dict[str, Any] = {}
    for column in getattr(processor, "OUTPUT_SCHEMA", {}):
        aggregate = processor.get_aggregation(column)
        if aggregate is not None:
            rolled[column] = aggregate(list(entity_rows), {})
    return rolled


def _object_processors(excluded: Iterable[str]) -> List[Any]:
    excluded = set(excluded or ())
    return [p for p in (InstanceProcessor(), ContactsProcessor(), MeshProcessor())
            if p.NAME not in excluded]


def analyse_object(folder: Path, group: str, excluded: Sequence[str] = ()) -> ObjectResult:
    """Load one object folder and measure it. Never raises: failures come back as `error`."""
    started = time.perf_counter()
    object_id = folder.name
    try:
        record = ObjectLoader().load(folder)
        meta = _image_meta(record)
        provenance = _provenance(folder, record)

        morphology = MorphologyProcessor()
        entity_rows = []
        for c_index, _ in enumerate(record.meta.get("channel_names") or []):
            measured = morphology.run_chunk(_entity_record(record, c_index))
            entity_rows.append({**meta, **provenance, **measured,
                                "obs_level": 1, "dim_c": c_index,
                                "imported_path_short": group})

        object_row = {**meta, **provenance, **_roll_up(morphology, entity_rows),
                      "obs_level": 0, "imported_path_short": group}
        for processor in _object_processors(excluded):
            object_row.update(processor.run_chunk(record))

        # The entity rows describe one channel, so nothing about the whole object belongs on
        # them; dropped after the roll-up, which reads them.
        for row in entity_rows:
            for column in _OBJECT_ONLY:
                row.pop(column, None)

        return ObjectResult(object_id, object_row, entity_rows,
                            seconds=time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - one bad object must not stop the batch
        logger.error("anatomy: %s failed: %s: %s", object_id, type(exc).__name__, exc)
        return ObjectResult(object_id, error=f"{type(exc).__name__}: {exc}",
                            seconds=time.perf_counter() - started)
    finally:
        CACHE.clear()


def _provenance(folder: Path, record: Record) -> Dict[str, Any]:
    """Where the object came from and how much was read.

    An object is a folder, so most of PixelPatrol's file columns (name, parent, depth, type)
    describe nothing here and are left out. `file_extension` is the format of the volumes
    themselves, which they share, so it stays: it is what tells a TIFF batch from any other.
    """
    files = [folder / name for name in (record.meta.get("entity_files") or [])]
    stats = [p.stat() for p in files if p.exists()]
    suffixes = {p.suffix.lower().lstrip(".") for p in files}
    return {
        "path": str(folder.resolve()),
        "file_extension": suffixes.pop() if len(suffixes) == 1 else None,
        "size_bytes": sum(s.st_size for s in stats) or None,
        "modification_date": (datetime.fromtimestamp(max(s.st_mtime for s in stats))
                              if stats else None),
    }


# ── the batch ─────────────────────────────────────────────────────────────────

@dataclass
class Report:
    """What a run produced, and what it could not."""
    rows: pl.DataFrame
    failures: Dict[str, str]
    seconds: float
    per_object_seconds: Dict[str, float]

    @property
    def n_objects(self) -> int:
        return int((self.rows["obs_level"] == 0).sum()) if self.rows.height else 0


def group_of(folder: Path, root: Path, paths: Sequence[str]) -> str:
    """The import path an object belongs to, which is the viewer's default grouping."""
    try:
        relative = folder.resolve().relative_to(root.resolve())
    except ValueError:
        return ""
    for path in paths:
        if relative == Path(path) or Path(path) in relative.parents:
            return path
    return relative.parts[0] if len(relative.parts) > 1 else ""


def worker_count(requested: Optional[int], n_objects: int, peak_gb: float) -> int:
    """As many objects at once as memory allows, never more than there are objects."""
    if requested is not None:
        return max(1, min(requested, n_objects))
    try:
        import psutil
        available = psutil.virtual_memory().available / 1024**3 * 0.8
    except Exception:
        return 1
    fit = int(available // peak_gb) if peak_gb > 0 else n_objects
    return max(1, min(fit, n_objects, os.cpu_count() or 1))


def analyse(
    folders: Sequence[Path],
    root: Path,
    paths: Sequence[str] = (),
    *,
    excluded: Sequence[str] = (),
    workers: Optional[int] = None,
    peak_gb: float = 4.0,
) -> Report:
    """Measure every object folder and return the rows plus whatever failed."""
    started = time.perf_counter()
    groups = [group_of(f, root, paths) for f in folders]
    n_workers = worker_count(workers, len(folders), peak_gb)
    # Meshing farms instances out to processes of its own, and this pool is sized by memory
    # rather than by cores, so on big objects most of the machine would otherwise sit idle.
    # Each object gets the share left over; an explicit --mesh-workers already set it.
    os.environ.setdefault("PP_ANATOMY_MESH_WORKERS",
                          str(max(1, (os.cpu_count() or 1) // n_workers)))
    logger.info("anatomy: %d object(s), %d worker(s), %s mesh process(es) each",
                len(folders), n_workers, os.environ["PP_ANATOMY_MESH_WORKERS"])

    if n_workers == 1:
        results = [analyse_object(f, g, excluded) for f, g in zip(folders, groups)]
    else:
        # spawn, not fork: the parent holds native threads (polars, DuckDB, BLAS) and a
        # forked child can deadlock on their locks. Workers inherit the environment, which
        # is how they get the analysis options.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as pool:
            results = list(pool.map(analyse_object, folders, groups,
                                    [tuple(excluded)] * len(folders)))

    rows: List[Dict[str, Any]] = []
    failures: Dict[str, str] = {}
    for result in results:
        if result.error:
            failures[result.object_id] = result.error
            continue
        rows.append(result.object_row)
        rows.extend(result.entity_rows)
        logger.info("anatomy: %s done in %.1f s", result.object_id, result.seconds)

    return Report(
        rows=_frame(rows),
        failures=failures,
        seconds=time.perf_counter() - started,
        per_object_seconds={r.object_id: r.seconds for r in results},
    )


# Identity first, then how the object was measured; everything else follows in the order the
# processors declared it.
_LEADING_COLUMNS = ("obs_level", "object_id", "imported_path_short", "dim_c",
                    "entity_name", "entity_kind", "spatial_dims")


def _frame(rows: Sequence[Dict[str, Any]]) -> pl.DataFrame:
    """Rows as one table, with every column every row can have and nothing that is all-null."""
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows, infer_schema_length=None, strict=False)
    keep = [c for c in frame.columns if frame[c].null_count() < frame.height]
    frame = frame.select(keep).with_columns(_as_float32(frame.schema, keep))
    ordered = [c for c in _LEADING_COLUMNS if c in frame.columns]
    return frame.select(ordered + [c for c in frame.columns if c not in ordered])


def _as_float32(schema: Dict[str, Any], columns: Sequence[str]) -> List[pl.Expr]:
    """Measurements as float32, scalars and lists alike.

    Seven significant digits is far more than a µm measurement carries, and the viewer loads
    the whole file: on a real batch this is ~12% of it.
    """
    casts = []
    for column in columns:
        dtype = schema[column]
        if dtype == pl.Float64:
            casts.append(pl.col(column).cast(pl.Float32))
        elif dtype == pl.List(pl.Float64):
            casts.append(pl.col(column).cast(pl.List(pl.Float32)))
    return casts


# ── writing ───────────────────────────────────────────────────────────────────

PRIVACY_SUMMARY = [
    "- object folder paths and the names of the volumes read",
    "- voxel size, extent and per-structure measurements",
    "- no pixel data: geometry is written beside the report, never into it",
]


def write(report: Report, output: Path, *, root: Path, paths: Sequence[str],
          flavor: str, project_name: Optional[str] = None,
          omit_base_dir: bool = False) -> Path:
    """Write the report parquet, with the provenance metadata the viewer's footer shows."""
    from pixel_patrol_base.core.project_metadata import ProjectMetadata
    from pixel_patrol_base.io.parquet_io import save_parquet

    metadata = ProjectMetadata(
        project_name=project_name or output.stem,
        flavor=flavor,
        loader=ObjectLoader.NAME,
        base_dir=str(root.resolve()),
        paths=list(paths),
        processing_stats={
            "wall_s": round(report.seconds, 3),
            "n_objects": report.n_objects,
            "n_failed": len(report.failures),
            "seconds_per_object": {k: round(v, 1) for k, v in report.per_object_seconds.items()},
        },
        omit_base_dir=omit_base_dir,
        privacy_summary=PRIVACY_SUMMARY,
    )
    save_parquet(report.rows, output, metadata)
    return output


def recolour(report: Path, colours: Dict[str, str]) -> int:
    """Write `colours` into an existing report's entity rows. Returns how many were coloured.

    Colouring is a presentation choice, and a run takes minutes: this lets a palette be tried
    and changed against a report that already exists. `process --colours` writes it the same way,
    so there is one path either way, and the footer metadata is carried over untouched.
    """
    import polars as pl
    import pyarrow.parquet as pq
    from pixel_patrol_base.core.project_metadata import ProjectMetadata
    from pixel_patrol_base.io.parquet_io import save_parquet

    table = pq.read_table(report)
    frame = pl.from_arrow(table)
    if "entity_name" not in frame.columns:
        raise ValueError(f"{report} has no entity rows to colour")

    coloured = frame.with_columns(
        pl.col("entity_name").replace_strict(colours, default=None).alias("entity_colour")
    )
    # The report is rewritten, so it has to keep the provenance it was written with: the
    # viewer's footer strip reads it, and it is the only record of how the run was made.
    footer = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()
              if k.startswith(b"pp_")}
    save_parquet(coloured, report, ProjectMetadata.from_parquet_meta(footer))
    return int(coloured["entity_colour"].is_not_null().sum())
