"""``cellsketch`` - PixelPatrol with the settings a cell folder needs.

Everything here is a thin wrapper over ``pixel-patrol``: it exists because two of that
command's options are not tuning knobs for this loader but requirements (a leaf block
must be one whole entity volume, and a cell must not be split to fit a memory budget),
and because PixelPatrol's CLI has no way to pass plugin options, so the analysis flags
travel as environment variables. ``pixel-patrol process --loader cellsketch`` still
works; you just have to remember the flags yourself.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Iterable, Tuple

import click
import tifffile

from pixel_patrol_cellsketch.config import CellSketchConfig
from pixel_patrol_cellsketch.discovery import inspect_cell_dir

logger = logging.getLogger(__name__)

# A leaf block must be one whole entity volume: C splits per entity, and every spatial
# axis keeps its full extent.
SLICE_SIZE = {"C": 1, "Z": -1, "Y": -1, "X": -1}

# Headroom over the largest cell's stacked size. Only has to be enough that PixelPatrol
# never splits a cell; peak memory is a separate estimate (see estimate_peak_gb).
_MB_PER_TASK_HEADROOM = 1.5
_MIN_MB_PER_TASK = 512.0

# Peak resident memory per worker, as a multiple of (stack + one distance transform).
# Calibrated against a real 133-megavoxel cell with five entities: predicted 4.7 GB,
# measured 4.4 GB. Rough by nature - raise --max-workers or lower it by hand if needed.
_PEAK_OVERHEAD = 2.5


def find_cell_dirs(root: Path) -> list[Path]:
    """Every folder under root that the loader would claim as a cell."""
    from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CellLoader

    loader = CellLoader()
    if loader.is_folder_supported(root):
        return [root]
    return sorted(d for d in root.rglob("*") if d.is_dir() and loader.is_folder_supported(d))


def _cell_extent(cell_dir: Path) -> Tuple[int, int]:
    """(voxels per entity, entity count) for one cell, from TIFF headers only."""
    d = inspect_cell_dir(cell_dir)
    if d.source is None:
        return 0, 0
    try:
        with tifffile.TiffFile(d.source) as tf:
            shape = tf.series[0].shape
    except Exception:
        return 0, 0
    return math.prod(int(s) for s in shape), max(1, len(d.entities))


def _stacked_mb(cell_dir: Path) -> float:
    """Megabytes one cell occupies as a CZYX stack, worst case (4 bytes per label id).

    The loader narrows the stack to the smallest integer type the labels need, usually
    uint16, so this is an over-estimate - which is the safe direction for a budget whose
    only job is to stay above the real size.
    """
    voxels, entities = _cell_extent(cell_dir)
    return voxels * entities * 4 / 1024 / 1024


def estimate_peak_gb(cell_dir: Path) -> float:
    """Rough peak resident memory for processing one cell, in GB.

    The stack (2 bytes per voxel per entity) plus one whole-volume float32 distance
    transform - the processors keep only one alive - times measured overhead.
    """
    voxels, entities = _cell_extent(cell_dir)
    return voxels * (2 * entities + 4) * _PEAK_OVERHEAD / 1024**3


# PixelPatrol sizes each worker at mb_per_task × this, on the assumption that a
# processor may expand its chunk that far, and caps the worker count by available RAM
# accordingly (processing._get_or_create_client). Ours expand ~3.5× - measured 4.4 GB
# peak on a 1.24 GB stack - but the budget has to clear the largest cell either way, so
# one big cell in a batch holds the whole run to few workers. Mirrored here so the
# number this command prints is the number PixelPatrol will actually use.
_PP_WORKER_MEMORY_FACTOR = 8


def suggest_max_workers(cell_dirs: Iterable[Path], mb_per_task: float | None = None) -> int:
    """How many workers will actually run: RAM over the larger of our peak and PP's."""
    cells = list(cell_dirs)
    peak_gb = max((estimate_peak_gb(d) for d in cells), default=0.0)
    if peak_gb <= 0:
        return 1
    budget_gb = (mb_per_task if mb_per_task is not None else auto_mb_per_task(cells)) / 1024
    per_worker_gb = max(peak_gb, budget_gb * _PP_WORKER_MEMORY_FACTOR)
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / 1024**3 * 0.8
    except Exception:
        return 1
    return max(1, min(os.cpu_count() or 1, int(available_gb // per_worker_gb)))


def auto_mb_per_task(cell_dirs: Iterable[Path]) -> float:
    """A budget no cell exceeds, so PixelPatrol never splits one."""
    largest = max((_stacked_mb(d) for d in cell_dirs), default=0.0)
    return max(_MIN_MB_PER_TASK, largest * _MB_PER_TASK_HEADROOM)


def mesh_options(**overrides: Any) -> "MeshOptions":
    """MeshOptions from the environment, so both commands read one configuration."""
    from pixel_patrol_cellsketch.mesh import MeshOptions

    cfg = CellSketchConfig.from_env()
    return MeshOptions(
        smooth_sigma=cfg.mesh_smooth_sigma,
        step_size=cfg.mesh_step_size,
        target_reduction=cfg.mesh_target_reduction,
        level=cfg.mesh_level,
        max_skeleton_voxels=cfg.max_skeleton_voxels,
        num_threads=cfg.num_threads,
        contact_max_um=cfg.contact_max_um,
        **overrides,
    )


def _apply_mesh_env(
    mesh_dir: Path | None,
    smooth_sigma: float | None,
    step_size: int | None,
    target_reduction: float | None,
    level: float | None,
) -> None:
    settings = {
        "CELLSKETCH_MESH_DIR": mesh_dir,
        "CELLSKETCH_MESH_SMOOTH_SIGMA": smooth_sigma,
        "CELLSKETCH_MESH_STEP_SIZE": step_size,
        "CELLSKETCH_MESH_TARGET_REDUCTION": target_reduction,
        "CELLSKETCH_MESH_LEVEL": level,
    }
    for key, value in settings.items():
        if value is not None:
            os.environ[key] = str(value)


def _mesh_flags(fn):
    """The --mesh-* options, identical on `process --with-mesh` and `mesh`."""
    for option in reversed([
        click.option("--mesh-smooth-sigma", type=float, default=None, metavar="SIGMA",
                     help="Gaussian sigma before marching cubes (default: 0.7; 0 disables)."),
        click.option("--mesh-step-size", type=int, default=None, metavar="N",
                     help="Marching-cubes step size; 1 = full resolution (default: 2)."),
        click.option("--mesh-target-reduction", type=float, default=None, metavar="F",
                     help="Decimation fraction (default: 0.8, keeping ~20%% of faces)."),
        click.option("--mesh-level", type=float, default=None, metavar="L",
                     help="Iso-surface level on the signed distance field (default: 0)."),
    ]):
        fn = option(fn)
    return fn


def _apply_analysis_env(
    voxel_size_um: str | None,
    auto_clip_to_pm: bool,
    auto_label_masks: bool,
    contact_max_um: float | None,
    max_skeleton_voxels: int | None,
    num_threads: int | None,
    polarity_spread: bool = False,
    distance_histograms: bool = False,
) -> None:
    """Plugin options travel as environment variables; see config.CellSketchConfig."""
    settings = {
        "CELLSKETCH_VOXEL_SIZE_UM": voxel_size_um,
        "CELLSKETCH_AUTO_CLIP_TO_PM": "1" if auto_clip_to_pm else None,
        "CELLSKETCH_AUTO_LABEL_MASKS": "1" if auto_label_masks else None,
        "CELLSKETCH_CONTACT_MAX_UM": contact_max_um,
        "CELLSKETCH_MAX_SKELETON_VOXELS": max_skeleton_voxels,
        "CELLSKETCH_NUM_THREADS": num_threads,
        "CELLSKETCH_POLARITY_SPREAD": "1" if polarity_spread else None,
        "CELLSKETCH_DISTANCE_HISTOGRAMS": "1" if distance_histograms else None,
    }
    for key, value in settings.items():
        if value is not None:
            os.environ[key] = str(value)


@click.group()
def cli() -> None:
    """Cell component spatial analysis, on PixelPatrol."""


@cli.command()
@click.argument("cell_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False, path_type=Path),
              help="Where to write the .parquet report.")
@click.option("--paths", "-p", multiple=True,
              help="Subdirectory to import as its own group (repeatable). Becomes the "
                   "default grouping in the viewer.")
@click.option("--voxel-size-um", default=None, metavar="Z,Y,X",
              help="Voxel size in µm. Inferred from the source TIFF metadata when omitted.")
@click.option("--auto-clip-to-pm", is_flag=True,
              help="Clip non-membrane entities to the plasma membrane before analysis.")
@click.option("--auto-label-masks", is_flag=True,
              help="Promote masks with several connected components to label entities.")
@click.option("--contact-max-um", type=float, default=None, metavar="T",
              help="Largest instance-pair gap recorded (default: 0.5).")
@click.option("--max-skeleton-voxels", type=int, default=None, metavar="N",
              help="Skip curve skeletons for instances above this voxel count (default: 500000).")
@click.option("--num-threads", type=int, default=None, metavar="N",
              help="kimimaro worker count (default: 1; cells already run in parallel).")
@click.option("--polarity-spread", is_flag=True,
              help="Also measure each instance's angular spread on the polarity sphere.")
@click.option("--distance-histograms", is_flag=True,
              help="Also measure per-instance distance distributions, not just the minimum.")
@click.option("--no-contacts", is_flag=True, help="Skip the instance contact edge list.")
@click.option("--no-instances", is_flag=True,
              help="Skip per-instance measurements: entity-level morphology only.")
@click.option("--mb-per-task", type=float, default=None,
              help="Work budget per task in MB. Sized from the largest cell when omitted.")
@click.option("--max-workers", type=int, default=None, help="Worker processes (default: auto).")
@click.option("--with-mesh", is_flag=True,
              help="Also write per-cell meshes and skeletons for the 3D viewer and Blender. "
                   "They go to <output>_meshes/, never into the parquet.")
@click.option("--mesh-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Where --with-mesh writes report_meshes.csv (default: <output>_meshes).")
@_mesh_flags
def process(
    cell_dir: Path, output: Path, paths: Tuple[str, ...], voxel_size_um: str | None,
    auto_clip_to_pm: bool, auto_label_masks: bool, contact_max_um: float | None,
    max_skeleton_voxels: int | None, num_threads: int | None, polarity_spread: bool,
    distance_histograms: bool, no_contacts: bool, no_instances: bool,
    mb_per_task: float | None, max_workers: int | None, with_mesh: bool,
    mesh_dir: Path | None, mesh_smooth_sigma: float | None, mesh_step_size: int | None,
    mesh_target_reduction: float | None, mesh_level: float | None,
) -> None:
    """Analyse every cell folder under CELL_DIR and write one report."""
    from pixel_patrol_base import api

    cells = find_cell_dirs(cell_dir)
    if not cells:
        raise click.ClickException(
            f"No cell folders found under {cell_dir}. A cell folder holds a source image "
            "plus <prefix>_<name>_label.tif / _mask.tif volumes; run 'cellsketch dry-run' to see "
            "what was rejected and why."
        )
    _apply_analysis_env(voxel_size_um, auto_clip_to_pm, auto_label_masks,
                        contact_max_um, max_skeleton_voxels, num_threads,
                        polarity_spread, distance_histograms)

    meshes_to = (mesh_dir or output.with_name(output.stem + "_meshes")) if with_mesh else None
    _apply_mesh_env(meshes_to, mesh_smooth_sigma, mesh_step_size, mesh_target_reduction, mesh_level)

    budget = mb_per_task if mb_per_task is not None else auto_mb_per_task(cells)
    workers = max_workers if max_workers is not None else suggest_max_workers(cells, budget)
    peak = max((estimate_peak_gb(d) for d in cells), default=0.0)
    excluded = {"cellsketch-contacts"} if no_contacts else set()
    if not with_mesh:
        excluded.add("cellsketch-mesh")
    if no_instances:
        excluded.add("cellsketch-instances")

    click.echo(
        f"{len(cells)} cell folder(s); {budget:,.0f} MB per task; {workers} worker(s) "
        f"(largest cell needs ~{peak:.1f} GB each)"
    )
    project = api.create_project(output.stem, cell_dir, loader="cellsketch", output_path=output)
    if paths:
        api.add_paths(project, list(paths))
    api.process_files(
        project,
        slice_size=SLICE_SIZE,
        mb_per_task=budget,
        max_workers=workers,
        processors_excluded=excluded or None,
    )
    click.echo(f"Report written to {output}")
    if meshes_to:
        click.echo(f"Meshes written to {meshes_to}/<cell>/report_meshes.csv")


@cli.command(name="dry-run")
@click.argument("cell_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def dry_run(cell_dir: Path) -> None:
    """Show which folders would be analysed, and what was ignored in each.

    Reads TIFF headers only, so it is fast even for a large batch. Exits non-zero if any
    folder that looks like a cell cannot be analysed.
    """
    cells = find_cell_dirs(cell_dir)
    if not cells:
        click.echo(f"No cell folders found under {cell_dir}.")
        raise SystemExit(1)

    problems = 0
    entity_presence: dict[str, int] = {}
    for cell in cells:
        d = inspect_cell_dir(cell)
        rel = cell.relative_to(cell_dir) if cell != cell_dir else Path(cell.name)
        click.echo(f"\n{rel}")
        click.echo(f"  source  {d.source.name if d.source else '(none)'}"
                   f"   [{_stacked_mb(cell):,.1f} MB stacked]")
        for kind in ("label", "mask"):
            names = sorted(
                e.name + ("*" if e.name == d.membrane_name else "")
                for e in d.entities.values() if e.kind == kind
            )
            click.echo(f"  {kind + 's':7s} {', '.join(names) if names else '(none)'}")
        for entity in d.entities.values():
            entity_presence[f"{entity.kind}:{entity.name}"] = (
                entity_presence.get(f"{entity.kind}:{entity.name}", 0) + 1
            )
        for path in d.unparsed:
            click.echo(f"  warn    ignored {path.name} — not <prefix>_<name>_label|labels|mask")
        for path, reason in d.rejected:
            click.echo(f"  warn    ignored {path.name} — {reason}")
        for error in d.errors:
            click.echo(f"  ERROR   {error}")
            problems += 1

    click.echo(f"\n===== {len(cells)} cell folder(s) =====  (* = plasma membrane)")
    for key, count in sorted(entity_presence.items()):
        missing = "" if count == len(cells) else "   ← missing in some cells"
        click.echo(f"  {key:24s} {count}/{len(cells)}{missing}")
    budget = auto_mb_per_task(cells)
    peak = max((estimate_peak_gb(d) for d in cells), default=0.0)
    workers = suggest_max_workers(cells, budget)
    click.echo(f"\nSuggested --mb-per-task: {budget:,.0f}   --max-workers: {workers}"
               f"   (~{peak:.1f} GB peak per worker)")
    if workers == 1 and len(cells) > 1:
        click.echo(f"  Cells run one at a time: PixelPatrol sizes each worker at "
                   f"{_PP_WORKER_MEMORY_FACTOR}x --mb-per-task, and the budget has to "
                   "clear the largest cell.")
    if problems:
        click.echo(f"{problems} problem(s) found.")
        raise SystemExit(1)


@cli.command()
@click.argument("cell_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", "-o", required=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Where to write <cell>/report_meshes.csv.")
@click.option("--voxel-size-um", default=None, metavar="Z,Y,X",
              help="Voxel size in µm. Inferred from the source TIFF metadata when omitted.")
@click.option("--auto-clip-to-pm", is_flag=True,
              help="Clip non-membrane entities to the plasma membrane first.")
@click.option("--no-skeletons", is_flag=True, help="Meshes only, no skeleton overlay.")
@click.option("--contact-max-um", type=float, default=None, metavar="T",
              help="Gap threshold for the contact rows the 3D viewer groups by (default: 0.5).")
@click.option("--no-contacts", is_flag=True, help="Leave the contact rows out of the CSV.")
@_mesh_flags
def mesh(
    cell_dir: Path, out_dir: Path, voxel_size_um: str | None, auto_clip_to_pm: bool,
    no_skeletons: bool, contact_max_um: float | None, no_contacts: bool,
    mesh_smooth_sigma: float | None, mesh_step_size: int | None,
    mesh_target_reduction: float | None, mesh_level: float | None,
) -> None:
    """Write per-cell meshes and skeletons for mesh_viewer.html and the Blender export.

    The same geometry `process --with-mesh` writes, for when you already have a report and
    only want the 3D files - or want to re-mesh with different settings.
    """
    from pixel_patrol_cellsketch.mesh import mesh_rows_for_cell, write_mesh_csv
    from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CellLoader
    from pixel_patrol_cellsketch.plugins.processors.instances import channel_view

    cells = find_cell_dirs(cell_dir)
    if not cells:
        raise click.ClickException(f"No cell folders found under {cell_dir}.")
    _apply_analysis_env(voxel_size_um, auto_clip_to_pm, False, contact_max_um, None, None)
    _apply_mesh_env(None, mesh_smooth_sigma, mesh_step_size, mesh_target_reduction, mesh_level)
    options = mesh_options(with_skeletons=not no_skeletons,
                           **({"contact_max_um": None} if no_contacts else {}))

    loader = CellLoader()
    for cell in cells:
        record = loader.load(cell)
        names = list(record.meta["channel_names"])
        c_axis = record.dim_order.index("C")
        rows = mesh_rows_for_cell(
            {name: channel_view(record.data, c_axis, i) for i, name in enumerate(names)},
            dict(zip(names, record.meta["entity_kinds"])),
            (record.meta["pixel_size_Z"], record.meta["pixel_size_Y"], record.meta["pixel_size_X"]),
            cell_id=record.meta["cell_id"],
            options=options,
        )
        path = write_mesh_csv(out_dir / record.meta["cell_id"] / "report_meshes.csv", rows)
        meshed = sum(1 for row in rows if row["mesh_b64"])
        click.echo(f"{record.meta['cell_id']}: {meshed}/{len(rows)} meshed → {path} "
                   f"({path.stat().st_size / 1024**2:.1f} MB)")


@cli.command()
@click.argument("report", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--significance", is_flag=True, help="Show significance brackets from the start.")
@click.option("--port", type=int, default=8052, show_default=True)
def view(report: Path, significance: bool, port: int) -> None:
    """Open a report in the PixelPatrol viewer, with the CellSketch widgets."""
    from pixel_patrol_base import api

    api.view(report, port=port, is_show_significance=significance)


if __name__ == "__main__":  # pragma: no cover
    cli()
