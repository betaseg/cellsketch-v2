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
from typing import Iterable, Tuple

import click
import tifffile

from pixel_patrol_cellsketch.discovery import inspect_cell_dir

logger = logging.getLogger(__name__)

# A leaf block must be one whole entity volume: C splits per entity, and every spatial
# axis keeps its full extent.
SLICE_SIZE = {"C": 1, "Z": -1, "Y": -1, "X": -1}

# Headroom over the largest cell's stacked size, for the copies a processor makes
# (distance transforms are float32 per target, skeletons hold their own arrays).
_MB_PER_TASK_HEADROOM = 4.0
_MIN_MB_PER_TASK = 512.0


def find_cell_dirs(root: Path) -> list[Path]:
    """Every folder under root that the loader would claim as a cell."""
    from pixel_patrol_cellsketch.plugins.loaders.cell_loader import CellLoader

    loader = CellLoader()
    if loader.is_folder_supported(root):
        return [root]
    return sorted(d for d in root.rglob("*") if d.is_dir() and loader.is_folder_supported(d))


def _stacked_mb(cell_dir: Path) -> float:
    """Megabytes one cell occupies as an int32 CZYX stack, from TIFF headers only."""
    d = inspect_cell_dir(cell_dir)
    if d.source is None:
        return 0.0
    try:
        with tifffile.TiffFile(d.source) as tf:
            shape = tf.series[0].shape
    except Exception:
        return 0.0
    voxels = math.prod(int(s) for s in shape)
    return voxels * max(1, len(d.entities)) * 4 / 1024 / 1024


def auto_mb_per_task(cell_dirs: Iterable[Path]) -> float:
    """A budget no cell exceeds, so PixelPatrol never splits one."""
    largest = max((_stacked_mb(d) for d in cell_dirs), default=0.0)
    return max(_MIN_MB_PER_TASK, largest * _MB_PER_TASK_HEADROOM)


def _apply_analysis_env(
    voxel_size_um: str | None,
    auto_clip_to_pm: bool,
    auto_label_masks: bool,
    contact_max_um: float | None,
    max_skeleton_voxels: int | None,
    num_threads: int | None,
) -> None:
    """Plugin options travel as environment variables; see config.CellSketchConfig."""
    settings = {
        "CELLSKETCH_VOXEL_SIZE_UM": voxel_size_um,
        "CELLSKETCH_AUTO_CLIP_TO_PM": "1" if auto_clip_to_pm else None,
        "CELLSKETCH_AUTO_LABEL_MASKS": "1" if auto_label_masks else None,
        "CELLSKETCH_CONTACT_MAX_UM": contact_max_um,
        "CELLSKETCH_MAX_SKELETON_VOXELS": max_skeleton_voxels,
        "CELLSKETCH_NUM_THREADS": num_threads,
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
@click.option("--no-contacts", is_flag=True, help="Skip the instance contact edge list.")
@click.option("--no-instances", is_flag=True,
              help="Skip per-instance measurements: entity-level morphology only.")
@click.option("--mb-per-task", type=float, default=None,
              help="Work budget per task in MB. Sized from the largest cell when omitted.")
@click.option("--max-workers", type=int, default=None, help="Worker processes (default: auto).")
def process(
    cell_dir: Path, output: Path, paths: Tuple[str, ...], voxel_size_um: str | None,
    auto_clip_to_pm: bool, auto_label_masks: bool, contact_max_um: float | None,
    max_skeleton_voxels: int | None, num_threads: int | None, no_contacts: bool,
    no_instances: bool, mb_per_task: float | None, max_workers: int | None,
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
                        contact_max_um, max_skeleton_voxels, num_threads)

    budget = mb_per_task if mb_per_task is not None else auto_mb_per_task(cells)
    excluded = {"cellsketch-contacts"} if no_contacts else set()
    if no_instances:
        excluded.add("cellsketch-instances")

    click.echo(f"{len(cells)} cell folder(s); {budget:,.0f} MB per task")
    project = api.create_project(output.stem, cell_dir, loader="cellsketch", output_path=output)
    if paths:
        api.add_paths(project, list(paths))
    api.process_files(
        project,
        slice_size=SLICE_SIZE,
        mb_per_task=budget,
        max_workers=max_workers,
        processors_excluded=excluded or None,
    )
    click.echo(f"Report written to {output}")


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
    click.echo(f"\nSuggested --mb-per-task: {auto_mb_per_task(cells):.0f}")
    if problems:
        click.echo(f"{problems} problem(s) found.")
        raise SystemExit(1)


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
