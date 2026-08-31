"""``pixel-patrol-anatomy``: survey a batch, measure it, and open the report.

``process`` drives the pipeline in :mod:`pixel_patrol_anatomy.pipeline` rather than
PixelPatrol's, which would split objects too large for its budget. Plugin options travel as
environment variables, since processors are constructed with no arguments; every flag here
sets one.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import click
import tifffile

if TYPE_CHECKING:                      # mesh.py pulls in scikit-image
    from pixel_patrol_anatomy.mesh import MeshOptions

from pixel_patrol_anatomy.config import AnatomyConfig
from pixel_patrol_anatomy import pipeline
from pixel_patrol_anatomy.discovery import inspect_object_dir
from pixel_patrol_anatomy.spatial import voxel_size

logger = logging.getLogger(__name__)

# Written into the parquet footer metadata (pp_flavor); the viewer shows it as a chip in the
# report-info strip at the foot of the page.
FLAVOR = "object anatomy"

# Peak resident memory per worker, as a multiple of (stack + one distance transform).
# Calibrated on a 133-megavoxel five-entity object: predicted 4.7 GB, measured 4.4 GB.
_PEAK_OVERHEAD = 2.5


def find_object_dirs(root: Path) -> list[Path]:
    """Every folder under root that the loader would claim as an object."""
    from pixel_patrol_anatomy.plugins.loaders.object_loader import ObjectLoader

    loader = ObjectLoader()
    if loader.is_folder_supported(root):
        return [root]
    return sorted(d for d in root.rglob("*") if d.is_dir() and loader.is_folder_supported(d))


def _object_extent(object_dir: Path) -> Tuple[int, int]:
    """(voxels per entity, entity count) for one object, from TIFF headers only."""
    d = inspect_object_dir(object_dir)
    if d.source is None:
        return 0, 0
    try:
        with tifffile.TiffFile(d.source) as tf:
            shape = tf.series[0].shape
    except Exception:
        return 0, 0
    return math.prod(int(s) for s in shape), max(1, len(d.entities))


def _stacked_mb(object_dir: Path) -> float:
    """Megabytes one object occupies as a CZYX stack, worst case (4 bytes per label id).

    The loader narrows the stack to the smallest integer type the labels need, usually
    uint16, so this is an over-estimate - which is the safe direction for a budget whose
    only job is to stay above the real size.
    """
    voxels, entities = _object_extent(object_dir)
    return voxels * entities * 4 / 1024 / 1024


def estimate_peak_gb(object_dir: Path) -> float:
    """Rough peak resident memory for processing one object, in GB.

    The stack (2 bytes per voxel per entity) plus one whole-volume float32 distance
    transform - the processors keep only one alive - times measured overhead.
    """
    voxels, entities = _object_extent(object_dir)
    return voxels * (2 * entities + 4) * _PEAK_OVERHEAD / 1024**3


def mesh_options(**overrides: Any) -> "MeshOptions":
    """MeshOptions from the environment, so both commands read one configuration."""
    from pixel_patrol_anatomy.mesh import MeshOptions

    cfg = AnatomyConfig.from_env()
    return MeshOptions(
        smooth_sigma=cfg.mesh_smooth_sigma,
        step_size=cfg.mesh_step_size,
        target_reduction=cfg.mesh_target_reduction,
        level=cfg.mesh_level,
        skeleton_entities=cfg.skeleton_entities,
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
    mesh_format: str | None = None,
) -> None:
    settings = {
        "PP_ANATOMY_MESH_DIR": mesh_dir,
        "PP_ANATOMY_MESH_FORMAT": mesh_format,
        "PP_ANATOMY_MESH_SMOOTH_SIGMA": smooth_sigma,
        "PP_ANATOMY_MESH_STEP_SIZE": step_size,
        "PP_ANATOMY_MESH_TARGET_REDUCTION": target_reduction,
        "PP_ANATOMY_MESH_LEVEL": level,
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
    object_mask: str | None,
    voxel_size_um: str | None,
    auto_clip: bool,
    auto_label_masks: bool,
    contact_max_um: float | None,
    max_skeleton_voxels: int | None,
    num_threads: int | None,
    polarity_spread: bool = False,
    distance_histograms: bool = False,
    skeleton_entities: str | None = None,
    skip_skeletons: bool = False,
) -> None:
    """Plugin options travel as environment variables; see config.AnatomyConfig."""
    settings = {
        "PP_ANATOMY_OBJECT_MASK": object_mask,
        "PP_ANATOMY_VOXEL_SIZE_UM": voxel_size_um,
        "PP_ANATOMY_AUTO_CLIP": "1" if auto_clip else None,
        "PP_ANATOMY_AUTO_LABEL_MASKS": "1" if auto_label_masks else None,
        "PP_ANATOMY_CONTACT_MAX_UM": contact_max_um,
        "PP_ANATOMY_MAX_SKELETON_VOXELS": max_skeleton_voxels,
        "PP_ANATOMY_NUM_THREADS": num_threads,
        "PP_ANATOMY_POLARITY_SPREAD": "1" if polarity_spread else None,
        "PP_ANATOMY_DISTANCE_HISTOGRAMS": "1" if distance_histograms else None,
        "PP_ANATOMY_SKELETON_ENTITIES": skeleton_entities,
        "PP_ANATOMY_NO_SKELETONS": "1" if skip_skeletons else None,
    }
    for key, value in settings.items():
        if value is not None:
            os.environ[key] = str(value)


def _colours_from_file(path: Path) -> dict[str, str]:
    """The palette in a settings file, validated. See config.entity_colours for the format."""
    os.environ["PP_ANATOMY_ENTITY_COLOURS"] = str(path)
    try:
        return AnatomyConfig.from_env().entity_colours
    except ValueError as error:
        raise click.ClickException(str(error)) from None


@click.group()
def cli() -> None:
    """The spatial anatomy of segmented objects, on PixelPatrol."""


@cli.command()
@click.argument("report", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("palette", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def colours(report: Path, palette: Path) -> None:
    """Colour an existing REPORT from a PALETTE file, without measuring anything again.

    PALETTE is JSON, one hex colour per structure:

        {"mito": "#d62728", "er": "#2ca02c"}

    The colours land on the report's entity rows, so they travel with it and every widget draws
    that structure the same. Structures the file does not name keep the built-in palette. Run it
    again with a different file to change your mind; nothing else in the report is touched.
    """
    named = pipeline.recolour(report, _colours_from_file(palette))
    click.echo(f"{report}: coloured {named} entity row(s) from {palette}")


@cli.command()
@click.argument("object_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False, path_type=Path),
              help="Where to write the .parquet report.")
@click.option("--paths", "-p", multiple=True,
              help="Subdirectory to import as its own group (repeatable). Becomes the "
                   "default grouping in the viewer.")
@click.option("--object-mask", required=True, metavar="NAME",
              help="Mask that bounds each object, e.g. pm. Required, and never guessed: "
                   "every measurement is relative to it. The region is cropped to it and "
                   "polarity is measured from its centroid. Run 'dry-run' to see the masks "
                   "each folder has.")
@click.option("--voxel-size-um", default=None, metavar="Z,Y,X",
              help="Voxel size in µm. Inferred from the source TIFF metadata when omitted.")
@click.option("--auto-clip", "auto_clip", is_flag=True,
              help="Clip the entities to the object mask before analysis.")
@click.option("--auto-label-masks", is_flag=True,
              help="Promote masks with several connected components to label entities.")
@click.option("--contact-max-um", type=float, default=None, metavar="T",
              help="Largest instance-pair gap recorded (default: 0.5).")
@click.option("--max-skeleton-voxels", type=int, default=None, metavar="N",
              help="Skip curve skeletons for instances above this voxel count (default: 500000).")
@click.option("--num-threads", type=int, default=None, metavar="N",
              help="kimimaro worker count (default: 1; objects already run in parallel).")
@click.option("--skeleton-entities", default=None, metavar="NAMES",
              help="Only skeletonise these entities, e.g. mito,er. Skeletonising dominates a "
                   "run and a blob's skeleton says nothing, so naming the filaments is the "
                   "single biggest saving. Default: every label entity.")
@click.option("--no-skeletons", "skip_skeletons", is_flag=True,
              help="Skeletonise nothing: no branches, length or tortuosity, and no overlay.")
@click.option("--polarity-spread", is_flag=True,
              help="Also measure each instance's angular spread on the polarity sphere.")
@click.option("--distance-histograms", is_flag=True,
              help="Also measure per-instance distance distributions, not just the minimum.")
@click.option("--colours", "--colors", "colours", metavar="FILE",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="JSON file of structure: hex colour pairs, e.g. {\"mito\": \"#d62728\"}. "
                   "The report carries them, so every widget colours the same way and a shared "
                   "report arrives coloured. Structures it does not name keep the built-in "
                   "palette.")
@click.option("--no-contacts", is_flag=True, help="Skip the instance contact edge list.")
@click.option("--no-instances", is_flag=True,
              help="Skip per-instance measurements: entity-level morphology only.")
@click.option("--max-workers", type=int, default=None, help="Worker processes (default: auto).")
@click.option("--with-mesh", is_flag=True,
              help="Also write per-object geometry for the 3D widgets and Blender: meshes "
                   "and skeletons for a volume, outlines and skeletons for a plane. It goes "
                   "to <output>_meshes/, never into the parquet.")
@click.option("--mesh-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Where --with-mesh writes the geometry (default: <output>_meshes).")
@_mesh_flags
def process(
    object_dir: Path, output: Path, paths: Tuple[str, ...], object_mask: str | None,
    voxel_size_um: str | None,
    auto_clip: bool, auto_label_masks: bool, contact_max_um: float | None,
    max_skeleton_voxels: int | None, num_threads: int | None,
    skeleton_entities: str | None, skip_skeletons: bool, polarity_spread: bool,
    distance_histograms: bool, colours: Path | None, no_contacts: bool, no_instances: bool,
    max_workers: int | None, with_mesh: bool,
    mesh_dir: Path | None, mesh_smooth_sigma: float | None, mesh_step_size: int | None,
    mesh_target_reduction: float | None, mesh_level: float | None,
) -> None:
    """Analyse every object folder under OBJECT_DIR and write one report."""
    objects = find_object_dirs(object_dir)
    if not objects:
        raise click.ClickException(
            f"No object folders found under {object_dir}. An object folder holds a source image "
            "plus <prefix>_<name>_label.tif / _mask.tif volumes; run 'dry-run' to see "
            "what was rejected and why."
        )
    _apply_analysis_env(object_mask, voxel_size_um, auto_clip, auto_label_masks,
                        contact_max_um, max_skeleton_voxels, num_threads,
                        polarity_spread, distance_histograms,
                        skeleton_entities, skip_skeletons)

    meshes_to = (mesh_dir or output.with_name(output.stem + "_meshes")) if with_mesh else None
    _apply_mesh_env(meshes_to, mesh_smooth_sigma, mesh_step_size, mesh_target_reduction, mesh_level)

    excluded = {"anatomy-contacts"} if no_contacts else set()
    if not with_mesh:
        excluded.add("anatomy-mesh")
    if no_instances:
        excluded.add("anatomy-instances")

    peak = max((estimate_peak_gb(d) for d in objects), default=0.0)
    workers = pipeline.worker_count(max_workers, len(objects), peak)
    click.echo(f"{len(objects)} object folder(s); {workers} worker(s) "
               f"(largest object needs ~{peak:.1f} GB each)")

    report = pipeline.analyse(objects, object_dir, list(paths),
                              excluded=sorted(excluded), workers=workers, peak_gb=peak)
    pipeline.write(report, output, root=object_dir, paths=list(paths), flavor=FLAVOR)
    if colours:
        # The same call the `colours` command makes: one path for one column, and it can be
        # run again later without redoing any of the measuring.
        named = pipeline.recolour(output, _colours_from_file(colours))
        click.echo(f"Coloured {named} entity row(s) from {colours}")
    click.echo(f"Report written to {output} "
               f"({report.n_objects} object(s) in {report.seconds:.0f} s)")
    for object_id, error in report.failures.items():
        click.echo(f"FAILED  {object_id}: {error}")
    if report.failures:
        raise SystemExit(1)
    if meshes_to:
        from pixel_patrol_anatomy.mesh import GEOMETRY_FILENAME

        click.echo(f"Meshes written to {meshes_to}/<object>/{GEOMETRY_FILENAME}")


@cli.command(name="dry-run")
@click.argument("object_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--object-mask", default=None, metavar="NAME",
              help="Check this mask is present in every folder, and mark it with * below. "
                   "Left out, the masks are only listed, which is where you find the name "
                   "to pass to 'process'.")
def dry_run(object_dir: Path, object_mask: str | None) -> None:
    """Show which folders would be analysed, and what was ignored in each.

    Reads TIFF headers only, so it is fast even for a large batch. Without --object-mask this
    is the survey you run first, to see which masks the folders have; with one, it also checks
    that every folder has that mask. Exits non-zero if any folder that looks like an object
    cannot be analysed.
    """
    objects = find_object_dirs(object_dir)
    if not objects:
        click.echo(f"No object folders found under {object_dir}.")
        raise SystemExit(1)

    problems = 0
    entity_presence: dict[str, int] = {}
    for folder in objects:
        # Without a name, a missing object mask is not a problem to report: listing the
        # masks is what this command is for.
        d = inspect_object_dir(folder, object_mask)
        if object_mask is None:
            d = replace(d, errors=[e for e in d.errors if "No object mask named" not in e])
        rel = folder.relative_to(object_dir) if folder != object_dir else Path(folder.name)
        click.echo(f"\n{rel}")
        click.echo(f"  source  {d.source.name if d.source else '(none)'}"
                   f"   [{_stacked_mb(folder):,.1f} MB stacked]")
        for kind in ("label", "mask"):
            names = sorted(
                e.name + ("*" if e.name == d.object_mask_name else "")
                for e in d.entities.values() if e.kind == kind
            )
            click.echo(f"  {kind + 's':7s} {', '.join(names) if names else '(none)'}")
        for entity in d.entities.values():
            entity_presence[f"{entity.kind}:{entity.name}"] = (
                entity_presence.get(f"{entity.kind}:{entity.name}", 0) + 1
            )
        for path in d.unparsed:
            click.echo(f"  warn    ignored {path.name}: not <prefix>_<name>_label|labels|mask")
        for path, reason in d.rejected:
            click.echo(f"  warn    ignored {path.name}: {reason}")
        for error in d.errors:
            click.echo(f"  ERROR   {error}")
            problems += 1

    legend = "  (* = object mask)" if object_mask else ""
    click.echo(f"\n===== {len(objects)} object folder(s) ====={legend}")
    for key, count in sorted(entity_presence.items()):
        missing = "" if count == len(objects) else "   ← missing in some objects"
        click.echo(f"  {key:24s} {count}/{len(objects)}{missing}")
    if not object_mask:
        masks = sorted(key.split(":", 1)[1] for key in entity_presence if key.startswith("mask:"))
        click.echo("\nPick the mask that bounds each object and pass it as --object-mask: "
                   f"{', '.join(masks) if masks else '(this batch has no mask entities)'}")
    peak = max((estimate_peak_gb(d) for d in objects), default=0.0)
    workers = pipeline.worker_count(None, len(objects), peak)
    click.echo(f"\nSuggested --max-workers: {workers}   "
               f"(largest object needs ~{peak:.1f} GB)")
    if problems:
        click.echo(f"{problems} problem(s) found.")
        raise SystemExit(1)


@cli.command()
@click.argument("object_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", "-o", required=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Where to write <object>/geometry.parquet.")
@click.option("--object-mask", required=True, metavar="NAME",
              help="Mask that bounds each object, e.g. pm. Required, as for 'process'.")
@click.option("--voxel-size-um", default=None, metavar="Z,Y,X",
              help="Voxel size in µm. Inferred from the source TIFF metadata when omitted.")
@click.option("--auto-clip", "auto_clip", is_flag=True,
              help="Clip the entities to the object mask first.")
@click.option("--no-skeletons", is_flag=True, help="Meshes only, no skeleton overlay.")
@click.option("--skeleton-entities", default=None, metavar="NAMES",
              help="Only overlay skeletons for these entities, e.g. mito,er.")
@click.option("--contact-max-um", type=float, default=None, metavar="T",
              help="Gap threshold for the contact rows the 3D viewer groups by (default: 0.5).")
@click.option("--no-contacts", is_flag=True, help="Leave the contact rows out of the CSV.")
@_mesh_flags
def mesh(
    object_dir: Path, out_dir: Path, object_mask: str | None,
    voxel_size_um: str | None, auto_clip: bool,
    no_skeletons: bool, skeleton_entities: str | None, contact_max_um: float | None,
    no_contacts: bool,
    mesh_smooth_sigma: float | None, mesh_step_size: int | None,
    mesh_target_reduction: float | None, mesh_level: float | None,
) -> None:
    """Write per-object geometry for the 3D widgets and the Blender export.

    Meshes and skeletons for a volume; for a plane, outlines and skeletons, since a plane
    has no surface to mesh.

    The same geometry `process --with-mesh` writes, for when you already have a report and
    only want the 3D files - or want to re-mesh with different settings.
    """
    from pixel_patrol_anatomy.mesh import mesh_rows_for_object, write_geometry
    from pixel_patrol_anatomy.plugins.loaders.object_loader import ObjectLoader
    from pixel_patrol_anatomy.plugins.processors.instances import channel_view

    objects = find_object_dirs(object_dir)
    if not objects:
        raise click.ClickException(f"No object folders found under {object_dir}.")
    _apply_analysis_env(object_mask, voxel_size_um, auto_clip, False, contact_max_um,
                        None, None, skeleton_entities=skeleton_entities)
    _apply_mesh_env(None, mesh_smooth_sigma, mesh_step_size, mesh_target_reduction, mesh_level)
    options = mesh_options(with_skeletons=not no_skeletons,
                           **({"contact_max_um": None} if no_contacts else {}))

    loader = ObjectLoader()
    for folder in objects:
        record = loader.load(folder)
        names = list(record.meta["channel_names"])
        c_axis = record.dim_order.index("C")
        rows = mesh_rows_for_object(
            {name: channel_view(record.data, c_axis, i) for i, name in enumerate(names)},
            dict(zip(names, record.meta["entity_kinds"])),
            voxel_size(record.meta, record.dim_order),
            object_id=record.meta["object_id"],
            object_mask_name=record.meta.get("object_mask_name"),
            options=options,
        )
        path = write_geometry(out_dir / record.meta["object_id"], rows)
        # A plane is outlined, not meshed.
        drawable = "outline" if record.meta.get("spatial_dims") == 2 else "mesh"
        drawn = sum(1 for row in rows if row.get(drawable))
        click.echo(f"{record.meta['object_id']}: {drawn}/{len(rows)} "
                   f"{'outlined' if drawable == 'outline' else 'meshed'} → {path} "
                   f"({path.stat().st_size / 1024**2:.1f} MB)")


@cli.command()
@click.argument("report", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--significance", is_flag=True, help="Show significance brackets from the start.")
@click.option("--port", type=int, default=8052, show_default=True)
def view(report: Path, significance: bool, port: int) -> None:
    """Open a report in the PixelPatrol viewer.

    A convenience alias: the widgets load from this package's entry point whichever viewer
    command opens the report, and `pixel-patrol view` takes more options (grouping,
    filtering, palette).
    """
    from pixel_patrol_base import api

    api.view(report, port=port, is_show_significance=significance)


if __name__ == "__main__":  # pragma: no cover
    cli()
