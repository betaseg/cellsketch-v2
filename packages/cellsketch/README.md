# pixel-patrol-cellsketch

CellSketch as a [PixelPatrol](https://pixelpatrol.app/) extension: a loader, a
processor, and (later) viewer widgets that replace `analyze_cell.py`'s own
discovery, reporting, and HTML viewers with PixelPatrol's pipeline and report.

**Status: in progress.** The loader, per-entity morphology, per-instance morphology,
distances, polarity and instance contacts are in place, plus two viewer widgets, so a
run produces a PixelPatrol table carrying everything `analyze_cell.py`'s
`report.parquet` does except meshes and skeleton geometry. Still to port: the mesh and
skeleton payloads (and with them `mesh_viewer.html`), and a contacts widget with the
gap-threshold slider and grouping that `stats_viewer.html` has. `analyze_cell.py`
remains the complete tool until then.

## Data model

A cell is a *folder* of volumes that must be measured together. The loader claims such
a folder via `is_folder_supported`, so PixelPatrol treats the directory as one dataset
and never descends into it — the TIFFs inside are not records of their own. Each record
then gets one row per dimension slice, which maps onto CellSketch like this:

| CellSketch concept | PixelPatrol representation |
| --- | --- |
| one cell folder | one record, entity volumes stacked along `C` (`CZYX`) |
| one entity (organelle) | one row at `obs_level=1`, `dim_c` → `channel_names` |
| whole cell | one row at `obs_level=0` |
| one instance (`row_type=instance`) | an element of the `instance_*` list columns, on the cell row |
| one contact (`row_type=contact`) | an element of the `contact_*` list columns, on the cell row |
| experimental group | a `-p` import path (`imported_path_short`) |

PixelPatrol emits one row per leaf block, and a leaf is one entity — which is what makes
`entity_name` a groupable column with scalar metrics the built-in widgets can plot. It
also means a leaf sees only its own entity, so anything measured *between* entities
(a distance, a contact) is computed by a cell-level processor that sees every channel,
and stored as parallel list columns on the cell row. There are three such groups, each
one `unnest` away, and several `unnest()` calls in one `SELECT` stay row-aligned:

**Instances** — `analyze_cell.py`'s `row_type=instance` table:

```sql
SELECT cell_id, unnest(instance_entity) AS entity_name, unnest(instance_label) AS label,
                unnest(instance_volume_um3) AS volume_um3,
                unnest(instance_sphericity) AS sphericity
FROM pp_data WHERE obs_level = 0
```

**Distances**, long format — one element per instance × target entity, because target
names come from the data and PixelPatrol drops columns a processor did not declare:

```sql
SELECT cell_id, unnest(distance_entity) AS entity_name, unnest(distance_label) AS label,
                unnest(distance_target) AS target, unnest(distance_um) AS distance_um
FROM pp_data WHERE obs_level = 0
```

**Contacts** — the pairwise edge list, thresholded on `gap_um` interactively:

```sql
SELECT cell_id, unnest(contact_entity_a) AS entity_a, unnest(contact_label_a) AS label_a,
                unnest(contact_entity_b) AS entity_b, unnest(contact_label_b) AS label_b,
                unnest(contact_gap_um)   AS gap_um
FROM pp_data WHERE obs_level = 0
```

Distances and gaps are measured voxel centre to voxel centre, so structures sharing a
face read *one voxel step* rather than zero, and with anisotropic voxels the smallest
non-overlapping reading depends on direction. Zero means genuine overlap. `analyze_cell.py`
computes exactly the same numbers.

## Install (development)

Folder discovery for suffix-less directories needs `pixel-patrol-base` newer than
0.8.0, so install both from a local checkout:

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e . \
  -e /path/to/pixel-patrol/packages/pixel-patrol-base pytest
```

## Run

```bash
pixel-patrol process <cells-root> -o report.parquet --loader cellsketch \
  --slice-size C=1 --slice-size Z=-1 --mb-per-task 4096
pixel-patrol view report.parquet
```

For experimental groups, import each group as its own path:

```bash
pixel-patrol process experiment/ -o report.parquet --loader cellsketch \
  -p control -p treated --slice-size C=1 --slice-size Z=-1 --mb-per-task 4096
```

Two flags matter and are not optional:

- `--slice-size C=1 --slice-size Z=-1` makes one leaf block one whole entity volume.
  Without it PixelPatrol defaults `Z` to 1 and measures 2D planes.
- `--mb-per-task` must exceed the stacked size of one cell
  (`n_entities × Z × Y × X × 4` bytes). Below that PixelPatrol splits the volume
  spatially and the processor refuses the fragment rather than reporting metrics
  measured on part of a cell.

## Viewer widgets

Two widgets ship with the package and load automatically in `pixel-patrol view`
(discovered through the `pixel_patrol.viewer_extensions` entry point):

| Widget | Shows |
| --- | --- |
| Instance Morphology | one violin per group for every per-instance metric, with a structure selector |
| Distances Between Structures | distance from each instance of one structure to each other structure |
| Reaching Two Structures At Once | one panel per pair of structures; for each instance the *larger* of its two distances, as a cumulative share — a curve that climbs early means most instances sit against both |

The reach curves are drawn from quantiles, so a group of 8000 instances costs 51
vertices rather than 8000 points, and the reading is the same at any *n*. They mirror
the panel matrix `stats_viewer.html` grew, computed from the long-format distances.

Both build their own source — a subquery that unnests the cell row's list columns —
and hand it to the viewer's own distribution engine, so instance-level data goes
through the same violins, palette, grouping and **Mann-Whitney significance brackets**
as the built-in widgets. Turn the brackets on with the sidebar's *Show significance*
(or `pixel-patrol view … --significance`).

That reuse is the reason for this package: the engine takes an arbitrary source table
expression, so it does not care that the rows came from an unnest.

## Configuration

PixelPatrol instantiates plugins with no arguments and its CLI cannot pass plugin
options, so the analysis knobs are environment variables (defaults match the
`analyze_cell.py` flags of the same name):

| Variable | Default | `analyze_cell.py` equivalent |
| --- | --- | --- |
| `CELLSKETCH_VOXEL_SIZE_UM` | inferred from TIFF metadata | `--voxel-size-um z,y,x` |
| `CELLSKETCH_AUTO_CLIP_TO_PM` | `0` | `--auto-clip-to-pm` |
| `CELLSKETCH_AUTO_LABEL_MASKS` | `0` | `--auto-label-masks` |
| `CELLSKETCH_MAX_SKELETON_VOXELS` | `500000` | `--max-skeleton-voxels` |
| `CELLSKETCH_NUM_THREADS` | `1` (cells already run in parallel) | `--num-threads` |
| `CELLSKETCH_CONTACT_MAX_UM` | `0.5` | `--contact-max-um` |

`--with-contacts` has no equivalent: contacts are a processor, so they are on by default
and skipped with `--processors-exclude cellsketch-contacts`.

## Tests

```bash
.venv/bin/python -m pytest
```

`tests/synthetic.py` builds a small grouped batch of synthetic cells (membrane,
nucleus, mitochondria) with voxel size in the TIFF metadata; it is also runnable as a
script to produce a dataset to try the CLI on:

```bash
.venv/bin/python tests/synthetic.py /tmp/cells
```
