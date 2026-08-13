# pixel-patrol-cellsketch

CellSketch as a [PixelPatrol](https://pixelpatrol.app/) extension: a loader, four
processors and four viewer widgets that replace `analyze_cell.py`'s own discovery,
reporting and HTML viewers with PixelPatrol's pipeline and report.

**Status:** every measurement `analyze_cell.py` makes is ported, including the mesh and
skeleton geometry for `mesh_viewer.html` and the Blender export. Still to come: `stats_viewer.html`'s
remaining sections (the overview cards and the structure table) as widgets.

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
cellsketch dry-run experiment/                       # what will be analysed, and why not
cellsketch process experiment/ -o report.parquet -p control -p treated
cellsketch view report.parquet --significance
```

On a real dataset (eight vEM alpha cells, 2.5–10.5 GB stacked, five entities each) that
prints:

```
8 cell folder(s); 15,691 MB per task; 4 worker(s) (largest cell needs ~17.9 GB each)
```

`cellsketch` is a thin wrapper over `pixel-patrol`. It exists because two of that
command's options are requirements of this loader rather than tuning knobs, and because
PixelPatrol's CLI cannot pass plugin options, so the analysis flags
(`--voxel-size-um`, `--auto-clip-to-pm`, `--contact-max-um`, …) travel as environment
variables. The wrapper sets:

- `--slice-size C=1 Z=-1 Y=-1 X=-1`, so one leaf block is one whole entity volume.
  Without it PixelPatrol defaults `Z` to 1 and hands the processors 2D planes.
- `--mb-per-task` sized from the largest cell (`n_entities × Z × Y × X × 4` bytes, with
  headroom). Below that PixelPatrol splits a cell to fit its budget, and the processors
  refuse the fragment rather than measuring part of a cell.
- `--max-workers` from how many cells fit in RAM at once. Measured peak on a
  133-megavoxel five-entity cell is 4.4 GB; the estimate predicted 4.7 GB.

### Speed

Measured on one real cell (133 megavoxels, five entities, 6340 instances), with
distances, histograms, contacts and polarity spread all on:

```
load          6s     instances    39s     contacts    16s     meshes   133s
```

Three things got it there, and the first is worth knowing about if you write your own
processor:

- **`scipy.ndimage.minimum` is unusable at this scale** — 54 s per call on a 197-megavoxel
  cell, and 57 s when asked for only 50 of the labels, because its cost follows the volume
  and label count rather than the foreground. Each entity's foreground is indexed once
  instead (positions sorted by label id), after which a measurement is a gather plus
  `np.minimum.reduceat`: 0.01 s, same answers.
- **`--skeleton-entities mito`** — skeletonising dominates everything else (103 s of a
  2-minute cell), and most of it is usually wasted: a granule's skeleton is one branch the
  length of its diameter. Naming just the filaments took that cell from 55 s to 36 s.
- **Skeletons and contacts are computed once per cell**, not once per reader, so
  `--with-mesh` no longer repeats the expensive parts for its geometry.

Two things keep peak memory down, both worth knowing if you profile a run: entity volumes
are stacked in the narrowest integer type their label ids need (usually `uint16`, halving
a `float32`/`int32` segmentation), and the instance processor holds **one** whole-volume
distance transform at a time, reducing it over every entity's labels with one pass before
freeing it.

The underlying command still works if you prefer it, flags and all:

```bash
pixel-patrol process experiment/ -o report.parquet --loader cellsketch \
  -p control -p treated --slice-size C=1 --slice-size Z=-1 --mb-per-task 4096
```

Skip the expensive parts with `--no-instances` / `--no-contacts` (equivalently,
`--processors-exclude cellsketch-instances`).

## Meshes for the 3D viewer and Blender

Geometry never enters the parquet — base64 meshes would multiply the size of the report
every stats query loads — so it goes to one `report_meshes.csv` per cell, exactly where
`analyze_cell.py --with-mesh` put it. Either during processing:

```bash
cellsketch process experiment/ -o report.parquet --with-mesh \
  --mesh-smooth-sigma 1 --mesh-step-size 1 --mesh-level 0.05
# → report.parquet  +  report_meshes/<cell>/report_meshes.csv
```

or afterwards, when you have a report already and want to re-mesh with other settings:

```bash
cellsketch mesh experiment/ -o geometry/ --mesh-step-size 1 --no-skeletons
```

Each CSV holds one row per label instance (with `mesh_b64` and, unless `--no-skeletons`,
`skeleton_b64`), one per whole-structure mask, and the contact edge list the 3D viewer
groups by — drop it on `mesh_viewer.html`, or pass it to `csv_to_blender.py`. The payload
formats are unchanged, and the geometry is identical to what `analyze_cell.py` produced
for the same cell: same vertex and face counts, same bounds.

Meshing is the most expensive thing here (it skeletonises again for the overlay), which
is why it is opt-in rather than part of every run.

## Viewer widgets

Four widgets ship with the package and load automatically in `pixel-patrol view`
(discovered through the `pixel_patrol.viewer_extensions` entry point):

| Widget | Shows |
| --- | --- |
| Instance Morphology | one violin per group for every per-instance metric, with a structure selector |
| Distances Between Structures | distance from each instance of one structure to each other structure |
| Reaching Two Structures At Once | one panel per pair of structures; for each instance the *larger* of its two distances, as a cumulative share — a curve that climbs early means most instances sit against both |
| Contacts & Groups | contact groups (clusters of instances chained by contacts) at a live gap threshold: group-size box plots and the share of instances touching anything, per group, with a summary table |

The reach curves are drawn from quantiles, so a group of 8000 instances costs 51
vertices rather than 8000 points, and the reading is the same at any *n*. They mirror
the panel matrix `stats_viewer.html` grew, computed from the long-format distances.

Contacts & Groups clusters instances with a union-find over the edge list at whatever gap
the slider is set to, seeding every instance as a singleton so "share of instances
touching" has the right denominator. An instance touches something exactly when it lands
in a group of two or more, so the two charts always agree. On real data - 103,314 contacts
within 0.1 µm across eight cells - that resolves to 141 clusters, the largest holding
8,158 instances, with 97.6% of instances touching something: dense enough that the slider
is the point, not the default.

The clustering runs in the browser, so it is covered by running it through node from the
Python suite (`tests/contact_groups_check.mjs`) - identity is cell + entity + label, and a
test pins that a label id repeated in two cells stays two instances.

Each builds its own source — a subquery that unnests the cell row's list columns —
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
| `CELLSKETCH_SKELETON_ENTITIES` | all label entities | — (`--skeleton-entities mito,er`) |
| `CELLSKETCH_EDT_THREADS` | `0` (all cores) | part of `--num-threads` |
| `CELLSKETCH_NUM_THREADS` | `1` (cells already run in parallel) | `--num-threads` |
| `CELLSKETCH_CONTACT_MAX_UM` | `0.5` | `--contact-max-um` |
| `CELLSKETCH_POLARITY_SPREAD` | `0` | `--polarity-spread-labels` (all label entities) |
| `CELLSKETCH_DISTANCE_HISTOGRAMS` | `0` | `--dist-histogram-labels` (all label entities) |

`cellsketch process` sets these from flags of the same name, so you only need the
variables when driving `pixel-patrol` directly. `--with-contacts` has no equivalent:
contacts are a processor, so they are on by default and skipped with `--no-contacts`.

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
