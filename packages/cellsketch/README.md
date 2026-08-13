# pixel-patrol-cellsketch

CellSketch as a [PixelPatrol](https://pixelpatrol.app/) extension: a loader, a
processor, and (later) viewer widgets that replace `analyze_cell.py`'s own
discovery, reporting, and HTML viewers with PixelPatrol's pipeline and report.

**Status: in progress.** The loader, per-entity morphology, and instance contacts are
in place, so a run produces a PixelPatrol table with one row per cell and one row per
entity. Per-instance distances, meshes and the viewer widgets are not ported yet;
`analyze_cell.py` remains the complete tool until they are.

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
| one instance (`row_type=instance`) | an element of the `instance_*` list columns |
| one contact (`row_type=contact`) | an element of the `contact_*` list columns, on the cell row |
| experimental group | a `-p` import path (`imported_path_short`) |

Instances have no row of their own because PixelPatrol has no granularity below a
dimension slice. Widgets unnest them in SQL instead:

```sql
SELECT cell_id, entity_name, unnest(instance_volume_um3) AS volume_um3
FROM pp_data
WHERE obs_level = 1 AND entity_kind = 'label'
```

Contacts are pairs, so they belong to no single entity and ride on the cell row as a
parallel edge list. Unnest them the same way (several `unnest()` calls in one `SELECT`
stay row-aligned) and threshold the gap interactively:

```sql
SELECT cell_id, unnest(contact_entity_a) AS entity_a, unnest(contact_label_a) AS label_a,
                unnest(contact_entity_b) AS entity_b, unnest(contact_label_b) AS label_b,
                unnest(contact_gap_um)   AS gap_um
FROM pp_data WHERE obs_level = 0
```

`gap_um` is measured in whole voxel steps, so instances sharing a face read *one voxel
step*, not zero, and with anisotropic voxels the smallest reportable gap depends on
direction. `analyze_cell.py` computes exactly the same numbers.

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
