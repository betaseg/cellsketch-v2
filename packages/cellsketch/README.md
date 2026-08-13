# pixel-patrol-cellsketch

CellSketch as a [PixelPatrol](https://pixelpatrol.app/) extension: a loader, a
processor, and (later) viewer widgets that replace `analyze_cell.py`'s own
discovery, reporting, and HTML viewers with PixelPatrol's pipeline and report.

**Status: phase 1.** The loader and the morphology processor are in place, so a run
produces a PixelPatrol table with one row per cell and one row per entity. Distances,
contacts, meshes and the viewer widgets are not ported yet; `analyze_cell.py` remains
the complete tool until they are.

## Data model

PixelPatrol discovers files and gives each record one row per dimension slice. A cell
is a *folder* of volumes that must be measured together, which maps onto that model
like this:

| CellSketch concept | PixelPatrol representation |
| --- | --- |
| one cell | one record, entity volumes stacked along `C` (`CZYX`) |
| one entity (organelle) | one row at `obs_level=1`, `dim_c` → `channel_names` |
| whole cell | one row at `obs_level=0` |
| one instance (`row_type=instance`) | an element of the `instance_*` list columns |
| experimental group | a `-p` import path (`imported_path_short`) |

Instances have no row of their own because PixelPatrol has no granularity below a
dimension slice. Widgets unnest them in SQL instead:

```sql
SELECT cell_id, entity_name, unnest(instance_volume_um3) AS volume_um3
FROM pp_data
WHERE obs_level = 1 AND entity_kind = 'label'
```

The entity TIFFs of a cell are discovered as files as well; the loader declines them
by returning `None` from `load()`, which PixelPatrol skips per file.

## Install (development)

Until PixelPatrol is released with the version this depends on, install both from a
local checkout:

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
| `CELLSKETCH_NUM_THREADS` | `0` (all cores) | `--num-threads` |

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
