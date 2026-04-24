# AlphaCells Spatial Analysis

`analyze_cell.py` is a standalone `uv --script` workflow for 3D TIFF label/mask analysis.

No Nextflow is required.

## Run

From `code`:

```bash
uv run --script analyze_cell.py \
  --cell-dir ../data/test \
  --out-dir results \
  --auto-clip-to-pm
```

## Input layout

Each cell folder should contain:

- one source TIFF (non-derived image)
- derived files with matching basename:
  - `..._NAME_label` or `..._NAME_labels`
  - `..._NAME_mask`

The membrane is the mask whose `NAME` contains one of:

- `pm`
- `plasma`
- `membrane`

## Input modes

- **Single-cell mode**: `--cell-dir` points to one folder with TIFFs.
- **Batch mode**: `--cell-dir` points to a parent folder; each subfolder with TIFFs is treated as one cell.

In batch mode, outputs go to `--out-dir/<cell_name>/`.

## Parameters

- `--cell-dir PATH` (required): single cell folder or batch parent folder.
- `--out-dir PATH` (required): output folder.
- `--voxel-size-um z,y,x` (optional): manual voxel size in um; if omitted, read from source TIFF metadata.
- `--auto-clip-to-pm` (optional): clip all non-membrane entities to the membrane mask.
- `--generated-masks-dirname NAME` (optional, default `masks_for_analysis`): folder name for clipped masks.
- `--num-threads N` (optional, default `0`): distance-transform threads (`0` = auto/all cores).
- `--skip-plots` (optional): disable QC plot generation.

## Output files (per cell)

- `overall_cell.csv`
- `individual_<name>.csv` for each label entity
- `<generated-masks-dirname>/` (default `masks_for_analysis/`)
- `plots/` (unless `--skip-plots`)
- `.dt_cache/` distance-transform cache files

### `overall_cell.csv` columns

- `cell_id`
- `cell_volume_um3` (from membrane mask volume)
- one column per mask: `mask_<name>_volume_um3`
- one count column per label: `label_<name>_count`
- one total volume column per label: `label_<name>_total_volume_um3`

### `individual_<name>.csv` columns

For each labeled object in that entity:

- `label` (label ID)
- `volume_um3`
- `surface_area_um2`
- `sphericity`
- `aspect_ratio_major_minor`
- `branches` (3D skeleton branch-point proxy)
- `distance_to_<target>_um` for all targets:
  - all masks (`<target>` is mask name)
  - all other label entities (`<target>` looks like `label_<other_name>`)
- `distance_to_closest_same_type_um`

## Input equals output

Supported:

- single-cell: `--cell-dir X --out-dir X`
- batch: `--cell-dir PARENT --out-dir PARENT`

Existing files with the same names are overwritten.
