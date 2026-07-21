# CellSketch V2 - Cell Component Spatial Analysis

CellSketch V2 analyzes 3D TIFF segmentations (labels + masks) for one cell or many cells and produces CSV reports for statistics, interactive mesh viewing, and Blender import.

## Workflow

1. Organize your TIFF files in one of the supported input layouts.
2. Run one command with `analyze_cell.py`.
3. Get per-cell reports (and optional mesh reports) in `--out-dir`.
4. Open the outputs in `stats_viewer.html`, `mesh_viewer.html`, or Blender.

## Install and run

Use [`uv`](https://github.com/astral-sh/uv). Script dependencies are declared inside `analyze_cell.py` and install automatically on first run.

Run command (single command form):

```bash
uv run --script analyze_cell.py --cell-dir <INPUT_ROOT> --out-dir <OUTPUT_ROOT> [options]
```

Arguments:

- `--cell-dir <INPUT_ROOT>` (required): input root (single cell, flat batch, or grouped batch)
- `--out-dir <OUTPUT_ROOT>` (required): where outputs are written
- `--with-mesh`: include `mesh_b64` mesh data for `mesh_viewer.html` and Blender
- `--auto-clip-to-pm`: clip non-membrane entities to plasma membrane before analysis
- `--voxel-size-um z,y,x`: manual voxel size in um (otherwise inferred from source TIFF metadata)
- `--num-threads N`: distance-transform thread count (`0` = all cores)
- `--force-reprocess`: rerun cells even if `report.csv` already exists
- `--max-skeleton-voxels N`: skip branch metrics above this size
- `--mesh-smooth-sigma SIGMA`: mesh smoothing before marching cubes
- `--mesh-step-size N`: mesh resolution control (`1` = highest detail)
- `--mesh-target-reduction F`: mesh decimation fraction (default keeps ~20% of faces)
- `--mesh-level L`: marching-cubes isosurface level override

## What happens during a run

For each detected cell, the script:

1. Detects source TIFF + entity TIFFs (`*_label.tif` or `*_labels.tif`, plus `*_mask.tif`).
2. Resolves voxel size (from `--voxel-size-um` or source metadata).
3. Loads masks/labels, optionally clips to membrane (if `--auto-clip-to-pm`), and promotes multi-component masks to labels.
4. Builds/uses distance-transform cache in `.dt_cache`.
5. Computes file-level and instance-level morphology + distance metrics.
6. Writes per-cell `report.csv` and (if `--with-mesh`) `report_meshes.csv`.
7. In batch mode, also writes a joint `report.csv` at the root of `--out-dir`.

## Input data layout options

### Naming rules

Entity files must match:

- `<prefix>_<name>_label.tif` or `<prefix>_<name>_labels.tif`
- `<prefix>_<name>_mask.tif`

`<prefix>` must match the source image basename. Example:

```text
sample.tif
sample_mito_label.tif
sample_nucleus_mask.tif
sample_membrane_mask.tif
```

Membrane naming must include `pm`, `plasma`, or `membrane`.

### Layout 1: Single cell

```text
my_cell/
  sample.tif
  sample_mito_label.tif
  sample_nucleus_mask.tif
  sample_membrane_mask.tif
```

### Layout 2: Flat batch

```text
cells/
  cell_a/
    ...
  cell_b/
    ...
```

### Layout 3: Grouped batch

```text
experiment/
  control/
    cell_a/
      ...
  treated/
    cell_b/
      ...
```

For grouped runs, the group folder name is recorded as `group_id`.

## Output format

Output tree:

```text
<OUTPUT_ROOT>/
  report.parquet             # batch mode only — joint report across all cells
  <group>/<cell>/            # grouped mode
  <cell>/                    # flat mode
    report.csv
    report_meshes.csv        # only when --with-mesh
    masks_for_analysis/
    .dt_cache/
```

Row types (in `report.csv` / `report.parquet`):

- `row_type = file`: one row per entity file per cell (entity summary metrics)
- `row_type = instance`: one row per labeled object (instance metrics + distances)

Important columns:

- identity: `cell_id`, `group_id`, `entity_name`, `entity_kind`, `row_type`
- morphology: `volume_um3`, `surface_area_um2`, `sphericity`, `aspect_ratio_major_minor`
- skeleton metrics: `branches`, `length_um`, `tortuosity`
- distance metrics: `distance_to_<target>_um`, `distance_to_closest_same_type_um`
- mesh payload: `mesh_b64` (only in `report_meshes.csv`)

## Viewers and how to use them

Live hosted viewers: [https://betaseg.github.io/cellsketch-v2/](https://betaseg.github.io/cellsketch-v2/)

- `stats_viewer.html` (multi-cell stats):
  - load: drag/drop a joint `report.parquet` (batch mode) or a per-cell `report.csv`
  - use for: overview plots, component distributions, table + filtering by group/cell/entity

- `mesh_viewer.html` (single-cell 3D):
  - load: drag/drop a per-cell `report_meshes.csv`
  - requirement: analysis must be run with `--with-mesh`
  - use for: interactive 3D meshes and per-instance inspection

## Blender

Use `csv_to_blender.py` to import `report_meshes.csv` into Blender.

Requirements:

- Blender with `pandas` available in Blender's Python
- `report_meshes.csv` generated with `--with-mesh`

Run headless:

```bash
blender --background --python csv_to_blender.py -- /path/to/report_meshes.csv [out.blend] [out_render.png]
```

Or run inside Blender Script Editor:

1. Open `csv_to_blender.py`.
2. Set `CSV_PATH` (optionally `OUT_BLEND`, `OUT_RENDER`).
3. Run script (`Alt+P`).
