# CellSketch V2 - Cell Component Spatial Analysis

Tools for 3D spatial analysis of segmented cell components. Given a set of TIFF label/mask volumes (one per structure), the pipeline computes per-instance morphology metrics and inter-component distances, and writes the results to CSV files that can be explored interactively.

**Viewers:** https://betaseg.github.io/cellsketch-v2/

## Overview of tools

| File | Purpose |
|------|---------|
| `analyze_cell.py` | Main analysis script — reads TIFFs, computes metrics, writes CSVs |
| `stats_viewer.html` | Browser-based stats explorer — loads the lightweight joint `report.csv` |
| `mesh_viewer.html` | Browser-based 3D viewer — loads a per-cell `report_meshes.csv` |
| `csv_to_blender.py` | Imports meshes from `report_meshes.csv` into a Blender scene |
| `explore_report.ipynb` | Jupyter notebook with worked analysis examples |

---

## Input data format

Each cell is a folder containing 3D TIFF files. Each TIFF represents one segmented structure (e.g. mitochondria, nucleus, plasma membrane) and must be either:

- a **label volume** — integer array where each unique nonzero value is one instance (e.g. each mitochondrion gets its own ID)
- a **mask volume** — binary array (one connected region or multiple that get auto-split)

Files are matched by their basename. Entity filenames must follow the pattern `<prefix>_<name>_labels.tif` or `<prefix>_<name>_mask.tif`, where the prefix matches the source image filename exactly:

```
cell_folder/
  experiment_cell1.tif                 ← source image — defines the common prefix
  experiment_cell1_mito_labels.tif     ← label volume for "mito"
  experiment_cell1_nucleus_mask.tif    ← mask for "nucleus"
  experiment_cell1_membrane_mask.tif   ← plasma membrane (name must contain pm/plasma/membrane)
```

The plasma membrane (`pm`, `plasma`, or `membrane` in the name) is used as the reference boundary. All other entities are optionally clipped to it with `--auto-clip-to-pm`.

Masks with more than one connected component are automatically promoted to label volumes — each component is treated as an individual instance.

### Directory modes

| Structure | Detected as |
|-----------|-------------|
| `cell-dir/` contains TIFFs | Single cell |
| `cell-dir/cell_a/`, `cell_b/` … each contain TIFFs | Flat batch |
| `cell-dir/group_a/cell_a/`, `group_b/cell_b/` … | Grouped batch |

In batch modes a joint `report.csv` (stats only) covering all cells is written to `--out-dir/`.

---

## Run

Requires [`uv`](https://github.com/astral-sh/uv) — dependencies are declared inline in the script and installed automatically on first run.

### Input folder layouts

The script detects which mode to use based on the structure of `--cell-dir`:

### Naming convention

Entity filenames must follow the pattern `<prefix>_<name>_labels.tif` or `<prefix>_<name>_mask.tif`. The source (raw) image filename must match that prefix exactly — it is the common root all other filenames are built from:

```
sample.tif                  ← source image — the prefix
sample_mito_labels.tif      ← label volume for "mito"
sample_nucleus_mask.tif     ← mask for "nucleus"
sample_membrane_mask.tif    ← plasma membrane mask
```

### Input folder layouts

The script detects which mode to use based on the structure of `--cell-dir`:

**Single cell** — the folder itself contains the TIFFs:

```
my_cell/
  sample.tif
  sample_mito_labels.tif
  sample_nucleus_mask.tif
  sample_membrane_mask.tif
```
```bash
uv run --script analyze_cell.py --cell-dir my_cell --out-dir results/my_cell
```

**Flat batch** — one subfolder per cell, all at the same level. Use this when you have multiple cells from one condition:

```
cells/
  cell_a/
    a.tif
    a_mito_labels.tif
    ...
  cell_b/
    b.tif
    b_mito_labels.tif
    ...
```
```bash
uv run --script analyze_cell.py --cell-dir cells --out-dir results
```
A joint `report.csv` is written to `results/` covering all cells, plus individual outputs under `results/cell_a/`, `results/cell_b/`, etc.

**Grouped batch** — two levels of subfolders: groups, then cells. Use this when you have multiple experimental conditions (e.g. control vs. treated) each containing several cells:

```
experiment/
  control/
    cell_a/
      ...
    cell_b/
      ...
  treated/
    cell_c/
      ...
    cell_d/
      ...
```
```bash
uv run --script analyze_cell.py --cell-dir experiment --out-dir results
```
The group folder name (`control`, `treated`) is recorded in the `group_id` column and used for grouping in the stats viewer.

### Common flags

```bash
# Add 3D meshes — needed for mesh_viewer.html and Blender export
uv run --script analyze_cell.py --cell-dir experiment --out-dir results \
  --with-mesh

# Clip everything to the plasma membrane before analysis
uv run --script analyze_cell.py --cell-dir experiment --out-dir results \
  --with-mesh --auto-clip-to-pm

# Re-run cells that were already processed
uv run --script analyze_cell.py --cell-dir experiment --out-dir results \
  --force-reprocess
```

### All parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--cell-dir PATH` | *(required)* | Cell folder, batch parent, or grouped root |
| `--out-dir PATH` | *(required)* | Output root folder |
| `--voxel-size-um z,y,x` | *(from TIFF metadata)* | Manual voxel size in µm |
| `--auto-clip-to-pm` | off | Clip all non-membrane entities to the plasma membrane before analysis |
| `--with-mesh` | off | Generate a 3D mesh per instance; required for Blender export and 3D viewer |
| `--num-threads N` | `0` (all cores) | Threads for distance-transform computation |
| `--force-reprocess` | off | Re-run even if `report.csv` already exists |
| `--max-skeleton-voxels N` | `500000` | Skip skeleton metrics for instances larger than this voxel count |

---

## Outputs

```
out-dir/
  report.csv                  ← joint stats for all cells (no mesh data) → stats_viewer.html
  <group>/<cell>/
    report.csv                ← per-cell stats (no mesh data)
    report_meshes.csv         ← per-cell stats + 3D meshes → mesh_viewer.html, Blender
    masks_for_analysis/       ← processed TIFF cache (clipped, CC-promoted)
    .dt_cache/                ← distance-transform cache (.npy files)
```

The two CSV files per cell have identical columns except that `report_meshes.csv` additionally contains `mesh_b64`. The joint `out-dir/report.csv` omits `mesh_b64` entirely so it stays small enough to load in a browser across many cells.

### report.csv structure

Each row is either a **file** summary row (one per entity per cell) or an **instance** row (one per labelled object).

**File rows** (`row_type = file`):

| Column | Description |
|--------|-------------|
| `cell_id` | Cell folder name |
| `group_id` | Parent group folder name (empty for flat/single runs) |
| `entity_name` | Component name (e.g. `mito`, `nucleus`) |
| `entity_kind` | `source`, `mask`, or `label` |
| `instance_count` | Number of label instances |
| `total_volume_um3` | Total occupied volume in µm³ |
| `surface_area_um2` | Surface area in µm² (mask entities only) |
| `sphericity` | Sphericity — 1 = perfect sphere (mask entities only) |
| `aspect_ratio_major_minor` | Ratio of largest to smallest PCA axis (mask entities only) |
| `file_size_bytes` | Size of the processed mask file |
| `file_mtime` | Modification time of the processed mask file |
| `file_name` | Filename |

**Instance rows** (`row_type = instance`):

| Column | Description |
|--------|-------------|
| `label_id` | Label integer ID |
| `volume_um3` | Instance volume in µm³ |
| `surface_area_um2` | Surface area in µm² |
| `sphericity` | Sphericity (1 = perfect sphere) |
| `aspect_ratio_major_minor` | Ratio of largest to smallest PCA axis |
| `branches` | 3D skeleton branch count |
| `length_um` | Total skeleton length in µm |
| `tortuosity` | Skeleton length / end-to-end distance (`NaN` for branching structures) |
| `distance_to_<target>_um` | Minimum distance to each other entity in µm |
| `distance_to_closest_same_type_um` | Distance to nearest instance of the same entity |
| `mesh_b64` | Base64-encoded 3D mesh — **`report_meshes.csv` only**, requires `--with-mesh` |

---

## Viewers

Two separate browser-based viewers — no server required, open directly as local files.

### `stats_viewer.html` — multi-cell stats

Drop the joint `report.csv` (or a per-cell `report.csv`) onto the page.

| Section | Description |
|---------|-------------|
| **Overview** | Cell count, entity presence heatmap, total volumes per cell |
| **File Structure** | Sunburst chart and sortable file table |
| **Component Stats** | Per-entity metric box plots and distance distributions |
| **Instance Viewer** | Sortable instance list with colour-coded entity dots |

**Global controls:** Group / Cell / Entity filters apply across all sections. "Group plots by Cell | Group" switches how Component Stats are grouped.

### `mesh_viewer.html` — single-cell 3D

Drop a per-cell `report_meshes.csv` (produced with `--with-mesh`) onto the page.

| Section | Description |
|---------|-------------|
| **Cell 3D View** | Interactive 3D scene coloured by entity or any metric |
| **Instance Viewer** | 3D-rendered thumbnail grid; click any thumbnail to open the full mesh modal |
| **Component Stats** | Per-entity metric and distance plots for the loaded cell |

The mesh viewer defaults to 3D mesh mode. Thumbnails in the instance grid are rendered in real time from the mesh data — there are no pre-computed thumbnail images.

---

## Blender export (`csv_to_blender.py`)

Imports all meshes from a `report_meshes.csv` into a Blender scene. Each entity gets its own glass material and collection. Masks (file rows) and label instances are imported as separate merged objects per cell.

**Requirements:** Blender with `pandas` available in its Python environment, and a `report_meshes.csv` produced with `--with-mesh`.

### Usage

**From the Blender Script Editor:**

1. Open `csv_to_blender.py` in the Script Editor.
2. Set `CSV_PATH` at the top of the file to your `report_meshes.csv`.
3. Optionally set `OUT_BLEND` and/or `OUT_RENDER` to save or render automatically.
4. Press **Alt+P** to run.

**Headless / CLI:**

```bash
blender --background --python csv_to_blender.py -- /path/to/report_meshes.csv [out.blend] [out_render.png]
```

### Config options (edit at the top of the script)

| Variable | Default | Description |
|----------|---------|-------------|
| `CSV_PATH` | `"/path/to/report_meshes.csv"` | Path to the mesh CSV |
| `OUT_BLEND` | `""` | Save `.blend` file here after import (empty = don't save) |
| `OUT_RENDER` | `""` | Render to this path after import (empty = don't render) |
| `CELLS` | `[]` | List of `cell_id` values to import; empty = all |
| `ENTITIES` | `[]` | List of entity names to include; empty = all |
| `EXCL_ENTITIES` | `[]` | Entity names to always exclude |
| `IMPORT_MASKS` | `True` | Import file-row (whole-mask) meshes |
| `IMPORT_LABELS` | `True` | Import instance-row meshes |

The scene is set up with Cycles, a dark background, and the camera auto-framed to fit all imported geometry.

---

## Notebook (`explore_report.ipynb`)

```bash
uv run --with jupyter --with pandas --with matplotlib --with numpy jupyter notebook explore_report.ipynb
```

Worked examples: file metadata table, volume pivot across cells, top-N instances rendered as 3D meshes, and distance histograms split by group.
