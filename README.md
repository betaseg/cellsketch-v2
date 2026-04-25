# AlphaCells Spatial Analysis

Tools for 3D spatial analysis of segmented cell components. Given a set of TIFF label/mask volumes (one per structure), the pipeline computes per-instance morphology metrics and inter-component distances, and writes everything to a `report.csv` that can be explored interactively.

## Overview of tools

| File | Purpose |
|------|---------|
| `analyze_cell.py` | Main analysis script — reads TIFFs, computes metrics, writes `report.csv` |
| `viewer.html` | Browser-based interactive explorer for `report.csv` |
| `csv_to_blender.py` | Imports meshes from `report.csv` into a Blender scene |
| `explore_report.ipynb` | Jupyter notebook with worked analysis examples |

---

## Input data format

Each cell is a folder containing 3D TIFF files. Each TIFF represents one segmented structure (e.g. mitochondria, nucleus, plasma membrane) and must be either:

- a **label volume** — integer array where each unique nonzero value is one instance (e.g. each mitochondrion gets its own ID)
- a **mask volume** — binary array (one connected region or multiple that get auto-split)

Files are matched by their basename. The script expects filenames to contain a `NAME` token followed by `_labels`, `_label`, or `_mask`:

```
cell_folder/
  experiment_cell1_mito_labels.tif     ← label volume for "mito"
  experiment_cell1_nucleus_mask.tif    ← mask for "nucleus"
  experiment_cell1_pm_mask.tif        ← plasma membrane (name must contain pm/plasma/membrane)
  experiment_cell1_raw.tif            ← source image (used for metadata, not analysed)
```

The plasma membrane (`pm`, `plasma`, or `membrane` in the name) is used as the reference boundary. All other entities are optionally clipped to it with `--auto-clip-to-pm`.

Masks with more than one connected component are automatically promoted to label volumes — each component is treated as an individual instance.

### Directory modes

| Structure | Detected as |
|-----------|-------------|
| `cell-dir/` contains TIFFs | Single cell |
| `cell-dir/cell_a/`, `cell_b/` … each contain TIFFs | Flat batch |
| `cell-dir/group_a/cell_a/`, `group_b/cell_b/` … | Grouped batch |

In batch modes a joint `report.csv` covering all cells is written to `--out-dir/`.

---

## Run

Requires [`uv`](https://github.com/astral-sh/uv) — dependencies are declared inline in the script.

```bash
# Single cell
uv run --script analyze_cell.py \
  --cell-dir data/high/high_c1 \
  --out-dir results/high/high_c1

# Flat batch — one subfolder per cell
uv run --script analyze_cell.py \
  --cell-dir data/high \
  --out-dir results/high

# Grouped batch — subfolders are groups, their subfolders are cells
uv run --script analyze_cell.py \
  --cell-dir data \
  --out-dir results

# Include 3D meshes (enables Blender export and interactive 3D viewer)
uv run --script analyze_cell.py \
  --cell-dir data \
  --out-dir results \
  --with-mesh --auto-clip-to-pm
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--cell-dir PATH` | *(required)* | Cell folder, batch parent, or grouped root |
| `--out-dir PATH` | *(required)* | Output root folder |
| `--voxel-size-um z,y,x` | *(from TIFF metadata)* | Manual voxel size in µm |
| `--auto-clip-to-pm` | off | Clip all non-membrane entities to the plasma membrane before analysis |
| `--with-mesh` | off | Generate a 3D mesh per instance; required for Blender export and 3D viewer |
| `--generated-masks-dirname NAME` | `masks_for_analysis` | Subfolder name for processed-mask cache |
| `--num-threads N` | `0` (all cores) | Threads for distance-transform computation |
| `--skip-plots` | off | Skip per-instance thumbnail generation |
| `--force-reprocess` | off | Re-run even if `report.csv` already exists |
| `--max-skeleton-voxels N` | `500000` | Skip skeleton metrics for instances larger than this voxel count |

---

## Outputs

```
out-dir/
  report.csv              ← all metrics (load into viewer.html or Blender)
  <cell>/
    report.csv            ← per-cell metrics
    masks_for_analysis/   ← processed TIFF cache (clipped, CC-promoted)
    .dt_cache/            ← distance-transform cache (.npy files)
```

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
| `thumbnail_b64` | Base64-encoded PNG depth-shaded projection |
| `mesh_b64` | Base64-encoded 3D mesh (only with `--with-mesh`) |

---

## viewer.html

Open in a modern browser and drop `report.csv` onto the page. No server required.

| Section | Description |
|---------|-------------|
| **Overview** | Cell count, entity presence heatmap, total volumes |
| **File Structure** | Sunburst chart and sortable file table |
| **Component Stats** | Per-entity metric box plots and distance plots |
| **Instance Viewer** | Thumbnail grid — sortable, filterable, optional 3D mesh previews |
| **Cell 3D View** | Interactive 3D scene for a single cell, coloured by entity or metric (requires `mesh_b64`) |

**Global controls:** Group / Cell / Entity filters apply across all sections. "Group plots by Cell | Group" switches how Component Stats are grouped.

---

## Blender export (`csv_to_blender.py`)

Imports all meshes from `report.csv` into a Blender scene. Each entity gets its own glass material and collection. Masks (file rows) and label instances are imported as separate merged objects per cell.

**Requirements:** Blender with `pandas` available in its Python environment, and a `report.csv` produced with `--with-mesh`.

### Usage

**From the Blender Script Editor:**

1. Open `csv_to_blender.py` in the Script Editor.
2. Set `CSV_PATH` at the top of the file to your `report.csv`.
3. Optionally set `OUT_BLEND` and/or `OUT_RENDER` to save or render automatically.
4. Press **Alt+P** to run.

**Headless / CLI:**

```bash
blender --background --python csv_to_blender.py -- /path/to/report.csv [out.blend] [out_render.png]
```

### Config options (edit at the top of the script)

| Variable | Default | Description |
|----------|---------|-------------|
| `CSV_PATH` | `"/path/to/report.csv"` | Path to the CSV |
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
