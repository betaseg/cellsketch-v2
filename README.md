# AlphaCells Spatial Analysis

`analyze_cell.py` is a standalone `uv --script` workflow for 3D TIFF label/mask spatial analysis.
Results are explored in `viewer.html` (open in a browser, load `report.csv`) or programmatically via `explore_report.ipynb`:

```bash
uv run --with jupyter --with pandas --with matplotlib jupyter notebook explore_report.ipynb
```

## Run

```bash
uv run --script analyze_cell.py \
  --cell-dir ../data/test \
  --out-dir results
```

## Input layout

Each cell folder must contain one source TIFF (non-derived image) and derived files sharing its basename:

- `..._NAME_label` or `..._NAME_labels` — multi-instance label volume
- `..._NAME_mask` — binary mask volume

The plasma membrane is the mask whose `NAME` contains `pm`, `plasma`, or `membrane`.

## Input modes

- **Single-cell mode**: `--cell-dir` points to one folder with TIFFs.
- **Batch mode**: `--cell-dir` points to a parent folder; each subfolder with TIFFs is one cell.

In batch mode outputs go to `--out-dir/<cell_name>/`, and a joint `report.csv` is written to `--out-dir/`.

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--cell-dir PATH` | *(required)* | Single cell folder or batch parent folder |
| `--out-dir PATH` | *(required)* | Output folder |
| `--voxel-size-um z,y,x` | *(from TIFF metadata)* | Manual voxel size in µm |
| `--auto-clip-to-pm` | off | Clip all non-membrane entities to the membrane mask before analysis |
| `--generated-masks-dirname NAME` | `masks_for_analysis` | Subfolder name for processed masks |
| `--num-threads N` | `0` (all cores) | Threads for distance-transform computation |
| `--skip-plots` | off | Skip per-instance thumbnail generation (faster) |
| `--force-reprocess` | off | Re-run even if `report.csv` already exists |
| `--max-skeleton-voxels N` | `500000` | Skip branch counting for label instances larger than this voxel count |

## Output files (per cell)

| Path | Description |
|------|-------------|
| `report.csv` | All metrics and embedded thumbnails; load into `viewer.html` |
| `masks_for_analysis/` | Processed (optionally clipped) copies of all input TIFFs |
| `.dt_cache/` | Distance-transform cache (reused on re-runs) |

In batch mode an additional joint `report.csv` is written to `--out-dir/` concatenating all cells.

## report.csv structure

Each row is either a **file** summary row or an **instance** row (one per labelled object).

### File row columns

| Column | Description |
|--------|-------------|
| `cell_id` | Cell folder name |
| `entity_name` | Component name (e.g. `mito`, `pm`) |
| `entity_kind` | `source`, `mask`, or `label` |
| `row_type` | `file` |
| `instance_count` | Number of label instances (label entities only) |
| `total_volume_um3` | Total occupied volume in µm³ |
| `file_size_bytes` | Size of the processed mask file |
| `file_mtime` | Modification time of the processed mask file |
| `file_name` | Filename |

### Instance row columns

| Column | Description |
|--------|-------------|
| `cell_id` | Cell folder name |
| `entity_name` | Label entity name |
| `entity_kind` | `label` |
| `row_type` | `instance` |
| `label_id` | Label integer ID |
| `volume_um3` | Instance volume in µm³ |
| `surface_area_um2` | Surface area in µm² |
| `sphericity` | Isoperimetric sphericity (1 = perfect sphere) |
| `aspect_ratio_major_minor` | Ratio of largest to smallest PCA axis |
| `branches` | 3D skeleton branch count |
| `distance_to_<target>_um` | Minimum distance to each mask/label entity in µm |
| `distance_to_closest_same_type_um` | Distance to nearest instance of the same entity |
| `thumbnail_b64` | Base64-encoded PNG depth-shaded projection |
