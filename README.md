# Anatomy: spatial analysis of segmented objects

Anatomy analyses TIFF segmentations (labels + masks), 2D or 3D, for one object or many, and
produces a single [PixelPatrol](https://pixelpatrol.app/) report you explore in an
interactive viewer: distributions, distances, contacts, and the objects themselves in 3D.

An **object** is one segmented thing measured as a whole, given as a folder: a source
image, one mask that bounds the object, and the label/mask volumes inside it. In this
project an object is a cell and its bounding mask is the plasma membrane, so `--object-mask pm`.
Nothing about the tool assumes that: the mask is always named explicitly, never guessed.

Objects can be volumes or planes. A 2D object is measured as a plane, so area, perimeter,
circularity and one polarity angle; a 3D one as a volume, so volume, surface area,
sphericity, azimuth and elevation. Nothing is measured with the wrong formula and no plane
is padded into a volume one voxel deep; see the package README's
[2D and 3D](packages/anatomy/README.md#2d-and-3d) for the full column split.

It measures with its own loader and four processors, and ships seven viewer widgets as a
PixelPatrol viewer extension, all in [`packages/anatomy`](packages/anatomy). That package's
[README](packages/anatomy/README.md) is the reference for the data model, the widgets
and every configuration knob; this page is the short path through a run.

## Workflow

1. Organise your TIFFs in one of the input layouts below.
2. `pixel-patrol-anatomy dry-run` to check what will be analysed, and what was ignored.
3. `pixel-patrol-anatomy process` to write `report.parquet` (and, with `--with-mesh`, the geometry).
4. `pixel-patrol-anatomy view` to explore it, or hand the parquet to anyone with the hosted viewer.
   `pixel-patrol-anatomy colours report.parquet palette.json` sets a colour per structure at any point.
5. Optionally import an object's geometry into Blender.

## Install and run

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e packages/anatomy \
  -e /path/to/pixel-patrol/packages/pixel-patrol-base
```

```bash
pixel-patrol-anatomy dry-run experiment/
pixel-patrol-anatomy process experiment/ -o report.parquet --object-mask pm \
    -p control -p treated --with-mesh
pixel-patrol-anatomy view report.parquet --significance
```

`-p` names the subdirectories to import as groups; that grouping becomes the default
comparison in every widget. `--with-mesh` is opt-in because meshing (and skeletonising
again for the overlay) is the most expensive part of a run.

Useful flags, see `pixel-patrol-anatomy process --help` for the rest:

- `--voxel-size-um z,y,x`: manual voxel size, otherwise inferred from the source TIFF
- `--object-mask NAME`: **required**, the mask that bounds each object, e.g. `pm`.
  Everything is measured relative to it, so it is never inferred; `dry-run` lists the
  masks each folder has
- `--auto-clip`: clip the entities to the object mask first
- `--colours palette.json`: a colour per structure, e.g. `{"mito": "#d62728"}`. It lands in
  the report, so every widget draws that structure the same and a shared report arrives
  coloured; unnamed structures keep the built-in palette. `pixel-patrol-anatomy colours
  report.parquet palette.json` does it to a report you already have, in about a second
- `--skeleton-entities mito,er`: only skeletonise the structures where branches, length
  and tortuosity mean something; the single biggest saving on a long run
- `--no-instances` / `--no-contacts`: skip the expensive per-instance work
- `--contact-max-um T`: largest surface-to-surface gap recorded as a contact
- `--mesh-smooth-sigma` / `--mesh-step-size` / `--mesh-target-reduction` / `--mesh-level`

## Checking your input first (`dry-run`)

Before a long batch, check that every object has the files you expect:

```bash
pixel-patrol-anatomy dry-run experiment/
```

It reads TIFF headers only, so no analysis, no output, seconds even for a large batch. It
prints, per object: the source image with its size, the label and mask entities found
(`*` marks the object mask), and anything that looks wrong. Then a cross-object summary
of which entities are **missing in which objects**, and a suggested `--max-workers` for the
run. Exit code is `1` if any object cannot be analysed.

```text
control/cell_b
  source  sample_b.tif   [12.4 MB stacked]
  labels  mito
  masks   nucleus, pm*
  warn    ignored readme_overlay.tif: not <prefix>_<name>_label|labels|mask

===== 3 object folder(s) =====  (* = object mask)
  label:mito               2/3   ← missing in some objects
  mask:pm                  3/3
```

An uneven batch is worth catching here, and again in the viewer: the **Objects &
Structures** widget shows the same thing for the report you actually produced, and warns
when a structure is missing from some of them.

## Input data layout

Entity files must match `<prefix>_<name>_label.tif` (or `_labels.tif`) and
`<prefix>_<name>_mask.tif`, where `<prefix>` is the source image basename:

```text
my_cell/
  sample.tif
  sample_mito_label.tif
  sample_nucleus_mask.tif
  sample_membrane_mask.tif
```

One of the masks bounds the object, and `--object-mask NAME` says which. Always, with no
name-based guessing, because every distance and polarity in the report is measured against
it. Naming a mask a folder does not have is an error rather than a silent fallback, and the
message lists what that folder does have. An object folder can sit on its own, in a flat batch
(`cells/cell_a/`, `cells/cell_b/`), or in a grouped batch (`experiment/control/cell_a/`,
`experiment/treated/cell_b/`), where the group folder is what `-p` imports.

## Output

```text
report.parquet                            # one row per object, one per structure
report_meshes/<object>/geometry.parquet   # only with --with-mesh
```

Everything measured lands in `report.parquet`: per-structure rows (`obs_level = 1`) with
volume, surface area, sphericity and instance counts, and per-object rows (`obs_level = 0`)
carrying the per-instance, distance and contact tables as list columns the widgets unnest.
See the [package README](packages/anatomy/README.md#data-model) for the full model.

Geometry never enters the report, because meshes would multiply the size of a table every
query loads. It goes beside it, one `geometry.parquet` per object holding a mesh (or, for a
2D object, an outline) and a skeleton per instance, plus the contact edge list. The object
row records where, in `mesh_geometry_file`, which is how the 3D widgets find it.

## Blender

Use `geometry_to_blender.py` to import an object's `geometry.parquet` into Blender.

Requirements:

- Blender with `pandas` and `pyarrow` available in Blender's Python
  (`<blender>/python/bin/python3 -m pip install pandas pyarrow`)
- `geometry.parquet` generated by `pixel-patrol-anatomy process --with-mesh` or
  `pixel-patrol-anatomy mesh`, from **3D** objects. A 2D object carries outlines rather
  than meshes, which the viewer's gallery draws and Blender has no use for

Run headless:

```bash
blender --background --python geometry_to_blender.py -- /path/to/geometry.parquet [out.blend] [out_render.png]
```

Or run inside Blender Script Editor:

1. Open `geometry_to_blender.py`.
2. Set `GEOMETRY_PATH` (optionally `OUT_BLEND`, `OUT_RENDER`).
3. Run script (`Alt+P`).
