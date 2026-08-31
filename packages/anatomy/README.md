# pixel-patrol-anatomy

The spatial anatomy of segmented objects, as a [PixelPatrol](https://pixelpatrol.app/)
flavour. It measures objects itself, with its own loader and four processors, and writes a
report that PixelPatrol's viewer reads: seven widgets ship with it as a viewer extension.

An *object* is one segmented thing measured as a whole: a cell, an organoid, a nucleus, a
tissue block. It is given as a folder holding a source image, one mask that bounds the object,
and any number of label/mask volumes inside it. Everything is measured relative to that one
bounding mask: the region is cropped to it, and every polarity metric is measured from its
centroid.

Which mask that is, you say: `--object-mask NAME` is required and never inferred. It decides
the origin of every distance and polarity in the report, so guessing it from file names
would mean a regex quietly choosing what the numbers are relative to. `dry-run` lists the
masks in each folder, which is where the name comes from.

Reports are stamped with the `object anatomy` flavour, which the viewer shows as a chip in
the report-info strip at the foot of the page, beside the project name, version and date.

## Data model

An object is a *folder* of volumes that must be measured together. The loader claims such a
folder via `is_folder_supported`, so the directory is one dataset and the TIFFs inside are
not records of their own. Each object becomes one record and a fixed set of rows:

| Anatomy concept | PixelPatrol representation |
| --- | --- |
| one object folder | one record, entity volumes stacked along `C` (`CZYX`, or `CYX` in 2D) |
| one entity (organelle) | one row at `obs_level=1`, `dim_c` → `channel_names` |
| whole object | one row at `obs_level=0` |
| one instance (`row_type=instance`) | an element of the `instance_*` list columns, on the object row |
| one contact (`row_type=contact`) | an element of the `contact_*` list columns, on the object row |
| experimental group | a `-p` import path (`imported_path_short`) |

One row per entity is what makes `entity_name` a groupable column with scalar metrics the
built-in widgets can plot. A per-entity processor sees only its own channel, so anything
measured *between* entities (a distance, a contact) comes from an object-level processor
that sees them all, and is stored as parallel list columns on the object row. There are three such groups, each
one `unnest` away, and several `unnest()` calls in one `SELECT` stay row-aligned:

**Instances**, one row per instance:

```sql
SELECT object_id, unnest(instance_entity) AS entity_name, unnest(instance_label) AS label,
                unnest(instance_volume_um3) AS volume_um3,
                unnest(instance_sphericity) AS sphericity
FROM pp_data WHERE obs_level = 0
```

**Distances**, long format, one element per instance x target entity, because target
names come from the data and PixelPatrol drops columns a processor did not declare:

```sql
SELECT object_id, unnest(distance_entity) AS entity_name, unnest(distance_label) AS label,
                unnest(distance_target) AS target, unnest(distance_um) AS distance_um
FROM pp_data WHERE obs_level = 0
```

**Contacts**, the pairwise edge list, thresholded on `gap_um` interactively:

```sql
SELECT object_id, unnest(contact_entity_a) AS entity_a, unnest(contact_label_a) AS label_a,
                unnest(contact_entity_b) AS entity_b, unnest(contact_label_b) AS label_b,
                unnest(contact_gap_um)   AS gap_um
FROM pp_data WHERE obs_level = 0
```

Distances and gaps are measured voxel centre to voxel centre, so structures sharing a
face read *one voxel step* rather than zero, and with anisotropic voxels the smallest
non-overlapping reading depends on direction. Zero means genuine overlap.

### 2D and 3D

An object is a volume or a plane, and each is measured by the metrics that mean something
for it. The loader reads the source image's dimensionality and builds a `CZYX` record for a
volume or a `CYX` one for a plane, never a volume one voxel deep, and `spatial_dims` on
every row says which it was.

| Measured | 3D object | 2D object |
| --- | --- | --- |
| extent | `volume_um3`, `total_volume_um3`, `object_volume_um3` | `area_um2`, `total_area_um2`, `object_area_um2` |
| boundary | `surface_area_um2` (voxel faces) | `perimeter_um` (pixel edges) |
| roundness | `sphericity` | `circularity` = 4πA/P² |
| direction from the object centre | `polar_az_deg`, `polar_el_deg`, `polar_nz/ny/nx` | `polar_angle_deg`, `polar_ny/nx` |
| geometry beside the report | marching-cubes meshes | closed outline loops |
| skeletons | TEASAR curve skeletons (kimimaro) | thinned medial axis (`skimage`) |

Boundary metrics are staircase estimates: counting voxel faces measures the steps, not the
surface they approximate. Against ITK and the closed forms, a sphere of radius 1 µm sampled
at 0.1 × 0.05 × 0.05 µm reads

| | volume | surface | roundness |
| --- | --- | --- | --- |
| Anatomy | 4.155 µm³ | 18.67 µm² | sphericity 0.67 |
| SimpleITK | 4.155 µm³ | 12.48 µm² | roundness 1.00 |
| analytic | 4.189 µm³ | 12.57 µm² | 1 |

Extents agree exactly; the boundary reads ~1.5× high in 3D and ~1.3× in 2D, so sphericity and
circularity are for comparing shapes, not for reading 1 as round. `tests/test_reference_agreement.py`
pins both the agreement and the factor.

Everything else is measured identically and keeps its name: instance counts, the PCA aspect
ratio, `branches` / `length_um` / `tortuosity`, every `distance_*` column, and the contact
edge list. Distances and gaps are in whole sample steps in both cases.

The other dimensionality's columns are declared but never filled, so one report can hold
both kinds of object. A report with only one kind simply has no columns for the other,
since PixelPatrol drops a column no row filled. `--voxel-size-um` follows the same rule:
`z,y,x` for a volume, `y,x` for a plane, and a mismatch with the images is refused rather
than reinterpreted.

## Install (development)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e . pytest
```

Everything below runs in that activated environment; `.venv/bin/<command>` works too, for
a one-off without activating.

`pixel-patrol-base` comes from git, pinned to a commit in this package's own
`dependencies`: folder discovery for suffix-less directories, which the loader needs to
claim an object folder as one record, is in no release. That pin is the only copy of the
commit in the repo - `.github/workflows/deploy-pages.yml` reads it back out of the
pyproject rather than repeating it, so the deploy cannot drift from what the tests ran
against. Swap it for a version once that work has landed.

The install above runs the whole suite, which never opens the viewer. Seeing the widgets
does, and the viewer is a build product that is not committed, so it takes a clone at the
same commit and one npm build:

```bash
git clone https://github.com/ida-mdc/pixel-patrol.git
git -C pixel-patrol checkout "$(grep -oE '[0-9a-f]{40}' pyproject.toml)"
(cd pixel-patrol/viewer && npm ci && npm run build)
uv pip install -e pixel-patrol/packages/pixel-patrol-base
```

## Run

`dry-run` first, to see what the folders hold and which masks you can choose from; then
`process` with the mask that bounds the object:

```bash
pixel-patrol-anatomy dry-run experiment/
# ... masks   nucleus, pm
# Pick the mask that bounds each object and pass it as --object-mask: nucleus, pm

pixel-patrol-anatomy process experiment/ -o report.parquet --object-mask pm \
    -p control -p treated
pixel-patrol-anatomy view report.parquet --significance    # or: pixel-patrol view ...
```

Naming a mask a folder does not have is an error rather than a fallback, and the message
lists the masks that folder actually has.

On a real dataset (eight vEM alpha objects, 2.5–10.5 GB stacked, five entities each)
`process` prints:

```
8 object folder(s); 3 worker(s) (largest object needs ~17.9 GB each)
```

### Why the processing is ours

`process` runs the processors itself rather than through PixelPatrol's pipeline, because an
object cannot be split. PixelPatrol splits a record that exceeds a memory budget; the axis
it can split here is `C`, and a chunk holding two of five entities can answer almost nothing
because distances are *to* another entity and contacts are between them. A split object then
loses its instances, contacts and geometry, and because a refused chunk is only a warning,
the run still reports success.

So each object is loaded whole and measured in a pool sized by what an object actually costs
(measured peak on a 133-megavoxel five-entity object: 4.4 GB). One object that fails is
reported by name and the batch continues; the command exits non-zero if any did.

What still comes from PixelPatrol: the record and processor model, the parquet writer and
its provenance footer, and the viewer the widgets run in.

### Speed

Measured on one real object (133 megavoxels, five entities, 6340 instances), with
distances, histograms, contacts and polarity spread all on:

```
load          6s     instances    39s     contacts    16s     meshes   133s
```

Those meshing and instance figures predate the parallel meshing and the allocation work
below, and have not been re-measured on that object since.

Three things got it there, and the first is worth knowing about if you write your own
processor:

- **`scipy.ndimage.minimum` is unusable at this scale**: 54 s per call on a 197-megavoxel
  object, and 57 s when asked for only 50 of the labels, because its cost follows the volume
  and label count rather than the foreground. Each entity's foreground is indexed once
  instead (positions sorted by label id), after which a measurement is a gather plus
  `np.minimum.reduceat`: 0.01 s, same answers.
- **`--skeleton-entities mito`**: skeletonising dominates everything else (103 s of a
  2-minute object), and most of it is usually wasted: a granule's skeleton is one branch the
  length of its diameter. Naming just the filaments took that object from 55 s to 36 s.
- **Nothing is measured twice per object.** Skeletons, contacts, ITK's shape statistics and
  `regionprops` are computed once and shared, so `--with-mesh` does not repeat for its
  geometry what the instance table already measured.
- **Objects run in parallel**, as many at once as their measured peak allows, and **an
  object's instances are meshed in parallel** within that. Two levels, because the object
  pool is sized by memory rather than by cores: on a batch of large objects it is three
  workers on a machine with far more cores than that, and meshing is the longest part of a
  run. Measured on 22 cores with 8 mesh workers: 1.7x at 400 instances, 2.7x at 900.

Two things keep peak memory down, both worth knowing if you profile a run: entity volumes
are stacked in the narrowest integer type their label ids need (usually `uint16`, halving
a `float32`/`int32` segmentation), and the instance processor holds **one** whole-volume
distance transform at a time, reducing it over every entity's labels with one pass before
freeing it.

Skip the expensive parts with `--no-instances` / `--no-contacts`.

## Geometry for the 3D widgets and Blender

Geometry never enters the parquet, because meshes would multiply the size of the report every
stats query loads: one real object is 36 MB of them against a 1 MB report. It goes to
one `geometry.parquet` per object instead. Either during processing:

```bash
pixel-patrol-anatomy process experiment/ -o report.parquet --object-mask pm --with-mesh \
  --mesh-smooth-sigma 1 --mesh-step-size 1 --mesh-level 0.05
# → report.parquet  +  report_meshes/<object>/geometry.parquet
```

or afterwards, when you have a report already and want to re-mesh with other settings:

```bash
pixel-patrol-anatomy mesh experiment/ -o geometry/ --object-mask pm \
  --mesh-step-size 1 --no-skeletons
```

Each file holds one row per label instance (with a `mesh` blob and, unless
`--no-skeletons`, a `skeleton` blob), one per whole-structure mask, and the contact edge
list the 3D view groups by. The payload is the quantised vertex/index container
the standalone viewer used, stored raw with parquet's own zstd doing the compressing,
`geometry_to_blender.py` decodes it, and so does `decodePayload` in the viewer plugin.

Being parquet rather than one long CSV is what makes it *queryable*, which is the whole
mechanism behind the 3D widgets: DuckDB, native under `pixel-patrol view` and WASM in the
browser, filters by object, structure and metric and returns only the rows about to be
drawn. The twelve roundest granules cost twelve meshes, not an object's worth. The header
counts (`mesh_vertices`, `mesh_faces`) are columns of their own, so a widget can budget
its draw calls before transferring any geometry at all.

The object row carries `mesh_geometry_file`, the path this was written to; it is the only
thing the mesh processor adds to the table, and it is how a widget finds the geometry.

Meshing is the most expensive thing here (it skeletonises again for the gallery), which
is why it is opt-in rather than part of every run.

## Viewer widgets

Seven widgets ship with the package and load automatically in `pixel-patrol view`
(discovered through the `pixel_patrol.viewer_extensions` entry point):

| Widget | Shows |
| --- | --- |
| Objects & Structures | one row per object, one column per structure: instance counts, ✓ for whole-structure masks, and **missing** where an object was not segmented the same way as its neighbours. Two more columns: the object mask's own volume (or area), so objects are comparable by size, and a stacked bar splitting it between the structures inside, with what is in none of them in grey |
| Instance Morphology | one violin per group for every per-instance metric the structure has, with a structure selector. A metric with one value per group is drawn as bars with no significance test, and a metric measured for only some instances says so |
| Distances Between Structures | distance from each instance of one structure to each other structure |
| Reaching Two Structures At Once | one panel per pair of structures; for each instance the *larger* of its two distances, as a cumulative share. A curve that climbs early means most instances sit against both |
| Contacts & Groups | contacts at a live gap threshold, in three blocks: one point per object for how much contact there is (the only block with a significance test, since objects are the replicates), how close a connected group of the focused structure gets to every other structure compared with the same-size groups chance would give, and one row-normalised what-touches-what matrix per group. Then the counts, including how many clusters hold more than one structure |
| Object in 3D | one object as it was segmented: orbit, structure toggles, colour by structure / metric / contact group at a live gap, and an explode slider along each instance's own direction from the object centre. A 2D object is drawn as outlines. Skeletons belong to the Instance Gallery, where one instance is large enough to see one |
| Instance Gallery | the instances behind the distributions: the highest, lowest or a random sample of any metric, across every object the filter leaves or one group, as meshes (outlines for 2D objects). Asked for in rows rather than a count, so the grid fills whatever the card is wide enough for. An instance with a skeleton is drawn see-through so the skeleton shows; click one to look at it properly |

**Objects & Structures** is the one to read first: objects are segmented by hand, so one is
often missing a structure or has a nucleus split into three labels, and a violin drawn across
objects hides that. It also warns when a batch mixes 2D and 3D objects, whose extents are not
comparable. It reads the per-entity rows (`obs_level = 1`) from `pp_all`, which carry the
count and extent per structure; the object rows the other widgets query do not.

Every widget that picks a structure shares one choice, kept in the URL as `structure=`, so a
link opens on what you were reading about and switching in one widget switches the rest.

### Colours per structure

What each structure should look like is a property of the study, not of the tool, so it is a
settings file:

```json
{
  "mito": "#d62728",
  "er": "#2ca02c",
  "granules": "#ff7f0e"
}
```

```bash
# during a run
pixel-patrol-anatomy process experiment/ -o report.parquet --object-mask pm --colours palette.json

# or onto a report that already exists, in about a second
pixel-patrol-anatomy colours report.parquet palette.json
```

Both write the same column through the same call, so a palette can be tried and changed without
measuring anything again: the second form re-reads no pixels, and carries the footer the run was
written with over untouched. Run it again with a different file to change your mind.

Colours land on the entity rows as `entity_colour`, so they travel with the report: hand the
parquet to someone and it arrives already coloured, with no second file to pass around and
nothing to configure in the viewer. Every widget reads that one column, so a swatch in the
table, a segment of a composition bar and a mesh in the 3D view are the same colour.

Hex only, `#rrggbb` or `#rgb`, expanded and lower-cased on the way in. A structure the file
does not name keeps its place in the built-in palette (Tableau 10), so one file can name the
structures a project cares about and leave the rest alone, and a name the batch does not have
is simply unused. A file that is not valid JSON, or gives something that is not a colour, stops
the run and says which structure it was.

PixelPatrol's own widgets appear alongside these where the columns support them (summary,
file statistics, sunburst, image table, metadata). The histogram and mosaic widgets do not:
a label map has no intensity histogram, and no thumbnails are written.

The two 3D widgets read the geometry sidecar rather than the report, and appear only when
the report has a `mesh_geometry_file` column, so a run without `--with-mesh` does not show
them. Because they are in the viewer, they share its filter, grouping and palette: "the
outliers in that violin" and "these twelve meshes" are the same instances.

They need three.js, imported lazily on first open, from a `three.module.js` in the
extension directory if there is one, from the jsDelivr CDN otherwise. An object is drawn as
one merged geometry with per-vertex colours, so 8,000 instances cost one draw call.

That one merged mesh casts shadows onto itself, which is what tells you which granule is in
front of which; a low ambient term, a hemisphere light and fog across the object's own depth do
the rest. The gallery leaves shadows off, since a thumbnail holds one instance and there are
hundreds of them.

The key light stands 35° off to one side of the camera. Fixed in the world it would leave whole
sides of the object unlit, so turning the back to the front would show you a dark shape; riding
with the camera at an offset, the surface you are looking at is always lit while the light still
moves relative to the object, so the shadows travel across it as you orbit. Head-on would
flatten everything.

Two things keep the colours from washing out. The light intensities sum to about one on the
brightest face, because past that a lit surface clips towards white and takes its hue with it.
And vertex colours are converted from sRGB to linear before they go into the buffer: a vertex
colour is read as linear and the renderer converts linear to sRGB on the way out, so a hex fed
in raw came out a pastel of itself, `#1f77b4` rendering as `#62b6db`. three's `Color` does that
conversion for a hex; a plain `BufferAttribute` has nobody to do it.

Reach curves are drawn from quantiles: 51 vertices instead of 8,000 points, with the same
reading at any *n*.

Contacts & Groups keeps one rule: a p-value is only ever about a difference between groups.
Its first block is one point per object, which is the level a condition has replicates at, and
that block is the only one the engine tests. The group-reach block compares group sizes inside a
condition, where the groups are not independent of each other, so it is drawn as medians against
a size-matched baseline and gets no brackets. The matrix is composition, which is counted rather
than compared. Each block answers a different question, so no panel is another panel's
arithmetic.

The baseline in the group-reach block is the point of it. A group of ten reaches closer to
anything than a group of two whatever it is made of, because the closest of ten draws is closer
than the closest of two, so group size cannot be an axis without the axis doing the talking.
Instead each group is scored against the groups of its own size that the same structure in the
same object could have formed: the share of them it beats, which is 50% when nothing is going on.
Exact rather than simulated, C(n-r, k)/C(n, k) over the object's own distances, with ties split
evenly so the score stays flat under the null. One box per condition and target structure, the
concrete distance under each label, and one line at 50%.

Partner counts, contact shares and the distance trend are SQL over the object row's list
columns. Clusters are not: they need a union-find over the edge list at the slider's gap,
seeded with every instance so a lone instance counts as a cluster of one. Those numbers come
back to the engine as a `VALUES` table of one row per object, which is how a number computed in
the browser still gets the violins, palette and significance brackets.

The clustering runs in the browser, so it is covered by running it through node from the
Python suite (`tests/contact_groups_check.mjs`) - identity is object + entity + label, and a
test pins that a label id repeated in two objects stays two instances. `tests/test_widget_sql.py`
goes further: it runs every query the widgets build against a real report and draws every chart
they export, so a binder error or a missing helper is a failing test rather than a message in
the browser console.

Each builds its own source, a subquery that unnests the object row's list columns,
and hand it to the viewer's own distribution engine, so instance-level data goes
through the same violins, palette, grouping and **Mann-Whitney significance brackets**
as the built-in widgets. Turn the brackets on with the sidebar's *Show significance*
(or `pixel-patrol view … --significance`).

That reuse is the reason for this package: the engine takes an arbitrary source table
expression, so it does not care that the rows came from an unnest.

## Configuration

PixelPatrol instantiates plugins with no arguments and its CLI cannot pass plugin
options, so the analysis knobs are environment variables, each set by the CLI flag of the
same name:

| Variable | Default | CLI flag |
| --- | --- | --- |
| `PP_ANATOMY_OBJECT_MASK` | *required* | `--object-mask NAME` |
| `PP_ANATOMY_VOXEL_SIZE_UM` | inferred from TIFF metadata | `--voxel-size-um z,y,x` (3D) or `y,x` (2D) |
| `PP_ANATOMY_AUTO_CLIP` | `0` | `--auto-clip` |
| `PP_ANATOMY_AUTO_LABEL_MASKS` | `0` | `--auto-label-masks` |
| `PP_ANATOMY_ENTITY_COLOURS` | built-in palette | `--colours FILE` |
| `PP_ANATOMY_MAX_SKELETON_VOXELS` | `500000` | `--max-skeleton-voxels` |
| `PP_ANATOMY_SKELETON_ENTITIES` | all label entities | `--skeleton-entities mito,er` |
| `PP_ANATOMY_NO_SKELETONS` | `0` | `--no-skeletons` |
| `PP_ANATOMY_EDT_THREADS` | `0` (all cores) | none |
| `PP_ANATOMY_NUM_THREADS` | `1` (objects already run in parallel) | `--num-threads` |
| `PP_ANATOMY_CONTACT_MAX_UM` | `0.5` | `--contact-max-um` |
| `PP_ANATOMY_POLARITY_SPREAD` | `0` | `--polarity-spread` |
| `PP_ANATOMY_DISTANCE_HISTOGRAMS` | `0` | `--distance-histograms` |
| `PP_ANATOMY_MESH_DIR` | unset (no geometry) | `--with-mesh` / `--mesh-dir` |
| `PP_ANATOMY_MESH_SMOOTH_SIGMA` | `0.7` | `--mesh-smooth-sigma` |
| `PP_ANATOMY_MESH_STEP_SIZE` | `2` | `--mesh-step-size` |
| `PP_ANATOMY_MESH_TARGET_REDUCTION` | `0.8` | `--mesh-target-reduction` |
| `PP_ANATOMY_MESH_LEVEL` | `0` | `--mesh-level` |
| `PP_ANATOMY_MESH_WORKERS` | the cores the object pool is not using | `--mesh-workers` |

`pixel-patrol-anatomy process` sets these from its own flags, so you only need the
variables when driving `pixel-patrol` directly. `--with-contacts` has no equivalent:
contacts are a processor, so they are on by default and skipped with `--no-contacts`.

## Tests

```bash
pytest
```

`tests/synthetic.py` builds a small grouped batch of synthetic objects (membrane,
nucleus, mitochondria) with voxel size in the TIFF metadata; it is also runnable as a
script to produce a dataset to try the CLI on:

```bash
python tests/synthetic.py /tmp/objects
```
