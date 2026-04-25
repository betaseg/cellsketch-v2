"""
csv_to_blender.py — Import an AlphaCells report.csv into a Blender scene.

Script editor: set CSV_PATH below, then run with Alt+P.

CLI (headless):
    blender --background --python csv_to_blender.py -- /path/to/report.csv [out.blend]

Mesh encoding (from analyze_cell.py):
  gzip( [uint32 nV][uint32 nF]
        [float32×3 min_xyz][float32×3 scale_xyz]
        [uint16 × nV×3  quantised XYZ vertices (µm)]
        [uint32 × nF×3  face indices] )
Vertices are in µm, XYZ order (Three.js convention).
"""

import base64
import gzip
import struct
import sys

import bpy
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — edit these when running from the script editor
# ---------------------------------------------------------------------------

CSV_PATH      = "/path/to/report.csv"
OUT_BLEND     = ""   # leave empty to skip saving
OUT_RENDER    = ""   # leave empty to skip rendering
CELLS         = []   # e.g. ["high_c1", "high_c2"] — empty = all
ENTITIES      = []   # e.g. ["granules", "mitochondria"] — empty = all
EXCL_ENTITIES = []   # e.g. ["source"] — always exclude these
IMPORT_MASKS  = True   # import file-row (whole-mask) meshes
IMPORT_LABELS = True   # import instance-row meshes

# ---------------------------------------------------------------------------
# Colour palette (cycles by entity index — no hardcoded names)
# ---------------------------------------------------------------------------

_PALETTE = [
    (0.15, 0.45, 0.90),
    (0.90, 0.25, 0.10),
    (0.20, 0.75, 0.20),
    (0.95, 0.80, 0.10),
    (0.85, 0.15, 0.75),
    (0.20, 0.80, 0.80),
    (0.40, 0.70, 1.00),
    (0.70, 0.30, 0.90),
    (1.00, 0.50, 0.05),
    (0.60, 0.90, 0.60),
    (0.90, 0.40, 0.10),
    (0.10, 0.60, 0.90),
]

# ---------------------------------------------------------------------------
# Mesh decoding
# ---------------------------------------------------------------------------

def decode_mesh_b64(mesh_b64: str):
    """Return (verts float32 (N,3), faces uint32 (M,3)) or (None, None)."""
    if not mesh_b64 or not isinstance(mesh_b64, str) or not mesh_b64.strip():
        return None, None
    try:
        raw = gzip.decompress(base64.b64decode(mesh_b64))
    except Exception:
        return None, None

    nV, nF = struct.unpack_from("<II", raw, 0)
    if nV == 0 or nF == 0:
        return None, None

    min_xyz   = np.frombuffer(raw, dtype=np.float32, count=3, offset=8)
    scale_xyz = np.frombuffer(raw, dtype=np.float32, count=3, offset=20)
    verts_q   = np.frombuffer(raw, dtype=np.uint16,  count=nV * 3, offset=32).reshape(nV, 3)
    faces     = np.frombuffer(raw, dtype=np.uint32,  count=nF * 3, offset=32 + nV * 6).reshape(nF, 3)

    verts = min_xyz + verts_q.astype(np.float32) / 65535.0 * scale_xyz
    return verts, faces

# ---------------------------------------------------------------------------
# Blender helpers
# ---------------------------------------------------------------------------

def get_or_create_collection(name: str, parent=None):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    target = parent if parent is not None else bpy.context.scene.collection
    target.children.link(col)
    return col


def make_material(name: str, rgb: tuple, roughness: float = 0.15):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out   = nodes.new("ShaderNodeOutputMaterial")
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value     = (*rgb, 1.0)
    glass.inputs["Roughness"].default_value = roughness
    glass.inputs["IOR"].default_value       = 1.45
    links.new(glass.outputs["BSDF"], out.inputs["Surface"])
    return mat


def add_mesh_object(name: str, all_verts: np.ndarray, all_faces: np.ndarray, collection, material):
    """Create one Blender mesh object from already-combined vertex/face arrays."""
    mesh = bpy.data.meshes.new(name)
    # XYZ µm → Blender: swap Y↔Z so the XY imaging plane is horizontal
    v_list = [(float(v[0]), float(v[2]), float(v[1])) for v in all_verts]
    f_list = [tuple(int(i) for i in f) for f in all_faces]
    mesh.from_pydata(v_list, [], f_list)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.active_material = material
    for poly in mesh.polygons:
        poly.use_smooth = True
    collection.objects.link(obj)
    return obj


def merge_meshes(rows_iter):
    """Decode and concatenate meshes from an iterable of CSV rows.

    Returns (verts, faces) as combined numpy arrays, or (None, None) if
    nothing decoded successfully.
    """
    all_verts = []
    all_faces = []
    vert_offset = 0
    for _, row in rows_iter:
        verts, faces = decode_mesh_b64(str(row["mesh_b64"]))
        if verts is None:
            continue
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)
        vert_offset += len(verts)
    if not all_verts:
        return None, None
    return np.concatenate(all_verts), np.concatenate(all_faces)

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def setup_scene():
    if "Cube" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Cube"])

    world = bpy.data.worlds.get("World")
    if world:
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value[:3] = (0.01, 0.01, 0.01)

    light = bpy.data.objects.get("Light")
    if light:
        light.data.type   = "SUN"
        light.data.energy = 5

    cam = bpy.context.scene.camera
    if cam:
        cam.data.clip_end = 100_000
        cam.data.lens     = 45

    bpy.context.scene.render.engine              = "CYCLES"
    bpy.context.scene.cycles.samples             = 256
    bpy.context.scene.cycles.preview_samples     = 64
    bpy.context.scene.render.resolution_percentage = 100


def frame_camera():
    """Move the active camera to frame all mesh objects."""
    cam = bpy.context.scene.camera
    if cam is None:
        return
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        return

    # Compute bounding box centre and radius across all meshes
    all_corners = []
    for obj in meshes:
        for corner in obj.bound_box:
            all_corners.append(obj.matrix_world @ bpy.context.scene.cursor.location.__class__(corner))
    if not all_corners:
        return

    coords = np.array([(c.x, c.y, c.z) for c in all_corners])
    centre = coords.mean(axis=0)
    radius = np.linalg.norm(coords - centre, axis=1).max()

    from mathutils import Vector
    cam.location = Vector(centre) + Vector((0, -radius * 2.5, radius * 0.5))
    cam.rotation_euler = (1.1, 0.0, 0.0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_config():
    """Return (csv_path, out_blend, out_render) from CLI args or CONFIG vars."""
    args = sys.argv
    if "--" in args:
        args = args[args.index("--") + 1:]
        csv_path  = args[0] if len(args) > 0 else CSV_PATH
        out_blend = args[1] if len(args) > 1 else OUT_BLEND
        out_render = args[2] if len(args) > 2 else OUT_RENDER
    else:
        csv_path   = CSV_PATH
        out_blend  = OUT_BLEND
        out_render = OUT_RENDER
    return csv_path, out_blend or None, out_render or None


def main():
    import pandas as pd

    csv_path, out_blend, out_render = resolve_config()

    print(f"[csv_to_blender] Reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    if CELLS:
        df = df[df["cell_id"].astype(str).isin(set(CELLS))]
    if ENTITIES:
        df = df[df["entity_name"].isin(set(ENTITIES))]
    if EXCL_ENTITIES:
        df = df[~df["entity_name"].isin(set(EXCL_ENTITIES))]

    if "mesh_b64" not in df.columns:
        raise RuntimeError("CSV has no 'mesh_b64' column — re-run analyze_cell.py with --with-mesh")

    df = df[df["mesh_b64"].notna() & (df["mesh_b64"].astype(str).str.strip() != "")]
    print(f"[csv_to_blender] {len(df)} rows with meshes after filtering")
    if df.empty:
        print("[csv_to_blender] Nothing to import.")
        return

    setup_scene()

    all_entities = sorted(df["entity_name"].unique())
    color_map = {e: _PALETTE[i % len(_PALETTE)] for i, e in enumerate(all_entities)}

    material_cache = {}
    def get_mat(entity_name, is_mask):
        key = (entity_name, is_mask)
        if key not in material_cache:
            rgb = color_map[entity_name]
            mat_name = f"{entity_name}_{'mask' if is_mask else 'label'}"
            material_cache[key] = make_material(mat_name, rgb, roughness=0.3 if is_mask else 0.15)
        return material_cache[key]

    root_col = get_or_create_collection("AlphaCells")
    imported = skipped = 0

    for cell_id, cell_df in df.groupby("cell_id"):
        cell_col = get_or_create_collection(str(cell_id), parent=root_col)

        for entity_name, ent_df in cell_df.groupby("entity_name"):
            if IMPORT_MASKS:
                mask_rows = ent_df[ent_df["row_type"] == "file"]
                verts, faces = merge_meshes(mask_rows.iterrows())
                if verts is not None:
                    add_mesh_object(f"{cell_id}_{entity_name}_mask", verts, faces,
                                    cell_col, get_mat(entity_name, is_mask=True))
                    imported += 1
                elif not mask_rows.empty:
                    skipped += 1

            if IMPORT_LABELS:
                inst_rows = ent_df[ent_df["row_type"] == "instance"]
                verts, faces = merge_meshes(inst_rows.iterrows())
                if verts is not None:
                    add_mesh_object(f"{cell_id}_{entity_name}_labels", verts, faces,
                                    cell_col, get_mat(entity_name, is_mask=False))
                    imported += 1
                elif not inst_rows.empty:
                    skipped += 1

    print(f"[csv_to_blender] Imported {imported} objects, skipped {skipped} empty")

    frame_camera()

    if out_blend:
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        print(f"[csv_to_blender] Saved → {out_blend}")
    if out_render:
        bpy.context.scene.render.filepath = out_render
        bpy.ops.render.render(write_still=True)
        print(f"[csv_to_blender] Rendered → {out_render}")


main()
