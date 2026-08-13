"""Cell-folder entity discovery, voxel size, and volume preparation.

Moved from ``analyze_cell.py`` (entity naming rules unchanged) so the loader can
turn one cell folder into one PixelPatrol record. ``analyze_cell.py`` still holds
its own copy while both paths coexist; it is the module to delete from once the
extension replaces the script.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tifffile
from scipy.ndimage import label as nd_label

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    kind: str  # "label" or "mask"
    path: Path


@dataclass
class Dataset:
    source: Path
    membrane_name: str
    entities: Dict[str, Entity]


@dataclass
class Discovery:
    """Everything entity discovery learned about a cell folder, including what it threw away."""
    source: Path | None
    entities: Dict[str, Entity]
    membrane_name: str | None
    unparsed: list[Path]
    rejected: list[Tuple[Path, str]]
    errors: list[str]


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def shared_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def parse_entity_file(stem: str) -> tuple[str, str, str] | None:
    s = normalize_name(stem)
    m = re.match(r"^(.*)_([a-z0-9_]+)_(labels?|mask)$", s)
    if not m:
        return None
    prefix = m.group(1)
    name = normalize_name(m.group(2))
    suffix = m.group(3)
    kind = "label" if suffix.startswith("label") else "mask"
    if not prefix or not name:
        return None
    return prefix, name, kind


def is_membrane_name(name: str) -> bool:
    n = normalize_name(name)
    return ("pm" == n) or ("plasma" in n) or ("membrane" in n)


def inspect_cell_dir(cell_dir: Path) -> Discovery:
    """Run entity discovery without raising, reporting what was accepted and rejected."""
    errors: list[str] = []
    tiffs = sorted(cell_dir.glob("*.tif*"))
    if not tiffs:
        return Discovery(None, {}, None, [], [], [f"No TIFF files found in {cell_dir}"])

    parsed = {}
    for p in tiffs:
        entry = parse_entity_file(p.stem)
        if entry:
            parsed[p] = entry

    if not parsed:
        errors.append("No NAME_label(s) or NAME_mask files found.")

    source_candidates = [p for p in tiffs if p not in parsed]
    if not source_candidates:
        errors.append("No source TIFF found (expected a non label/mask TIFF).")
        return Discovery(None, {}, None, [], [], errors)

    # Choose source TIFF with strongest shared prefix with derived entity prefixes.
    derived_prefixes = [pref for pref, _, _ in parsed.values()]

    def source_score(p: Path) -> tuple[int, int]:
        s = normalize_name(p.stem)
        score = sum(shared_prefix_len(s, pref) for pref in derived_prefixes)
        return score, int(p.stat().st_size)

    source = max(source_candidates, key=source_score)
    source_norm = normalize_name(source.stem)

    # Adaptive prefix matching: accept only entities whose shared prefix length with
    # the source name is maximal, with a word-boundary guard for clean matches so
    # "c1" does not match a "c10" prefix.
    shared_lens = {p: shared_prefix_len(source_norm, pref) for p, (pref, _, _) in parsed.items()}
    max_shared = max(shared_lens.values()) if shared_lens else 0

    entities: Dict[str, Entity] = {}
    rejected: list[Tuple[Path, str]] = []
    for p, (pref, name, kind) in parsed.items():
        sl = shared_lens[p]
        if sl < max_shared:
            rejected.append((p, f"prefix '{pref}' matches source '{source_norm}' less closely than the other entity files"))
            continue
        min_len = min(len(source_norm), len(pref))
        if sl >= min_len:
            longer = source_norm if len(source_norm) >= len(pref) else pref
            if sl < len(longer) and longer[sl] != "_":
                rejected.append((p, f"prefix '{pref}' is not a word-boundary match for source '{source_norm}'"))
                continue
        key = f"{kind}:{name}"
        if key in entities:
            rejected.append((p, f"duplicate {kind} '{name}' — keeping {entities[key].path.name}"))
            continue
        entities[key] = Entity(name=name, kind=kind, path=p)

    if not entities and not errors:
        errors.append("No NAME_label(s) or NAME_mask entities matching source basename were found.")

    membrane_candidates = [e.name for e in entities.values() if e.kind == "mask" and is_membrane_name(e.name)]
    membrane_name = sorted(membrane_candidates)[0] if membrane_candidates else None
    if membrane_name is None:
        errors.append("No membrane mask found (expected NAME with pm/plasma/membrane).")

    unparsed = [p for p in tiffs if p not in parsed and p != source]
    return Discovery(
        source=source,
        entities=entities,
        membrane_name=membrane_name,
        unparsed=unparsed,
        rejected=rejected,
        errors=errors,
    )


def discover_dataset(cell_dir: Path) -> Dataset:
    d = inspect_cell_dir(cell_dir)
    if d.errors:
        raise FileNotFoundError(d.errors[0])
    assert d.source is not None and d.membrane_name is not None
    return Dataset(source=d.source, membrane_name=d.membrane_name, entities=d.entities)


def infer_voxel_size_um_from_source(source_path: Path) -> Tuple[float, float, float]:
    with tifffile.TiffFile(source_path) as tf:
        page = tf.pages[0]
        ij = tf.imagej_metadata or {}

        z_um = float(ij["spacing"]) if "spacing" in ij else None

        x_um = None
        y_um = None
        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        unit_tag = page.tags.get("ResolutionUnit")
        unit_code = int(unit_tag.value) if unit_tag is not None else 1
        unit_to_um = {2: 25400.0, 3: 10000.0}.get(unit_code)
        if unit_to_um and xres_tag is not None and yres_tag is not None:
            x_num, x_den = xres_tag.value
            y_num, y_den = yres_tag.value
            if x_num:
                x_um = unit_to_um * float(x_den) / float(x_num)
            if y_num:
                y_um = unit_to_um * float(y_den) / float(y_num)
        if x_um is None or y_um is None:
            if xres_tag is not None and yres_tag is not None:
                x_num, x_den = xres_tag.value
                y_num, y_den = yres_tag.value
                if x_num:
                    x_um = float(x_den) / float(x_num)
                if y_num:
                    y_um = float(y_den) / float(y_num)

        if z_um is None or x_um is None or y_um is None:
            raise ValueError(f"Could not infer voxel size from source metadata: {source_path.name}")

    return (z_um, y_um, x_um)


def load_volume(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume in {path}, got shape {arr.shape}")
    return arr


def clip_to_membrane(arr: np.ndarray, membrane: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[~membrane] = 0
    return out


def crop_to_membrane_bbox(volumes: Dict[str, np.ndarray], membrane_key: str) -> Dict[str, np.ndarray]:
    membrane = volumes[membrane_key] > 0
    coords = np.argwhere(membrane)
    if coords.size == 0:
        return volumes
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    sl = tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))
    return {k: v[sl] for k, v in volumes.items()}


def promote_multicomponent_masks(
    volumes: Dict[str, np.ndarray],
    entities: Dict[str, Entity],
    membrane_key: str,
) -> None:
    """Promote masks with >1 connected component (in the cropped volume) to label entities.

    Mutates ``volumes`` and ``entities`` in place. Must be called after
    ``crop_to_membrane_bbox`` so the component count reflects what is actually
    inside the cell, not the full image.
    """
    cc_struct = np.ones((3, 3, 3), dtype=np.uint8)
    for key in list(volumes.keys()):
        entity = entities[key]
        if key == membrane_key:
            continue
        if entity.kind == "label" and int(volumes[key].max()) > 1:
            continue

        binary = volumes[key] > 0
        if not binary.any():
            continue

        labeled, n = nd_label(binary, structure=cc_struct)
        if n <= 1:
            continue

        volumes[key] = labeled.astype(np.int32)
        entities[key] = Entity(name=entity.name, kind="label", path=entity.path)
        logger.info("cellsketch: auto-label '%s' — %d components → label entity", entity.name, n)
