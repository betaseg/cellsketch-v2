"""Build a small grouped batch of synthetic cells for tests and CLI trials.

Each cell folder gets a source image plus three entities — a plasma-membrane mask, a
nucleus mask, and an instance-segmented mitochondria label volume — with the voxel
size written into the TIFF metadata so voxel-size inference is exercised too.

Run as a script to produce a dataset to point the CLI at:

    python tests/synthetic.py /tmp/cells
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import tifffile

SHAPE = (20, 40, 40)                 # Z, Y, X
VOXEL_SIZE_UM = (0.1, 0.02, 0.02)    # Z, Y, X


def _ellipsoid(shape: Tuple[int, int, int], center, radii) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0


def _write(path: Path, arr: np.ndarray) -> None:
    tifffile.imwrite(
        path,
        arr,
        imagej=True,
        resolution=(1.0 / VOXEL_SIZE_UM[2], 1.0 / VOXEL_SIZE_UM[1]),
        metadata={"spacing": VOXEL_SIZE_UM[0], "unit": "um", "axes": "ZYX"},
    )


def make_cell(
    cell_dir: Path,
    prefix: str = "sample",
    n_mito: int = 4,
    mito_radii: Sequence[float] = (2.0, 3.0, 3.0),
) -> Dict[str, Path]:
    """Write one synthetic cell folder; return the paths written by role."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    z, y, x = SHAPE
    center = (z / 2, y / 2, x / 2)

    membrane = _ellipsoid(SHAPE, center, (z / 2 - 1, y / 2 - 2, x / 2 - 2))
    nucleus = _ellipsoid(SHAPE, (center[0], center[1] - 6, center[2] - 6), (4, 6, 6))

    mito = np.zeros(SHAPE, dtype=np.uint16)
    for i in range(n_mito):
        angle = 2 * np.pi * i / n_mito
        cy = center[1] + 9 * np.sin(angle)
        cx = center[2] + 9 * np.cos(angle)
        blob = _ellipsoid(SHAPE, (center[0], cy, cx), mito_radii) & membrane
        mito[blob] = i + 1

    rng = np.random.default_rng(abs(hash(cell_dir.name)) % (2**32))
    source = (rng.normal(60, 8, SHAPE) + 120 * membrane + 60 * (mito > 0)).clip(0, 255).astype(np.uint8)

    paths = {
        "source": cell_dir / f"{prefix}.tif",
        "membrane": cell_dir / f"{prefix}_pm_mask.tif",
        "nucleus": cell_dir / f"{prefix}_nucleus_mask.tif",
        "mito": cell_dir / f"{prefix}_mito_label.tif",
    }
    _write(paths["source"], source)
    _write(paths["membrane"], membrane.astype(np.uint8))
    _write(paths["nucleus"], nucleus.astype(np.uint8))
    _write(paths["mito"], mito)
    return paths


def make_dataset(root: Path) -> Path:
    """Two groups of two cells each; treated cells have fewer, larger mitochondria."""
    make_cell(root / "control" / "cell_a", prefix="sample_a", n_mito=4, mito_radii=(2.0, 3.0, 3.0))
    make_cell(root / "control" / "cell_b", prefix="sample_b", n_mito=4, mito_radii=(2.0, 2.5, 2.5))
    make_cell(root / "treated" / "cell_c", prefix="sample_c", n_mito=3, mito_radii=(3.0, 4.5, 4.5))
    make_cell(root / "treated" / "cell_d", prefix="sample_d", n_mito=3, mito_radii=(3.0, 5.0, 5.0))
    return root


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "cells").resolve()
    make_dataset(target)
    print(f"wrote synthetic dataset to {target}")
