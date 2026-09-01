"""Build a small grouped batch of synthetic objects for tests and CLI trials.

Each object folder gets a source image plus three entities — a plasma-membrane mask, a
nucleus mask, and an instance-segmented mitochondria label volume — with the voxel
size written into the TIFF metadata so voxel-size inference is exercised too.

Run as a script to produce a dataset to point the CLI at:

    python tests/synthetic.py /tmp/objects
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import tifffile

SHAPE = (20, 40, 40)                 # Z, Y, X
VOXEL_SIZE_UM = (0.1, 0.02, 0.02)    # Z, Y, X

SHAPE_2D = (60, 60)                  # Y, X
PIXEL_SIZE_UM = (0.02, 0.02)         # Y, X


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


def make_object(
    object_dir: Path,
    prefix: str = "sample",
    n_mito: int = 4,
    mito_radii: Sequence[float] = (2.0, 3.0, 3.0),
) -> Dict[str, Path]:
    """Write one synthetic object folder; return the paths written by role."""
    object_dir.mkdir(parents=True, exist_ok=True)
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

    rng = np.random.default_rng(abs(hash(object_dir.name)) % (2**32))
    source = (rng.normal(60, 8, SHAPE) + 120 * membrane + 60 * (mito > 0)).clip(0, 255).astype(np.uint8)

    paths = {
        "source": object_dir / f"{prefix}.tif",
        "membrane": object_dir / f"{prefix}_pm_mask.tif",
        "nucleus": object_dir / f"{prefix}_nucleus_mask.tif",
        "mito": object_dir / f"{prefix}_mito_label.tif",
    }
    _write(paths["source"], source)
    _write(paths["membrane"], membrane.astype(np.uint8))
    _write(paths["nucleus"], nucleus.astype(np.uint8))
    _write(paths["mito"], mito)
    return paths


def _disc(shape: Tuple[int, int], center, radii) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (((yy - center[0]) / radii[0]) ** 2 + ((xx - center[1]) / radii[1]) ** 2) <= 1.0


def _write_2d(path: Path, arr: np.ndarray) -> None:
    """A plain 2D TIFF: resolution tags, and no ImageJ spacing, because there is no Z."""
    tifffile.imwrite(
        path,
        arr,
        imagej=True,
        resolution=(1.0 / PIXEL_SIZE_UM[1], 1.0 / PIXEL_SIZE_UM[0]),
        metadata={"unit": "um", "axes": "YX"},
    )


def make_object_2d(
    object_dir: Path,
    prefix: str = "sample",
    n_mito: int = 4,
    mito_radii: Sequence[float] = (3.0, 3.0),
) -> Dict[str, Path]:
    """Write one synthetic 2D object folder; return the paths written by role.

    The same three entities as the 3D case - a boundary mask, a nucleus mask and an
    instance-segmented label image - so the two dimensionalities exercise the same code
    paths with the same names.
    """
    object_dir.mkdir(parents=True, exist_ok=True)
    y, x = SHAPE_2D
    center = (y / 2, x / 2)

    boundary = _disc(SHAPE_2D, center, (y / 2 - 2, x / 2 - 2))
    nucleus = _disc(SHAPE_2D, (center[0] - 8, center[1] - 8), (7, 7))

    mito = np.zeros(SHAPE_2D, dtype=np.uint16)
    for i in range(n_mito):
        angle = 2 * np.pi * i / n_mito
        cy = center[0] + 16 * np.sin(angle)
        cx = center[1] + 16 * np.cos(angle)
        blob = _disc(SHAPE_2D, (cy, cx), mito_radii) & boundary
        mito[blob] = i + 1

    rng = np.random.default_rng(abs(hash(object_dir.name)) % (2**32))
    source = (rng.normal(60, 8, SHAPE_2D) + 120 * boundary + 60 * (mito > 0)).clip(0, 255)

    paths = {
        "source": object_dir / f"{prefix}.tif",
        "membrane": object_dir / f"{prefix}_pm_mask.tif",
        "nucleus": object_dir / f"{prefix}_nucleus_mask.tif",
        "mito": object_dir / f"{prefix}_mito_label.tif",
    }
    _write_2d(paths["source"], source.astype(np.uint8))
    _write_2d(paths["membrane"], boundary.astype(np.uint8))
    _write_2d(paths["nucleus"], nucleus.astype(np.uint8))
    _write_2d(paths["mito"], mito)
    return paths


def make_dataset_2d(root: Path) -> Path:
    """Two groups of two 2D objects each, shaped like make_dataset's 3D batch."""
    make_object_2d(root / "control" / "object_a", prefix="sample_a", n_mito=4, mito_radii=(3.0, 3.0))
    make_object_2d(root / "control" / "object_b", prefix="sample_b", n_mito=4, mito_radii=(2.5, 2.5))
    make_object_2d(root / "treated" / "object_c", prefix="sample_c", n_mito=3, mito_radii=(4.5, 4.5))
    make_object_2d(root / "treated" / "object_d", prefix="sample_d", n_mito=3, mito_radii=(5.0, 5.0))
    return root


def make_dataset(root: Path) -> Path:
    """Two groups of two objects each; treated objects have fewer, larger mitochondria."""
    make_object(root / "control" / "object_a", prefix="sample_a", n_mito=4, mito_radii=(2.0, 3.0, 3.0))
    make_object(root / "control" / "object_b", prefix="sample_b", n_mito=4, mito_radii=(2.0, 2.5, 2.5))
    make_object(root / "treated" / "object_c", prefix="sample_c", n_mito=3, mito_radii=(3.0, 4.5, 4.5))
    make_object(root / "treated" / "object_d", prefix="sample_d", n_mito=3, mito_radii=(3.0, 5.0, 5.0))
    return root


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "objects").resolve()
    make_dataset(target)
    print(f"wrote synthetic dataset to {target}")
