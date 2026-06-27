"""
Slice preprocessed HSI cubes into (patch_size x patch_size x C) patches and
save them to disk under data/processed/.

Split strategy: **region-disjoint** along the long (H) axis of each cube.
    - Each cube is carved into 3 contiguous row-regions (train / valid / test)
    *before* patching — no patch ever crosses a region boundary.
    - Split ratios: 70 / 15 / 15 (configurable in settings).
    - This prevents spatial autocorrelation leakage between splits (important
    for a rigorous evaluation in the paper).

Patch sampling:
    - Stride = settings.patch_stride (48 by default — 25% overlap).
    - Partial patches that would extend beyond the region boundary are dropped.
    - Sensor width W = 250 is fixed; with patch_size=64 and stride=48:
        column starts: 0, 48, 96, 144, 192  (5 columns per row group).

Output layout:
    data/processed/<folder_name>/<split>/patch_NNNNN.npy   (64x64x108) float32

Usage:
    python utils/dataset/slice.py [--data-root data/original] [--out-root data/processed] [--overwrite]
"""

import argparse
import os
from pathlib import Path

import numpy as np

from utils.dataset.preprocess import preprocess_cube
from utils.config import settings


# ---------------------------------------------------------------------------
# Patch extraction helpers
# ---------------------------------------------------------------------------
def _region_bounds(H: int, ratios: tuple) -> list:
    """
    Compute (start, end) row indices for each split region.

    Args:
        H      : total number of rows in the cube
        ratios : (train_frac, valid_frac, test_frac) — must sum to 1.0

    Returns:
        List of (start, end) tuples: [(0, train_end), (train_end, valid_end), (valid_end, H)]
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    train_end = int(H * ratios[0])
    valid_end = int(H * (ratios[0] + ratios[1]))
    return [(0, train_end), (train_end, valid_end), (valid_end, H)]


def _extract_patches(
    cube_hwc: np.ndarray,
    row_start: int,
    row_end: int,
    patch_size: int,
    stride: int,
) -> list:
    """
    Extract non-partial patches from a row region of the cube.

    Args:
        cube_hwc  : (H, W, C) float32
        row_start : first row of the region (inclusive)
        row_end   : last  row of the region (exclusive)
        patch_size: height and width of each patch
        stride    : step size (controls overlap)

    Returns:
        List of (patch_size, patch_size, C) float32 arrays.
    """
    H, W, C = cube_hwc.shape
    patches = []

    r = row_start
    while r + patch_size <= row_end:
        c = 0
        while c + patch_size <= W:
            patch = cube_hwc[r : r + patch_size, c : c + patch_size, :]
            patches.append(patch.copy())
            c += stride
        r += stride

    return patches


# ---------------------------------------------------------------------------
# Per-folder slice
# ---------------------------------------------------------------------------

def slice_folder(
    folder_path: str,
    out_root: str,
    patch_size: int = settings.patch_size,
    stride: int = settings.patch_stride,
    split_ratios: tuple = settings.split_ratios,
    overwrite: bool = False,
) -> dict:
    """
    Preprocess one data folder and save all patches.

    Args:
        folder_path  : path to one `data/original/<name>/` directory
        out_root     : root for processed output (e.g. `data/processed/`)
        patch_size   : spatial size of patches (square)
        stride       : sampling stride
        split_ratios : (train, valid, test) fractions
        overwrite    : re-process even if output already exists

    Returns:
        Dict mapping split name → number of patches saved.
    """
    folder_name = Path(folder_path).name
    split_names = ("train", "valid", "test")

    # Build output directories
    out_dirs = {}
    for split in split_names:
        out_dir = Path(out_root) / folder_name / split
        out_dirs[split] = out_dir

        if not overwrite and out_dir.exists() and any(out_dir.iterdir()):
            print(f"  [skip] {folder_name}/{split} already exists (use --overwrite to redo)")
            # Return existing counts
            counts = {s: len(list((Path(out_root) / folder_name / s).glob("*.npy"))) for s in split_names}
            return counts

        out_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess: (C, H, W)
    print(f"  Preprocessing {folder_name} ...")
    cube_chw = preprocess_cube(folder_path)           # (C, H, W)
    cube_hwc = cube_chw.transpose(1, 2, 0)            # (H, W, C) — model-ready

    H, W, C = cube_hwc.shape
    bounds = _region_bounds(H, split_ratios)

    counts = {}
    for split, (r_start, r_end) in zip(split_names, bounds):
        patches = _extract_patches(cube_hwc, r_start, r_end, patch_size, stride)
        out_dir = out_dirs[split]

        for idx, patch in enumerate(patches):
            # zero-padded index for lexicographic sort in the dataloader
            fname = out_dir / f"patch_{idx:05d}.npy"
            np.save(str(fname), patch)

        counts[split] = len(patches)
        print(
            f"    {split}: rows [{r_start}, {r_end}) → "
            f"{len(patches)} patches saved to {out_dir}"
        )

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice IIRS hyperspectral cubes into patches for training."
    )
    parser.add_argument(
        "--data-root",
        default=settings.data_original_root,
        help=f"Root directory with one sub-folder per observation (default: {settings.data_original_root})",
    )
    parser.add_argument(
        "--out-root",
        default=settings.data_processed_root,
        help=f"Root directory for processed patch output (default: {settings.data_processed_root})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process folders whose output already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    if not data_root.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")

    folders = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not folders:
        print(f"No sub-folders found in {data_root}")
        return

    print(f"Found {len(folders)} folders in {data_root}")
    print(f"Output root: {out_root}")
    print(
        f"Patch: {settings.patch_size}×{settings.patch_size}, "
        f"stride: {settings.patch_stride}, "
        f"split: {settings.split_ratios}"
    )
    print()

    total = {"train": 0, "valid": 0, "test": 0}
    for folder in folders:
        print(f"[{folder.name}]")
        counts = slice_folder(
            str(folder),
            str(out_root),
            overwrite=args.overwrite,
        )
        for split, n in counts.items():
            total[split] += n

    print()
    print("=== Summary ===")
    for split, n in total.items():
        print(f"  {split:6s}: {n:,} patches")
    print(f"  total : {sum(total.values()):,} patches")


if __name__ == "__main__":
    main()
