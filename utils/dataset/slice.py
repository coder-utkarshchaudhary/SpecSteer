"""
utils/dataset/slice.py
-----------------------
Slice preprocessed HSI cubes (IIRS, M3, AVIRIS) into (patch_size x
patch_size x C) patches and save them to disk under separate per-dataset
roots.

Split strategy: **region-disjoint** along the long (H) axis of each cube.
    - Each cube is carved into 3 contiguous row-regions (train / valid / test)
      *before* patching — no patch ever crosses a region boundary.
    - Split ratios: 70 / 15 / 15 (configurable in settings).
    - This prevents spatial autocorrelation leakage between splits.

Patch sampling:
    - Stride = settings.patch_stride (48 by default — 25% overlap).
    - Partial patches that would extend beyond the region boundary are dropped.
    - Any patch whose fraction of fill/invalid pixels (per the
        preprocessor's valid_mask) exceeds settings.fill_fraction_threshold
        is dropped — matters most for AVIRIS (orthorectified, large
        off-swath fill corners) and M3 (`data ignore value = -999`).

Output layout:
    data/processed/<DATASET>/<scene>/<split>/patch_NNNNN.npy   (64x64xC) float32

Usage:
    python utils/dataset/slice.py --dataset {iirs,m3,aviris,all} [--overwrite]
    python utils/dataset/slice.py --dataset m3 --data-root "data/original - m3" --out-root data/processed/M3
"""

import argparse
import json
from pathlib import Path

import numpy as np

from utils.config import DATASETS, settings
from utils.dataset.preprocess import (
    AVIRISPreprocessor,
    IIRSPreprocessor,
    M3Preprocessor,
)


# ---------------------------------------------------------------------------
# Patch extraction helpers (shared by all slicers)
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
    valid_mask: np.ndarray,
    row_start: int,
    row_end: int,
    patch_size: int,
    stride: int,
    fill_fraction_threshold: float,
) -> tuple:
    """
    Extract non-partial patches from a row region of the cube, dropping any
    patch whose fill fraction exceeds `fill_fraction_threshold`.

    Args:
        cube_hwc    : (H, W, C) float32
        valid_mask  : (H, W) bool — True where the pixel is not fill/invalid
        row_start   : first row of the region (inclusive)
        row_end     : last  row of the region (exclusive)
        patch_size  : height and width of each patch
        stride      : step size (controls overlap)
        fill_fraction_threshold : max allowed fraction of invalid pixels

    Returns:
        (patches, dropped) — list of (patch_size, patch_size, C) float32
        arrays, and the count of patches dropped for excess fill.
    """
    H, W, C = cube_hwc.shape
    patches = []
    dropped = 0

    r = row_start
    while r + patch_size <= row_end:
        c = 0
        while c + patch_size <= W:
            mask_patch = valid_mask[r : r + patch_size, c : c + patch_size]
            fill_fraction = 1.0 - float(mask_patch.mean())
            if fill_fraction <= fill_fraction_threshold:
                patch = cube_hwc[r : r + patch_size, c : c + patch_size, :]
                patches.append(patch.copy())
            else:
                dropped += 1
            c += stride
        r += stride

    return patches, dropped


# ---------------------------------------------------------------------------
# Base slicer
# ---------------------------------------------------------------------------

class BaseHSISlicer:
    """
    Shared region-split + patch + save orchestration.

    Subclasses set `dataset_name` (key into utils.config.DATASETS),
    `preprocessor_cls`, and implement `iter_sources(raw_root)` to yield
    (scene_name, source) pairs.
    """

    dataset_name: str = None
    preprocessor_cls = None

    def __init__(
        self,
        out_root: str = None,
        patch_size: int = None,
        stride: int = None,
        split_ratios: tuple = None,
        fill_fraction_threshold: float = None,
        overwrite: bool = False,
    ):
        cfg = DATASETS[self.dataset_name]
        self.out_root = Path(out_root) if out_root else Path(cfg["processed_root"])
        self.patch_size = patch_size or settings.patch_size
        self.stride = stride or settings.patch_stride
        self.split_ratios = split_ratios or settings.split_ratios
        self.fill_fraction_threshold = (
            fill_fraction_threshold
            if fill_fraction_threshold is not None
            else settings.fill_fraction_threshold
        )
        self.overwrite = overwrite
        self.preprocessor = self.preprocessor_cls()

    def iter_sources(self, raw_root: Path):
        """Yield (scene_name, source) pairs for every acquisition under raw_root."""
        raise NotImplementedError

    def slice_source(self, scene_name: str, source) -> tuple:
        """
        Preprocess one source and save all of its patches.

        Returns:
            (counts, patch_maxes) where counts is {split: n} and patch_maxes is
            a dict mapping the patch's path relative to self.out_root (as a
            POSIX string) to its per-patch max reflectance. Consumed by the
            dataloader to skip the per-item .max() scan.
        """
        split_names = ("train", "valid", "test")
        out_dirs = {s: self.out_root / scene_name / s for s in split_names}

        if not self.overwrite and all(d.exists() and any(d.iterdir()) for d in out_dirs.values()):
            print(f"  [skip] {scene_name} already exists (use --overwrite to redo)")
            counts = {s: len(list(out_dirs[s].glob("*.npy"))) for s in split_names}
            # Re-scan existing patches so the manifest we're about to write
            # covers *all* patches on disk, not just ones we produced this run.
            maxes = self._scan_existing_maxes(out_dirs)
            return counts, maxes

        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        print(f"  Preprocessing {scene_name} ...")
        cube_chw, valid_mask = self.preprocessor.preprocess(source)   # (C, H, W), (H, W)
        cube_hwc = cube_chw.transpose(1, 2, 0)                        # (H, W, C) — model-ready

        H, W, C = cube_hwc.shape
        bounds = _region_bounds(H, self.split_ratios)

        counts: dict = {}
        maxes: dict = {}
        for split, (r_start, r_end) in zip(split_names, bounds):
            patches, dropped = _extract_patches(
                cube_hwc,
                valid_mask,
                r_start,
                r_end,
                self.patch_size,
                self.stride,
                self.fill_fraction_threshold,
            )
            out_dir = out_dirs[split]
            for idx, patch in enumerate(patches):
                fname = out_dir / f"patch_{idx:05d}.npy"
                np.save(str(fname), patch)
                rel = fname.relative_to(self.out_root).as_posix()
                maxes[rel] = float(patch.max())

            counts[split] = len(patches)
            print(
                f"    {split}: rows [{r_start}, {r_end}) → {len(patches)} patches saved "
                f"({dropped} dropped for >{self.fill_fraction_threshold:.0%} fill) to {out_dir}"
            )

        return counts, maxes

    def _scan_existing_maxes(self, out_dirs: dict) -> dict:
        """Fallback used on --skip to rebuild manifest for already-processed scenes."""
        maxes = {}
        for d in out_dirs.values():
            for p in sorted(d.glob("*.npy")):
                rel = p.relative_to(self.out_root).as_posix()
                maxes[rel] = float(np.load(p).max())
        return maxes

    def run(self, raw_root=None, limit: int = None) -> dict:
        """Process every source under raw_root. Returns aggregate per-split counts."""
        raw_root = Path(raw_root) if raw_root else Path(DATASETS[self.dataset_name]["raw_root"])
        if not raw_root.exists():
            raise FileNotFoundError(f"raw root not found: {raw_root}")

        sources = list(self.iter_sources(raw_root))
        if limit is not None:
            sources = sources[:limit]
        if not sources:
            print(f"  No sources found in {raw_root}")
            return {"train": 0, "valid": 0, "test": 0}

        total = {"train": 0, "valid": 0, "test": 0}
        manifest_maxes: dict = {}
        for scene_name, source in sources:
            print(f"[{scene_name}]")
            counts, scene_maxes = self.slice_source(scene_name, source)
            for split, n in counts.items():
                total[split] += n
            manifest_maxes.update(scene_maxes)

        # Write / merge the sidecar manifest at the dataset root so the
        # dataloader can skip the per-item .max() scan.
        self._write_manifest(manifest_maxes)
        return total

    def _write_manifest(self, patch_maxes: dict) -> None:
        """Write logs/manifest.json alongside the processed patches."""
        if not patch_maxes:
            return
        self.out_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.out_root / "manifest.json"
        existing = {}
        if manifest_path.is_file():
            try:
                with manifest_path.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except (OSError, json.JSONDecodeError):
                existing = {}
        merged_maxes = {**(existing.get("patch_max") or {}), **patch_maxes}
        payload = {
            "dataset": self.dataset_name,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "split_ratios": list(self.split_ratios),
            "fill_fraction_threshold": self.fill_fraction_threshold,
            "patch_max": merged_maxes,
        }
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  manifest: {manifest_path} ({len(merged_maxes)} patches indexed)")


# ---------------------------------------------------------------------------
# Per-dataset slicers
# ---------------------------------------------------------------------------

class IIRSSlicer(BaseHSISlicer):
    """One source per acquisition sub-folder; scene name = sub-folder name."""

    dataset_name = "IIRS"
    preprocessor_cls = IIRSPreprocessor

    def iter_sources(self, raw_root: Path):
        for p in sorted(raw_root.iterdir()):
            if p.is_dir():
                yield p.name, p


class M3Slicer(BaseHSISlicer):
    """
    One source per `*_rfl.img` file; scene name = the code between `m3g`
    and the first `_` (e.g. `m3g20090814t021520_v01_rfl.img` → `20090814t021520`).
    """

    dataset_name = "M3"
    preprocessor_cls = M3Preprocessor

    def iter_sources(self, raw_root: Path):
        for p in sorted(raw_root.glob("*_rfl.img")):
            stem = p.stem
            after_prefix = stem[len("m3g") :] if stem.startswith("m3g") else stem
            scene = after_prefix.split("_")[0]
            yield scene, p


class AVIRISSlicer(BaseHSISlicer):
    """One source per `*.nc` file; scene name = the full filename stem."""

    dataset_name = "AVIRIS"
    preprocessor_cls = AVIRISPreprocessor

    def iter_sources(self, raw_root: Path):
        for p in sorted(raw_root.glob("*.nc")):
            yield p.stem, p


SLICERS = {
    "IIRS": IIRSSlicer,
    "M3": M3Slicer,
    "AVIRIS": AVIRISSlicer,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice IIRS / M3 / AVIRIS hyperspectral cubes into patches for training."
    )
    parser.add_argument(
        "--dataset",
        choices=["iirs", "m3", "aviris", "all"],
        default="all",
        help="Which dataset to process (default: all)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the raw root (only valid together with a single --dataset).",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Override the processed output root (only valid together with a single --dataset).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process scenes whose output already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sources per dataset (debug / smoke-test).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    names = list(SLICERS) if args.dataset == "all" else [args.dataset.upper()]

    if args.data_root and len(names) > 1:
        raise SystemExit("--data-root can only be used together with a single --dataset")
    if args.out_root and len(names) > 1:
        raise SystemExit("--out-root can only be used together with a single --dataset")

    grand_total = {"train": 0, "valid": 0, "test": 0}
    for name in names:
        print(f"=== {name} ===")
        slicer = SLICERS[name](out_root=args.out_root, overwrite=args.overwrite)
        raw_root = args.data_root or DATASETS[name]["raw_root"]
        print(f"  raw root : {raw_root}")
        print(f"  out  root: {slicer.out_root}")
        print(
            f"  patch: {slicer.patch_size}×{slicer.patch_size}, stride: {slicer.stride}, "
            f"split: {slicer.split_ratios}, fill_threshold: {slicer.fill_fraction_threshold:.0%}"
        )

        totals = slicer.run(raw_root=raw_root, limit=args.limit)
        for split, n in totals.items():
            grand_total[split] += n
        print()

    print("=== Summary (all datasets processed this run) ===")
    for split, n in grand_total.items():
        print(f"  {split:6s}: {n:,} patches")
    print(f"  total : {sum(grand_total.values()):,} patches")


if __name__ == "__main__":
    main()
