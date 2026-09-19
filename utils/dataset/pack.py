"""
utils/dataset/pack.py
---------------------
Pack the ~15,000 per-patch ``.npy`` files of each dataset into ONE contiguous
float16 memmap per split.

Why this exists
===============
Profiling the 5-epoch smoke run showed that three of the four models were not
compute-bound at all — they were bound by reading patches off disk. One IIRS
epoch touches 14,624 + 3,084 separate files totalling ~74 GB, and on the lab
machine that lives on an external drive (~150 MB/s), giving a hard floor of
~8 min/epoch *regardless of which model runs*. On top of that,
``utils/training/dataloader.py`` re-read the entire dataset before every run to
recompute per-patch maxima, because the ``manifest.json`` it looks for was never
actually written.

This script fixes all of it at once:

  * **float16** halves the bytes. Patches are divided by their own max here (so
    each tops out at 1.0), and fp16's ~5e-4 relative precision is *finer* than
    the bfloat16 autocast the training loop already runs in — so this cannot
    degrade the math. Measured worst-case round-trip error: 2.4e-4.
  * **One file replaces ~15,000.** Per-file open overhead disappears, reads go
    sequential, and the OS page cache starts working across epochs.
  * **Normalisation moves here**, permanently removing the startup rescan.
  * **The band crop is applied here.** ``crop_bands`` is otherwise only honoured
    by ``slice.py``, which never runs for CRIMS (it ships pre-processed) — so
    CRIMS's 457 -> 456 crop has to happen at pack time or not at all.
  * **The training-patch cap is applied here**, stratified across scenes with a
    fixed seed, so every model in the ablation sees the identical subset and the
    packed output is small enough to zip for Kaggle.

Output layout
=============
    data/packed/<DATASET>/
        train.npy    (N, 64, 64, C) float16, per-patch max-normalised
        train.json   metadata + provenance (source file list, per-patch maxima)
        valid.npy / valid.json
        test.npy  / test.json

Usage
=====
    # everything, honouring settings.train_patch_cap
    PYTHONPATH=. python utils/dataset/pack.py

    # one dataset, re-pack from scratch, then verify against the sources
    PYTHONPATH=. python utils/dataset/pack.py --dataset CRIMS --overwrite --verify

    # uncapped (full training set)
    PYTHONPATH=. python utils/dataset/pack.py --cap 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import DATASETS, settings  # noqa: E402

SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# Patch discovery + capping
# ---------------------------------------------------------------------------

def find_patches(processed_root: Path, split: str) -> list[Path]:
    """All patch files for a split, sorted so the packing order is reproducible."""
    return sorted(processed_root.glob(f"**/{split}/*.npy"))


def scene_of(path: Path) -> str:
    """Scene directory name — the parent of the split directory."""
    return path.parent.parent.name


def apply_cap(files: list[Path], cap: int | None, seed: int) -> list[Path]:
    """
    Deterministically subsample to at most ``cap`` patches, proportionally across
    scenes so no scene is dropped entirely.

    Proportional (rather than uniform-random) allocation matters: a flat random
    draw over a dataset whose scenes differ 10x in patch count would silently
    over-represent the big scenes. Every model in the grid gets the identical
    subset because the seed is fixed in Settings.
    """
    if not cap or len(files) <= cap:
        return files

    by_scene: dict[str, list[Path]] = {}
    for f in files:
        by_scene.setdefault(scene_of(f), []).append(f)

    total = len(files)
    rng = np.random.default_rng(seed)

    # Every scene keeps at least one patch, so the cap cannot go below the scene
    # count. That only matters for absurd caps (18 scenes, cap=13), but it must
    # not be silent — the caller asked for N and would get more.
    if cap < len(by_scene):
        print(f"  [warn] cap={cap} is below the scene count ({len(by_scene)}); "
              f"keeping 1 patch per scene = {len(by_scene)} patches instead.")

    # Largest-remainder allocation: floor the proportional share, then hand out
    # the leftovers to the scenes with the biggest fractional parts. Guarantees
    # the quotas sum to exactly `cap` and that no scene gets zero.
    scenes = sorted(by_scene)
    exact = {s: cap * len(by_scene[s]) / total for s in scenes}
    quota = {s: max(1, int(exact[s])) for s in scenes}

    # Trim/extend to hit `cap` exactly, never exceeding a scene's own supply.
    def _capacity(s: str) -> int:
        return len(by_scene[s])

    while sum(quota.values()) > cap:
        # Take from whoever is furthest above its exact share (and above 1).
        s = max((s for s in scenes if quota[s] > 1),
                key=lambda s: quota[s] - exact[s], default=None)
        if s is None:
            break
        quota[s] -= 1
    while sum(quota.values()) < cap:
        s = min((s for s in scenes if quota[s] < _capacity(s)),
                key=lambda s: quota[s] - exact[s], default=None)
        if s is None:
            # Every scene is exhausted — can only happen if cap > len(files),
            # which the early return above already excludes.
            break
        quota[s] += 1

    picked: list[Path] = []
    for s in scenes:
        pool = by_scene[s]
        k = min(quota[s], len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        picked.extend(pool[i] for i in sorted(idx))

    # Sort so the packed file's row order is stable and reads stay sequential.
    return sorted(picked)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def normalise(arr: np.ndarray, crop: int | None) -> tuple[np.ndarray, float]:
    """
    Crop bands, replace non-finite values, and divide by the per-patch max.

    Returns (float32 normalised patch, the max divisor used). Mirrors exactly
    what the old ``HSIPatchDataset.__getitem__`` did per item, so packed and
    unpacked training see the same numbers.
    """
    if crop:
        arr = arr[..., :crop]
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(arr.max())
    if m <= 0:
        m = 1.0
    return arr / m, m


def pack_split(dataset: str, split: str, processed_root: Path, packed_root: Path,
               cap: int | None, seed: int, overwrite: bool) -> dict | None:
    """Pack one (dataset, split). Returns its metadata dict, or None if skipped."""
    out_npy = packed_root / f"{split}.npy"
    out_json = packed_root / f"{split}.json"

    if out_npy.is_file() and out_json.is_file() and not overwrite:
        meta = json.loads(out_json.read_text())
        print(f"  [skip] {split:5s} already packed: {meta['n']} patches, "
              f"C={meta['C']}  ({out_npy})")
        return meta

    files = find_patches(processed_root, split)
    if not files:
        print(f"  [warn] {split:5s} no patches under {processed_root}/**/{split}/")
        return None

    n_found = len(files)
    # Only the training split is capped; valid/test stay whole so evaluation is
    # never subsampled.
    if split == "train":
        files = apply_cap(files, cap, seed)

    meta_ds = DATASETS[dataset]
    crop = meta_ds.get("crop_bands")
    probe = np.load(files[0], mmap_mode="r")
    h, w, raw_c = probe.shape
    c = int(crop) if crop else int(raw_c)

    expected_raw = meta_ds.get("raw_channels")
    if expected_raw is not None and raw_c != expected_raw:
        raise SystemExit(
            f"{dataset}/{split}: patches have {raw_c} bands, config says "
            f"raw_channels={expected_raw}. Refusing to pack a mismatched cube — "
            f"fix utils/config.py first (see utils/dataset/inspect_channels.py)."
        )
    if c != meta_ds["input_channels"]:
        raise SystemExit(
            f"{dataset}/{split}: post-crop band count {c} != input_channels "
            f"{meta_ds['input_channels']}. Fix utils/config.py first."
        )

    n = len(files)
    packed_root.mkdir(parents=True, exist_ok=True)
    gb = n * h * w * c * 2 / 1e9
    print(f"  [pack] {split:5s} {n:>6,} / {n_found:>6,} patches  "
          f"({h}x{w}x{c} fp16 = {gb:.1f} GB)  -> {out_npy}")

    # Write to a temp name and rename at the end, so an interrupted run never
    # leaves a half-written shard that looks complete to the dataloader.
    tmp_npy = out_npy.with_suffix(".npy.partial")
    mm = np.lib.format.open_memmap(
        tmp_npy, mode="w+", dtype=np.float16, shape=(n, h, w, c)
    )

    maxes = np.empty(n, dtype=np.float32)
    t0 = time.time()
    for i, f in enumerate(files):
        norm, m = normalise(np.load(f), crop)
        mm[i] = norm.astype(np.float16)
        maxes[i] = m
        if (i + 1) % 500 == 0 or i + 1 == n:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-6)
            print(f"         {i + 1:>6,}/{n:,}  {el:6.1f}s  "
                  f"({rate:5.1f} patch/s, eta {(n - i - 1) / max(rate, 1e-6):5.0f}s)",
                  end="\r", flush=True)
    mm.flush()
    del mm
    tmp_npy.replace(out_npy)
    print()

    meta = {
        "dataset": dataset,
        "split": split,
        "n": n,
        "n_available": n_found,
        "H": int(h),
        "W": int(w),
        "C": int(c),
        "raw_C": int(raw_c),
        "crop_bands": crop,
        "dtype": "float16",
        "normalisation": "divide by per-patch max (max -> 1.0; not clipped)",
        "cap": cap if split == "train" else None,
        "cap_seed": seed if split == "train" else None,
        "processed_root": str(processed_root),
        "patch_max": [float(x) for x in maxes],
        "source_files": [str(f.relative_to(processed_root)) for f in files],
    }
    out_json.write_text(json.dumps(meta))
    print(f"         wrote {out_json.name} ({time.time() - t0:.1f}s total)")
    return meta


def verify_split(dataset: str, split: str, processed_root: Path,
                 packed_root: Path, n_samples: int = 12, seed: int = 0) -> bool:
    """
    Re-read a random sample of packed rows and compare against the source
    ``.npy`` put through the same normalisation. Catches crop-off-by-one, row
    misalignment, and fp16 range problems.
    """
    out_npy = packed_root / f"{split}.npy"
    out_json = packed_root / f"{split}.json"
    if not (out_npy.is_file() and out_json.is_file()):
        print(f"  [verify] {split}: nothing packed")
        return False

    meta = json.loads(out_json.read_text())
    mm = np.load(out_npy, mmap_mode="r")
    if mm.shape[0] != meta["n"]:
        print(f"  [verify] {split}: FAIL row count {mm.shape[0]} != {meta['n']}")
        return False

    rng = np.random.default_rng(seed)
    idx = rng.choice(meta["n"], size=min(n_samples, meta["n"]), replace=False)
    crop = meta["crop_bands"]
    worst = 0.0
    # Normalisation divides by the per-patch *signed* max, so the positive peak
    # is 1.0 but negative reflectances are left un-clipped and can be several
    # times larger in magnitude (see utils/training/dataloader.py). The fp16
    # tolerance must therefore track the actual value range, not assume [0, 1].
    peak = 1.0
    for i in idx:
        src = np.load(processed_root / meta["source_files"][i])
        norm, m = normalise(src, crop)
        got = np.asarray(mm[i], dtype=np.float32)
        if got.shape != norm.shape:
            print(f"  [verify] {split}: FAIL row {i} shape {got.shape} != {norm.shape}")
            return False
        worst = max(worst, float(np.abs(got - norm).max()))
        peak = max(peak, float(np.abs(norm).max()))
        if abs(m - meta["patch_max"][i]) > 1e-3 * max(m, 1.0):
            print(f"  [verify] {split}: FAIL row {i} max {m} != {meta['patch_max'][i]}")
            return False

    # fp16 keeps ~11 bits of mantissa, so its worst-case round-trip error is
    # |value| * 2**-11. Scale the tolerance to the observed peak magnitude
    # (2**-10 = one ULP of headroom over the 2**-11 half-ULP bound). A genuine
    # row misalignment or crop error produces an O(1) error — ~1000x this tol —
    # so this stays just as sensitive to real corruption as the old flat 1e-3,
    # while no longer failing on legitimate rounding of large negative bands.
    tol = peak * 2 ** -10
    ok = worst < tol
    print(f"  [verify] {split}: {'OK ' if ok else 'FAIL'}  "
          f"{len(idx)} rows sampled, max abs err {worst:.2e} "
          f"(tol {tol:.2e} @ peak |v|={peak:.2f})")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=sorted(DATASETS),
                    help="Dataset to pack (repeatable). Default: all.")
    ap.add_argument("--split", action="append", choices=SPLITS,
                    help="Split to pack (repeatable). Default: all.")
    ap.add_argument("--processed-root", default=None,
                    help="Override the processed root for the single --dataset given.")
    ap.add_argument("--packed-root", default=None,
                    help="Override the packed output root (default: data/packed/<DS>).")
    ap.add_argument("--cap", type=int, default=None,
                    help="Max training patches per dataset. 0 disables the cap. "
                         f"Default: settings.train_patch_cap ({settings.train_patch_cap}).")
    ap.add_argument("--seed", type=int, default=settings.patch_cap_seed)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-pack even if a shard already exists.")
    ap.add_argument("--verify", action="store_true",
                    help="Spot-check packed rows against their source patches.")
    args = ap.parse_args()

    datasets = args.dataset or sorted(DATASETS)
    splits = args.split or list(SPLITS)
    cap = settings.train_patch_cap if args.cap is None else (args.cap or None)

    if args.processed_root and len(datasets) > 1:
        raise SystemExit("--processed-root requires exactly one --dataset")

    print("=" * 70)
    print(" pack: processed .npy patches -> fp16 memmap shards")
    print(f"  datasets : {', '.join(datasets)}")
    print(f"  splits   : {', '.join(splits)}")
    print(f"  cap      : {cap if cap else '<none>'} (train only, seed {args.seed})")
    print("=" * 70)

    failures = 0
    total_gb = 0.0
    for ds in datasets:
        meta_ds = DATASETS[ds]
        processed_root = Path(args.processed_root or meta_ds["processed_root"])
        packed_root = Path(args.packed_root or meta_ds["packed_root"])
        print(f"\n{ds}  ({processed_root} -> {packed_root})")
        if not processed_root.exists():
            print(f"  [warn] processed root missing: {processed_root} — skipping")
            failures += 1
            continue

        for split in splits:
            try:
                meta = pack_split(ds, split, processed_root, packed_root,
                                  cap, args.seed, args.overwrite)
            except SystemExit as e:
                print(f"  [FAIL] {split}: {e}")
                failures += 1
                continue
            if meta:
                total_gb += meta["n"] * meta["H"] * meta["W"] * meta["C"] * 2 / 1e9
            if args.verify and meta:
                if not verify_split(ds, split, processed_root, packed_root):
                    failures += 1

    print("\n" + "=" * 70)
    print(f" packed total: {total_gb:.1f} GB   failures: {failures}")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
