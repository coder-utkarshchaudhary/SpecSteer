"""
utils/training/dataloader.py
-----------------------------
PyTorch Dataset and DataLoader for the preprocessed HSI patches.

Two storage backends, picked automatically:

  1. **Packed (preferred)** — one fp16 memmap per split, written by
     ``utils/dataset/pack.py``:

         data/packed/<DATASET>/
             train.npy   (N, H, W, C) float16, already max-normalised (max -> 1.0)
             train.json  metadata + provenance

  2. **Legacy (fallback)** — the original per-patch layout, used automatically
     when no packed shard exists so nothing breaks mid-migration:

         data/processed/<DATASET>/<scene>/<split>/patch_00000.npy

Why the packed backend exists
=============================
The 5-epoch smoke run showed three of four models were disk-bound, not
compute-bound: ``vae-standard`` needs 40x fewer FLOPs than ``vae-our`` yet took
the same 40 minutes. One IIRS epoch opened ~17,700 files totalling ~74 GB off an
external drive. Packing collapses that to one sequential fp16 read (~15 GB), and
``--cache-ram`` removes it entirely after the first epoch.

The legacy path also re-read the *entire* dataset at startup to recompute
per-patch maxima, because the ``manifest.json`` it looked for was never written
by ``slice.py`` — roughly 10 minutes of dead time before every one of the 28
grid runs. Packing pre-applies the normalisation, so that scan is gone.

Both backends yield identical (H, W, C) float32 tensors, per-patch
max-normalised so each patch's maximum is 1.0. Values are NOT clipped to
[0, 1] — real cubes carry small negative reflectances, and the legacy path
never clipped either, so the packed path must not start.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils.config import DATASETS, settings


_LOG = logging.getLogger(__name__)


def _load_manifest(root: Path) -> dict:
    """Read the optional manifest.json at the dataset root; empty dict if missing."""
    p = root / "manifest.json"
    if not p.is_file():
        return {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        _LOG.warning("manifest.json at %s unreadable: %s", p, e)
        return {}


# ---------------------------------------------------------------------------
# Packed backend
# ---------------------------------------------------------------------------

class PackedPatchDataset(Dataset):
    """
    Rows of a single fp16 memmap shard produced by ``utils/dataset/pack.py``.

    Args:
        packed_root : directory holding ``<split>.npy`` / ``<split>.json``
        split       : 'train' | 'valid' | 'test'
        cache_ram   : read the whole shard into RAM up front. Worth it when the
                      split fits: the capped train splits are 5-25 GB, and it
                      takes disk out of the loop entirely from epoch 2 onward.
        limit       : optional further cap on row count (deterministic prefix of
                      a seeded permutation). Normally unused — pack.py already
                      applied ``settings.train_patch_cap`` — but lets you shrink
                      a run without re-packing.
    """

    def __init__(self, packed_root: str | Path, split: str,
                 cache_ram: bool = False, limit: int | None = None,
                 seed: int | None = None):
        root = Path(packed_root)
        self.npy_path = root / f"{split}.npy"
        self.json_path = root / f"{split}.json"
        if not self.npy_path.is_file():
            raise FileNotFoundError(f"No packed shard at {self.npy_path}")

        self.meta = (
            json.loads(self.json_path.read_text())
            if self.json_path.is_file() else {}
        )

        data = np.load(self.npy_path, mmap_mode=None if cache_ram else "r")
        self.cached = bool(cache_ram)
        self._data = data

        n = data.shape[0]
        self.indices: np.ndarray | None = None
        if limit and limit < n:
            rng = np.random.default_rng(
                settings.patch_cap_seed if seed is None else seed
            )
            self.indices = np.sort(rng.choice(n, size=limit, replace=False))

        self.C = int(data.shape[-1])
        _LOG.info(
            "dataloader: packed %s split=%s rows=%d C=%d dtype=%s%s%s",
            root, split, len(self), self.C, data.dtype,
            " (RAM-cached)" if cache_ram else " (memmap)",
            f" [limited from {n}]" if self.indices is not None else "",
        )

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else self._data.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        """(H, W, C) float32, per-patch max-normalised."""
        if self.indices is not None:
            idx = int(self.indices[idx])
        # Cast to float32 here, not later: leaving the batch in fp16 would put an
        # fp16 target against a bf16 reconstruction inside mse_loss_fn, which is
        # a silent-precision trap for no bandwidth gain (the disk read is done).
        return torch.from_numpy(np.asarray(self._data[idx], dtype=np.float32))


# ---------------------------------------------------------------------------
# Legacy backend
# ---------------------------------------------------------------------------

class HSIPatchDataset(Dataset):
    """
    Iterable dataset over all `.npy` patch files for a given split (legacy layout).

    Kept as a fallback for machines where ``pack.py`` has not run yet. Prefer the
    packed backend: this path re-reads every patch at construction time when no
    ``manifest.json`` is present, which costs ~10 minutes per run on IIRS.

    Args:
        processed_root : path to the root produced by slice.py (e.g. ``data/processed/IIRS``)
        split          : one of ``'train'``, ``'valid'``, ``'test'``
        crop_bands     : optional band crop, mirroring what pack.py applies
    """

    def __init__(self, processed_root: str, split: str, crop_bands: int | None = None):
        assert split in ("train", "valid", "test"), (
            f"split must be one of 'train', 'valid', 'test'; got '{split}'"
        )

        root = Path(processed_root)
        if not root.exists():
            raise FileNotFoundError(
                f"Processed root not found: {root}\n"
                "Run utils/dataset/slice.py (or scripts/preprocess.sh) first."
            )

        self.crop_bands = crop_bands
        self.patch_files: List[Path] = sorted(root.glob(f"**/{split}/*.npy"))
        if len(self.patch_files) == 0:
            raise FileNotFoundError(
                f"No .npy patches found under {root}/**/{split}/\n"
                "Run utils/dataset/slice.py (or scripts/preprocess.sh) first."
            )

        # Try the sidecar first (populated by slice.py). Keys are relative
        # POSIX paths from the dataset root (e.g. "scene/train/patch_00000.npy").
        manifest = _load_manifest(root)
        maxes_map = (manifest.get("patch_max") or {}) if isinstance(manifest, dict) else {}

        maxes = np.empty(len(self.patch_files), dtype=np.float32)
        missing = 0
        for i, p in enumerate(self.patch_files):
            rel = p.relative_to(root).as_posix()
            v = maxes_map.get(rel)
            if v is not None:
                maxes[i] = float(v) or 1.0
            else:
                maxes[i] = -1.0  # sentinel; will scan below
                missing += 1

        if missing:
            _LOG.warning(
                "dataloader: manifest.json missing %d/%d patch maxes under %s; "
                "scanning every patch (SLOW — ~10 min on IIRS, and it repeats "
                "every run). Run `python utils/dataset/pack.py` to avoid this.",
                missing, len(self.patch_files), root,
            )
            for i, p in enumerate(self.patch_files):
                if maxes[i] > 0:
                    continue
                m = float(np.load(p).max())
                maxes[i] = m if m > 0 else 1.0

        maxes = np.where(maxes > 0, maxes, 1.0).astype(np.float32)
        self._maxes: np.ndarray = maxes

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Tensor:
        """(H, W, C) float32 after per-patch max normalization."""
        arr = np.load(self.patch_files[idx])
        if self.crop_bands:
            arr = arr[..., : self.crop_bands]
        t = torch.from_numpy(np.ascontiguousarray(arr))
        m = float(self._maxes[idx])
        if m != 1.0:
            t = t.div(m)
        return t


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def build_dataset(
    dataset: str,
    split: str,
    processed_root: str | None = None,
    packed_root: str | None = None,
    cache_ram: bool = False,
    limit: int | None = None,
) -> Dataset:
    """
    Return the packed dataset when a shard exists, else the legacy one.

    ``processed_root`` (i.e. ``--data-root``) forces the legacy backend, since it
    names a per-patch tree explicitly.
    """
    key = dataset.upper()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASETS)}.")
    meta = DATASETS[key]

    if processed_root is None:
        proot = Path(packed_root or meta["packed_root"])
        if (proot / f"{split}.npy").is_file():
            return PackedPatchDataset(proot, split, cache_ram=cache_ram, limit=limit)
        _LOG.warning(
            "dataloader: no packed shard at %s/%s.npy — falling back to the "
            "per-patch layout. Run `python utils/dataset/pack.py --dataset %s` "
            "for a large speedup.", proot, split, key,
        )

    return HSIPatchDataset(
        processed_root or meta["processed_root"],
        split,
        crop_bands=meta.get("crop_bands"),
    )


class IIRSDataset(HSIPatchDataset):
    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["IIRS"]["processed_root"], split,
                         crop_bands=DATASETS["IIRS"].get("crop_bands"))


class M3Dataset(HSIPatchDataset):
    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["M3"]["processed_root"], split,
                         crop_bands=DATASETS["M3"].get("crop_bands"))


class AVIRISDataset(HSIPatchDataset):
    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["AVIRIS"]["processed_root"], split,
                         crop_bands=DATASETS["AVIRIS"].get("crop_bands"))


class CRIMSDataset(HSIPatchDataset):
    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["CRIMS"]["processed_root"], split,
                         crop_bands=DATASETS["CRIMS"].get("crop_bands"))


DATASET_CLASSES = {
    "IIRS": IIRSDataset,
    "M3": M3Dataset,
    "AVIRIS": AVIRISDataset,
    "CRIMS": CRIMSDataset,
}


def build_dataloader(
    dataset: str,
    split: str,
    processed_root: str = None,
    packed_root: str = None,
    batch_size: int = None,
    shuffle: bool = True,
    num_workers: int = None,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    cache_ram: bool = False,
    limit: int = None,
) -> DataLoader:
    """
    Convenience factory. Returns a DataLoader whose batches are (B, H, W, C)
    float32 tensors (C is dataset-specific).
    """
    if batch_size is None:
        batch_size = settings.batch_size
    if num_workers is None:
        num_workers = settings.num_workers

    ds = build_dataset(
        dataset, split,
        processed_root=processed_root,
        packed_root=packed_root,
        cache_ram=cache_ram,
        limit=limit,
    )

    # NOTE: workers are kept even when cache_ram is on. The fp16 -> fp32 cast in
    # __getitem__ is real per-item CPU work (~7 MB/patch on AVIRIS) and is the
    # binding constraint once disk is out of the picture; single-process it caps
    # out around 120 patch/s. Linux forks workers, so they share the cached
    # array's pages copy-on-write and cost no extra RAM for the payload.
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **kwargs)
