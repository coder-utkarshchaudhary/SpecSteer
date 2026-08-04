"""
utils/training/dataloader.py
-----------------------------
PyTorch Dataset and DataLoader for the preprocessed HSI patch files.

Layout produced by utils/dataset/slice.py:

  data/processed/<DATASET>/
    <scene_name>/
      train/   patch_00000.npy  patch_00001.npy  ...
      valid/   ...
      test/    ...
      manifest.json      (optional; per-split per-patch max values)

Each .npy file contains a single (patch_size, patch_size, C) float32 array.
The dataset returns (H, W, C) float32 tensors.

Perf notes:
    - ``persistent_workers=True`` avoids re-forking workers every epoch.
    - ``prefetch_factor=4`` hides disk-read latency behind GPU compute.
    - Per-patch max values are cached in ``__init__``: either read from the
      sidecar ``manifest.json`` slice.py writes, or computed in a one-pass
      scan the first time the dataset is instantiated (this scan itself is
      what makes the *previous* per-item ``.max()`` disappear from the hot
      path). __getitem__ becomes ``np.load → torch.from_numpy → /= max`` with
      no per-item allocations beyond what ``np.load`` already does.
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


class HSIPatchDataset(Dataset):
    """
    Iterable dataset over all `.npy` patch files for a given split.

    Args:
        processed_root : path to the root produced by slice.py (e.g. ``data/processed/IIRS``)
        split          : one of ``'train'``, ``'valid'``, ``'test'``
    """

    def __init__(self, processed_root: str, split: str):
        assert split in ("train", "valid", "test"), (
            f"split must be one of 'train', 'valid', 'test'; got '{split}'"
        )

        root = Path(processed_root)
        if not root.exists():
            raise FileNotFoundError(
                f"Processed root not found: {root}\n"
                "Run utils/dataset/slice.py (or scripts/preprocess.sh) first."
            )

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
            _LOG.info(
                "dataloader: manifest.json missing %d/%d patch maxes under %s; "
                "scanning once (this only happens on first run).",
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
        """
        Returns:
            Tensor of shape (H, W, C) float32, values in [0, 1] after
            per-patch max normalization.
        """
        arr = np.load(self.patch_files[idx])
        # np.load already returns float32 for our patches; from_numpy is zero-copy.
        t = torch.from_numpy(arr)
        m = float(self._maxes[idx])
        if m != 1.0:
            t = t.div(m)
        return t


class IIRSDataset(HSIPatchDataset):
    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["IIRS"]["processed_root"], split)


class M3Dataset(HSIPatchDataset):
    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["M3"]["processed_root"], split)


class AVIRISDataset(HSIPatchDataset):
    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["AVIRIS"]["processed_root"], split)


class CRIMSDataset(HSIPatchDataset):
    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["CRIMS"]["processed_root"], split)


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
    batch_size: int = None,
    shuffle: bool = True,
    num_workers: int = None,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
) -> DataLoader:
    """
    Convenience factory. Returns a DataLoader whose batches are (B, H, W, C)
    float32 tensors (C is dataset-specific).
    """
    key = dataset.upper()
    if key not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASET_CLASSES)}.")

    if batch_size is None:
        batch_size = settings.batch_size
    if num_workers is None:
        num_workers = settings.num_workers

    ds = DATASET_CLASSES[key](split, processed_root=processed_root)

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
