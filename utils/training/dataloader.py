"""
utils/training/dataloader.py
-----------------------------
PyTorch Dataset and DataLoader for the preprocessed HSI patch files.

Expects a processed root directory with the layout produced by
utils/dataset/slice.py:

  data/processed/<DATASET>/
    <scene_name>/
      train/   patch_00000.npy  patch_00001.npy  ...
      valid/   patch_00000.npy  ...
      test/    patch_00000.npy  ...
    <scene_name>/
      ...

Each .npy file contains a single (patch_size, patch_size, C) float32 array,
where C is dataset-specific (IIRS=256, M3=84, AVIRIS=424, CRIMS=544 — see
utils.config.DATASETS). The dataset returns (H, W, C) float32 tensors — the
shape expected by the Dual-Stream PI-VAE's forward pass. Since band count
differs per dataset, each dataset needs its own model instance (built via
utils.config.make_settings(dataset)); the dataloader just needs the matching
processed root.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils.config import DATASETS, settings


class HSIPatchDataset(Dataset):
    """
    Iterable dataset over all `.npy` patch files for a given split.

    Args:
        processed_root : path to the root produced by slice.py (e.g. ``data/processed/``)
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

        # Collect all patches across all observation sub-folders for the split
        self.patch_files: List[Path] = sorted(root.glob(f"**/{split}/*.npy"))

        if len(self.patch_files) == 0:
            raise FileNotFoundError(
                f"No .npy patches found under {root}/**/{split}/\n"
                "Run utils/dataset/slice.py (or scripts/preprocess.sh) first."
            )

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Tensor:
        """
        Returns:
            Tensor of shape (H, W, C) == (patch_size, patch_size, input_channels)
            dtype: float32, values in [0, 1].

        Each patch is max-normalized on the fly (patch / patch.max()) so model
        inputs live in [0, 1] — this matches the sigmoid reconstruction head and
        keeps the MSE term coherent across datasets with different reflectance
        scales.
        """
        patch: np.ndarray = np.load(self.patch_files[idx]).astype(np.float32)   # (H, W, C)
        patch_max = float(patch.max())
        if patch_max > 0:
            patch = patch / patch_max
        return torch.from_numpy(patch)                         # no copy if C-contiguous


class IIRSDataset(HSIPatchDataset):
    """HSIPatchDataset rooted at DATASETS["IIRS"]["processed_root"]."""

    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["IIRS"]["processed_root"], split)


class M3Dataset(HSIPatchDataset):
    """HSIPatchDataset rooted at DATASETS["M3"]["processed_root"]."""

    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["M3"]["processed_root"], split)


class AVIRISDataset(HSIPatchDataset):
    """HSIPatchDataset rooted at DATASETS["AVIRIS"]["processed_root"]."""

    def __init__(self, split: str, processed_root: str = None):
        super().__init__(processed_root or DATASETS["AVIRIS"]["processed_root"], split)


class CRIMSDataset(HSIPatchDataset):
    """HSIPatchDataset rooted at DATASETS["CRIMS"]["processed_root"]."""

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
    batch_size: int = settings.batch_size,
    shuffle: bool = True,
    num_workers: int = settings.num_workers,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Convenience factory for the per-dataset HSIPatchDataset subclasses.

    Args:
        dataset        : "IIRS" | "M3" | "AVIRIS" (case-insensitive)
        split          : 'train' | 'valid' | 'test'
        processed_root : override the dataset's default processed root
        batch_size     : samples per batch
        shuffle        : True for training, False for evaluation
        num_workers    : DataLoader worker processes
        pin_memory     : speeds up CPU to GPU transfer when True

    Returns:
        torch.utils.data.DataLoader whose batches are (B, H, W, C) float32
        tensors (C is dataset-specific — see utils.config.DATASETS).
    """
    key = dataset.upper()
    if key not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASET_CLASSES)}.")

    ds = DATASET_CLASSES[key](split, processed_root=processed_root)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # drop_last avoids a size-1 batch at the end of the training set which
        # can cause issues with BatchNorm if ever added; safe to set for training.
        drop_last=(split == "train"),
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=False,
    )
    return loader
