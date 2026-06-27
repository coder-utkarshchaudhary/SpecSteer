"""
utils/training/dataloader.py
-----------------------------
PyTorch Dataset and DataLoader for the preprocessed HSI patch files.

Expects a processed root directory with the layout produced by
utils/dataset/slice.py:

  data/processed/
    <folder_name>/
      train/   patch_00000.npy  patch_00001.npy  ...
      valid/   patch_00000.npy  ...
      test/    patch_00000.npy  ...
    <folder_name>/
      ...

Each .npy file contains a single (patch_size, patch_size, C) float32 array.
The dataset returns (H, W, C) float32 tensors — the shape expected by the
Dual-Stream PI-VAE's forward pass.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from utils.config import settings


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
            dtype: float32
        """
        patch: np.ndarray = np.load(self.patch_files[idx])   # (H, W, C)
        return torch.from_numpy(patch)                         # no copy if C-contiguous


def build_dataloader(
    processed_root: str,
    split: str,
    batch_size: int = settings.batch_size,
    shuffle: bool = True,
    num_workers: int = settings.num_workers,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Convenience factory for the HSIPatchDataset.

    Args:
        processed_root : path to data/processed/
        split          : 'train' | 'valid' | 'test'
        batch_size     : samples per batch
        shuffle        : True for training, False for evaluation
        num_workers    : DataLoader worker processes
        pin_memory     : speeds up CPU to GPU transfer when True

    Returns:
        torch.utils.data.DataLoader whose batches are (B, H, W, C) float32 tensors.
    """
    dataset = HSIPatchDataset(processed_root, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # drop_last avoids a size-1 batch at the end of the training set which
        # can cause issues with BatchNorm if ever added; safe to set for training.
        drop_last=(split == "train"),
    )
    return loader
