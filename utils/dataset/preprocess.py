"""
utils/dataset/preprocess.py
----------------------------
Vectorised preprocessing for Chandrayaan-2 IIRS hyperspectral cubes.

Pipeline (applied in this order):
  1. load_cube       — read *_rfl_d18_srd.qub (BSQ float32) + sibling .hdr
  2. select_bands    — keep bands [band_start : band_end]  (default 7:115 = 108 bands)
  3. normalize       — divide each pixel spectrum by its value at norm_band_idx (≈1500 nm)
  4. smooth          — Savitzky-Golay filter along the band axis

Skipped (already applied upstream):
  - Radiance → reflectance conversion  (input files are *_rfl_* products)
  - OSF-band interpolation             (out of scope for this pipeline)

The public entry point is `preprocess_cube(folder_path)` which returns a
(C, H, W) float32 ndarray where C = band_end - band_start (default 108).

Reference: docs/file_processing.py
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.signal import savgol_filter

from utils.config import settings


# ---------------------------------------------------------------------------
# HDR parsing
# ---------------------------------------------------------------------------

def read_hdr(hdr_path: str) -> Dict[str, object]:
    """
    Parse an ENVI-format .hdr file.

    Returns a dict with at minimum:
        bands       (int)
        lines       (int)
        samples     (int)
        data_type   (int)   4 → float32
        interleave  (str)   'bsq' | 'bil' | 'bip'
    """
    with open(hdr_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    # Strip multi-line brace blocks (e.g. wavelength = { ... }) so they don't
    # confuse the simple key = value parser below.
    text_no_blocks = re.sub(r"\{[^}]*\}", "", text, flags=re.DOTALL)

    hdr: Dict[str, object] = {}
    for line in text_no_blocks.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower().replace(" ", "_")
        val = val.strip()
        hdr[key] = val

    # Cast numeric fields
    for int_key in ("bands", "lines", "samples", "header_offset", "data_type", "byte_order"):
        if int_key in hdr:
            try:
                hdr[int_key] = int(hdr[int_key])
            except ValueError:
                pass

    return hdr


# ---------------------------------------------------------------------------
# Cube loading
# ---------------------------------------------------------------------------

def _find_rfl_files(folder: str) -> Tuple[str, str]:
    """
    Locate the *_rfl_*.qub and its sibling .hdr inside *folder*.
    Raises FileNotFoundError if either is missing.
    """
    folder_path = Path(folder)
    qub_files = list(folder_path.glob("*_rfl_*.qub"))
    hdr_files = list(folder_path.glob("*_rfl_*.hdr"))

    if not qub_files:
        raise FileNotFoundError(f"No *_rfl_*.qub file found in {folder}")
    if not hdr_files:
        raise FileNotFoundError(f"No *_rfl_*.hdr file found in {folder}")

    # If multiple matches exist, prefer the one whose stem matches the qub stem
    qub_path = qub_files[0]
    # Find the hdr whose stem matches the qub stem
    matching_hdr = [h for h in hdr_files if h.stem == qub_path.stem]
    hdr_path = matching_hdr[0] if matching_hdr else hdr_files[0]

    return str(qub_path), str(hdr_path)


def load_cube(folder: str) -> np.ndarray:
    """
    Read the *_rfl_*.qub file from *folder* and return it as a (C, H, W)
    float32 ndarray.

    The .qub is raw binary, BSQ-interleaved, float32.  Shape is taken from
    the sibling .hdr (bands × lines × samples).
    """
    qub_path, hdr_path = _find_rfl_files(folder)
    hdr = read_hdr(hdr_path)

    bands = int(hdr.get("bands", 256))
    lines = int(hdr.get("lines", 0))
    samples = int(hdr.get("samples", 0))

    if lines == 0 or samples == 0:
        raise ValueError(
            f"HDR parsing failed for {hdr_path}: got bands={bands}, "
            f"lines={lines}, samples={samples}"
        )

    # BSQ: all pixels of band 0, then band 1, etc.
    cube = np.fromfile(qub_path, dtype=np.float32)
    expected = bands * lines * samples
    if cube.size != expected:
        raise ValueError(
            f"{qub_path}: expected {expected} float32 values "
            f"({bands}×{lines}×{samples}), got {cube.size}"
        )

    cube = cube.reshape(bands, lines, samples)   # (C, H, W) — BSQ order
    return cube


# ---------------------------------------------------------------------------
# Preprocessing steps
# ---------------------------------------------------------------------------

def select_bands(
    cube: np.ndarray,
    band_start: int = settings.band_start,
    band_end: int = settings.band_end,
) -> np.ndarray:
    """
    Slice the cube along the band axis (axis 0).

    Args:
        cube       : (C_full, H, W) float32
        band_start : first band index, inclusive  (default 7)
        band_end   : last  band index, exclusive  (default 115)

    Returns:
        (C_sub, H, W) float32  where C_sub = band_end - band_start (default 108)
    """
    return cube[band_start:band_end, :, :].copy()


def normalize(
    cube: np.ndarray,
    norm_band_idx: int = settings.norm_band_idx,
) -> np.ndarray:
    """
    Normalise each pixel's spectrum by the reflectance at *norm_band_idx*
    (index within the already-selected sub-band array, corresponding to ≈1500 nm).

    Vectorised broadcast:  cube / ref[np.newaxis, :, :]  where ref is
    the single-band slice at norm_band_idx.

    Args:
        cube          : (C, H, W) float32  (after select_bands)
        norm_band_idx : band index in *cube* that is the normalisation reference

    Returns:
        (C, H, W) float32  — normalised cube
    """
    ref = cube[norm_band_idx:norm_band_idx + 1, :, :]   # (1, H, W) for broadcast
    normalised = cube / (ref + 1e-8)
    return normalised.astype(np.float32)


def smooth(
    cube: np.ndarray,
    window: int = settings.savgol_window,
    polyorder: int = settings.savgol_polyorder,
) -> np.ndarray:
    """
    Apply a Savitzky-Golay smoothing filter along the spectral (band) axis.

    Args:
        cube      : (C, H, W) float32
        window    : Savitzky-Golay window length (must be odd, > polyorder)
        polyorder : polynomial order

    Returns:
        (C, H, W) float32  — smoothed cube
    """
    smoothed = savgol_filter(
        cube,
        window_length=window,
        polyorder=polyorder,
        axis=0,          # band axis
    )
    return smoothed.astype(np.float32)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess_cube(folder: str) -> np.ndarray:
    """
    Full preprocessing pipeline for one data folder.

    Steps:
      1. load_cube    → (256, H, W) float32
      2. select_bands → (108, H, W) float32
      3. normalize    → divide each spectrum by its 1500-nm value
      4. smooth       → Savitzky-Golay filter along the band axis

    Memory note: each raw cube is ~3.6 GB (float32).  Process one at a time.

    Args:
        folder : path to one of the `data/original/<name>/` directories

    Returns:
        (C, H, W) float32  where C = settings.band_end - settings.band_start (108)
    """
    cube = load_cube(folder)          # (256, H, W)
    cube = select_bands(cube)         # (108, H, W)
    cube = normalize(cube)            # (108, H, W) — normalised
    cube = smooth(cube)               # (108, H, W) — smoothed
    return cube
