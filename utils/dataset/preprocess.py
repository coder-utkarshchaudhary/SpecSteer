"""
utils/dataset/preprocess.py
----------------------------
Per-sensor preprocessing for hyperspectral cubes: IIRS, M3, AVIRIS.

Pipeline (applied in this order by every preprocessor class):
    1. load       — read the sensor's native format into (C, H, W) float32,
                    per-band wavelengths (nm), and a (H, W) valid-pixel mask.
    2. normalize  — divide each pixel spectrum by the band nearest
                    settings.norm_target_nm (≈1500 nm, chosen per-cube from
                    that cube's own wavelength array); invalid pixels → 0.
    3. smooth     — Savitzky-Golay filter along the band axis; invalid
                    pixels stay 0 (smoothing is per-pixel along bands, so it
                    never leaks across the spatial dimensions).

Unlike the original IIRS-only pipeline, the **whole cube** is processed —
no band selection/cropping — except M3, whose native 85 bands are cropped
to 84 so the spectral branch's Conv1d/ConvTranspose1d stack round-trips
exactly (85 → 21 → 84 ≠ 85; 84 → 21 → 84). See utils/config.py DATASETS.

"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.signal import savgol_filter

from utils.config import DATASETS, settings


# ---------------------------------------------------------------------------
# ENVI .hdr parsing (shared by IIRS + M3)
# ---------------------------------------------------------------------------

def read_hdr(hdr_path) -> Dict[str, object]:
    """
    Parse an ENVI-format .hdr file.

    Returns a dict with at minimum:
        bands       (int)
        lines       (int)
        samples     (int)
        data_type   (int)   4 → float32
        interleave  (str)   'bsq' | 'bil' | 'bip'
        _blocks     (dict)  brace-delimited lists, e.g. _blocks['wavelength']
    """
    with open(hdr_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    # Capture brace blocks (e.g. `wavelength = { ... }`, possibly multi-line)
    # before stripping them from the flat key=value text below.
    block_pattern = re.compile(r"([A-Za-z_ ]+?)\s*=\s*\{([^}]*)\}", re.DOTALL)
    blocks: Dict[str, List[str]] = {}
    for key, body in block_pattern.findall(text):
        key_norm = key.strip().lower().replace(" ", "_")
        blocks[key_norm] = [v.strip() for v in body.split(",") if v.strip()]

    text_no_blocks = block_pattern.sub("", text)

    hdr: Dict[str, object] = {}
    for line in text_no_blocks.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower().replace(" ", "_")
        val = val.strip()
        if val:
            hdr[key] = val

    for int_key in ("bands", "lines", "samples", "header_offset", "data_type", "byte_order"):
        if int_key in hdr:
            try:
                hdr[int_key] = int(hdr[int_key])
            except ValueError:
                pass

    hdr["_blocks"] = blocks
    return hdr


def read_hdr_wavelengths(hdr: Dict[str, object]) -> Optional[np.ndarray]:
    """Extract the `wavelength = {...}` block from a parsed .hdr, if present."""
    raw = hdr.get("_blocks", {}).get("wavelength")
    if not raw:
        return None
    return np.array([float(v) for v in raw], dtype=np.float64)


# ---------------------------------------------------------------------------
# Base preprocessor: shared normalize / smooth logic
# ---------------------------------------------------------------------------

class BaseHSIPreprocessor:
    """
    Shared normalize/smooth pipeline for all sensors.

    Subclasses implement `load(source) -> (cube, wavelengths, valid_mask)`:
        cube        : (C, H, W) float32
        wavelengths : (C,) float, nanometers
        valid_mask  : (H, W) bool — True where the pixel spectrum is usable
    """

    def __init__(
        self,
        norm_target_nm: Optional[float] = None,
        savgol_window: Optional[int] = None,
        savgol_polyorder: Optional[int] = None,
        fill_value: Optional[float] = None,
    ):
        self.norm_target_nm = norm_target_nm if norm_target_nm is not None else settings.norm_target_nm
        self.savgol_window = savgol_window if savgol_window is not None else settings.savgol_window
        self.savgol_polyorder = savgol_polyorder if savgol_polyorder is not None else settings.savgol_polyorder
        self.fill_value = fill_value

    def load(self, source) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _valid_mask_from_fill(self, cube: np.ndarray) -> np.ndarray:
        """
        (H, W) bool mask: True where the pixel's full spectrum is finite and,
        if this sensor has a sentinel fill value, none of its bands equal it.
        """
        valid = np.all(np.isfinite(cube), axis=0)
        if self.fill_value is not None:
            valid &= np.all(cube != self.fill_value, axis=0)
        return valid

    def normalize(self, cube: np.ndarray, wavelengths: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Divide each pixel spectrum by the band nearest `norm_target_nm`.
        Invalid pixels are zeroed so they don't corrupt normalization stats
        downstream (and don't propagate through smoothing).
        """
        ref_idx = int(np.argmin(np.abs(wavelengths - self.norm_target_nm)))
        ref = cube[ref_idx : ref_idx + 1, :, :]   # (1, H, W)
        normalised = (cube / (ref + 1e-8)).astype(np.float32)
        normalised[:, ~valid_mask] = 0.0
        return normalised

    def smooth(self, cube: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Savitzky-Golay filter along the band axis (axis=0). This is a
        per-pixel operation across bands only, so it cannot leak fill
        contamination across the spatial dimensions.
        """
        smoothed = savgol_filter(
            cube, window_length=self.savgol_window, polyorder=self.savgol_polyorder, axis=0
        ).astype(np.float32)
        smoothed[:, ~valid_mask] = 0.0
        return smoothed

    def preprocess(self, source) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline for one raw source (folder / file, sensor-dependent).

        Returns:
            cube       : (C, H, W) float32 — normalised + smoothed
            valid_mask : (H, W) bool
        """
        cube, wavelengths, valid_mask = self.load(source)
        cube = self.normalize(cube, wavelengths, valid_mask)
        cube = self.smooth(cube, valid_mask)
        return cube, valid_mask


# ---------------------------------------------------------------------------
# IIRS — *_rfl_*.qub (ENVI BSQ, float32)
# ---------------------------------------------------------------------------

class IIRSPreprocessor(BaseHSIPreprocessor):
    """
    source = path to one IIRS acquisition sub-folder (contains
    *_rfl_d18_srd.qub + sibling .hdr).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("fill_value", DATASETS["IIRS"]["fill_value"])
        super().__init__(**kwargs)

    @staticmethod
    def _find_rfl_files(folder) -> Tuple[str, str]:
        folder_path = Path(folder)
        qub_files = list(folder_path.glob("*_rfl_*.qub"))
        hdr_files = list(folder_path.glob("*_rfl_*.hdr"))

        if not qub_files:
            raise FileNotFoundError(f"No *_rfl_*.qub file found in {folder}")
        if not hdr_files:
            raise FileNotFoundError(f"No *_rfl_*.hdr file found in {folder}")

        qub_path = qub_files[0]
        matching_hdr = [h for h in hdr_files if h.stem == qub_path.stem]
        hdr_path = matching_hdr[0] if matching_hdr else hdr_files[0]
        return str(qub_path), str(hdr_path)

    def load(self, folder) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        qub_path, hdr_path = self._find_rfl_files(folder)
        hdr = read_hdr(hdr_path)

        bands = int(hdr.get("bands", 256))
        lines = int(hdr.get("lines", 0))
        samples = int(hdr.get("samples", 0))
        if lines == 0 or samples == 0:
            raise ValueError(
                f"HDR parsing failed for {hdr_path}: got bands={bands}, "
                f"lines={lines}, samples={samples}"
            )

        cube = np.fromfile(qub_path, dtype=np.float32)
        expected = bands * lines * samples
        if cube.size != expected:
            raise ValueError(
                f"{qub_path}: expected {expected} float32 values "
                f"({bands}×{lines}×{samples}), got {cube.size}"
            )
        cube = cube.reshape(bands, lines, samples)   # BSQ: (C, H, W)

        wavelengths = read_hdr_wavelengths(hdr)
        if wavelengths is None:
            raise ValueError(f"No wavelength block found in {hdr_path}")

        valid_mask = self._valid_mask_from_fill(cube)
        return cube, wavelengths, valid_mask


# ---------------------------------------------------------------------------
# M3 — *_rfl.img (ENVI BIL, float32)
# ---------------------------------------------------------------------------

class M3Preprocessor(BaseHSIPreprocessor):
    """
    source = path to one M3 `*_v01_rfl.img` file (sibling `.hdr` required).

    Cropped to `DATASETS["M3"]["crop_bands"]` (84) bands so the spectral
    branch's Conv1d/ConvTranspose1d stack round-trips exactly.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("fill_value", DATASETS["M3"]["fill_value"])
        super().__init__(**kwargs)
        self.crop_bands = DATASETS["M3"]["crop_bands"]

    @staticmethod
    def _find_rfl_files(img_path) -> Tuple[str, str]:
        img_path = Path(img_path)
        hdr_path = img_path.with_suffix(".hdr")
        if not hdr_path.exists():
            raise FileNotFoundError(f"No sibling .hdr for {img_path}")
        return str(img_path), str(hdr_path)

    def load(self, source) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_path, hdr_path = self._find_rfl_files(source)
        hdr = read_hdr(hdr_path)

        bands = int(hdr.get("bands", 0))
        lines = int(hdr.get("lines", 0))
        samples = int(hdr.get("samples", 0))
        if bands == 0 or lines == 0 or samples == 0:
            raise ValueError(
                f"HDR parsing failed for {hdr_path}: got bands={bands}, "
                f"lines={lines}, samples={samples}"
            )

        raw = np.fromfile(img_path, dtype=np.float32)
        expected = bands * lines * samples
        if raw.size != expected:
            raise ValueError(
                f"{img_path}: expected {expected} float32 values "
                f"({bands}×{lines}×{samples}), got {raw.size}"
            )

        # BIL on disk: (lines, bands, samples). `flip` in the .hdr is a
        # display-orientation hint only — irrelevant for reconstruction
        # training, so it's intentionally ignored here.
        cube = raw.reshape(lines, bands, samples).transpose(1, 0, 2)   # (C, H, W)
        cube = np.ascontiguousarray(cube)

        wavelengths = read_hdr_wavelengths(hdr)
        if wavelengths is None:
            raise ValueError(f"No wavelength block found in {hdr_path}")

        if self.crop_bands is not None and self.crop_bands < cube.shape[0]:
            cube = np.ascontiguousarray(cube[: self.crop_bands])
            wavelengths = wavelengths[: self.crop_bands]

        valid_mask = self._valid_mask_from_fill(cube)
        return cube, wavelengths, valid_mask


# ---------------------------------------------------------------------------
# AVIRIS — *_RFL_ORT.nc (NetCDF-4 / HDF5)
# ---------------------------------------------------------------------------

class AVIRISPreprocessor(BaseHSIPreprocessor):
    """
    source = path to one AVIRIS `*_RFL_ORT.nc` file.

    Orthorectified products place the flight-line swath diagonally inside a
    rectangular grid, so large off-swath regions (typically the corners) are
    filled with `_FillValue = -9999`. `valid_mask` marks those pixels invalid
    so `slice.py` can drop patches that are mostly fill.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("fill_value", DATASETS["AVIRIS"]["fill_value"])
        super().__init__(**kwargs)

    def load(self, source) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(source, "r") as f:
            cube = f["reflectance/reflectance"][:].astype(np.float32)      # (C, H, W)
            wavelengths = f["reflectance/wavelength"][:].astype(np.float64)

        valid_mask = self._valid_mask_from_fill(cube)
        return cube, wavelengths, valid_mask


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREPROCESSORS = {
    "IIRS": IIRSPreprocessor,
    "M3": M3Preprocessor,
    "AVIRIS": AVIRISPreprocessor,
}


if __name__ == "__main__":
    def _smoke_test(name: str, source) -> None:
        print(f"--- {name}: {source} ---")
        proc = PREPROCESSORS[name]()
        cube, valid_mask = proc.preprocess(source)
        fill_fraction = 1.0 - float(valid_mask.mean())
        print(f"  cube shape: {cube.shape}  fill_fraction: {fill_fraction:.4f}")

    iirs_root = Path(DATASETS["IIRS"]["raw_root"])
    m3_root = Path(DATASETS["M3"]["raw_root"])
    aviris_root = Path(DATASETS["AVIRIS"]["raw_root"])

    if iirs_root.exists():
        candidates = sorted(p for p in iirs_root.iterdir() if p.is_dir())
        if candidates:
            _smoke_test("IIRS", candidates[0])

    if m3_root.exists():
        candidates = sorted(m3_root.glob("*_rfl.img"))
        if candidates:
            _smoke_test("M3", candidates[0])

    if aviris_root.exists():
        candidates = sorted(aviris_root.glob("*.nc"))
        if candidates:
            _smoke_test("AVIRIS", candidates[0])
