"""
modules/metrics.py
------------------
The single implementation of PSNR and SSIM for the whole repo.

WHY THIS EXISTS
===============
There used to be two incompatible SSIMs:

  * ``inference/inference.py`` used a GLOBAL single-scale SSIM — one mean and one
    variance per sample over the entire flattened cube, no windowing at all.
  * ``notebooks/*.ipynb`` used a proper 11x11 Gaussian-windowed, per-band SSIM.

They are not the same statistic and their numbers were never comparable, so a
notebook SSIM and a script SSIM for the same checkpoint disagreed for reasons
nobody had written down. This module keeps the windowed version (the one that
matches the literature) and every caller now imports it.

CONVENTIONS
===========
* Inputs are channels-last ``(B, H, W, C)``, the repo-wide layout. Pass
  ``channels_last=False`` for ``(B, C, H, W)``.
* ``psnr``/``ssim`` return **0-dim tensors on the input device** so callers can
  accumulate them on-GPU without a sync (train/train.py's one-``.item()``-per-
  epoch invariant). ``compute_psnr``/``compute_ssim`` are float-returning shims
  for the existing inference call sites.
* ``data_range=1.0`` because ``utils/dataset/pack.py`` normalises each patch by
  its own maximum. Note the data is NOT clipped to [0, 1] — real cubes carry
  negative reflectance (AVIRIS is ~12% negative) — so 1.0 is the positive peak,
  not a true dynamic range. Kept at 1.0 anyway so train-time and test-time
  numbers stay comparable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Gaussian windows are rebuilt per (channels, window, sigma, device, dtype) at
# most once. The notebook version rebuilt one on every call, inside the val loop.
_WINDOW_CACHE: dict = {}

# SSIM materialises several (B, C, H, W) temporaries. At B=32, C=456 that is
# ~240 MB each, and there are six or so live at the peak. Chunking the batch
# bounds it without changing the result (SSIM is averaged over samples).
_SSIM_CHUNK = 8


def _to_nchw(t: torch.Tensor, channels_last: bool) -> torch.Tensor:
    return t.permute(0, 3, 1, 2) if channels_last else t


def _gaussian_window(channels: int, window_size: int, sigma: float,
                     device, dtype) -> torch.Tensor:
    key = (channels, window_size, sigma, str(device), dtype)
    win = _WINDOW_CACHE.get(key)
    if win is None:
        coords = torch.arange(window_size, device=device, dtype=dtype)
        g = torch.exp(-((coords - window_size // 2) ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w2d = torch.outer(g, g)[None, None]
        win = w2d.expand(channels, 1, window_size, window_size).contiguous()
        _WINDOW_CACHE[key] = win
    return win


def psnr(x: torch.Tensor, recon: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio, dB. Returns a 0-dim tensor."""
    mse = F.mse_loss(recon.float(), x.float())
    # A clamp rather than a branch: an exact-zero MSE would give -inf and poison
    # an epoch accumulator, and branching on tensor value forces a device sync.
    mse = mse.clamp_min(1e-12)
    return 10.0 * torch.log10(torch.tensor(data_range ** 2, device=mse.device) / mse)


def ssim(x: torch.Tensor, recon: torch.Tensor, data_range: float = 1.0,
         window_size: int = 11, sigma: float = 1.5,
         channels_last: bool = True) -> torch.Tensor:
    """
    Mean Gaussian-windowed SSIM, computed per band and averaged.

    Returns a 0-dim tensor. Always evaluated in fp32: the val loop runs under a
    bf16 autocast, and the variance reduction here spans ~2M elements per sample,
    where bf16's 8-bit mantissa accumulates visible error.
    """
    a = _to_nchw(x, channels_last).float()
    b = _to_nchw(recon, channels_last).float()
    channels = a.shape[1]
    win = _gaussian_window(channels, window_size, sigma, a.device, a.dtype)
    pad = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    total = torch.zeros((), device=a.device, dtype=a.dtype)
    n = 0
    for i in range(0, a.shape[0], _SSIM_CHUNK):
        a_c, b_c = a[i:i + _SSIM_CHUNK], b[i:i + _SSIM_CHUNK]
        mu1 = F.conv2d(a_c, win, padding=pad, groups=channels)
        mu2 = F.conv2d(b_c, win, padding=pad, groups=channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1 = F.conv2d(a_c * a_c, win, padding=pad, groups=channels) - mu1_sq
        s2 = F.conv2d(b_c * b_c, win, padding=pad, groups=channels) - mu2_sq
        s12 = F.conv2d(a_c * b_c, win, padding=pad, groups=channels) - mu1_mu2
        m = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
        total = total + m.mean() * a_c.shape[0]
        n += a_c.shape[0]
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Float-returning shims for the existing inference/ call sites.
# ---------------------------------------------------------------------------
def compute_mse(x, recon) -> float:
    return F.mse_loss(recon.float(), x.float()).item()


def compute_psnr(x, recon, max_val: float = 1.0) -> float:
    return psnr(x, recon, data_range=max_val).item()


def compute_ssim(x, recon, max_val: float = 1.0) -> float:
    return ssim(x, recon, data_range=max_val).item()
