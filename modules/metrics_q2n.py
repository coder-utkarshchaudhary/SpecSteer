"""
modules/metrics_q2n.py
-----------------------
Q2^n -- the hypercomplex generalisation of the Universal Image Quality Index
(Wang & Bovik 2002) to n-band images via Cayley-Dickson algebras (Garzelli &
Nencini, IEEE GRSL 2009; the original Q4 is this construction at n=2,
i.e. quaternions, for 4-band imagery).

Kept OUT of modules/metrics.py deliberately: PSNR/SSIM in that file are
train-loop-safe (0-dim tensors, no device sync, no masking) because they are
genuinely used during training; Q2^n is eval-only, has no training-loop
caller, and its hypercomplex algebra + band-padding policy would bloat that
file's otherwise simple, fast implementations.

BAND PADDING (a REGISTERED decision, inference/preregistration.yaml's
recon_metrics_ext section): the Cayley-Dickson construction requires the
number of components -- here, spectral bands -- to be a power of 2. This
repo's datasets are IIRS=256 (native), M3=84, AVIRIS=424, CRIMS=456 (all
need padding). Bands are zero-padded up to the next power of 2, matching the
reference MATLAB q2n.m convention. |mu| and sigma are norms over components,
so all-zero padded bands leave them unchanged; whether the cross term
|sigma_{z,zhat}| is similarly unaffected is NOT proven here -- this is the
reference implementation's own convention, not a proof of invariance.

CONSEQUENCE FOR COMPARABILITY: the padding fraction differs per dataset
(IIRS 0%, CRIMS 11%, AVIRIS 17%, M3 34%). Q2^n values are therefore only
comparable WITHIN a dataset (across models), never ACROSS datasets -- the
same discipline this repo already applies to raw (non-valid-masked) SAM on
CRIMS. Do not rank datasets against each other on this metric.

BLOCK SIZE: the literature-standard 32x32 non-overlapping block (Vivone et
al.'s pansharpening toolbox default). This repo's patches are 64x64, so
that's exactly 4 blocks per patch -- no remainder handling needed. Each
block's hypercomplex covariance is estimated from only 1024 pixels against
up to 512 padded bands: a thin estimate by literature standards (which
assume full scenes, not 64x64 patches). Reported as a known caveat, not
silently smoothed over.

MEMORY, the reason this file exists as carefully as it does: the covariance
term requires a full Cayley-Dickson (hypercomplex) product of two length-C
vectors, which is O(C^2) in FLOPs. A naive implementation might materialise
an explicit (..., C, C) multiplication table to get there -- exactly the
"materialise a big tensor when only a small accumulator is needed" mistake
that caused the 2026-09-17 AVIRIS OOM in inference/probes.py. The RECURSIVE
Cayley-Dickson formula below never does this: each recursion level splits
the band dimension in half and combines two half-length hypercomplex
products, so every intermediate tensor stays O(chunk_size * block_h *
block_w * C) -- the same order of magnitude as the input patch itself, not
its square. The cost lands entirely in FLOPs (recursion depth
log2(C_padded), each level roughly doubling the elementwise work), not
memory, and is meaningfully more expensive than SSIM -- budget wall-clock
accordingly (see docs/preregistration.md's methodology notes and the
plan's Verification section for a single-cell timing check before a full
sweep).

Usage
=====
    from modules.metrics_q2n import q2n, compute_q2n
    value = compute_q2n(x, recon)   # x, recon: (B, H, W, C), channels-last
"""

from __future__ import annotations

import torch

_Q2N_CHUNK = 8         # patches per chunk; matches modules/metrics.py's _SSIM_CHUNK
_Q2N_BLOCK = 32        # literature-standard non-overlapping block size


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pad_bands(x: torch.Tensor, c_pad: int) -> torch.Tensor:
    """Zero-pad the last (band) dimension of a channels-last tensor to c_pad."""
    c = x.shape[-1]
    if c == c_pad:
        return x
    pad_shape = list(x.shape[:-1]) + [c_pad - c]
    zeros = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, zeros], dim=-1)


def _cd_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Cayley-Dickson product of two hypercomplex numbers, each represented as
    a real vector of length C = 2^n along the last dimension (the standard
    doubling construction: a hypercomplex number of order n is a pair of
    order-(n-1) numbers). Recursive, so every intermediate tensor stays the
    same shape as the inputs (O(C) elements), not O(C^2) -- see the module
    docstring. Base case n=0 (C=1) is ordinary real multiplication.

    a, b: (..., C) real tensors, C a power of 2. Returns (..., C).
    """
    c = a.shape[-1]
    if c == 1:
        return a * b
    half = c // 2
    a1, a2 = a[..., :half], a[..., half:]
    b1, b2 = b[..., :half], b[..., half:]
    # (a1 + a2*e)(b1 + b2*e) = (a1*b1 - conj(b2)*a2) + (b2*a1 + a2*conj(b1))*e
    # using the standard Cayley-Dickson doubling with conjugation folded in
    # via _cd_conjugate below (kept separate so it is reusable and testable).
    b1c = _cd_conjugate(b1)
    a1c = _cd_conjugate(a1)
    term1 = _cd_multiply(a1, b1) - _cd_multiply(_cd_conjugate(b2), a2)
    term2 = _cd_multiply(b2, a1) + _cd_multiply(a2, b1c)
    return torch.cat([term1, term2], dim=-1)


def _cd_conjugate(a: torch.Tensor) -> torch.Tensor:
    """Hypercomplex conjugate: negate every imaginary component, keep the
    first (real) component. a: (..., C), C a power of 2 (C=1 is a no-op)."""
    if a.shape[-1] == 1:
        return a
    out = -a
    out = out.clone()
    out[..., 0] = a[..., 0]
    return out


def _cd_norm_sq(a: torch.Tensor) -> torch.Tensor:
    """Squared modulus |a|^2 = sum of squared components. (..., C) -> (...)."""
    return (a ** 2).sum(dim=-1)


def _q2n_blocks(x: torch.Tensor, recon: torch.Tensor, c_pad: int,
                block: int) -> torch.Tensor:
    """
    Q2^n per non-overlapping block, for one chunk of patches.

    x, recon: (chunk, H, W, C) channels-last, NOT yet band-padded.
    Returns (chunk, n_blocks_h * n_blocks_w) -- one Q2^n value per block.
    """
    B, H, W, C = x.shape
    xp = _pad_bands(x, c_pad)
    rp = _pad_bands(recon, c_pad)

    nbh, nbw = H // block, W // block
    # (B, nbh, block, nbw, block, C) -> (B, nbh*nbw, block*block, C)
    xb = xp[:, :nbh * block, :nbw * block, :] \
        .reshape(B, nbh, block, nbw, block, c_pad) \
        .permute(0, 1, 3, 2, 4, 5) \
        .reshape(B, nbh * nbw, block * block, c_pad)
    rb = rp[:, :nbh * block, :nbw * block, :] \
        .reshape(B, nbh, block, nbw, block, c_pad) \
        .permute(0, 1, 3, 2, 4, 5) \
        .reshape(B, nbh * nbw, block * block, c_pad)

    n = block * block
    mu_x = xb.mean(dim=2)                              # (B, nblk, C)
    mu_r = rb.mean(dim=2)
    dx = xb - mu_x.unsqueeze(2)                         # (B, nblk, n, C)
    dr = rb - mu_r.unsqueeze(2)

    var_x = _cd_norm_sq(dx).sum(dim=2) / max(n - 1, 1)  # (B, nblk)
    var_r = _cd_norm_sq(dr).sum(dim=2) / max(n - 1, 1)
    sigma_x = torch.sqrt(var_x.clamp_min(0))
    sigma_r = torch.sqrt(var_r.clamp_min(0))

    # Hypercomplex cross-covariance: full Cayley-Dickson product per pixel,
    # then summed over the block and normalised. Only the modulus is used,
    # but the full product must be computed to get it (does not factor into
    # |sigma_x||sigma_r|).
    cov_terms = _cd_multiply(dx, _cd_conjugate(dr))     # (B, nblk, n, C)
    cov = cov_terms.sum(dim=2) / max(n - 1, 1)           # (B, nblk, C)
    sigma_xr = torch.sqrt(_cd_norm_sq(cov).clamp_min(0))

    mu_x_mod = torch.sqrt(_cd_norm_sq(mu_x).clamp_min(0))
    mu_r_mod = torch.sqrt(_cd_norm_sq(mu_r).clamp_min(0))

    correlation = sigma_xr / (sigma_x * sigma_r).clamp_min(1e-12)
    contrast = (2 * sigma_x * sigma_r) / (sigma_x ** 2 + sigma_r ** 2).clamp_min(1e-12)
    luminance = (2 * mu_x_mod * mu_r_mod) / (mu_x_mod ** 2 + mu_r_mod ** 2).clamp_min(1e-12)
    return correlation * contrast * luminance            # (B, nblk)


def q2n(x: torch.Tensor, recon: torch.Tensor, block_size: int = _Q2N_BLOCK,
        chunk: int = _Q2N_CHUNK) -> torch.Tensor:
    """
    Q2^n (Garzelli & Nencini 2009), mean over blocks and patches.

    x, recon: (B, H, W, C) channels-last. Bands are zero-padded to the next
    power of 2 (see module docstring for why, and the comparability caveat
    that travels with the result). Chunked over the batch axis -- every
    intermediate tensor stays O(chunk * n_blocks * block^2 * C_pad), never
    O(C_pad^2), so this is memory-safe regardless of chunk size; chunk only
    trades wall-clock parallelism, unlike SSIM/SID/SCC where chunk size is
    the actual memory safety valve.

    Returns a 0-dim tensor (sample-count-weighted mean over all blocks in
    all chunks -- exact, not an approximation, by the same weighted-mean
    argument as modules/metrics.py's ssim()/scc()).
    """
    B, H, W, C = x.shape
    c_pad = _next_pow2(C)
    nbh, nbw = H // block_size, W // block_size
    if nbh == 0 or nbw == 0:
        raise ValueError(f"block_size={block_size} exceeds patch dims {H}x{W}")

    total = torch.zeros((), device=x.device, dtype=torch.float32)
    n = 0
    for i in range(0, B, chunk):
        xc = x[i:i + chunk].float()
        rc = recon[i:i + chunk].float()
        q_blocks = _q2n_blocks(xc, rc, c_pad, block_size)   # (chunk, nblk)
        total = total + q_blocks.sum()
        n += q_blocks.numel()
    return total / max(n, 1)


def compute_q2n(x, recon, block_size: int = _Q2N_BLOCK) -> float:
    return q2n(x, recon, block_size=block_size).item()
