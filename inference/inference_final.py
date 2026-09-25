"""
inference/inference_final.py
-----------------------------
The FROZEN, paper-locking evaluation run. Training is frozen; this script
produces the exact numbers the paper cites, in five CSVs, one row per
(model, dataset, loss). It is a separate, self-contained pipeline from
inference.py/probes.py/downstream.py/verdict.py/aggregate.py — those were
built for iterating on a 512-patch preregistered sample across many small
process invocations; this evaluates the COMPLETE valid+test split, in one
process, and never touches results/ (the exploratory sweep's output dir).

Scope (locked, see the session's plan):
    datasets : IIRS, AVIRIS, CRIMS                          (no M3)
    models   : vae-our-nl, vae-standard,
               vae-1d-pixelwise, vae-3d-spatio-spectral      (no vae-our, no SpecViT variants)
    losses   : vae-our-nl -> physics only (PHYSICS_ONLY);
               the 3 baselines -> {physics, standard}
    -> 3 datasets x (1 + 3x2) = 21 cells, one row per cell in every CSV.

    checkpoint selection : select=sam, always (matches every other part of
        this project). Checkpoint SEED is a separate, explicit axis: every
        cell is evaluated against BOTH the seed-67 and the seed-69 trained
        checkpoint, and every number is averaged over the two — consistent
        with how every existing paper table was built. `resolve_checkpoint`
        is always called with an explicit seed=67 / seed=69, never the
        seedless auto-detect path (which raises when >1 seed is on disk,
        which is exactly this project's situation).

    inference population : the COMBINED valid+test split (ConcatDataset of
        two PackedPatchDataset instances, memmapped — nothing is read until
        actually iterated), FULL SIZE, no subsampling, for every experiment.

Five output CSVs (under --out-dir, default results/final/):
    model-validity-probes.csv   P1 (trivial-floor lift, reported against BOTH
                                 trivial predictors — the best zero-rate mean
                                 and the per-patch mean — as PSNR dB, SSIM,
                                 absolute SAM rad AND relative SAM) + P2 (latent-rate
                                 audit) + P3 (posterior collapse) + P4
                                 (spatial-reliance shuffle) — validity
                                 diagnostics, no pass/fail verdict.
    reconstruction-quality.csv  MSE, SAM (rad), SAM-valid, PSNR, SSIM, SID,
                                 SCC, Q2^n on the full inference set.
    noise-recovery.csv          SAM/PSNR of the reconstruction after Gaussian
                                 noise is added to the encoded latent, at
                                 sigma in {0.1, 0.5, 1.0}, averaged over 2
                                 checkpoint seeds x 3 RNG seeds (67, 69, 1234)
                                 per sigma.
    chemical-interpolation.csv  Jaggedness + path length of latent-space
                                 interpolation between patch pairs, plus an
                                 eigenvalue-entropy effective-occupancy score
                                 per latent stream (a collapse/hallucination-
                                 risk indicator: a low-occupancy stream can
                                 only drive a handful of independent output
                                 directions, so any apparent spectral
                                 diversity in its decoded output is more
                                 likely templated than genuinely latent-driven).
    missing-pixel-recovery.csv  Whole-patch SAM before vs after zeroing a
                                 random 10% of a patch's pixels (all bands),
                                 averaged over 2 checkpoint seeds x 3 RNG
                                 seeds (67, 69, 1234) for the mask draw.

A running Telegram log (utils.notify.TelegramNotifier) accompanies every
stage; see send_document() calls at the end of each experiment and the
step-4.2 "lift" summary right after reconstruction-quality.csv completes.

COST NOTE (read before launching): with the checkpoint-seed averaging and
the full-set-no-cap policy, this script makes roughly 8 full streaming
passes over each dataset's combined valid+test split per (cell, checkpoint
seed) -- 21 cells x 2 seeds x 8 passes = ~336 full-set passes total across
the whole run. Run --dry-run first to see the exact cell grid and patch
counts before committing to the real pass.

Run from the repo root with PYTHONPATH set:
    PYTHONPATH=. python inference/inference_final.py --dry-run
    PYTHONPATH=. python inference/inference_final.py --ckpt-dir model --out-dir results/final
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
# Repo root must come FIRST: this file's own directory is sys.path[0] when run
# as a script, and it contains inference.py, which would otherwise shadow the
# `inference` package (same guard as inference/probes.py).
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from modules.losses import spectral_angle_mapper_loss  # noqa: E402
from modules.metrics import (  # noqa: E402
    compute_mse, compute_psnr, compute_ssim, compute_sid,
    compute_sid_clamped_frac, compute_scc,
)
from modules.metrics_q2n import compute_q2n  # noqa: E402
from modules.registry import PHYSICS_ONLY, resolve_checkpoint  # noqa: E402
from utils.config import DATASETS, apply_dataset, settings  # noqa: E402
from utils.hyperparams import apply_cli_overrides, apply_hyperparams, load_hyperparams  # noqa: E402
from utils.training.dataloader import PackedPatchDataset  # noqa: E402
from utils.notify import TelegramNotifier  # noqa: E402

from inference.inference import (  # noqa: E402
    load_model, sam_valid_sums, compute_psnr_from_mse, _recon_metrics_ext_cfg,
)
from inference.downstream import add_latent_noise, lerp_latents  # noqa: E402
from inference.probes import (  # noqa: E402
    load_prereg, train_band_statistics, p1_shared_floors, p1_report,
    metrics as small_sample_metrics, _sam_per_pixel,
)


# ---------------------------------------------------------------------------
# Fixed scope (see module docstring)
# ---------------------------------------------------------------------------

DATASETS_SCOPE = ["IIRS", "AVIRIS", "CRIMS"]
MODELS_SCOPE = ["vae-our-nl", "vae-standard", "vae-1d-pixelwise", "vae-3d-spatio-spectral"]
CHECKPOINT_SEEDS = [67, 69]
RNG_SEEDS = [67, 69, 1234]          # noise / mask draws — unrelated to which checkpoint is loaded
SIGMAS = [0.01, 0.05, 0.2]
LEGACY_CACHE_SIGMAS = [0.01, 0.05, 0.2]
PIXEL_MASK_FRACTION = 0.10
N_INTERP_PAIRS = 100
N_ALPHA = 11
INTERP_PIXEL = (32, 32)
P1_FLOOR_SAMPLE = 2000              # subsample size for the trivial-floor characterisation



def noise_columns(sigmas: list[float]) -> list[str]:
    """noise-recovery.csv header, derived from the sigma list so the two can't drift."""
    cols = ["dataset", "model", "loss"]
    for sigma in sigmas:
        cols += [f"sam_recovery_s{sigma}", f"sam_rad_recovery_s{sigma}", f"psnr_recovery_s{sigma}"]
    return cols


CSV_COLUMNS = {"noise-recovery.csv": noise_columns(SIGMAS)}

# Every CSV except dataset-floors.csv gets rows per cell; each is cached per
# cell as a LIST of rows so a resumed run rebuilds the CSVs exactly.
PER_CELL_CSVS = ["noise-recovery.csv"]
# Bump whenever a cached cell's schema or meaning changes, so a stale
# results/final/.cache/*.json from an older run is recomputed, not reused.
CACHE_VERSION = 4


# ---------------------------------------------------------------------------
# Telegram helpers — thin wrapper so --no-telegram is a single guard
# ---------------------------------------------------------------------------

class Logger:
    """Console + Telegram, one object threaded through the whole run."""

    def __init__(self, enabled: bool):
        self.tg = TelegramNotifier(enabled=enabled) if enabled else None

    def send(self, text: str) -> None:
        print(text)
        if self.tg is not None:
            try:
                self.tg.send(text)
            except Exception as e:  # noqa: BLE001 — Telegram must never break the run
                print(f"  (telegram send failed: {e})")

    def send_pre(self, header: str, body: str) -> None:
        # Escaped: tracebacks carry "<module>", which Telegram's HTML parse
        # mode rejects outright — the failure message would never arrive.
        self.send(f"{header}\n<pre>{html.escape(body)}</pre>")

    def send_pre_long(self, header: str, body: str, limit: int = 3500) -> None:
        """Like send_pre, but split on line boundaries into several messages,
        each with its own <pre> — the notifier's own splitter would cut one
        <pre> in half, and Telegram rejects the unbalanced halves."""
        chunk: list[str] = []
        size, part = 0, 1
        for line in body.splitlines():
            if chunk and size + len(line) + 1 > limit:
                self.send_pre(f"{header} ({part})", "\n".join(chunk))
                chunk, size, part = [], 0, part + 1
            chunk.append(line)
            size += len(line) + 1
        if chunk:
            self.send_pre(f"{header} ({part})" if part > 1 else header, "\n".join(chunk))

    def send_csv_as_text(self, path: Path, title: str, columns: list[tuple[str, str]],
                         group_by: str = "dataset") -> None:
        """
        Post a results CSV as fixed-width text, one message per dataset.
        Always sent, alongside send_document: a file attachment needs the
        direct bot credentials, and over the relay only a one-line
        '[attachment unavailable]' notice arrives — the numbers would never
        reach Telegram. columns = [(csv_column, short_header), ...].
        """
        if not path.is_file():
            return
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return

        def cell(v: str) -> str:
            try:
                return f"{float(v):.4g}"
            except (TypeError, ValueError):
                return (v or "")[:22]

        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r.get(group_by, ""), []).append(r)
        for g, grows in groups.items():
            table = [[h for _, h in columns]] + [[cell(r.get(c, "")) for c, _ in columns] for r in grows]
            widths = [max(len(row[j]) for row in table) for j in range(len(columns))]
            body = "\n".join("  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in table)
            self.send_pre_long(f"{title} — {g}", body)

    def send_document(self, path: Path, caption: str = "") -> None:
        print(f"  -> {path}")
        if self.tg is not None:
            try:
                if path.is_file() and path.stat().st_size > 0:
                    self.tg.send_document(path, caption=caption)
            except Exception as e:  # noqa: BLE001
                print(f"  (telegram send_document failed: {e})")


# ---------------------------------------------------------------------------
# Step 1 — packed-dataset audit
# ---------------------------------------------------------------------------

def audit_dataset(dataset: str, packed_root: Path, log: Logger) -> dict:
    """
    Read the {train,valid,test}.json sidecars + the cheap on-disk shape for
    each split. Hard-fails (raises) if valid.npy or test.npy is missing —
    build_dataset() falls back to the slow legacy per-patch tree silently
    otherwise, and a frozen run must never do that unnoticed.
    """
    report = {}
    for split in ("train", "valid", "test"):
        npy_path = packed_root / f"{split}.npy"
        json_path = packed_root / f"{split}.json"
        if split in ("valid", "test") and not npy_path.is_file():
            raise FileNotFoundError(
                f"{dataset}: missing packed shard {npy_path} — refusing to run "
                f"(build_dataset would silently fall back to the slow legacy "
                f"per-patch tree, which a frozen run must not do unnoticed)."
            )
        if not npy_path.is_file():
            report[split] = {"n_on_disk": 0, "n_available": None, "cap": None, "cap_seed": None}
            continue
        n_on_disk = int(np.load(npy_path, mmap_mode="r").shape[0])
        meta = json.loads(json_path.read_text()) if json_path.is_file() else {}
        report[split] = {
            "n_on_disk": n_on_disk,
            "n_available": meta.get("n_available"),
            "cap": meta.get("cap"),
            "cap_seed": meta.get("cap_seed"),
        }
    body = "\n".join(
        f"{split:5}: n={report[split]['n_on_disk']:>6,}"
        f"  available={report[split]['n_available']}"
        f"  cap={report[split]['cap']}  cap_seed={report[split]['cap_seed']}"
        for split in ("train", "valid", "test")
    )
    log.send_pre(f"Inference final - dataset audit - DONE - {dataset}", body)
    return report


# ---------------------------------------------------------------------------
# Step 2 — the combined inference set
# ---------------------------------------------------------------------------

def build_inference_set(dataset: str, packed_root: Path):
    """ConcatDataset(valid, test), memmapped — costs nothing until iterated."""
    ds_valid = PackedPatchDataset(packed_root, "valid")
    ds_test = PackedPatchDataset(packed_root, "test")
    combined = ConcatDataset([ds_valid, ds_test])

    def _labels(ds) -> list[str]:
        meta = getattr(ds, "meta", None)
        if meta and meta.get("source_files"):
            return [Path(p).parts[0] for p in meta["source_files"]]
        return ["unknown"] * len(ds)

    scenes = _labels(ds_valid) + _labels(ds_test)
    return combined, scenes


# ---------------------------------------------------------------------------
# Extended streaming accumulator — mse/sam/ssim/sam_valid/sid/scc/q2n,
# sample-weighted exactly like inference/inference.py's own loop (including
# the SID-clamped-fraction's pixel-count weighting quirk and "PSNR derived
# once from pooled MSE, never averaged in dB").
# ---------------------------------------------------------------------------

class ReconAccumulator:
    def __init__(self, min_energy: float, ext_cfg: dict):
        self.min_energy = min_energy
        self.ext_cfg = ext_cfg
        self.mse_wsum = self.sam_wsum = self.ssim_wsum = 0.0
        self.sid_wsum = self.scc_wsum = self.q2n_wsum = 0.0
        self.sam_valid_sum = 0.0
        self.n_valid_px = 0
        self.n_total_px = 0
        self.sid_clamped_sum = 0.0
        self.n_sid_px = 0
        self.n_samples = 0

    def update(self, x: torch.Tensor, recon: torch.Tensor) -> None:
        b = x.shape[0]
        self.mse_wsum += compute_mse(x, recon) * b
        self.sam_wsum += spectral_angle_mapper_loss(x, recon).item() * b
        self.ssim_wsum += compute_ssim(x, recon) * b
        sv_sum, sv_n, sv_total = sam_valid_sums(x, recon, self.min_energy)
        self.sam_valid_sum += sv_sum
        self.n_valid_px += sv_n
        self.n_total_px += sv_total
        eps = self.ext_cfg["sid_clamp_epsilon"]
        self.sid_wsum += compute_sid(x, recon, epsilon=eps) * b
        self.sid_clamped_sum += compute_sid_clamped_frac(x, epsilon=eps) * x[..., 0].numel()
        self.n_sid_px += x[..., 0].numel()
        self.scc_wsum += compute_scc(x, recon) * b
        self.q2n_wsum += compute_q2n(x, recon, block_size=self.ext_cfg["q2n_block_size"]) * b
        self.n_samples += b

    def result(self) -> dict:
        n = max(self.n_samples, 1)
        mse = self.mse_wsum / n
        return {
            "mse": mse,
            "sam_rad": self.sam_wsum / n,
            "sam_valid": (self.sam_valid_sum / self.n_valid_px) if self.n_valid_px else float("nan"),
            "valid_pixel_frac": (self.n_valid_px / self.n_total_px) if self.n_total_px else float("nan"),
            "psnr": float(compute_psnr_from_mse(mse)),
            "ssim": self.ssim_wsum / n,
            "sid": self.sid_wsum / n,
            "sid_clamped_frac": (self.sid_clamped_sum / self.n_sid_px) if self.n_sid_px else float("nan"),
            "scc": self.scc_wsum / n,
            "q2n": self.q2n_wsum / n,
            "n_samples": self.n_samples,
        }


class SimpleAccumulator:
    """mse/sam/ssim/sam_valid only — for P4's intact/shuffled and step 7's
    masked-input scoring, which don't need SID/SCC/Q2^n."""

    def __init__(self, min_energy: float):
        self.min_energy = min_energy
        self.sam_wsum = self.ssim_wsum = self.mse_wsum = 0.0
        self.sam_valid_sum = 0.0
        self.n_valid_px = 0
        self.n_pixels = 0
        self.n_samples = 0
        self.psnr_sum = 0.0
        self.has_divisors = False

    def update(self, x: torch.Tensor, recon: torch.Tensor, divisors: torch.Tensor | None = None) -> None:
        b = x.shape[0]
        if divisors is not None:
            self.has_divisors = True
            x = x * divisors
            recon = recon * divisors

        self.mse_wsum += compute_mse(x, recon) * b
        angle = _sam_per_pixel(x, recon)
        self.sam_wsum += float(angle.sum())
        self.n_pixels += angle.numel()
        self.ssim_wsum += compute_ssim(x, recon) * b

        # Compute per-sample PSNR and accumulate
        for k in range(b):
            dk = 1.0 if divisors is None else float(divisors[k].item())
            msek = compute_mse(x[k:k+1], recon[k:k+1])
            self.psnr_sum += (20.0 * math.log10(dk) - 10.0 * math.log10(max(msek, 1e-12)))

        energy = (x ** 2).sum(dim=-1)
        if divisors is not None:
            # Mask logic adjusted for scaled energy
            mask = torch.zeros_like(energy, dtype=torch.bool)
            for k in range(b):
                dk = float(divisors[k].item())
                mask[k] = energy[k] >= (self.min_energy * (dk ** 2))
        else:
            mask = energy >= self.min_energy

        nvalid = int(mask.sum())
        if nvalid:
            self.sam_valid_sum += float(angle[mask].sum())
            self.n_valid_px += nvalid
        self.n_samples += b

    def result(self) -> dict:
        n = max(self.n_samples, 1)
        if self.has_divisors:
            psnr_val = self.psnr_sum / n
        else:
            psnr_val = 10.0 * math.log10(1.0 / max(self.mse_wsum / n, 1e-12))

        return {
            "mse": self.mse_wsum / n,
            "sam": self.sam_wsum / max(self.n_pixels, 1),
            "sam_valid": (self.sam_valid_sum / self.n_valid_px) if self.n_valid_px else float("nan"),
            "ssim": self.ssim_wsum / n,
            "psnr": psnr_val,
        }


# ---------------------------------------------------------------------------
# Latent-flattening + eigenvalue-entropy effective occupancy
# ---------------------------------------------------------------------------

def _flatten_latent_for_occupancy(t: torch.Tensor, model_name: str) -> torch.Tensor:
    """
    (N, d) with d = the latent's own feature/channel width. Every registered
    model except vae-1d-pixelwise puts the channel axis at dim 1 (spatially-
    structured latents, NCHW-like); vae-1d-pixelwise's latent is
    (B, H, W, Z) — channels already last.
    """
    if model_name == "vae-1d-pixelwise":
        d = t.shape[-1]
        return t.reshape(-1, d)
    d = t.shape[1]
    perm = (0,) + tuple(range(2, t.ndim)) + (1,)
    return t.permute(*perm).reshape(-1, d)


def effective_occupancy(flat: torch.Tensor) -> float:
    """
    Eigenvalue-entropy effective rank of a (N, d) latent sample:
    covariance -> eigenvalues -> normalise to a distribution -> Shannon
    entropy -> effective_rank = exp(entropy) -> occupancy = effective_rank/d.
    1.0 = every latent direction carries equal variance (fully used); near 0
    = variance is concentrated in a handful of directions (collapse-prone,
    limited capacity to represent chemically diverse spectra without
    hallucinating along the directions it doesn't actually use).
    """
    d = flat.shape[1]
    if d <= 1 or flat.shape[0] < 2:
        return float("nan")
    flat = flat.float()
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    cov = (centered.t() @ centered) / max(flat.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    total = eigvals.sum()
    if float(total) <= 1e-12:
        return 0.0
    p = eigvals / total
    p_nz = p[p > 1e-12]
    entropy = -(p_nz * torch.log(p_nz)).sum()
    effective_rank = torch.exp(entropy)
    return float(effective_rank / d)


# ---------------------------------------------------------------------------
# Per-(cell, checkpoint-seed) evaluation
# ---------------------------------------------------------------------------

def _make_loader(dataset_obj, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


class MissingPixelAccumulator:
    def __init__(self, min_energy: float):
        self.min_energy = min_energy
        self.sam_clean_sum = 0.0
        self.sam_masked_sum = 0.0
        self.sam_valid_clean_sum = 0.0
        self.sam_valid_masked_sum = 0.0
        
        self.sam_rad_masked_only_sum = 0.0
        self.sam_valid_masked_only_sum = 0.0
        self.mse_masked_only_sum = 0.0
        self.n_masked_elements = 0
        self.n_masked_pixels = 0
        self.n_valid_masked_pixels = 0
        self.n_samples = 0
        self.n_valid_pixels = 0
        self.n_total_pixels = 0

    def update(self, x: torch.Tensor, recon_clean: torch.Tensor, recon_masked: torch.Tensor, m: torch.Tensor) -> None:
        b, H, W, C = x.shape
        self.n_samples += b
        
        angles_clean = _sam_per_pixel(x, recon_clean)
        angles_masked = _sam_per_pixel(x, recon_masked)
        
        self.sam_clean_sum += float(angles_clean.sum())
        self.sam_masked_sum += float(angles_masked.sum())
        self.n_total_pixels += angles_clean.numel()
        
        energy = (x ** 2).sum(dim=-1)
        valid_mask = energy >= self.min_energy
        nvalid = int(valid_mask.sum())
        if nvalid:
            self.sam_valid_clean_sum += float(angles_clean[valid_mask].sum())
            self.sam_valid_masked_sum += float(angles_masked[valid_mask].sum())
            self.n_valid_pixels += nvalid
            
        if m.any():
            self.sam_rad_masked_only_sum += float(angles_masked[m].sum())
            self.n_masked_pixels += int(m.sum())
            
            valid_masked = m & valid_mask
            if valid_masked.any():
                self.sam_valid_masked_only_sum += float(angles_masked[valid_masked].sum())
                self.n_valid_masked_pixels += int(valid_masked.sum())
                
            diff_sq = (recon_masked.float() - x.float()) ** 2
            m_expanded = m.unsqueeze(-1).expand_as(diff_sq)
            self.mse_masked_only_sum += float(diff_sq[m_expanded].sum())
            self.n_masked_elements += int(m_expanded.sum())

    def result(self) -> dict:
        n_tot_px = max(self.n_total_pixels, 1)
        n_val_px = max(self.n_valid_pixels, 1)
        n_mask_px = max(self.n_masked_pixels, 1)
        n_val_mask_px = max(self.n_valid_masked_pixels, 1)
        n_mask_el = max(self.n_masked_elements, 1)
        
        sam_clean = self.sam_valid_clean_sum / n_val_px if self.n_valid_pixels else float("nan")
        sam_masked = self.sam_valid_masked_sum / n_val_px if self.n_valid_pixels else float("nan")
        sam_drop = sam_masked - sam_clean
        
        sam_rad_clean = self.sam_clean_sum / n_tot_px
        sam_rad_masked = self.sam_masked_sum / n_tot_px
        sam_rad_drop = sam_rad_masked - sam_rad_clean
        
        sam_rad_masked_only = self.sam_rad_masked_only_sum / n_mask_px
        sam_valid_masked_only = self.sam_valid_masked_only_sum / n_val_mask_px if self.n_valid_masked_pixels else float("nan")
        
        mse_masked_only = self.mse_masked_only_sum / n_mask_el
        psnr_masked_only = 10.0 * math.log10(1.0 / max(mse_masked_only, 1e-12))
        
        return {
            "sam_clean": sam_clean,
            "sam_masked": sam_masked,
            "sam_drop": sam_drop,
            "sam_rad_clean": sam_rad_clean,
            "sam_rad_masked": sam_rad_masked,
            "sam_rad_drop": sam_rad_drop,
            "sam_valid_masked_only": sam_valid_masked_only,
            "sam_rad_masked_only": sam_rad_masked_only,
            "psnr_masked_only": psnr_masked_only,
        }


def check_convex_hull_fraction(Y: torch.Tensor, T: torch.Tensor) -> float:
    try:
        T_mean = T.mean(dim=0, keepdim=True)
        _, _, V = torch.pca_lowrank(T - T_mean, q=3)
        T_proj = ((T - T_mean) @ V).cpu().numpy()
        Y_proj = ((Y - T_mean) @ V).cpu().numpy()
        
        from scipy.spatial import Delaunay
        tri = Delaunay(T_proj)
        inside = tri.find_simplex(Y_proj) >= 0
        return float(inside.mean())
    except Exception as e:
        print(f"Warning: Convex hull fraction check failed: {e}")
        return float("nan")


def get_training_spectra_sample(dataset_name: str, packed_root: Path, num_spectra: int = 1000, seed: int = 42) -> torch.Tensor:
    train_ds = PackedPatchDataset(packed_root, "train")
    n = len(train_ds)
    rng = np.random.default_rng(seed)
    patch_indices = rng.choice(n, size=min(100, n), replace=False)
    pixels = []
    for idx in patch_indices:
        patch = train_ds[int(idx)]  # (H, W, C) float32 tensor already
        flat = patch.reshape(-1, patch.shape[-1])
        k = max(1, num_spectra // len(patch_indices))
        pixel_indices = rng.choice(flat.shape[0], size=min(k, flat.shape[0]), replace=False)
        pixels.append(flat[pixel_indices])
    sampled_spectra = torch.cat(pixels, dim=0)[:num_spectra]
    return sampled_spectra


def reconstruct_zero_spatial(model, x: torch.Tensor) -> torch.Tensor:
    """
    Spatial-stream ablation for the dual-stream models (vae-our / vae-our-nl):
    mirrors HSI_DualStream_PI_VAE.forward exactly, except the spatial latent is
    replaced by zeros (the prior mean) before the spatial decoder — so the
    spatial stream carries no information about x, and the spectral stream
    and the gated fusion run unchanged. Local to this script on purpose: the
    model code is frozen training code.
    """
    with torch.no_grad():
        mu_s, _ = torch.chunk(model.spatial_stream.encoder(x), 2, dim=1)
        recon_s = model.spatial_stream.decoder(torch.zeros_like(mu_s))
        z_p, _, _ = model.reparameterize(model.spectral_stream.encoder(x))
        recon_p = model.spectral_stream.decoder(z_p)
        return model.fusion(recon_s, recon_p)


def compute_per_sample_reconstruction_metrics(model, loader: DataLoader, device,
                                              min_energy: float, ext_cfg: dict,
                                              zero_spatial: bool = False) -> list[dict]:
    results = []
    with torch.inference_mode():
        for x in tqdm(loader, desc="Per-sample Recon Metrics", leave=False):
            x = x.to(device, non_blocking=True)
            b = x.shape[0]
            recon = reconstruct_zero_spatial(model, x) if zero_spatial else model.reconstruct(x)
            
            for k in range(b):
                xk = x[k:k+1]
                rk = recon[k:k+1]
                
                mse_v = compute_mse(xk, rk)
                psnr_v = float(compute_psnr_from_mse(mse_v))
                ssim_v = compute_ssim(xk, rk)
                
                sv_sum, sv_n, _ = sam_valid_sums(xk, rk, min_energy)
                sam_valid_v = (sv_sum / sv_n) if sv_n else float("nan")
                
                sam_rad_v = spectral_angle_mapper_loss(xk, rk).item()
                
                energy = (xk ** 2).sum(dim=-1)
                valid_mask = energy >= min_energy
                valid_pixel_frac_v = float(valid_mask.float().mean())
                
                eps = ext_cfg["sid_clamp_epsilon"]
                sid_v = compute_sid(xk, rk, epsilon=eps)
                sid_clamped_frac_v = compute_sid_clamped_frac(xk, epsilon=eps)
                
                scc_v = compute_scc(xk, rk)
                q2n_v = compute_q2n(xk, rk, block_size=ext_cfg["q2n_block_size"])
                
                results.append({
                    "mse": mse_v,
                    "sam_rad": sam_rad_v,
                    "sam_valid": sam_valid_v,
                    "valid_pixel_frac": valid_pixel_frac_v,
                    "psnr": psnr_v,
                    "ssim": ssim_v,
                    "sid": sid_v,
                    "sid_clamped_frac": sid_clamped_frac_v,
                    "scc": scc_v,
                    "q2n": q2n_v,
                })
            del recon, x
    return results


def get_divisor_for_dataset(ds, local_idx: int) -> float:
    if ds.__class__.__name__ == "PackedPatchDataset":
        actual_idx = local_idx
        if ds.indices is not None:
            actual_idx = int(ds.indices[local_idx])
        if ds.meta and "patch_max" in ds.meta:
            return float(ds.meta["patch_max"][actual_idx])
    elif ds.__class__.__name__ == "HSIPatchDataset":
        return float(ds._maxes[local_idx])
    return 1.0


def get_divisor_for_global_idx(dataset_obj, global_idx: int) -> float:
    if isinstance(dataset_obj, ConcatDataset):
        for ds in dataset_obj.datasets:
            if global_idx < len(ds):
                return get_divisor_for_dataset(ds, global_idx)
            global_idx -= len(ds)
    else:
        return get_divisor_for_dataset(dataset_obj, global_idx)
    return 1.0


def unnormalize_batch(x: torch.Tensor, loader: DataLoader, batch_start_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    b = x.shape[0]
    divisors = []
    for k in range(b):
        divisors.append(get_divisor_for_global_idx(loader.dataset, batch_start_idx + k))
    div_tensor = torch.tensor(divisors, device=x.device, dtype=x.dtype).view(b, 1, 1, 1)
    return x * div_tensor, div_tensor


def evaluate_one_checkpoint(model, loader: DataLoader, device, cfg: dict,
                            min_energy: float, ext_cfg: dict, model_name: str,
                            ds: str) -> dict:
    """
    Runs noise recovery evaluation against the full inference set.
    Returns a dict of raw (not seed-averaged) results.
    """
    out: dict = {}
    torch.cuda.empty_cache()

    # --- Step 5: noise recovery (one pass per RNG seed) -------
    noise_raw = {sigma: {"sam_rad": [], "sam_valid": [], "psnr": []} for sigma in SIGMAS}
    for rng_seed in RNG_SEEDS:
        gen = torch.Generator(device=device).manual_seed(rng_seed)
        accs = {sigma: SimpleAccumulator(min_energy) for sigma in SIGMAS}
        with torch.inference_mode():
            batch_start_idx = 0
            for x in tqdm(loader, desc=f"[{ds}|{model_name}] Noise Recov (rng={rng_seed})", leave=False):
                x = x.to(device, non_blocking=True)
                b = x.shape[0]
                per_band_std = x.std(dim=(1, 2), keepdim=True)
                # Compute divisors for clean un-normalised target
                _, div_tensor = unnormalize_batch(x, loader, batch_start_idx)
                for sigma in SIGMAS:
                    if batch_start_idx == 0 and rng_seed == RNG_SEEDS[0]:
                        print(f"[Interpretability Log] Adding noise σ={sigma} as explicit multiplier of per-band signal std.")
                    noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * (sigma * per_band_std)
                    x_corrupted = x + noise
                    recon = model.reconstruct(x_corrupted)
                    accs[sigma].update(x, recon, divisors=div_tensor)
                    del noise, x_corrupted, recon
                batch_start_idx += b
                del x
        for sigma in SIGMAS:
            r = accs[sigma].result()
            noise_raw[sigma]["sam_rad"].append(r["sam"])
            noise_raw[sigma]["sam_valid"].append(r["sam_valid"])
            noise_raw[sigma]["psnr"].append(r["psnr"])
        torch.cuda.empty_cache()
    out["noise"] = {
        sigma: {
            "sam_rad": float(np.mean(noise_raw[sigma]["sam_rad"])),
            "sam_valid": float(np.mean(noise_raw[sigma]["sam_valid"])),
            "psnr": float(np.mean(noise_raw[sigma]["psnr"]))
        }
        for sigma in SIGMAS
    }

    return out


def interpolation_and_occupancy(latents_full: list[torch.Tensor], model, device,
                                model_name: str, n_pairs: int, n_alpha: int,
                                pixel: tuple[int, int], dataset_seed: int,
                                T_train: torch.Tensor, dataset_obj) -> dict:
    """
    Step 6: population-level interpolation smoothness (n_pairs random pairs,
    batched across all pairs per alpha step) + eigenvalue-entropy occupancy
    per latent stream, from the already-cached full-set latents.
    """
    n_total = latents_full[0].shape[0]
    rng = np.random.default_rng(dataset_seed)
    n_pairs = min(n_pairs, n_total // 2) or 1
    idx_a = rng.choice(n_total, size=n_pairs, replace=False)
    remaining = np.setdiff1d(np.arange(n_total), idx_a)
    idx_b = rng.choice(remaining, size=n_pairs, replace=(len(remaining) < n_pairs))

    la = [t[idx_a] for t in latents_full]
    lb = [t[idx_b] for t in latents_full]
    r, c = pixel
    alphas = np.linspace(0.0, 1.0, n_alpha)
    spectra = []  # (n_alpha, n_pairs, C)
    interp_batch_size = 8  # Decode in small batches to prevent OOM in SpectralBranch (B*4096 px spectra)
    with torch.no_grad():
        for alpha in alphas:
            mix = lerp_latents(la, lb, float(alpha))
            pair_chunks = []
            for p_start in range(0, n_pairs, interp_batch_size):
                p_end = min(p_start + interp_batch_size, n_pairs)
                mix_chunk = [t[p_start:p_end].to(device, non_blocking=True) for t in mix]
                recon_chunk = model.decode_latents(mix_chunk)          # (chunk_size, H, W, C)
                pair_chunks.append(recon_chunk[:, r, c, :].detach().cpu().numpy())
            spectra.append(np.concatenate(pair_chunks, axis=0))
            torch.cuda.empty_cache()
    spectra = np.stack(spectra, axis=0)                 # (n_alpha, n_pairs, C)

    if n_alpha >= 3:
        second_diff = spectra[:-2] - 2.0 * spectra[1:-1] + spectra[2:]
        jaggedness_per_pair = np.linalg.norm(second_diff, axis=-1).mean(axis=0)  # (n_pairs,)
    else:
        jaggedness_per_pair = np.full(n_pairs, float("nan"))
    steps = np.linalg.norm(np.diff(spectra, axis=0), axis=-1)   # (n_alpha-1, n_pairs)
    path_length_per_pair = steps.sum(axis=0)

    # --- Endpoint fidelity: reconstruction SAM at t=0 (patch A) and t=1 (patch B) ---
    x_A = []
    x_B = []
    for p in range(n_pairs):
        patch_a = dataset_obj[int(idx_a[p])]  # (H, W, C) float32 tensor already
        patch_b = dataset_obj[int(idx_b[p])]
        x_A.append(patch_a[r, c])
        x_B.append(patch_b[r, c])
    x_A = torch.stack(x_A).to(device)  # (n_pairs, C)
    x_B = torch.stack(x_B).to(device)  # (n_pairs, C)

    # lerp_latents is alpha*z_A + (1-alpha)*z_B: alpha=1 (spectra[-1]) decodes
    # patch A's latent, alpha=0 (spectra[0]) decodes patch B's. t=0 is the
    # patch-A endpoint, t=1 the patch-B endpoint.
    recon_t0 = torch.from_numpy(spectra[-1]).to(device)  # (n_pairs, C) — decodes z_A
    recon_t1 = torch.from_numpy(spectra[0]).to(device)   # (n_pairs, C) — decodes z_B

    def sam_angle(y1, y2):
        y1_norm = y1 / y1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y2_norm = y2 / y2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cos_sim = (y1_norm * y2_norm).sum(dim=-1)
        return torch.acos(cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

    endpoint_sam_t0 = float(sam_angle(x_A, recon_t0).mean().item())
    endpoint_sam_t1 = float(sam_angle(x_B, recon_t1).mean().item())

    # --- On-manifold validity of each interpolant ---
    # Flat all interpolants to shape (N_alpha * N_pairs, C)
    interpolants = torch.from_numpy(spectra).to(device).reshape(-1, spectra.shape[-1]) # (N_alpha * N_pairs, C)
    T = T_train.to(device) # (M, C)

    # 1. Spectral angle to nearest training spectrum
    interpolants_norm = interpolants / interpolants.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    T_norm = T / T.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cos_sim = interpolants_norm @ T_norm.t() # (N_alpha * N_pairs, M)
    angles = torch.acos(cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) # (N_alpha * N_pairs, M)
    on_manifold_angle = float(angles.min(dim=-1).values.mean().item())

    # 2. Fraction of interpolants inside the convex hull of training spectra (using 3D PCA projection)
    on_manifold_convex_hull_frac = check_convex_hull_fraction(interpolants, T)

    occ = {}
    stream_names = ["spatial", "spectral"] if len(latents_full) == 2 else ["spectral"]
    for name, t in zip(stream_names, latents_full):
        flat = _flatten_latent_for_occupancy(t, model_name)
        occ[name] = effective_occupancy(flat)
    if len(latents_full) == 1:
        occ["spatial"] = occ["spectral"]

    return {
        "jaggedness": float(np.mean(jaggedness_per_pair)),
        "path_length": float(np.mean(path_length_per_pair)),
        "occupancy_spatial": occ.get("spatial", float("nan")),
        "occupancy_spectral": occ.get("spectral", float("nan")),
        "occupancy_mean": float(np.nanmean([occ.get("spatial", float("nan")),
                                            occ.get("spectral", float("nan"))])),
        "on_manifold_angle": on_manifold_angle,
        "on_manifold_convex_hull_frac": on_manifold_convex_hull_frac,
        "endpoint_sam_t0": endpoint_sam_t0,
        "endpoint_sam_t1": endpoint_sam_t1,
    }


# ---------------------------------------------------------------------------
# P1 (trivial-predictor floors) — reused as-is on a bounded subsample
# ---------------------------------------------------------------------------

def get_or_build_floors(dataset: str, cfg: dict, x_sample: torch.Tensor,
                        scenes_sample: list[str], stats: dict, device,
                        cache_path: Path) -> dict:
    sig = {"dataset": dataset, "n": int(x_sample.shape[0]), "C": int(x_sample.shape[-1])}
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("signature") == sig:
                return cached["floors"]
        except (json.JSONDecodeError, KeyError):
            pass
    eps = cfg["p1_trivial_floors"]["sam_valid_min_energy"]
    floors = p1_shared_floors(x_sample, scenes_sample, stats, cfg, eps, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"signature": sig, "floors": floors}, indent=1))
    return floors


def stratified_subsample(dataset_obj, scenes: list[str], n: int, seed: int):
    total = len(dataset_obj)
    n = min(n, total)
    rng = np.random.default_rng(seed)
    by: dict[str, list[int]] = {}
    for i, s in enumerate(scenes):
        by.setdefault(s, []).append(i)
    idx = []
    for s in sorted(by):
        k = max(1, round(n * len(by[s]) / total))
        idx.extend(rng.choice(by[s], size=min(k, len(by[s])), replace=False))
    idx = np.sort(np.asarray(idx[:n]))
    x = torch.stack([dataset_obj[int(i)] for i in idx])
    lbl = [scenes[int(i)] for i in idx]
    return x, lbl


# ---------------------------------------------------------------------------
# P1 lift — absolute (rad) companions to probes.py's relative SAM lifts
# ---------------------------------------------------------------------------

def p1_lift_columns(p1r: dict, floor_key: str, lift_key: str,
                    stat_prefix: str, lift_prefix: str) -> dict:
    """
    `inference.probes.p1_report` reports PSNR lift in an absolute physical unit
    (dB) but SAM lift only as a fraction of the floor's own SAM. That makes the
    two halves of a P1 row unquotable side by side: "+6.2 dB and +31 %" says
    nothing about how many radians of spectral angle were actually recovered.

    This adds the absolute deltas, in radians, computed here rather than in
    `probes.py` — `p1_report` already returns the raw `model` / floor metric
    dicts, and the exploratory sweep (probes.py:p1_report's other caller) must
    keep its existing schema.

    Sign convention matches `psnr_db` and `sam_relative`: **floor minus model**,
    so a POSITIVE lift always means the model beat the trivial predictor. Both
    SAM variants are emitted; `sam_valid` is the one to quote (raw `sam` carries
    the pi/2 contamination from sub-epsilon-energy pixels — CLAUDE.md section 12).

    NOTE on population: the floors are characterised on a P1_FLOOR_SAMPLE-patch
    subsample while the model score is the full-set, seed-averaged number, so
    these deltas are a diagnostic against a subsampled floor, not a full-set
    quantity. This is pre-existing — `p1_lift_psnr_db` has always had it.
    """
    floor = p1r.get(floor_key, {}) or {}
    model = p1r.get("model", {}) or {}
    lift = p1r.get(lift_key, {}) or {}
    nan = float("nan")

    def delta(key: str) -> float:
        f, m = floor.get(key, nan), model.get(key, nan)
        if f is None or m is None:
            return nan
        try:
            d = float(f) - float(m)
        except (TypeError, ValueError):
            return nan
        return d

    return {
        f"{stat_prefix}_psnr": floor.get("psnr", nan),
        f"{stat_prefix}_ssim": floor.get("ssim", nan),
        f"{stat_prefix}_sam": floor.get("sam", nan),
        f"{stat_prefix}_sam_valid": floor.get("sam_valid", nan),
        f"{lift_prefix}_psnr_db": lift.get("psnr_db", nan),
        f"{lift_prefix}_ssim": lift.get("ssim_absolute", nan),
        f"{lift_prefix}_sam_rad": delta("sam"),
        f"{lift_prefix}_sam_rel": lift.get("sam_relative", nan),
        f"{lift_prefix}_sam_valid_rad": delta("sam_valid"),
        f"{lift_prefix}_sam_valid_rel": lift.get("sam_valid_relative", nan),
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", restval="")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def flush_all_csvs(all_rows: dict[str, list[dict]], columns_map: dict[str, list[str]], out_dir: Path) -> None:
    """Incrementally persist all non-empty CSVs so partial progress is never lost."""
    for name, cols in columns_map.items():
        if all_rows.get(name):
            path = out_dir / name
            write_csv(all_rows[name], cols, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def resolve_packed_root(ds: str, packed_root_arg: str | None) -> Path:
    """
    --packed-root, when given, names the COMMON parent directory (one level
    above the per-dataset subfolders — e.g. 'data/packed', so IIRS is read
    from '<packed-root>/IIRS'), because this script loops over multiple
    datasets in one process, unlike inference.py/probes.py (which are
    invoked once per dataset and so take an already-per-dataset root).
    """
    if packed_root_arg:
        return Path(packed_root_arg) / ds.upper()
    return Path(DATASETS[ds.upper()]["packed_root"])


def cell_grid(models: list[str], datasets: list[str]) -> list[tuple]:
    cells = []
    for ds in datasets:
        for model in models:
            losses = ["physics"] if model in PHYSICS_ONLY else ["physics", "standard"]
            for loss in losses:
                cells.append((ds, model, loss))
    return cells


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default="model")
    p.add_argument("--out-dir", default="results/final")
    p.add_argument("--packed-root", default=None,
                   help="Override the COMMON packed-data root, one level above "
                        "the per-dataset subdirectories (e.g. 'data/packed', so "
                        "IIRS is read from '<packed-root>/IIRS') — this script "
                        "loops over multiple datasets in one process, unlike "
                        "inference.py/probes.py. Default: DATASETS[ds]['packed_root'].")
    p.add_argument("--datasets", default=",".join(DATASETS_SCOPE))
    p.add_argument("--models", default=",".join(MODELS_SCOPE))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--n-interp-pairs", type=int, default=N_INTERP_PAIRS)
    p.add_argument("--n-alpha", type=int, default=N_ALPHA)
    p.add_argument("--pixel", type=int, nargs=2, default=list(INTERP_PIXEL))
    p.add_argument("--p1-floor-sample", type=int, default=P1_FLOOR_SAMPLE)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute all cells even if cached results exist on disk.")
    p.add_argument("--no-telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the cell grid + patch-audit counts; load no model.")
    p.add_argument("--set", action="append", default=None, metavar="KEY=VALUE")
    p.add_argument("--sigmas", default=",".join(str(s) for s in SIGMAS),
                   help="Comma list of input-noise sigmas (multiples of each band's own "
                        "std). Use a separate --out-dir per sigma set: a cell cache is only "
                        "reused when its sigma list matches exactly.")
    return p.parse_args()


def compute_noisy_input_reference(loader: DataLoader, device, min_energy: float) -> dict:
    """Computes reference PSNR and SAM for corrupted input x against clean target x."""
    noise_raw = {sigma: {"sam_rad": [], "sam_valid": [], "psnr": []} for sigma in SIGMAS}
    for rng_seed in RNG_SEEDS:
        gen = torch.Generator(device=device).manual_seed(rng_seed)
        accs = {sigma: SimpleAccumulator(min_energy) for sigma in SIGMAS}
        with torch.inference_mode():
            batch_start_idx = 0
            for x in tqdm(loader, desc=f"Ref Noise Reference (seed={rng_seed})", leave=False):
                x = x.to(device, non_blocking=True)
                b = x.shape[0]
                per_band_std = x.std(dim=(1, 2), keepdim=True)
                _, div_tensor = unnormalize_batch(x, loader, batch_start_idx)
                for sigma in SIGMAS:
                    if batch_start_idx == 0 and rng_seed == RNG_SEEDS[0]:
                        print(f"[Interpretability Log] Adding reference noise σ={sigma} as explicit multiplier of per-band signal std.")
                    noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * (sigma * per_band_std)
                    x_corrupted = x + noise
                    accs[sigma].update(x, x_corrupted, divisors=div_tensor)
                    del noise, x_corrupted
                batch_start_idx += b
                del x
        for sigma in SIGMAS:
            r = accs[sigma].result()
            noise_raw[sigma]["sam_rad"].append(r["sam"])
            noise_raw[sigma]["sam_valid"].append(r["sam_valid"])
            noise_raw[sigma]["psnr"].append(r["psnr"])
            
    out = {}
    for sigma in SIGMAS:
        out[f"sam_recovery_s{sigma}"] = float(np.mean(noise_raw[sigma]["sam_valid"]))
        out[f"sam_rad_recovery_s{sigma}"] = float(np.mean(noise_raw[sigma]["sam_rad"]))
        out[f"psnr_recovery_s{sigma}"] = float(np.mean(noise_raw[sigma]["psnr"]))
    return out


def main() -> int:
    global SIGMAS
    args = parse_args()
    SIGMAS = [float(s) for s in args.sigmas.split(",") if s.strip()]
    CSV_COLUMNS["noise-recovery.csv"] = noise_columns(SIGMAS)
    datasets = [d.strip().upper() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log = Logger(enabled=not args.no_telegram)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_prereg()

    cells = cell_grid(models, datasets)
    total_cells = len(cells)
    log.send(f"Inference final started — sigmas={SIGMAS}, {total_cells} cells "
             f"({len(datasets)} datasets x models/losses), device={device}, "
             f"out_dir={out_dir}")

    if args.dry_run:
        for ds, model, loss in cells:
            print(f"  {ds:8}{model:24}{loss}")
        for ds in datasets:
            packed_root = resolve_packed_root(ds, args.packed_root)
            try:
                audit_dataset(ds, packed_root, log)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
        for name, cols in CSV_COLUMNS.items():
            print(f"{name}: {','.join(cols)}")
        return 0

    all_rows: dict[str, list[dict]] = {name: [] for name in CSV_COLUMNS}
    t0 = time.time()
    cell_idx = 0
    had_fatal_error = False

    try:
        for ds in datasets:
            packed_root = resolve_packed_root(ds, args.packed_root)
            try:
                audit_dataset(ds, packed_root, log)
                apply_dataset(ds, verify=True)
                apply_hyperparams(settings, load_hyperparams(ds))
                overrides = apply_cli_overrides(settings, args.set)
                if overrides:
                    log.send(f"--set overrides active for {ds}: {overrides}")

                inference_set, scenes = build_inference_set(ds, packed_root)
                n_total = len(inference_set)
                batch_size = args.batch_size or min(settings.batch_size, 16)
                num_workers = args.num_workers if args.num_workers is not None else settings.num_workers
                min_energy = cfg["p1_trivial_floors"]["sam_valid_min_energy"]
                ext_cfg = _recon_metrics_ext_cfg()

                ref_loader = _make_loader(inference_set, batch_size, num_workers)
                ref_metrics = compute_noisy_input_reference(ref_loader, device, min_energy)
                ref_row = {
                    "dataset": ds,
                    "model": "noisy_input_no_model",
                    "loss": "none",
                }
                ref_row.update(ref_metrics)
                all_rows["noise-recovery.csv"].append(ref_row)
                log.send(f"Computed 'noisy input, no model' reference for {ds}")
                
            except Exception as e:
                tb = traceback.format_exc()
                log.send_pre(f"❌ Inference final - dataset setup FAILED on {ds}: {html.escape(str(e))}", tb[-2000:])
                print(f"Error setting up dataset {ds}:\n{tb}", file=sys.stderr)
                continue

            log.send(f"Inference final - noise-recovery - START - {ds}")

            ds_cells = [(m, l) for d, m, l in cells if d == ds]
            pbar_cells = tqdm(ds_cells, desc=f"Eval Cells ({ds})", leave=True)
            for model_name, loss in pbar_cells:
                cell_idx += 1
                pbar_cells.set_description(f"Eval [{cell_idx}/{total_cells}] {ds} | {model_name} | {loss}")
                cell_cache_path = cache_dir / f"{ds}__{model_name}__{loss}.json"

                # ---- Resume / cache check --------------------------------
                if not args.overwrite and cell_cache_path.is_file():
                    try:
                        cached_cell = json.loads(cell_cache_path.read_text(encoding="utf-8"))
                        cached_rows = cached_cell.get("rows", {})
                        if (cached_cell.get("version") == CACHE_VERSION
                                # caches written before --sigmas existed carry no list;
                                # they were all computed at the original [0.01, 0.05, 0.2]
                                and cached_cell.get("sigmas", LEGACY_CACHE_SIGMAS) == SIGMAS
                                and all(isinstance(cached_rows.get(k), list) for k in PER_CELL_CSVS)):
                            for name in PER_CELL_CSVS:
                                all_rows[name].extend(cached_rows[name])
                            log.send(f"Inference final [{cell_idx}/{total_cells}] - {ds} | {model_name} | {loss} - LOADED FROM CACHE, skipping")
                            flush_all_csvs(all_rows, CSV_COLUMNS, out_dir)
                            continue
                    except Exception as e:
                        print(f"Failed to read cache {cell_cache_path}: {e}")

                log.send(f"Inference final [{cell_idx}/{total_cells}] - START - {ds} | {model_name} | {loss}")

                try:
                    per_seed = []
                    for seed in CHECKPOINT_SEEDS:
                        try:
                            ckpt_path = resolve_checkpoint(args.ckpt_dir, ds, model_name, loss,
                                                           seed=seed, select="sam")
                            if not ckpt_path.is_file():
                                raise FileNotFoundError(f"missing checkpoint {ckpt_path}")
                            model, _ = load_model(model_name, ckpt_path, device)
                            loader = _make_loader(inference_set, batch_size, num_workers)
                            res = evaluate_one_checkpoint(
                                model, loader, device, cfg, min_energy, ext_cfg, model_name, ds)
                            per_seed.append(res)
                            del model
                            torch.cuda.empty_cache()
                        except Exception as e:  # noqa: BLE001
                            tb = traceback.format_exc()
                            log.send_pre(f"❌ Inference final - {model_name}|{ds}|{loss}|seed{seed} FAILED: {html.escape(str(e))}", tb[-2000:])
                            print(tb, file=sys.stderr)

                    if not per_seed:
                        log.send(f"Inference final - {model_name}|{ds}|{loss} - NO USABLE CHECKPOINT, skipped")
                        continue

                    def avg(key_path, seeds=per_seed):
                        vals = []
                        for r in seeds:
                            d = r
                            for k in key_path:
                                d = d[k]
                            if d is not None and not (isinstance(d, float) and math.isnan(d)):
                                vals.append(d)
                        return float(np.mean(vals)) if vals else float("nan")

                    noise_row = {"dataset": ds, "model": model_name, "loss": loss}
                    for sigma in SIGMAS:
                        noise_row[f"sam_recovery_s{sigma}"] = avg(["noise", sigma, "sam_valid"])
                        noise_row[f"sam_rad_recovery_s{sigma}"] = avg(["noise", sigma, "sam_rad"])
                        noise_row[f"psnr_recovery_s{sigma}"] = avg(["noise", sigma, "psnr"])

                    cell_rows = {
                        "noise-recovery.csv": [noise_row],
                    }
                    assert set(cell_rows) == set(PER_CELL_CSVS)
                    for name, rows in cell_rows.items():
                        all_rows[name].extend(rows)

                    cell_cache = {
                        "version": CACHE_VERSION,
                        "sigmas": SIGMAS,
                        "dataset": ds,
                        "model": model_name,
                        "loss": loss,
                        "rows": cell_rows,
                    }
                    cell_cache_path.write_text(json.dumps(cell_cache, indent=2), encoding="utf-8")
                    flush_all_csvs(all_rows, CSV_COLUMNS, out_dir)

                    log.send(f"Inference final [{cell_idx}/{total_cells}] - DONE (Noise Recovery) - {ds} | {model_name} | {loss}")
                except Exception as e:
                    tb = traceback.format_exc()
                    log.send_pre(f"❌ Inference final - {model_name}|{ds}|{loss} CELL PROCESSING FAILED: {html.escape(str(e))}", tb[-2000:])
                    print(f"Error processing cell {model_name}|{ds}|{loss}:\n{tb}", file=sys.stderr)
                    continue

    except Exception as e:
        had_fatal_error = True
        tb = traceback.format_exc()
        log.send_pre(f"❌ Inference final UNHANDLED FATAL ERROR: {html.escape(str(e))}", tb[-2500:])
        print(f"Fatal unhandled exception:\n{tb}", file=sys.stderr)

    flush_all_csvs(all_rows, CSV_COLUMNS, out_dir)
    for name, cols in CSV_COLUMNS.items():
        path = out_dir / name
        if path.is_file():
            log.send_document(path, caption=name)
    text_cols = [("model", "model"), ("loss", "loss")]
    for sigma in SIGMAS:
        text_cols += [(f"psnr_recovery_s{sigma}", f"PSNR@{sigma}"),
                      (f"sam_recovery_s{sigma}", f"SAMv@{sigma}")]
    log.send_csv_as_text(out_dir / "noise-recovery.csv",
                         "Noise recovery (input-space; SAMv = SAM-valid, rad)", text_cols)

    elapsed = time.time() - t0
    summary = "\n".join(f"{name}: {len(rows)} rows" for name, rows in all_rows.items())
    log.send_pre(f"Inference final complete — {elapsed/60:.1f} min", summary)

    return 1 if had_fatal_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
