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
    model-validity-probes.csv   P1 (trivial-floor lift) + P2 (latent-rate
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
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

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
SIGMAS = [0.1, 0.5, 1.0]
PIXEL_MASK_FRACTION = 0.10
N_INTERP_PAIRS = 100
N_ALPHA = 11
INTERP_PIXEL = (32, 32)
P1_FLOOR_SAMPLE = 2000              # subsample size for the trivial-floor characterisation

CSV_COLUMNS = {
    "model-validity-probes.csv": [
        "dataset", "model", "loss",
        "latent_elements", "compression_ratio", "rate_dev_pct", "rate_matched_frac",
        "active_units", "mean_kl_per_dim", "latent_swap_delta",
        "recon_std_across_batch", "collapsed_frac",
        "sri", "sam_intact", "sam_shuffled", "psnr_intact", "psnr_shuffled",
        "uses_spatial_context_frac",
        "p1_floor_psnr", "p1_floor_ssim", "p1_floor_sam_valid",
        "p1_lift_psnr_db", "p1_lift_ssim", "p1_lift_sam_rel", "p1_lift_sam_valid_rel",
        "p1_meanpatch_psnr", "p1_meanpatch_sam_valid",
        "p1_lift_vs_meanpatch_psnr_db", "p1_lift_vs_meanpatch_sam_valid_rel",
        "p1_headroom_psnr",
    ],
    "reconstruction-quality.csv": [
        "dataset", "model", "loss",
        "mse", "sam_rad", "sam_valid", "valid_pixel_frac",
        "psnr", "ssim", "sid", "sid_clamped_frac", "scc", "q2n", "n_samples",
    ],
    "noise-recovery.csv": [
        "dataset", "model", "loss",
        "sam_recovery_s0.1", "psnr_recovery_s0.1",
        "sam_recovery_s0.5", "psnr_recovery_s0.5",
        "sam_recovery_s1.0", "psnr_recovery_s1.0",
    ],
    "chemical-interpolation.csv": [
        "dataset", "model", "loss",
        "jaggedness", "path_length",
        "occupancy_spatial", "occupancy_spectral", "occupancy_mean",
    ],
    "missing-pixel-recovery.csv": [
        "dataset", "model", "loss", "sam_clean", "sam_masked", "sam_drop",
    ],
}


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
        self.send(f"{header}\n<pre>{body}</pre>")

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

    def update(self, x: torch.Tensor, recon: torch.Tensor) -> None:
        b = x.shape[0]
        self.mse_wsum += compute_mse(x, recon) * b
        angle = _sam_per_pixel(x, recon)
        self.sam_wsum += float(angle.sum())
        self.n_pixels += angle.numel()
        self.ssim_wsum += compute_ssim(x, recon) * b
        energy = (x ** 2).sum(dim=-1)
        mask = energy >= self.min_energy
        nvalid = int(mask.sum())
        if nvalid:
            self.sam_valid_sum += float(angle[mask].sum())
            self.n_valid_px += nvalid
        self.n_samples += b

    def result(self) -> dict:
        return {
            "mse": self.mse_wsum / max(self.n_samples, 1),
            "sam": self.sam_wsum / max(self.n_pixels, 1),
            "sam_valid": (self.sam_valid_sum / self.n_valid_px) if self.n_valid_px else float("nan"),
            "ssim": self.ssim_wsum / max(self.n_samples, 1),
            "psnr": 10.0 * math.log10(1.0 / max(self.mse_wsum / max(self.n_samples, 1), 1e-12)),
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
                      num_workers=num_workers, pin_memory=True)


def evaluate_one_checkpoint(model, loader: DataLoader, device, cfg: dict,
                            min_energy: float, ext_cfg: dict, model_name: str,
                            shuffle_perm: torch.Tensor,
                            mask_by_rng: dict[int, np.ndarray]) -> dict:
    """
    Runs every experiment that depends on ONE loaded checkpoint against the
    full inference set. Returns a dict of raw (not seed-averaged) results;
    the caller averages across the two checkpoint seeds.

    Pass plan (see module docstring's cost note):
      A  reconstruct(x) for the clean input AND reconstruct(x_shuffled)
         -> feeds reconstruction-quality (clean only) + P4 (both) in one pass
      B  encode_latents(x) -> latents_full, cached for P3/step5/step6
      C  P3's latent-swap: decode base + rolled latents, compare to x
      D-F  step 5, one pass per RNG seed (all 3 sigmas per batch)
      G-I  step 7, one pass per RNG seed (masked reconstruct only; the clean
           SAM already came out of pass A)
    """
    out: dict = {}

    # --- Pass A: reconstruction-quality + P4 (intact & shuffled) ----------
    recon_acc = ReconAccumulator(min_energy, ext_cfg)
    p4_intact = SimpleAccumulator(min_energy)
    p4_shuffled = SimpleAccumulator(min_energy)
    with torch.no_grad():
        for x in loader:
            x = x.to(device, non_blocking=True)
            b, H, W, C = x.shape
            recon = model.reconstruct(x)
            recon_acc.update(x, recon)
            p4_intact.update(x, recon)
            x_sh = x.reshape(b, H * W, C)[:, shuffle_perm, :].reshape(b, H, W, C).contiguous()
            recon_sh = model.reconstruct(x_sh)
            p4_shuffled.update(x_sh, recon_sh)
    out["reconstruction"] = recon_acc.result()
    m_int, m_sh = p4_intact.result(), p4_shuffled.result()
    sri = (m_sh["sam"] - m_int["sam"]) / max(m_int["sam"], 1e-12)
    out["p4"] = {"sri": sri, "sam_intact": m_int["sam"], "sam_shuffled": m_sh["sam"],
                "psnr_intact": m_int["psnr"], "psnr_shuffled": m_sh["psnr"],
                "uses_spatial_context": sri >= cfg["p4_spatial_reliance"]["min_sri_for_spatial_use"]}

    # --- Pass B: encode -> latents_full (small; cached for P3/5/6) --------
    parts: list[list[torch.Tensor]] = None
    with torch.no_grad():
        for x in loader:
            x = x.to(device, non_blocking=True)
            lat = model.encode_latents(x)
            if parts is None:
                parts = [[] for _ in lat]
            for j, t in enumerate(lat):
                parts[j].append(t)
    latents_full = [torch.cat(p, dim=0) for p in parts]
    out["latents_full"] = latents_full  # consumed by steps 5/6, popped before JSON-ing

    # P2 — trivial, one patch's worth of latent already in latents_full
    elements = int(sum(t[:1].numel() for t in latents_full))
    inp = H * W * C
    p2p = cfg["p2_latent_budget"]
    ratio = inp / elements
    dev = 100 * (ratio - p2p["target_ratio"]) / p2p["target_ratio"]
    out["p2"] = {"latent_elements": elements, "compression_ratio": ratio,
                "deviation_pct": dev,
                "rate_matched": abs(dev) <= p2p["match_tolerance_pct"]}

    # P3 — per-dim KL from the aggregate posterior (cheap, latents are tiny)
    kls = []
    for t in latents_full:
        flat = t.reshape(t.shape[0], -1).double()
        var = flat.var(dim=0, unbiased=False)
        mean = flat.mean(dim=0)
        kl = 0.5 * (var + mean ** 2 - 1.0 - torch.log(var + 1e-12))
        kls.append(kl)
    kl_all = torch.cat(kls)
    p3p = cfg["p3_collapse"]
    active = (kl_all > p3p["active_unit_kl_nats"]).double().mean().item()

    # --- Pass C: P3's latent-swap (decode base + rolled, score vs x) ------
    lat_rolled = [torch.roll(t, 1, dims=0) for t in latents_full]
    sam_base_sum = sam_swap_sum = 0.0
    n_pixels = 0
    sum_x = sum_x2 = None
    n_total = 0
    with torch.no_grad():
        i = 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            b = x.shape[0]
            base = model.decode_latents([t[i:i + b] for t in latents_full])
            swapped = model.decode_latents([t[i:i + b] for t in lat_rolled])
            angle_base = _sam_per_pixel(x, base)
            angle_swap = _sam_per_pixel(x, swapped)
            sam_base_sum += float(angle_base.sum())
            sam_swap_sum += float(angle_swap.sum())
            n_pixels += angle_base.numel()
            if sum_x is None:
                sum_x = base.sum(dim=0)
                sum_x2 = (base ** 2).sum(dim=0)
            else:
                sum_x += base.sum(dim=0)
                sum_x2 += (base ** 2).sum(dim=0)
            n_total += b
            i += b
    sam_base = sam_base_sum / max(n_pixels, 1)
    sam_swap = sam_swap_sum / max(n_pixels, 1)
    delta = abs(sam_swap - sam_base) / max(sam_base, 1e-12)
    mean_b = sum_x / max(n_total, 1)
    var_b = (sum_x2 - n_total * mean_b ** 2) / max(n_total - 1, 1)
    std_b = torch.sqrt(torch.clamp(var_b, min=0))
    collapsed = (active < p3p["min_active_fraction"] or delta < p3p["latent_swap_min_delta_sam"])
    out["p3"] = {"active_unit_fraction": active, "mean_kl_per_dim": float(kl_all.mean()),
                "latent_swap_delta": delta, "recon_std_across_batch": float(std_b.mean()),
                "collapsed": bool(collapsed)}

    # --- Passes D-F: step 5, noise recovery (one pass per RNG seed) -------
    noise_raw = {sigma: {"sam": [], "psnr": []} for sigma in SIGMAS}
    for rng_seed in RNG_SEEDS:
        gen = torch.Generator(device=device).manual_seed(rng_seed)
        accs = {sigma: SimpleAccumulator(min_energy) for sigma in SIGMAS}
        with torch.no_grad():
            i = 0
            for x in loader:
                x = x.to(device, non_blocking=True)
                b = x.shape[0]
                chunk = [t[i:i + b] for t in latents_full]
                for sigma in SIGMAS:
                    noisy = add_latent_noise(chunk, sigma, generator=gen)
                    recon = model.decode_latents(noisy)
                    accs[sigma].update(x, recon)
                i += b
        for sigma in SIGMAS:
            r = accs[sigma].result()
            noise_raw[sigma]["sam"].append(r["sam"])
            noise_raw[sigma]["psnr"].append(r["psnr"])
    out["noise"] = {sigma: {"sam": float(np.mean(noise_raw[sigma]["sam"])),
                            "psnr": float(np.mean(noise_raw[sigma]["psnr"]))}
                    for sigma in SIGMAS}

    # --- Passes G-I: step 7, missing-pixel recovery (masked pass only) ----
    drop_raw = []
    for rng_seed in RNG_SEEDS:
        mask_np = mask_by_rng[rng_seed]  # (N_total, H, W) bool, dataset-level, model-independent
        acc = SimpleAccumulator(min_energy)
        with torch.no_grad():
            i = 0
            for x in loader:
                x = x.to(device, non_blocking=True)
                b, H, W, C = x.shape
                m = torch.from_numpy(mask_np[i:i + b]).to(device)
                x_masked = torch.where(m.unsqueeze(-1), torch.zeros_like(x), x)
                recon = model.reconstruct(x_masked)
                acc.update(x, recon)
                i += b
        drop_raw.append(acc.result()["sam"])
    out["missing_pixel_sam_masked"] = float(np.mean(drop_raw))

    return out


def interpolation_and_occupancy(latents_full: list[torch.Tensor], model, device,
                                model_name: str, n_pairs: int, n_alpha: int,
                                pixel: tuple[int, int], dataset_seed: int) -> dict:
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
    with torch.no_grad():
        for alpha in alphas:
            mix = lerp_latents(la, lb, float(alpha))
            recon = model.decode_latents(mix)          # (n_pairs, H, W, C)
            spectra.append(recon[:, r, c, :].detach().cpu().numpy())
    spectra = np.stack(spectra, axis=0)                 # (n_alpha, n_pairs, C)

    if n_alpha >= 3:
        second_diff = spectra[:-2] - 2.0 * spectra[1:-1] + spectra[2:]
        jaggedness_per_pair = np.linalg.norm(second_diff, axis=-1).mean(axis=0)  # (n_pairs,)
    else:
        jaggedness_per_pair = np.full(n_pairs, float("nan"))
    steps = np.linalg.norm(np.diff(spectra, axis=0), axis=-1)   # (n_alpha-1, n_pairs)
    path_length_per_pair = steps.sum(axis=0)

    occ = {}
    stream_names = ["spatial", "spectral"] if len(latents_full) == 2 else ["spectral"]
    for name, t in zip(stream_names, latents_full):
        flat = _flatten_latent_for_occupancy(t, model_name)
        occ[name] = effective_occupancy(flat)
    if len(latents_full) == 1:
        occ.setdefault("spatial", float("nan"))

    return {
        "jaggedness": float(np.mean(jaggedness_per_pair)),
        "path_length": float(np.mean(path_length_per_pair)),
        "occupancy_spatial": occ.get("spatial", float("nan")),
        "occupancy_spectral": occ.get("spectral", float("nan")),
        "occupancy_mean": float(np.nanmean([occ.get("spatial", float("nan")),
                                            occ.get("spectral", float("nan"))])),
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
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
    log.send(f"Inference final started — {total_cells} cells "
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
                batch_size = args.batch_size or settings.batch_size
                num_workers = args.num_workers if args.num_workers is not None else settings.num_workers
                min_energy = cfg["p1_trivial_floors"]["sam_valid_min_energy"]
                ext_cfg = _recon_metrics_ext_cfg()
                H, W = settings.input_height, settings.input_width

                g = torch.Generator(device="cpu").manual_seed(cfg["sampling"]["seed"] + 3)
                shuffle_perm = torch.randperm(H * W, generator=g).to(device)

                mask_by_rng = {}
                for rng_seed in RNG_SEEDS:
                    rng = np.random.default_rng(rng_seed)
                    mask_by_rng[rng_seed] = rng.random((n_total, H, W)) < PIXEL_MASK_FRACTION

                stats = train_band_statistics(ds, cfg, packed_root=str(packed_root))
                floor_x, floor_scenes = stratified_subsample(
                    inference_set, scenes, args.p1_floor_sample, cfg["sampling"]["seed"])
                floors_cache = out_dir / "model-validity-probes" / f"floors_{ds}.json"
                floors = get_or_build_floors(ds, cfg, floor_x, floor_scenes, stats, device, floors_cache)
            except Exception as e:
                tb = traceback.format_exc()
                log.send_pre(f"❌ Inference final - dataset setup FAILED on {ds}: {e}", tb[-2000:])
                print(f"Error setting up dataset {ds}:\n{tb}", file=sys.stderr)
                continue

            for exp in ("model-validity-probes", "reconstruction-quality", "noise-recovery",
                       "chemical-interpolation", "missing-pixel-recovery"):
                log.send(f"Inference final - {exp} - START - {ds}")

            ds_cells = [(m, l) for d, m, l in cells if d == ds]
            for model_name, loss in ds_cells:
                cell_idx += 1
                cell_cache_path = cache_dir / f"{ds}__{model_name}__{loss}.json"

                # ---- Resume / cache check --------------------------------
                if not args.overwrite and cell_cache_path.is_file():
                    try:
                        cached_cell = json.loads(cell_cache_path.read_text(encoding="utf-8"))
                        cached_rows = cached_cell.get("rows", {})
                        if all(k in cached_rows for k in CSV_COLUMNS):
                            for name in CSV_COLUMNS:
                                all_rows[name].append(cached_rows[name])
                            log.send(f"Inference final [{cell_idx}/{total_cells}] - {ds} | {model_name} | {loss} - LOADED FROM CACHE, skipping")
                            flush_all_csvs(all_rows, CSV_COLUMNS, out_dir)
                            continue
                    except Exception as e:
                        print(f"Failed to read cache {cell_cache_path}: {e}")

                log.send(f"Inference final [{cell_idx}/{total_cells}] - START - {ds} | {model_name} | {loss}")

                try:
                    per_seed = []
                    occ_per_seed = []
                    for seed in CHECKPOINT_SEEDS:
                        try:
                            ckpt_path = resolve_checkpoint(args.ckpt_dir, ds, model_name, loss,
                                                           seed=seed, select="sam")
                            if not ckpt_path.is_file():
                                raise FileNotFoundError(f"missing checkpoint {ckpt_path}")
                            model, _ = load_model(model_name, ckpt_path, device)
                            loader = _make_loader(inference_set, batch_size, num_workers)
                            res = evaluate_one_checkpoint(
                                model, loader, device, cfg, min_energy, ext_cfg, model_name,
                                shuffle_perm, mask_by_rng)
                            occ_res = interpolation_and_occupancy(
                                res["latents_full"], model, device, model_name,
                                args.n_interp_pairs, args.n_alpha, tuple(args.pixel),
                                cfg["sampling"]["seed"])
                            res.pop("latents_full")
                            per_seed.append(res)
                            occ_per_seed.append(occ_res)
                            del model
                            torch.cuda.empty_cache()
                        except Exception as e:  # noqa: BLE001
                            tb = traceback.format_exc()
                            log.send_pre(f"❌ Inference final - {model_name}|{ds}|{loss}|seed{seed} FAILED: {e}", tb[-2000:])
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

                    def frac(key_path, seeds=per_seed):
                        vals = [1.0 if r[key_path[0]][key_path[1]] else 0.0 for r in seeds]
                        return float(np.mean(vals))

                    recon_row = {k: avg(["reconstruction", k]) for k in
                                ("mse", "sam_rad", "sam_valid", "valid_pixel_frac", "psnr",
                                 "ssim", "sid", "sid_clamped_frac", "scc", "q2n", "n_samples")}
                    recon_row.update({"dataset": ds, "model": model_name, "loss": loss})

                    recon_for_p1 = {"psnr": recon_row["psnr"], "ssim": recon_row["ssim"],
                                    "sam": recon_row["sam_rad"], "sam_valid": recon_row["sam_valid"]}
                    n_sub = floors.get("random_null_patches", 8) if not floors.get("error") else 8
                    n_sub = min(n_sub, floor_x.shape[0])
                    sub_recon = None
                    try:
                        ckpt_path = resolve_checkpoint(args.ckpt_dir, ds, model_name, loss,
                                                       seed=CHECKPOINT_SEEDS[0], select="sam")
                        probe_model, _ = load_model(model_name, ckpt_path, device)
                        with torch.no_grad():
                            sub_x = floor_x[:n_sub].to(device)
                            sub_recon_t = probe_model.reconstruct(sub_x)
                        sub_recon = small_sample_metrics(sub_x, sub_recon_t, min_energy)
                        del probe_model
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        sub_recon = {"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan"),
                                    "sam": float("nan"), "sam_valid": float("nan")}
                    p1r = p1_report(recon_for_p1, sub_recon, floors, cfg) if not floors.get("error") else {"error": floors["error"]}

                    validity_row = {
                        "dataset": ds, "model": model_name, "loss": loss,
                        "latent_elements": avg(["p2", "latent_elements"]),
                        "compression_ratio": avg(["p2", "compression_ratio"]),
                        "rate_dev_pct": avg(["p2", "deviation_pct"]),
                        "rate_matched_frac": frac(["p2", "rate_matched"]),
                        "active_units": avg(["p3", "active_unit_fraction"]),
                        "mean_kl_per_dim": avg(["p3", "mean_kl_per_dim"]),
                        "latent_swap_delta": avg(["p3", "latent_swap_delta"]),
                        "recon_std_across_batch": avg(["p3", "recon_std_across_batch"]),
                        "collapsed_frac": frac(["p3", "collapsed"]),
                        "sri": avg(["p4", "sri"]), "sam_intact": avg(["p4", "sam_intact"]),
                        "sam_shuffled": avg(["p4", "sam_shuffled"]),
                        "psnr_intact": avg(["p4", "psnr_intact"]), "psnr_shuffled": avg(["p4", "psnr_shuffled"]),
                        "uses_spatial_context_frac": frac(["p4", "uses_spatial_context"]),
                        "p1_floor_psnr": p1r.get("best_zero_rate_floor", {}).get("psnr", float("nan")),
                        "p1_floor_ssim": p1r.get("best_zero_rate_floor", {}).get("ssim", float("nan")),
                        "p1_floor_sam_valid": p1r.get("best_zero_rate_floor", {}).get("sam_valid", float("nan")),
                        "p1_lift_psnr_db": p1r.get("lift_over_zero_rate", {}).get("psnr_db", float("nan")),
                        "p1_lift_ssim": p1r.get("lift_over_zero_rate", {}).get("ssim_absolute", float("nan")),
                        "p1_lift_sam_rel": p1r.get("lift_over_zero_rate", {}).get("sam_relative", float("nan")),
                        "p1_lift_sam_valid_rel": p1r.get("lift_over_zero_rate", {}).get("sam_valid_relative", float("nan")),
                        "p1_meanpatch_psnr": p1r.get("mean_patch", {}).get("psnr", float("nan")),
                        "p1_meanpatch_sam_valid": p1r.get("mean_patch", {}).get("sam_valid", float("nan")),
                        "p1_lift_vs_meanpatch_psnr_db": p1r.get("lift_over_mean_patch", {}).get("psnr_db", float("nan")),
                        "p1_lift_vs_meanpatch_sam_valid_rel": p1r.get("lift_over_mean_patch", {}).get("sam_valid_relative", float("nan")),
                        "p1_headroom_psnr": p1r.get("headroom_captured_psnr", float("nan")),
                    }

                    noise_row = {"dataset": ds, "model": model_name, "loss": loss}
                    for sigma in SIGMAS:
                        noise_row[f"sam_recovery_s{sigma}"] = avg(["noise", sigma, "sam"])
                        noise_row[f"psnr_recovery_s{sigma}"] = avg(["noise", sigma, "psnr"])

                    def occ_avg(key, seeds=occ_per_seed):
                        vals = [r[key] for r in seeds if not math.isnan(r[key])]
                        return float(np.mean(vals)) if vals else float("nan")

                    interp_row = {
                        "dataset": ds, "model": model_name, "loss": loss,
                        "jaggedness": float(np.mean([r["jaggedness"] for r in occ_per_seed])),
                        "path_length": float(np.mean([r["path_length"] for r in occ_per_seed])),
                        "occupancy_spatial": occ_avg("occupancy_spatial"),
                        "occupancy_spectral": occ_avg("occupancy_spectral"),
                        "occupancy_mean": occ_avg("occupancy_mean"),
                    }

                    sam_masked = avg(["missing_pixel_sam_masked"])
                    sam_clean = recon_row["sam_rad"]
                    mp_row = {"dataset": ds, "model": model_name, "loss": loss,
                             "sam_clean": sam_clean, "sam_masked": sam_masked,
                             "sam_drop": sam_masked - sam_clean}

                    # Append to current memory rows
                    all_rows["reconstruction-quality.csv"].append(recon_row)
                    all_rows["model-validity-probes.csv"].append(validity_row)
                    all_rows["noise-recovery.csv"].append(noise_row)
                    all_rows["chemical-interpolation.csv"].append(interp_row)
                    all_rows["missing-pixel-recovery.csv"].append(mp_row)

                    # Save intermediate cache & write CSVs
                    cell_cache = {
                        "dataset": ds,
                        "model": model_name,
                        "loss": loss,
                        "rows": {
                            "reconstruction-quality.csv": recon_row,
                            "model-validity-probes.csv": validity_row,
                            "noise-recovery.csv": noise_row,
                            "chemical-interpolation.csv": interp_row,
                            "missing-pixel-recovery.csv": mp_row,
                        },
                    }
                    cell_cache_path.write_text(json.dumps(cell_cache, indent=2), encoding="utf-8")
                    flush_all_csvs(all_rows, CSV_COLUMNS, out_dir)

                    log.send(f"Inference final [{cell_idx}/{total_cells}] - DONE - {ds} | {model_name} | {loss} "
                             f"(SAM={recon_row['sam_rad']:.4f}, PSNR={recon_row['psnr']:.2f}dB)")
                except Exception as e:
                    tb = traceback.format_exc()
                    log.send_pre(f"❌ Inference final - {model_name}|{ds}|{loss} CELL PROCESSING FAILED: {e}", tb[-2000:])
                    print(f"Error processing cell {model_name}|{ds}|{loss}:\n{tb}", file=sys.stderr)
                    continue

    except Exception as e:
        had_fatal_error = True
        tb = traceback.format_exc()
        log.send_pre(f"❌ Inference final UNHANDLED FATAL ERROR: {e}", tb[-2500:])
        print(f"Fatal unhandled exception:\n{tb}", file=sys.stderr)

    # Persist and send whatever CSV rows we have collected
    flush_all_csvs(all_rows, CSV_COLUMNS, out_dir)
    for name, cols in CSV_COLUMNS.items():
        path = out_dir / name
        if path.is_file():
            log.send_document(path, caption=name)

    # --- Step 4.2: the lift Telegram message ------------------------------
    recon_rows = all_rows.get("reconstruction-quality.csv", [])
    ours = [r for r in recon_rows if r.get("model") == "vae-our-nl"]
    others = [r for r in recon_rows if r.get("model") != "vae-our-nl"]
    if ours and others:
        ours_sorted = sorted(ours, key=lambda r: (r.get("sam_rad", 999), -r.get("psnr", 0)))
        others_sorted = sorted(others, key=lambda r: (r.get("sam_rad", 999), -r.get("psnr", 0)))
        our_best, other_best = ours_sorted[0], others_sorted[0]
        sam_lift = other_best["sam_rad"] - our_best["sam_rad"]
        psnr_lift = our_best["psnr"] - other_best["psnr"]
        msg = (
            f"Inference final — lift summary\n"
            f"vae-our-nl best: {our_best['dataset']} "
            f"(SAM={our_best['sam_rad']:.4f} rad, PSNR={our_best['psnr']:.2f} dB)\n"
            f"strongest other: {other_best['model']}|{other_best['dataset']}|{other_best['loss']} "
            f"(SAM={other_best['sam_rad']:.4f} rad, PSNR={other_best['psnr']:.2f} dB)\n"
            f"SAM lift: {sam_lift:+.4f} rad ({'WIN' if sam_lift > 0 else 'LOSS'})\n"
            f"PSNR lift: {psnr_lift:+.2f} dB ({'WIN' if psnr_lift > 0 else 'LOSS'})"
        )
        log.send(msg)

    elapsed = time.time() - t0
    summary = "\n".join(f"{name}: {len(rows)} rows" for name, rows in all_rows.items())
    log.send_pre(f"Inference final complete — {elapsed/60:.1f} min", summary)

    return 1 if had_fatal_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
