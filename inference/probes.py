"""
inference/probes.py
-------------------
Mechanism DIAGNOSTICS on frozen checkpoints (formerly the falsification suite).

2026-09-04 DEMOTION (docs/new_plan.md): the suite-level PASS/FAIL/INVALID
adjudication and four probes were retired. P1's trivial-predictor floor was
miscalibrated (a copy-through "trivial" predictor out-scored trained models on
clean data, flagging every cell INVALID), and P5/P6/P7 shipped with scaling
bugs (NPR at 13-27 against a ~1.0 legend, physics R^2 at -1e9). Rather than
patch an adjudication layer the paper no longer leans on, the probes that are
correct and directly support the paper's claims are kept as DIAGNOSTICS, with
no pass/fail semantics:

  P2  latent budget / rate       the fairness certificate: every model encodes
                                 to the common budget T (+ vae-our per-branch
                                 MSE decomposition)
  P3  posterior collapse         active units + latent-swap; a collapsed cell
                                 is excluded from ranking (`collapsed: true`)
  P4  spatial-reliance shuffle   SRI — does the model use spatial context?
                                 (the Iteration-1 before/after figure)

plus `sam_valid` (pi/2-excluded SAM), which inference/inference.py now also
reports in the headline table. The paired-statistics layer (bootstrap CIs,
permutation p, Holm) lives in inference/stats.py and is unchanged — rankings
still need it; they just no longer pass through a verdict gate.

Usage
=====
    PYTHONPATH=. python inference/probes.py --dataset IIRS --model vae-our --loss physics
    PYTHONPATH=. python inference/probes.py --dataset IIRS --all-models
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
# Repo root must come FIRST: this file's own directory is sys.path[0] when run as
# a script, and it contains inference.py, which would otherwise shadow the
# `inference` package and break `from inference.inference import ...`.
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from inference.inference import compute_psnr, compute_ssim, load_model  # noqa: E402
from modules.losses import spectral_angle_mapper_loss  # noqa: E402
from modules.registry import MODEL_NAMES, PHYSICS_ONLY, checkpoint_name, resolve_checkpoint  # noqa: E402
from utils.config import DATASETS, apply_dataset, settings  # noqa: E402
from utils.hyperparams import apply_cli_overrides, apply_hyperparams, load_hyperparams  # noqa: E402
from utils.training.dataloader import build_dataset  # noqa: E402

PREREG_PATH = REPO_ROOT / "inference" / "preregistration.yaml"


def load_prereg() -> dict:
    if not PREREG_PATH.is_file():
        raise SystemExit(
            f"Preregistration file missing: {PREREG_PATH}\n"
            "Probes read every threshold from it and will not run without it — "
            "that is the point of preregistering."
        )
    return yaml.safe_load(PREREG_PATH.read_text())


def _gpu_processes() -> str:
    """Compute processes currently on the GPU, via nvidia-smi. Best effort."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "(none reported)"
    except Exception:
        return "(nvidia-smi unavailable)"


def preflight_vram(device: torch.device, min_free_gb: float = 2.0) -> None:
    """
    Warn (never abort) when the card is already occupied before probes starts.

    Mirrors train/train.py:preflight_vram, at a lower threshold -- probes.py's
    peak footprint per cell is now a few hundred MB to a couple GB (see the
    chunked accumulators below), not a training batch's worth. This does not
    explain the 2026-09-17 AVIRIS OOM (the card was idle at start; that crash
    was probes.py's own unchunked peak inside a single cell -- see the module
    docstring below -- now fixed). It exists for the other failure mode
    training's preflight already guards against: a leaked/co-resident process
    that a file lock alone cannot see.
    """
    if device.type != "cuda":
        return
    try:
        free_b, total_b = torch.cuda.mem_get_info()
    except Exception:
        return
    free_gb, total_gb = free_b / 1024 ** 3, total_b / 1024 ** 3
    used_gb = total_gb - free_gb
    if used_gb > 1.0:
        print(f"WARNING: {used_gb:.1f} GB of {total_gb:.1f} GB GPU already in "
              f"use before this run starts (free: {free_gb:.1f} GB free). "
              f"If this OOMs, that is likely why.\n"
              f"  compute apps: {_gpu_processes()}")


# ---------------------------------------------------------------------------
# Chunked model calls
# ---------------------------------------------------------------------------
# Probes evaluate up to `max_patches` (512) patches per cell. Pushing that
# through a model in one forward pass would OOM even a 24 GB card -- the
# training batch size for these same models is 16-32. Every model call in this
# module therefore goes through these helpers, which chunk along the batch axis
# and reassemble. Purely a memory-management concern: results are identical to
# an unchunked call because none of these models mix information across the
# batch dimension.
#
# 2026-09-17 AVIRIS OOM: chunking the MODEL CALL (below) was not enough by
# itself. The old batched_decode/batched_forward helpers still `torch.cat`
# their chunks back into one full (B,H,W,C) tensor before returning -- at
# AVIRIS's 512 patches x 424 bands, that is 3.3 GiB PER TENSOR, and
# p2_latent_budget/p3_collapse/p4_spatial_reliance each held several such
# tensors live at once (a peak of ~20-26 GiB against a 23.4 GB card), even
# though none of their final numbers need more than a scalar or a small
# running accumulator. `ChunkedMetricAccumulator` below and the chunked
# rewrites of P2/P3/P4 fix this by never materialising the full tensor --
# every number they produce is mathematically identical to the old
# full-tensor computation (see the class docstring for exactly why), not an
# approximation, and the preregistered 512-patch sample stays unchanged.
# batched_decode/batched_forward were removed once nothing called them any
# more; batched_reconstruct/batched_encode are still used (run_cell's one
# necessary full reconstruction, and P3's small latent encode).

PROBE_BATCH = 8


@torch.no_grad()
def batched_reconstruct(model, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([model.reconstruct(x[i:i + PROBE_BATCH])
                      for i in range(0, x.shape[0], PROBE_BATCH)], dim=0)


@torch.no_grad()
def batched_encode(model, x: torch.Tensor) -> list[torch.Tensor]:
    parts = [model.encode_latents(x[i:i + PROBE_BATCH])
             for i in range(0, x.shape[0], PROBE_BATCH)]
    return [torch.cat([p[j] for p in parts], dim=0) for j in range(len(parts[0]))]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sam_per_pixel(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Per-pixel spectral angle, (B, H, W). Same formula as
    modules/losses.spectral_angle_mapper_loss and sam_valid below, factored
    out so the chunked accumulator and the whole-tensor path agree exactly."""
    dot = (x * recon).sum(dim=-1)
    nt = torch.sqrt((x ** 2).sum(dim=-1) + 1e-8)
    np_ = torch.sqrt((recon ** 2).sum(dim=-1) + 1e-8)
    cos = torch.clamp(dot / (nt * np_ + 1e-8), -1 + 1e-8, 1 - 1e-8)
    return torch.acos(cos)


def sam_valid(x: torch.Tensor, recon: torch.Tensor, min_energy: float) -> float:
    """
    SAM restricted to pixels carrying real signal.

    SAM normalises by sqrt(sum(x^2) + 1e-8). For a pixel whose spectral energy is
    far below that epsilon the norm is dominated by it, cos_sim collapses to ~0,
    and the pixel contributes exactly pi/2 NO MATTER WHAT THE MODEL PREDICTED.
    CRIMS has ~24% such pixels, so its raw SAM carries a hard floor of about
    0.24 * pi/2 ~= 0.377 rad that has nothing to do with model quality. Excluding
    them is what makes SAM comparable across datasets.
    """
    energy = (x ** 2).sum(dim=-1)
    mask = energy >= min_energy
    if mask.sum() == 0:
        return float("nan")
    angle = _sam_per_pixel(x, recon)
    return float(angle[mask].mean())


def metrics(x: torch.Tensor, recon: torch.Tensor, min_energy: float) -> dict:
    return {
        "mse": float(F.mse_loss(recon, x)),
        "psnr": float(compute_psnr(x, recon)),
        "ssim": float(compute_ssim(x, recon)),
        "sam": float(spectral_angle_mapper_loss(x, recon)),
        "sam_valid": sam_valid(x, recon, min_energy),
    }


class ChunkedMetricAccumulator:
    """
    Streaming equivalent of metrics(x, recon, min_energy): feed
    (x_chunk, recon_chunk) pairs instead of holding a full (B,H,W,C) tensor.

    Each running total below reproduces its single-tensor counterpart EXACTLY
    (up to float summation order, ~1e-6 relative -- not an approximation):
      mse / psnr : modules/metrics.py's psnr() calls F.mse_loss on the WHOLE
                   tensor -- a global mean of squared error over every
                   element. Accumulate sum-of-squared-error and element count,
                   and take log10 only once at the end from the aggregate MSE.
                   Averaging per-chunk PSNR values would be WRONG here (log is
                   nonlinear, mean-of-logs != log-of-mean).
      ssim       : modules/metrics.py's ssim() is ALREADY a sample-count-
                   weighted mean over its own internal chunks
                   (`total = total + m.mean() * a_c.shape[0]`) -- weighting
                   our own chunk-level compute_ssim() calls by chunk size
                   reproduces that identically.
      sam        : spectral_angle_mapper_loss's reduction is torch.mean over
                   every (B,H,W) pixel -- accumulate the angle sum and pixel
                   count.
      sam_valid  : same, restricted to the energy-valid mask.
    """

    def __init__(self, min_energy: float):
        self.min_energy = min_energy
        self.sse = 0.0
        self.n_elem = 0
        self.ssim_wsum = 0.0
        self.n_samples = 0
        self.sam_sum = 0.0
        self.n_pixels = 0
        self.sam_valid_sum = 0.0
        self.n_valid_pixels = 0

    def update(self, x_c: torch.Tensor, recon_c: torch.Tensor) -> None:
        b = x_c.shape[0]
        self.sse += float(((recon_c - x_c) ** 2).sum())
        self.n_elem += x_c.numel()
        self.ssim_wsum += compute_ssim(x_c, recon_c) * b
        self.n_samples += b
        angle = _sam_per_pixel(x_c, recon_c)
        self.sam_sum += float(angle.sum())
        self.n_pixels += angle.numel()
        energy = (x_c ** 2).sum(dim=-1)
        mask = energy >= self.min_energy
        nvalid = int(mask.sum())
        if nvalid:
            self.sam_valid_sum += float(angle[mask].sum())
            self.n_valid_pixels += nvalid

    def result(self) -> dict:
        mse = self.sse / max(self.n_elem, 1)
        psnr_v = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        ssim_v = self.ssim_wsum / max(self.n_samples, 1)
        sam_v = self.sam_sum / max(self.n_pixels, 1)
        sam_valid_v = (self.sam_valid_sum / self.n_valid_pixels
                       if self.n_valid_pixels else float("nan"))
        return {"mse": mse, "psnr": psnr_v, "ssim": ssim_v,
                "sam": sam_v, "sam_valid": sam_valid_v}


def per_patch_metrics(x: torch.Tensor, recon: torch.Tensor, min_energy: float) -> dict:
    """Same metrics but one value per patch — what the paired statistics need."""
    out = {k: [] for k in ("mse", "psnr", "ssim", "sam", "sam_valid")}
    for i in range(x.shape[0]):
        m = metrics(x[i:i + 1], recon[i:i + 1], min_energy)
        for k, v in m.items():
            out[k].append(v)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_patches(dataset: str, split: str, cfg: dict, packed_root=None,
                 data_root=None) -> tuple[torch.Tensor, list[str]]:
    """Sampled test patches plus their scene labels (for P7's nuisance probe)."""
    ds = build_dataset(dataset, split, processed_root=data_root, packed_root=packed_root)
    n = len(ds)
    cap = cfg["sampling"]["max_patches"] or n
    rng = np.random.default_rng(cfg["sampling"]["seed"])

    scenes = _scene_labels(ds, n)
    if cap < n and cfg["sampling"].get("stratify_by_scene", True) and scenes:
        # Proportional draw per scene: a flat sample over a dataset whose scenes
        # differ several-fold in patch count would silently over-represent the
        # big ones, and P7's scene probe would then be measuring the imbalance.
        by = {}
        for i, s in enumerate(scenes):
            by.setdefault(s, []).append(i)
        idx = []
        for s in sorted(by):
            k = max(1, round(cap * len(by[s]) / n))
            idx.extend(rng.choice(by[s], size=min(k, len(by[s])), replace=False))
        idx = np.sort(np.asarray(idx[:cap]))
    else:
        idx = np.sort(rng.choice(n, size=min(cap, n), replace=False))

    x = torch.stack([ds[int(i)] for i in idx])
    lbl = [scenes[int(i)] if scenes else "unknown" for i in idx]
    return x, lbl


def _scene_labels(ds, n: int) -> list[str]:
    meta = getattr(ds, "meta", None)
    if meta and meta.get("source_files"):
        return [Path(p).parts[0] for p in meta["source_files"]]
    files = getattr(ds, "patch_files", None)
    if files:
        return [p.parent.parent.name for p in files]
    return []


# NOTE (2026-09-04): `train_statistics` and the P1 trivial-floor machinery
# (p1_shared_floors / p1_trivial_floors), the P5 inpainting probe, the P6
# purification probe and the P7 linear probe were REMOVED in the falsification
# demotion — P1's floor was miscalibrated and P5-P7 had scaling bugs; see the
# module docstring and docs/new_plan.md. git history has the code.


# ---------------------------------------------------------------------------
# P2 — Latent budget / rate control
# ---------------------------------------------------------------------------

def p2_latent_budget(model, x, model_name, cfg) -> dict:
    """
    Verify the rate match held, and report per-branch MSE for vae-our.

    After the exact budget matching this is a verification rather than a
    diagnosis: if rates are equal, a reconstruction difference is attributable
    to architecture. The per-branch split exists because vae-our's total_mse
    mixes three terms (0.5:w:w, normalised — see modules/vae_our.py), and each
    stream alone reconstructs the cube worse than the fusion — so a large
    mse_spatial/mse_spectral is a property of the auxiliary objective, not
    evidence about the model's reconstruction quality. Quote mse_final.
    """
    p = cfg["p2_latent_budget"]
    with torch.no_grad():
        lat = model.encode_latents(x[:1])
    elements = int(sum(t.numel() for t in lat))
    inp = int(np.prod(x.shape[1:]))
    ratio = inp / elements
    dev = 100 * (ratio - p["target_ratio"]) / p["target_ratio"]
    out = {
        "latent_elements": elements,
        "latent_shapes": [tuple(t.shape[1:]) for t in lat],
        "input_elements": inp,
        "compression_ratio": ratio,
        "deviation_pct": dev,
        "bits_per_pixel_per_band": 32.0 * elements / inp,
        "rate_matched": abs(dev) <= p["match_tolerance_pct"],
    }
    if model_name in ("vae-our", "vae-our-nl") and p.get("report_per_branch_mse", True):
        # Chunked, not batched_forward + F.mse_loss on the full tensor: the
        # model's 7-tuple forward (3 full-size reconstructions + 4 latents)
        # concatenated to (B,H,W,C) is 3.3 GiB per tensor at AVIRIS scale, and
        # this call held three of them live at once (~20 GiB transient) --
        # the single biggest contributor to the 2026-09-17 OOM. Only a scalar
        # per branch is needed, so accumulate sum-of-squared-error per chunk
        # instead; algebraically identical to F.mse_loss's global mean (see
        # ChunkedMetricAccumulator's docstring above for the same argument).
        sse_f = sse_s = sse_p = 0.0
        n_elem = 0
        with torch.no_grad():
            for i in range(0, x.shape[0], PROBE_BATCH):
                xb = x[i:i + PROBE_BATCH]
                rf, rs, rp, *_ = model(xb)
                sse_f += float(((rf - xb) ** 2).sum())
                sse_s += float(((rs - xb) ** 2).sum())
                sse_p += float(((rp - xb) ** 2).sum())
                n_elem += xb.numel()
                del rf, rs, rp
        out["per_branch_mse"] = {
            "mse_final": sse_f / max(n_elem, 1),
            "mse_spatial": sse_s / max(n_elem, 1),
            "mse_spectral": sse_p / max(n_elem, 1),
        }
        t = out["per_branch_mse"]
        # Mirror the ACTUAL training mix (modules/vae_our.py loss_terms):
        # (0.5*final + w*spatial + w*spectral) / (0.5 + 2w), w from settings.
        w_aux = settings.vae_our_aux_mse_weight
        denom = 0.5 + 2.0 * w_aux
        t["total_mse"] = (0.5 * t["mse_final"] + w_aux * t["mse_spatial"]
                          + w_aux * t["mse_spectral"]) / denom
        t["final_share_of_total"] = (0.5 / denom) * t["mse_final"] / max(t["total_mse"], 1e-12)
    out["verdict"] = "PASS" if out["rate_matched"] else "FAIL"
    return out


# ---------------------------------------------------------------------------
# P3 — Posterior collapse / latent usage
# ---------------------------------------------------------------------------

def p3_collapse(model, x, cfg, min_energy) -> dict:
    """
    Is the latent used at all?

    Three independent signals, because each alone can mislead: per-dimension KL
    (a unit carrying no information has KL ~ 0), the variance ratio, and a
    latent-swap. The swap is the decisive one — decode patch i's latent into
    patch j's slot and see whether the output moves. If it barely does, the
    decoder is ignoring the latent and producing a constant, which is exactly
    the failure that made two IIRS cells report SAM = pi/2.
    """
    p = cfg["p3_collapse"]
    out = {}
    lat = batched_encode(model, x)

    # Per-dimension KL from the deterministic latents, treating the aggregate
    # posterior's spread as the signal: a dead unit has near-zero variance
    # across the batch and contributes no information. Latents are small
    # (16k-29k elements here vs. a 3.3 GiB reconstruction), nowhere near the
    # memory problem below -- left as a whole-tensor computation.
    kls = []
    for t in lat:
        flat = t.reshape(t.shape[0], -1).double()
        var = flat.var(dim=0, unbiased=False)
        mean = flat.mean(dim=0)
        kl = 0.5 * (var + mean ** 2 - 1.0 - torch.log(var + 1e-12))
        kls.append(kl)
    kl_all = torch.cat(kls)
    active = (kl_all > p["active_unit_kl_nats"]).double().mean().item()
    out["n_latent_dims"] = int(kl_all.numel())
    out["active_unit_fraction"] = active
    out["mean_kl_per_dim"] = float(kl_all.mean())

    # Latent swap: roll the batch so every patch is decoded from another's
    # code. The roll is an information-mixing op across the batch, so it must
    # happen once on the (small) full latent tensors, before chunking --
    # everything downstream of that only ever touches PROBE_BATCH-sized
    # decoded chunks. batched_decode's un-chunked `torch.cat` result (base,
    # swapped) used to hold two full 3.3 GiB reconstructions live at once at
    # AVIRIS scale; only two scalar SAM values and a per-(H,W,C) running
    # moment (for recon_std_across_batch) are needed, so accumulate those
    # instead of materialising either full tensor.
    B = x.shape[0]
    lat_rolled = [torch.roll(t, 1, dims=0) for t in lat]
    sam_base_sum = sam_swap_sum = 0.0
    n_pixels = 0
    sum_x = sum_x2 = None
    with torch.no_grad():
        for i in range(0, B, PROBE_BATCH):
            x_c = x[i:i + PROBE_BATCH]
            base_c = model.decode_latents([t[i:i + PROBE_BATCH] for t in lat])
            swapped_c = model.decode_latents([t[i:i + PROBE_BATCH] for t in lat_rolled])

            angle_base = _sam_per_pixel(x_c, base_c)
            angle_swap = _sam_per_pixel(x_c, swapped_c)
            sam_base_sum += float(angle_base.sum())
            sam_swap_sum += float(angle_swap.sum())
            n_pixels += angle_base.numel()

            # recon_std_across_batch = base.std(dim=0).mean() over the FULL
            # batch: keep a running sum/sum-of-squares per (H,W,C) location
            # (~7 MB, not 3.3 GiB) instead of the full tensor, and reduce to
            # torch.std's default unbiased (correction=1) variance at the end.
            if sum_x is None:
                sum_x = base_c.sum(dim=0)
                sum_x2 = (base_c ** 2).sum(dim=0)
            else:
                sum_x += base_c.sum(dim=0)
                sum_x2 += (base_c ** 2).sum(dim=0)
            del base_c, swapped_c

    sam_base = sam_base_sum / max(n_pixels, 1)
    sam_swap = sam_swap_sum / max(n_pixels, 1)
    delta = abs(sam_swap - sam_base) / max(sam_base, 1e-12)
    out.update({"sam_own_latent": sam_base, "sam_swapped_latent": sam_swap,
                "latent_swap_delta": delta})

    # Output constancy: a collapsed decoder emits near-identical patches.
    mean_b = sum_x / B
    var_b = (sum_x2 - B * mean_b ** 2) / max(B - 1, 1)
    std_b = torch.sqrt(torch.clamp(var_b, min=0))
    out["recon_std_across_batch"] = float(std_b.mean())

    collapsed = (active < p["min_active_fraction"]
                 or delta < p["latent_swap_min_delta_sam"])
    out["collapsed"] = bool(collapsed)
    out["verdict"] = p["verdict_on_collapse"] if collapsed else "PASS"
    return out


# ---------------------------------------------------------------------------
# P4 — Spatial-reliance shuffle (shortcut test 1)
# ---------------------------------------------------------------------------

def p4_spatial_reliance(model, x, model_name, cfg, min_energy) -> dict:
    """
    Does the model actually use spatial context, or is it pixelwise in disguise?

    Permute the H*W pixel grid, keeping each pixel's spectrum intact, and score
    the shuffled reconstruction against the shuffled input. A model that only
    ever looks at one pixel at a time is EXACTLY permutation-equivariant, so its
    score cannot change; a model using neighbourhood context degrades.

    vae-1d is the positive control: its two scores must be bit-identical. Any
    deviation there means the probe itself is wrong — almost certainly the
    permutation applied to the input but not to the target when scoring — and is
    reported as a probe bug, never as a finding about the model.
    """
    p = cfg["p4_spatial_reliance"]
    B, H, W, C = x.shape
    g = torch.Generator(device="cpu").manual_seed(cfg["sampling"]["seed"] + 3)
    perm = torch.randperm(H * W, generator=g).to(x.device)

    flat = x.reshape(B, H * W, C)
    x_sh = flat[:, perm, :].reshape(B, H, W, C).contiguous()

    # Chunked reconstruct + streaming metrics instead of batched_reconstruct's
    # full torch.cat: r_int and r_sh were two full 3.3 GiB tensors held live
    # at once at AVIRIS scale (this alone was enough to OOM vae-standard,
    # which has no other large allocation in this module). Only the five
    # scalar metrics per variant are needed -- see ChunkedMetricAccumulator.
    acc_int = ChunkedMetricAccumulator(min_energy)
    acc_sh = ChunkedMetricAccumulator(min_energy)
    with torch.no_grad():
        for i in range(0, B, PROBE_BATCH):
            xb = x[i:i + PROBE_BATCH]
            rb = model.reconstruct(xb)
            acc_int.update(xb, rb)
            del rb
            xb_sh = x_sh[i:i + PROBE_BATCH]
            rb_sh = model.reconstruct(xb_sh)
            acc_sh.update(xb_sh, rb_sh)
            del rb_sh

    # Score each against ITS OWN input — the question is whether the model got
    # worse at the task, not whether the output moved.
    m_int = acc_int.result()
    m_sh = acc_sh.result()
    sri = (m_sh["sam"] - m_int["sam"]) / max(m_int["sam"], 1e-12)

    out = {
        "sam_intact": m_int["sam"], "sam_shuffled": m_sh["sam"],
        "psnr_intact": m_int["psnr"], "psnr_shuffled": m_sh["psnr"],
        "sri": sri,
        "uses_spatial_context": sri >= p["min_sri_for_spatial_use"],
    }

    if model_name == p["positive_control_model"]:
        drift = abs(m_sh["sam"] - m_int["sam"])
        out["positive_control_drift"] = drift
        out["positive_control_ok"] = drift <= p["positive_control_tolerance"]
        if not out["positive_control_ok"]:
            out["verdict"] = "PROBE_BUG"
            out["note"] = (
                f"vae-1d is exactly permutation-equivariant, so intact and "
                f"shuffled SAM must match to {p['positive_control_tolerance']:.0e}; "
                f"observed drift {drift:.3e}. The probe is wrong, not the model.")
            return out
        out["verdict"] = "PASS"   # control behaved; SRI ~ 0 is the expected result
        out["note"] = "positive control: pixelwise by construction, SRI ~ 0 expected"
        return out

    out["verdict"] = "PASS" if out["uses_spatial_context"] else "FAIL"
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def resolve_ckpt(model_name, dataset, loss, ckpt_dir, seed=None, select="sam"):
    """Locate one cell's checkpoint, honouring the seed axis and the two
    selection criteria (see modules/registry.py:resolve_checkpoint)."""
    return resolve_checkpoint(ckpt_dir, dataset, model_name, loss,
                              seed=seed, select=select)


def run_cell(model_name: str, dataset: str, loss: str, args, cfg,
             x, scenes, device) -> dict:
    ckpt = (Path(args.ckpt) if args.ckpt else
            resolve_ckpt(model_name, dataset, loss, args.ckpt_dir,
                         seed=args.seed, select=args.select))
    if not ckpt.is_file():
        return {"model": model_name, "dataset": dataset, "loss": loss,
                "error": f"missing checkpoint {ckpt}"}

    # load_model returns (model, checkpoint_dict); it already calls .eval().
    # A shape mismatch (e.g. a checkpoint trained under different --set
    # capacity overrides than are active now) must not abort the remaining
    # cells in the sweep — report it as an error cell instead.
    try:
        model, ckpt_meta = load_model(model_name, ckpt, device)
    except RuntimeError as e:
        return {"model": model_name, "dataset": dataset, "loss": loss,
                "error": f"load_state_dict failed for {ckpt}: {e}"}
    xd = x.to(device)
    eps = cfg["p1_trivial_floors"]["sam_valid_min_energy"]

    recon = batched_reconstruct(model, xd)
    recon_m = metrics(xd, recon, eps)
    torch.cuda.empty_cache()

    res = {
        "model": model_name, "dataset": dataset, "loss": loss,
        "seed": args.seed, "select": args.select,
        "checkpoint": str(ckpt), "n_patches": int(xd.shape[0]),
        "trained_epochs": ckpt_meta.get("epoch"),
        "best_val_loss": ckpt_meta.get("loss"),
        "preregistration": cfg.get("registered_on"),
        "reconstruction": recon_m,
    }
    # empty_cache() between P2/P3/P4: each is already peak-bounded to a few
    # hundred MB by the chunked accumulators (see their docstrings), but
    # clearing the allocator's cache between them keeps fragmentation from
    # compounding across the sequence of mid-size allocations in one cell.
    res["P2_latent_budget"] = p2_latent_budget(model, xd, model_name, cfg)
    torch.cuda.empty_cache()
    res["P3_collapse"] = p3_collapse(model, xd, cfg, eps)
    torch.cuda.empty_cache()
    res["P4_spatial_reliance"] = p4_spatial_reliance(model, xd, model_name, cfg, eps)
    torch.cuda.empty_cache()
    res["per_patch"] = {k: v.tolist() for k, v in
                        per_patch_metrics(xd, recon, eps).items()}

    # No suite-level PASS/FAIL verdict any more (2026-09-04 demotion). The one
    # exclusion that survives is collapse: a collapsed decoder emits a constant,
    # so its reconstruction metrics describe a trivial predictor, not a model —
    # rankings and pairwise stats must skip such cells.
    res["collapsed"] = bool(res["P3_collapse"]["collapsed"])
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mechanism diagnostics (P2 rate / P3 collapse / P4 SRI) "
                    "on frozen checkpoints.")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--model", choices=list(MODEL_NAMES))
    p.add_argument("--all-models", action="store_true")
    p.add_argument("--loss", default=None, choices=["standard", "physics"])
    p.add_argument("--losses", nargs="+", choices=["standard", "physics"], default=None,
                   help="With --all-models: restrict to these loss regimes. "
                        "Used to run only the physics cells for seeds that have "
                        "no standard-loss checkpoints (the manifest trains those "
                        "at the first seed only).")
    p.add_argument("--ckpt-dir", default="model")
    p.add_argument("--seed", type=int, default=None,
                    help="Which training seed's checkpoint to evaluate. Omit when only one seed exists; required once several do, since picking implicitly would make the result depend on file order.")
    p.add_argument("--select", choices=("sam", "mse"), default="sam",
                    help="Which checkpoint to load: the epoch selected on best val SAM (default) or on best val reconstruction MSE. Every cell writes both; a comparison must read the SAME criterion for every model.")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--packed-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--max-patches", type=int, default=None,
                   help="Override the preregistered sampling cap (0 = whole split).")
    p.add_argument("--n-random-draws", type=int, default=None,
                   help="DEPRECATED no-op (was P1's random-null draw count; P1 "
                        "was removed 2026-09-04). Accepted so existing scripts "
                        "don't break.")
    p.add_argument("--probe-batch", type=int, default=None,
                   help="Chunk size for model forwards inside probes "
                        "(memory only; does not change results).")
    p.add_argument("--out-dir", default="results/probes")
    p.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                   help="One-off Settings override, repeatable (e.g. --set "
                        "vae_3d_base_ch=30). Applied after the dataset YAML, "
                        "same semantics as train/train.py --set. Must match "
                        "whatever the checkpoint was actually trained with.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_prereg()
    if args.max_patches is not None:
        cfg["sampling"]["max_patches"] = args.max_patches
    global PROBE_BATCH
    PROBE_BATCH = (args.probe_batch
                   or cfg["sampling"].get("probe_batch", PROBE_BATCH))
    split = args.split or cfg.get("split", "test")

    apply_dataset(args.dataset, verify=True, processed_root=args.data_root)
    apply_hyperparams(settings, load_hyperparams(args.dataset))
    overrides = apply_cli_overrides(settings, args.set)
    if overrides:
        print(f"--set overrides active: {overrides}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preflight_vram(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"diagnostics | {args.dataset} | split={split} | preregistered {cfg['registered_on']}")
    try:
        x, scenes = load_patches(args.dataset, split, cfg, args.packed_root, args.data_root)
    except Exception as e:  # noqa: BLE001
        # Uncaught here used to kill the whole process before a single cell's
        # JSON was written and with no trace anywhere else -- probes.py sends
        # nothing to Telegram, so a crash at this line (e.g. the 2026-09-15
        # AVIRIS run, which failed identically on all 16 model/loss/seed
        # cells -- consistent with a shared-setup failure here rather than a
        # per-model bug) left zero evidence of what broke. Write it down
        # instead, so the next attempt is diagnosable without the original
        # terminal/log.
        tb = traceback.format_exc()
        print(f"FATAL: load_patches failed for {args.dataset}/{split}: {e}\n{tb}")
        (out_dir / f"{args.dataset}__LOAD_FAILURE.json").write_text(json.dumps({
            "dataset": args.dataset, "split": split, "stage": "load_patches",
            "error": str(e), "traceback": tb,
        }, indent=1))
        return 1
    print(f"  {x.shape[0]} patches, C={x.shape[-1]}, {len(set(scenes))} scenes")

    cells = []
    if args.all_models:
        for m in MODEL_NAMES:
            losses = ["physics"] if m in PHYSICS_ONLY else ["standard", "physics"]
            if args.losses:
                losses = [l for l in losses if l in args.losses]
            cells += [(m, l) for l in losses]
    else:
        m = args.model or "vae-our"
        losses = ["physics"] if m in PHYSICS_ONLY else [args.loss or "physics"]
        cells = [(m, l) for l in losses]

    rc = 0
    for m, l in cells:
        try:
            res = run_cell(m, args.dataset, l, args, cfg, x, scenes, device)
        except Exception as e:  # noqa: BLE001
            # A crash inside one cell (P2/P3/P4, outside run_cell's own
            # load_model guard) used to abort every remaining cell in an
            # --all-models invocation. Record it as an error cell instead and
            # keep going -- the caller (scripts/run_clean_grid.sh) already
            # treats an "error" result as a per-cell failure.
            tb = traceback.format_exc()
            print(f"  {m:<24} {l:<9} CRASHED  ({e})")
            res = {"model": m, "dataset": args.dataset, "loss": l,
                   "error": f"run_cell crashed: {e}", "traceback": tb}
        name = checkpoint_name(m, l, seed=args.seed, select=args.select).replace(".pt", "")
        (out_dir / f"{args.dataset}__{name}.json").write_text(json.dumps(res, indent=1))
        if res.get("error"):
            print(f"  {m:<24} {l:<9} MISSING  ({res['error']})")
            rc = 1
            continue
        p2, p3, p4 = (res["P2_latent_budget"], res["P3_collapse"],
                      res["P4_spatial_reliance"])
        print(f"  {m:<24} {l:<9} "
              f"{'COLLAPSED' if res['collapsed'] else 'ok':<10} "
              f"rate={p2['latent_elements']:>7,} ({p2['deviation_pct']:+.1f}%) "
              f"active={p3['active_unit_fraction']:.2f} "
              f"swapDSAM={p3['latent_swap_delta']:.3f} "
              f"SRI={p4['sri']:+.3f} "
              f"SAMv={res['reconstruction']['sam_valid']:.4f}")
    print(f"\nwrote {len(cells)} result file(s) to {out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
