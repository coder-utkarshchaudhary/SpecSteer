"""
inference/probes.py
-------------------
Falsification suite. Seven probes, each answering a specific way the headline
result could be fake, run against FROZEN checkpoints.

The suite is adversarial by design: every probe is a chance for the model to
fail, and the thresholds are fixed in inference/preregistration.yaml BEFORE the
run. This module refuses to start if that file is missing, so the rules cannot
be tuned to fit the results.

  P1  trivial-predictor floors   is it better than predicting the mean?
  P2  latent budget / rate       is the win capacity rather than architecture?
  P3  posterior collapse         is the latent used at all?
  P4  spatial-reliance shuffle   does it use spatial context, or just pixels?
  P5  spectral inpainting        does it have a spectral prior, or copy input?
  P6  noise pass-through         does it actually purify?  (the paper's claim)
  P7  linear probe on latents    does the latent encode chemistry or nuisance?

Verdicts are PASS / FAIL / INVALID. INVALID means the cell is not a usable
result at all (e.g. a collapsed decoder) and must be reported as a finding
rather than scored.

Usage
=====
    PYTHONPATH=. python inference/probes.py --dataset IIRS --model vae-our --loss physics
    PYTHONPATH=. python inference/probes.py --dataset IIRS --all-models
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from utils.hyperparams import apply_hyperparams, load_hyperparams  # noqa: E402
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


@torch.no_grad()
def batched_decode(model, latents: list[torch.Tensor]) -> torch.Tensor:
    n = latents[0].shape[0]
    return torch.cat([model.decode_latents([t[i:i + PROBE_BATCH] for t in latents])
                      for i in range(0, n, PROBE_BATCH)], dim=0)


@torch.no_grad()
def batched_forward(model, x: torch.Tensor) -> tuple:
    """Full forward, chunked. Returns tensors concatenated along the batch axis."""
    outs = [model(x[i:i + PROBE_BATCH]) for i in range(0, x.shape[0], PROBE_BATCH)]
    return tuple(torch.cat([o[j] for o in outs], dim=0) if torch.is_tensor(outs[0][j])
                 else outs[0][j] for j in range(len(outs[0])))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    dot = (x * recon).sum(dim=-1)
    nt = torch.sqrt((x ** 2).sum(dim=-1) + 1e-8)
    np_ = torch.sqrt((recon ** 2).sum(dim=-1) + 1e-8)
    cos = torch.clamp(dot / (nt * np_ + 1e-8), -1 + 1e-8, 1 - 1e-8)
    return float(torch.acos(cos)[mask].mean())


def metrics(x: torch.Tensor, recon: torch.Tensor, min_energy: float) -> dict:
    return {
        "mse": float(F.mse_loss(recon, x)),
        "psnr": float(compute_psnr(x, recon)),
        "ssim": float(compute_ssim(x, recon)),
        "sam": float(spectral_angle_mapper_loss(x, recon)),
        "sam_valid": sam_valid(x, recon, min_energy),
    }


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


def train_statistics(dataset: str, cfg: dict, packed_root=None, data_root=None) -> dict:
    """
    Mean/std spectra from the TRAIN split, for P1's floors and P5's baseline.

    Computed from train only and never from valid/test, so a "trivial" predictor
    is genuinely leakage-free and the comparison is honest.
    """
    ds = build_dataset(dataset, "train", processed_root=data_root, packed_root=packed_root)
    n = len(ds)
    rng = np.random.default_rng(cfg["sampling"]["seed"] + 1)
    k = min(cfg["sampling"]["max_patches"] or n, n)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    scenes = _scene_labels(ds, n)

    acc, sq, cnt = None, None, 0
    per_scene: dict[str, list] = {}
    for i in idx:
        a = ds[int(i)].numpy()
        m = a.mean(axis=(0, 1))
        acc = m if acc is None else acc + m
        sq = m ** 2 if sq is None else sq + m ** 2
        cnt += 1
        if scenes:
            per_scene.setdefault(scenes[int(i)], []).append(m)
    mean = acc / cnt
    std = np.sqrt(np.maximum(sq / cnt - mean ** 2, 1e-12))
    return {
        "mean_spectrum": mean.astype(np.float32),
        "std_spectrum": std.astype(np.float32),
        "scene_means": {k2: np.mean(v, axis=0).astype(np.float32)
                        for k2, v in per_scene.items()},
        "n_used": cnt,
    }


# ---------------------------------------------------------------------------
# P1 — Trivial-predictor floors
# ---------------------------------------------------------------------------

def p1_shared_floors(x, stats, scenes, cfg) -> dict:
    """
    The MODEL-INDEPENDENT half of P1: the trivial-predictor floors, the
    1000-draw random null, and the identity oracle. All of it is a function of
    the shared patch sample and the train statistics only, so it is computed
    ONCE per (dataset, seed) and reused across that cell's seven models --
    exactly as inference/scripts/inference.sh already claims in its comments.
    The random null alone is ~20 min of a run; recomputing it per cell was
    multiplying that by five.
    """
    p = cfg["p1_trivial_floors"]
    eps = p["sam_valid_min_energy"]
    B, H, W, C = x.shape
    shared = {"baselines": {}, "random_arrays": {}}

    def bcast(spec):
        return torch.as_tensor(spec, device=x.device).view(1, 1, 1, C).expand(B, H, W, C).contiguous()

    cands = {
        "mean_global": bcast(stats["mean_spectrum"]),
        "mean_fold": bcast(stats["mean_spectrum"]),   # train-fold only, by construction
        "mean_patch": x.mean(dim=(1, 2), keepdim=True).expand(B, H, W, C).contiguous(),
    }
    if stats["scene_means"]:
        sm = torch.stack([
            torch.as_tensor(stats["scene_means"].get(s, stats["mean_spectrum"]),
                            device=x.device) for s in scenes
        ])
        cands["mean_region"] = sm.view(B, 1, 1, C).expand(B, H, W, C).contiguous()

    for name, pred in cands.items():
        shared["baselines"][name] = metrics(x, pred, eps)

    # Random null: an empirical distribution rather than a single draw, so the
    # model's score gets an exact percentile instead of a hand-waved "better
    # than noise". The per-draw metric ARRAYS are kept (not just their means)
    # because the per-cell percentile test compares them against that cell's
    # own reconstruction score.
    rng = np.random.default_rng(cfg["sampling"]["seed"] + 2)
    lo, hi = float(x.min()), float(x.max())
    mu = torch.as_tensor(stats["mean_spectrum"], device=x.device).view(1, 1, 1, C)
    sd = torch.as_tensor(stats["std_spectrum"], device=x.device).view(1, 1, 1, C)
    draws = {"random_uniform": [], "random_normal": []}
    n_draw = p["n_random_draws"]
    sub = x[: min(8, B)]      # a few patches per draw keeps 1000 draws affordable
    for d in range(n_draw):
        g = torch.Generator(device="cpu").manual_seed(int(rng.integers(1 << 30)))
        u = (torch.rand(sub.shape, generator=g).to(x.device) * (hi - lo) + lo)
        nrm = (torch.randn(sub.shape, generator=g).to(x.device) * sd + mu)
        draws["random_uniform"].append(metrics(sub, u, eps))
        draws["random_normal"].append(metrics(sub, nrm, eps))

    for name, ds_ in draws.items():
        arr = {k: np.array([d[k] for d in ds_]) for k in ds_[0]}
        shared["random_arrays"][name] = {k: v.tolist() for k, v in arr.items()}
        shared["baselines"][name] = {k: float(np.nanmean(v)) for k, v in arr.items()}
        shared[f"{name}_best"] = {
            "psnr": float(np.nanmax(arr["psnr"])),
            "ssim": float(np.nanmax(arr["ssim"])),
            "sam": float(np.nanmin(arr["sam"])),
        }

    shared["identity_oracle"] = metrics(x, x.clone(), eps)

    fl = shared["baselines"]
    shared["best_floor"] = {
        "psnr": max(v["psnr"] for v in fl.values()),
        "ssim": max(v["ssim"] for v in fl.values()),
        "sam": min(v["sam"] for v in fl.values()),
        "sam_valid": min(v["sam_valid"] for v in fl.values()),
    }
    return shared


def p1_trivial_floors(model_recon, shared, cfg) -> dict:
    """
    The PER-CELL half of P1: score this model's reconstruction against the
    shared floors and the random null, and apply the preregistered lift test.

    A model that cannot beat "broadcast the mean spectrum" has learned nothing,
    however good its absolute numbers look. `mean_patch` -- the patch's own
    spatial-mean spectrum -- is the strongest of these and needs no training set
    whatsoever, so it is the one that really bites.

    The identity oracle in `shared` is the true ceiling (a perfect copy scores
    ~0.0223 on IIRS SAM, not 0, because of the norm epsilon), so "headroom
    captured" is measured against it rather than against zero.
    """
    p = cfg["p1_trivial_floors"]
    out = {
        "baselines": shared["baselines"],
        "identity_oracle": shared["identity_oracle"],
        "best_floor": shared["best_floor"],
    }
    for name in ("random_uniform", "random_normal"):
        arr = {k: np.asarray(v) for k, v in shared["random_arrays"][name].items()}
        out[f"{name}_best"] = shared[f"{name}_best"]
        out[f"{name}_percentile"] = {
            "psnr": float(np.mean(arr["psnr"] >= model_recon["psnr"])),
            "sam": float(np.mean(arr["sam"] <= model_recon["sam"])),
        }

    best = shared["best_floor"]
    lift = p["min_lift_over_best_floor"]
    checks = {
        "psnr": model_recon["psnr"] - best["psnr"] >= lift["psnr_db"],
        "ssim": model_recon["ssim"] - best["ssim"] >= lift["ssim_absolute"],
        "sam": (best["sam"] - model_recon["sam"]) / max(best["sam"], 1e-12) >= lift["sam_relative"],
        "beats_all_random": max(out["random_uniform_percentile"]["sam"],
                                out["random_normal_percentile"]["sam"]) <= p["max_random_percentile"],
    }
    out["checks"] = checks
    out["lift"] = {
        "psnr_db": model_recon["psnr"] - best["psnr"],
        "ssim": model_recon["ssim"] - best["ssim"],
        "sam_relative": (best["sam"] - model_recon["sam"]) / max(best["sam"], 1e-12),
    }
    # Fraction of the achievable gap the model actually closed.
    denom = out["identity_oracle"]["psnr"] - best["psnr"]
    out["headroom_captured_psnr"] = float(
        (model_recon["psnr"] - best["psnr"]) / denom) if denom > 0 else float("nan")
    out["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return out


# ---------------------------------------------------------------------------
# P2 — Latent budget / rate control
# ---------------------------------------------------------------------------

def p2_latent_budget(model, x, model_name, cfg) -> dict:
    """
    Verify the rate match held, and report per-branch MSE for vae-our.

    After the 64:1 matching this is a verification rather than a diagnosis: if
    rates are equal, a reconstruction difference is attributable to architecture.
    The per-branch split exists because vae-our's total_mse sums three terms, and
    its 256-dim whole-patch spatial bottleneck cannot reconstruct a full cube —
    so a large mse_spatial is an artifact of the auxiliary loss weighting rather
    than evidence about reconstruction quality.
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
    if model_name == "vae-our" and p.get("report_per_branch_mse", True):
        rf, rs, rp, *_ = batched_forward(model, x)
        out["per_branch_mse"] = {
            "mse_final": float(F.mse_loss(rf, x)),
            "mse_spatial": float(F.mse_loss(rs, x)),
            "mse_spectral": float(F.mse_loss(rp, x)),
        }
        t = out["per_branch_mse"]
        t["total_mse"] = t["mse_final"] + 0.5 * t["mse_spatial"] + 0.5 * t["mse_spectral"]
        t["final_share_of_total"] = t["mse_final"] / max(t["total_mse"], 1e-12)
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
    # across the batch and contributes no information.
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

    # Latent swap: roll the batch so every patch is decoded from another's code.
    base = batched_decode(model, lat)
    swapped = batched_decode(model, [torch.roll(t, 1, dims=0) for t in lat])
    sam_base = float(spectral_angle_mapper_loss(x, base))
    sam_swap = float(spectral_angle_mapper_loss(x, swapped))
    delta = abs(sam_swap - sam_base) / max(sam_base, 1e-12)
    out.update({"sam_own_latent": sam_base, "sam_swapped_latent": sam_swap,
                "latent_swap_delta": delta})

    # Output constancy: a collapsed decoder emits near-identical patches.
    out["recon_std_across_batch"] = float(base.std(dim=0).mean())

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

    r_int = batched_reconstruct(model, x)
    r_sh = batched_reconstruct(model, x_sh)

    # Score each against ITS OWN input — the question is whether the model got
    # worse at the task, not whether the output moved.
    m_int = metrics(x, r_int, min_energy)
    m_sh = metrics(x_sh, r_sh, min_energy)
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
# P5 — Spectral band-masking inpainting (shortcut test 2)
# ---------------------------------------------------------------------------

def p5_spectral_inpainting(model, x, stats, cfg, min_energy) -> dict:
    """
    Does the model hold a spectral prior, or does it just copy the input through?

    Zero a contiguous block of bands and score ONLY those bands. A pass-through
    autoencoder reproduces the zeros it was given; a model that has learned the
    shape of a spectrum fills them in. The bar to clear is filling with the
    band-wise train mean, which requires no model at all.

    Directly relevant to "hallucination-free purification": a model with no
    spectral prior has no mechanism by which to purify anything.
    """
    p = cfg["p5_spectral_inpainting"]
    B, H, W, C = x.shape
    k = max(1, int(round(p["mask_fraction"] * C)))
    starts = np.linspace(0, C - k, p["n_mask_positions"]).astype(int)
    mean_spec = torch.as_tensor(stats["mean_spectrum"], device=x.device)

    model_err, base_err = [], []
    for s0 in starts:
        sl = slice(int(s0), int(s0) + k)
        xm = x.clone()
        xm[..., sl] = 0.0
        r = batched_reconstruct(model, xm)
        model_err.append(float(F.mse_loss(r[..., sl], x[..., sl])))
        fill = mean_spec[sl].view(1, 1, 1, k).expand(B, H, W, k)
        base_err.append(float(F.mse_loss(fill, x[..., sl])))

    me, be = float(np.mean(model_err)), float(np.mean(base_err))
    gain = (be - me) / max(be, 1e-12)
    return {
        "mask_bands": k, "n_positions": len(starts),
        "model_masked_mse": me, "meanfill_masked_mse": be,
        "relative_gain": gain,
        "has_spectral_prior": gain >= p["min_relative_gain_over_baseline"],
        "verdict": "PASS" if gain >= p["min_relative_gain_over_baseline"] else "FAIL",
    }


# ---------------------------------------------------------------------------
# P6 — Noise pass-through / purification
# ---------------------------------------------------------------------------

def p6_purification(model, x, cfg, min_energy) -> dict:
    """
    The paper's core claim, tested directly.

    Corrupt the INPUT (downstream.py perturbs the latent, which is a different
    question), encode-decode, and measure how much of the corruption survives:

        NPR = ||D(E(x_noisy)) - x_clean|| / ||x_noisy - x_clean||

    NPR < 1 means the model removed noise. NPR ~ 1 means it passed the noise
    straight through — the signature of an autoencoder that has learned identity
    rather than a prior. NPR > 1 means it added error of its own.
    """
    p = cfg["p6_purification"]
    sd = float(x.std())
    g = torch.Generator(device="cpu").manual_seed(cfg["sampling"]["seed"] + 4)
    out = {"sigmas": {}}
    for s in p["sigmas"]:
        noise = (torch.randn(x.shape, generator=g) * (s * sd)).to(x.device)
        xn = x + noise
        r = batched_reconstruct(model, xn)
        num = float(torch.linalg.vector_norm(r - x))
        den = float(torch.linalg.vector_norm(xn - x))
        npr = num / max(den, 1e-12)
        clean = metrics(x, r, min_energy)
        out["sigmas"][str(s)] = {
            "npr": npr,
            "behaviour": ("denoises" if npr < p["denoising_max_npr"]
                          else "pass_through" if npr <= p["passthrough_npr_range"][1]
                          else "harmful"),
            "psnr_vs_clean": clean["psnr"],
            "sam_vs_clean": clean["sam"],
            "sam_valid_vs_clean": clean["sam_valid"],
        }
    nprs = [v["npr"] for v in out["sigmas"].values()]
    out["mean_npr"] = float(np.mean(nprs))
    out["denoises_at_all_sigmas"] = all(n < p["denoising_max_npr"] for n in nprs)
    out["verdict"] = "PASS" if out["denoises_at_all_sigmas"] else "FAIL"
    return out


# ---------------------------------------------------------------------------
# P7 — Linear probe on frozen latents
# ---------------------------------------------------------------------------

def _continuum_removed_depths(x: torch.Tensor, n_features: int,
                              mean_spec: np.ndarray) -> np.ndarray:
    """
    Continuum-removed absorption depth at the deepest features of the mean
    spectrum, per patch.

    Wavelengths are deliberately not used: the packed shards carry no wavelength
    vector (pack.py's sidecar records n/H/W/C/crop_bands/patch_max/source_files
    and nothing spectral), and the probes read only the frozen artifacts. So the
    features are located by BAND INDEX, at the deepest minima of the train-mean
    continuum-removed spectrum. That is dataset-comparable and computable today;
    adding wavelengths to the sidecar later would let these be *named*
    (1900 nm hydration, etc.) without changing the maths.
    """
    C = x.shape[-1]
    ms = np.asarray(mean_spec, dtype=np.float64)
    # Straight-line continuum across the full range, removed multiplicatively.
    cont = np.linspace(ms[0], ms[-1], C)
    cr = ms / np.maximum(cont, 1e-8)
    interior = np.arange(2, C - 2)
    order = interior[np.argsort(cr[interior])]
    picks, min_sep = [], max(2, C // 20)
    for b in order:
        if all(abs(b - q) >= min_sep for q in picks):
            picks.append(int(b))
        if len(picks) == n_features:
            break

    xn = x.detach().cpu().numpy().astype(np.float64)
    B = xn.shape[0]
    pc = xn.mean(axis=(1, 2))                       # (B, C) patch mean spectra
    contb = np.linspace(pc[:, :1], pc[:, -1:], C, axis=1).squeeze(-1)
    crb = pc / np.maximum(contb, 1e-8)
    return np.stack([1.0 - crb[:, b] for b in picks], axis=1)   # (B, n_features)


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> float:
    """Closed-form ridge with a 50/50 split; returns held-out R^2."""
    n = X.shape[0]
    if n < 8:
        return float("nan")
    k = n // 2
    Xtr, Xte, ytr, yte = X[:k], X[k:], y[:k], y[k:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
    Xte = np.hstack([Xte, np.ones((Xte.shape[0], 1))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _logreg_accuracy(X: np.ndarray, labels: list[str], epochs: int = 200) -> float:
    """Multinomial logistic regression accuracy, held out 50/50."""
    uniq = sorted(set(labels))
    if len(uniq) < 2:
        return float("nan")
    y = np.array([uniq.index(l) for l in labels])
    n = X.shape[0]
    k = n // 2
    mu, sd = X[:k].mean(0), X[:k].std(0) + 1e-8
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    lin = torch.nn.Linear(Xt.shape[1], len(uniq))
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(Xt[:k]), yt[:k]).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(Xt[k:]).argmax(1) == yt[k:]).float().mean())


def _pca(X: np.ndarray, dim: int) -> np.ndarray:
    """
    Project to a common width so latent SIZE does not decide the probe.

    Without this the probe is confounded by exactly the thing P2 controls for:
    a wider latent hands a linear model more columns to fit with, and would look
    like richer content. PCA equalises the width so the probe measures
    information, not dimensionality.
    """
    Xc = X - X.mean(0, keepdims=True)
    d = min(dim, min(Xc.shape) - 1)
    if d < 1:
        return Xc
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:d].T


def p7_linear_probe(model, x, scenes, stats, cfg) -> dict:
    """
    Does the frozen latent encode chemistry, or just which scene it came from?

    Two linear probes on the same PCA-equalised features: absorption depth
    (physics) and scene identity (nuisance). High nuisance with low physics is
    the "nonsensical latent" case — the code memorised acquisition conditions
    rather than what is in the scene. Raw pixels give an upper bound and a
    shuffled latent gives the chance floor, so the numbers can be read against
    something.
    """
    p = cfg["p7_linear_probe"]
    lat = batched_encode(model, x)
    Z = torch.cat([t.reshape(t.shape[0], -1) for t in lat], dim=1).cpu().numpy().astype(np.float64)
    Zp = _pca(Z, p["pca_dim"])

    y = _continuum_removed_depths(x, p["n_features"], stats["mean_spectrum"])
    raw = _pca(x.reshape(x.shape[0], -1).cpu().numpy().astype(np.float64), p["pca_dim"])
    rng = np.random.default_rng(cfg["sampling"]["seed"] + 5)
    Zsh = Zp[rng.permutation(Zp.shape[0])]

    r2 = _ridge(Zp, y)
    out = {
        "latent_dims_raw": int(Z.shape[1]),
        "pca_dim": int(Zp.shape[1]),
        "physics_r2": r2,
        "physics_r2_raw_pixels_upper_bound": _ridge(raw, y),
        "physics_r2_shuffled_chance_floor": _ridge(Zsh, y),
        "n_scenes": len(set(scenes)),
    }
    acc = _logreg_accuracy(Zp, scenes)
    out["scene_id_accuracy"] = acc
    out["scene_id_chance"] = 1.0 / max(len(set(scenes)), 1)
    out["encodes_physics"] = bool(r2 >= p["min_physics_r2"])
    out["nuisance_dominated"] = bool(
        (acc == acc) and acc > p["high_nuisance_accuracy"] and r2 < p["min_physics_r2"])
    out["verdict"] = "PASS" if out["encodes_physics"] else "FAIL"
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
             x, scenes, stats, device, p1_shared) -> dict:
    ckpt = (Path(args.ckpt) if args.ckpt else
            resolve_ckpt(model_name, dataset, loss, args.ckpt_dir,
                         seed=args.seed, select=args.select))
    if not ckpt.is_file():
        return {"model": model_name, "dataset": dataset, "loss": loss,
                "error": f"missing checkpoint {ckpt}", "verdict": "MISSING"}

    # load_model returns (model, checkpoint_dict); it already calls .eval().
    model, ckpt_meta = load_model(model_name, ckpt, device)
    xd = x.to(device)
    eps = cfg["p1_trivial_floors"]["sam_valid_min_energy"]

    recon = batched_reconstruct(model, xd)
    recon_m = metrics(xd, recon, eps)

    res = {
        "model": model_name, "dataset": dataset, "loss": loss,
        "seed": args.seed, "select": args.select,
        "checkpoint": str(ckpt), "n_patches": int(xd.shape[0]),
        "trained_epochs": ckpt_meta.get("epoch"),
        "best_val_loss": ckpt_meta.get("loss"),
        "preregistration": cfg.get("registered_on"),
        "reconstruction": recon_m,
        "P1_trivial_floors": p1_trivial_floors(recon_m, p1_shared, cfg),
        "P2_latent_budget": p2_latent_budget(model, xd, model_name, cfg),
        "P3_collapse": p3_collapse(model, xd, cfg, eps),
        "P4_spatial_reliance": p4_spatial_reliance(model, xd, model_name, cfg, eps),
        "P5_spectral_inpainting": p5_spectral_inpainting(model, xd, stats, cfg, eps),
        "P6_purification": p6_purification(model, xd, cfg, eps),
        "P7_linear_probe": p7_linear_probe(model, xd, scenes, stats, cfg),
    }
    res["per_patch"] = {k: v.tolist() for k, v in
                        per_patch_metrics(xd, recon, eps).items()}

    # A collapsed cell is not a score. Nothing downstream should rank it.
    if res["P3_collapse"]["verdict"] == "INVALID":
        res["verdict"] = "INVALID"
        res["verdict_reason"] = (
            "posterior collapse: the decoder ignores the latent, so the "
            "reconstruction metrics describe a constant predictor, not a model")
    elif res["P1_trivial_floors"]["verdict"] == "FAIL":
        res["verdict"] = "INVALID"
        res["verdict_reason"] = (
            "does not beat a trivial predictor by the preregistered margin")
    else:
        fails = [k for k in ("P2_latent_budget", "P4_spatial_reliance",
                             "P5_spectral_inpainting", "P6_purification",
                             "P7_linear_probe")
                 if res[k]["verdict"] not in ("PASS",)]
        res["verdict"] = "PASS" if not fails else "FAIL"
        res["failed_probes"] = fails
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Falsification probes on frozen checkpoints.")
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
                   help="Override p1_trivial_floors.n_random_draws. For the smoke "
                        "harness only — the real run must use the preregistered 1000.")
    p.add_argument("--probe-batch", type=int, default=None,
                   help="Chunk size for model forwards inside probes "
                        "(memory only; does not change results).")
    p.add_argument("--out-dir", default="results/probes")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_prereg()
    if args.max_patches is not None:
        cfg["sampling"]["max_patches"] = args.max_patches
    if args.n_random_draws is not None:
        cfg["p1_trivial_floors"]["n_random_draws"] = args.n_random_draws
    global PROBE_BATCH
    PROBE_BATCH = (args.probe_batch
                   or cfg["sampling"].get("probe_batch", PROBE_BATCH))
    split = args.split or cfg.get("split", "test")

    apply_dataset(args.dataset, verify=True, processed_root=args.data_root)
    apply_hyperparams(settings, load_hyperparams(args.dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"probes | {args.dataset} | split={split} | preregistered {cfg['registered_on']}")
    x, scenes = load_patches(args.dataset, split, cfg, args.packed_root, args.data_root)
    print(f"  {x.shape[0]} patches, C={x.shape[-1]}, {len(set(scenes))} scenes")
    stats = train_statistics(args.dataset, cfg, args.packed_root, args.data_root)
    print(f"  train statistics from {stats['n_used']} patches")

    # The model-independent half of P1 (trivial floors + 1000-draw random null +
    # identity oracle): computed once here, reused across every cell below.
    xd = x.to(device)
    p1_shared = p1_shared_floors(xd, stats, scenes, cfg)
    print(f"  P1 shared floors + {cfg['p1_trivial_floors']['n_random_draws']}-draw "
          f"random null computed once")

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for m, l in cells:
        res = run_cell(m, args.dataset, l, args, cfg, x, scenes, stats, device, p1_shared)
        name = checkpoint_name(m, l, seed=args.seed, select=args.select).replace(".pt", "")
        (out_dir / f"{args.dataset}__{name}.json").write_text(json.dumps(res, indent=1))
        if res.get("error"):
            print(f"  {m:<24} {l:<9} {res['verdict']}  ({res['error']})")
            continue
        print(f"  {m:<24} {l:<9} {res['verdict']:<8} "
              f"P1={res['P1_trivial_floors']['verdict']:<4} "
              f"P2={res['P2_latent_budget']['verdict']:<4} "
              f"P3={res['P3_collapse']['verdict']:<7} "
              f"P4={res['P4_spatial_reliance']['verdict']:<9} "
              f"P5={res['P5_spectral_inpainting']['verdict']:<4} "
              f"P6={res['P6_purification']['verdict']:<4} "
              f"P7={res['P7_linear_probe']['verdict']}")
        if res.get("verdict_reason"):
            print(f"      -> {res['verdict_reason']}")
    print(f"\nwrote {len(cells)} result file(s) to {out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
