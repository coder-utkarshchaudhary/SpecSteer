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

2026-09-19 REINSTATEMENT of P1 and P5, both DIAGNOSTIC ONLY (no pass/fail —
see the "reporting: diagnostic_only" keys in inference/preregistration.yaml):

  P1  trivial-predictor floors   redesigned to be TIERED BY RATE (mean_global/
                                 mean_fold/mean_region carry zero per-patch
                                 information and form `best_zero_rate_floor`;
                                 `mean_patch` is a much stronger, separate,
                                 rate-aware predictor reported on its own axis
                                 — conflating the two is exactly what made the
                                 old gate miscalibrated) and computed ONCE PER
                                 DATASET, cached to `floors_<DS>_<split>.json`
                                 (verdict.load_cells already skips files named
                                 `floors_*`), not once per cell.
  P5  spectral band-masking      a contiguous 10%-of-bands block is zeroed at
                                 the model's input at 5 positions along the
                                 spectrum; reconstruction is scored ONLY on the
                                 masked bands, against filling them with the
                                 band-wise training-set mean. Deliberately BAND
                                 masking, not pixel masking, because a pixel
                                 mask would be unanswerable by
                                 vae-1d-pixelwise (no spatial context at all)
                                 and so would not discriminate among the
                                 spatial models.

Both are wrapped by `_isolated()` and write their error INSIDE their own
sub-dict (`P1_trivial_floors["error"]` / `P5_spectral_inpainting["error"]`),
never at the cell root — `verdict.load_cells()` drops any cell whose top-level
`error` is truthy, so a bug in either new probe must not delete that cell's
P2/P3/P4 rows too.

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


# NOTE (2026-09-04 / reinstated 2026-09-19): the P6 purification probe and the
# P7 linear probe remain removed (scaling bugs at the time — NPR at 13-27
# against a ~1.0 legend, physics R^2 at -1e9; git history has the code). P1
# and P5 are reinstated below, redesigned as diagnostics — see the module
# docstring.


def _isolated(name: str, fn):
    """
    Run one probe function, converting any exception into an in-place error
    record instead of propagating it.

    MUST NOT set a top-level `error` key on the cell result: verdict.py's
    load_cells() drops any cell dict whose top-level `error` is truthy, so a
    bug in a NEW probe (P1/P5) would silently delete that cell's P2/P3/P4 rows
    from probes.csv too. The error therefore lives inside the probe's own
    sub-dict; verdict.flatten() reads it defensively (`.get()`) and reports
    NaN for that probe's columns plus the error string, leaving every other
    probe's columns for that cell intact.
    """
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"      {name} FAILED ({e}) — writing NaN for its columns, cell continues")
        return {"error": f"{name} failed: {e}", "traceback": tb}


# ---------------------------------------------------------------------------
# P1 — Trivial-predictor floors (reinstated 2026-09-19, diagnostic only)
# ---------------------------------------------------------------------------

def train_band_statistics(dataset: str, cfg: dict, packed_root=None,
                          data_root=None) -> dict:
    """
    Per-band training-set statistics, for P1's zero-rate floors and P5's
    mean-fill baseline. TRAIN split only — never valid/test, or the floors
    would leak test-set information into their own bar.

    Also used by P5's `bandwise_train_mean` baseline, so both probes share one
    pass over the training split rather than each rescanning it.
    """
    p1 = cfg["p1_trivial_floors"]
    ds = build_dataset(dataset, "train", processed_root=data_root, packed_root=packed_root)
    n = len(ds)
    cap = min(p1.get("train_stat_patches", 512), n) or n
    rng = np.random.default_rng(cfg["sampling"]["seed"] + 1)
    idx = np.sort(rng.choice(n, size=cap, replace=False))

    scenes = _scene_labels(ds, n)
    train_scenes = sorted({scenes[int(i)] for i in idx}) if scenes else []

    sum_spec = None
    sumsq_spec = None
    scene_sum: dict[str, torch.Tensor] = {}
    scene_n: dict[str, int] = {}
    n_pixels = 0
    for i in idx:
        patch = ds[int(i)]  # (H, W, C)
        flat = patch.reshape(-1, patch.shape[-1]).double()
        s = flat.sum(dim=0)
        sq = (flat ** 2).sum(dim=0)
        if sum_spec is None:
            sum_spec, sumsq_spec = s, sq
        else:
            sum_spec += s
            sumsq_spec += sq
        n_pixels += flat.shape[0]
        if scenes:
            sc = scenes[int(i)]
            scene_sum[sc] = scene_sum.get(sc, 0) + s
            scene_n[sc] = scene_n.get(sc, 0) + flat.shape[0]

    mean_spectrum = (sum_spec / max(n_pixels, 1)).float()
    var_spectrum = (sumsq_spec / max(n_pixels, 1) - mean_spectrum.double() ** 2).clamp(min=0)
    std_spectrum = torch.sqrt(var_spectrum).float()
    scene_means = {sc: (scene_sum[sc] / max(scene_n[sc], 1)).float()
                  for sc in scene_sum}

    return {
        "mean_spectrum": mean_spectrum, "std_spectrum": std_spectrum,
        "scene_means": scene_means, "train_scenes": train_scenes,
        "n_used": int(len(idx)),
    }


def _nanmax_dict(dicts: list[dict], keys: list[str]) -> dict:
    return {k: float(np.nanmax([d[k] for d in dicts])) for k in keys}


def _nanmin_dict(dicts: list[dict], keys: list[str]) -> dict:
    return {k: float(np.nanmin([d[k] for d in dicts])) for k in keys}


def p1_shared_floors(x: torch.Tensor, scenes: list[str], stats: dict,
                     cfg: dict, min_energy: float, device) -> dict:
    """
    Model-independent trivial-predictor floors, computed ONCE per dataset (not
    once per cell — the old version's cost was 1000 random draws x 28 cells for
    a number that never depends on the model).

    Every floor is scored CHUNKED against the test sample `x`: the old,
    deleted version built the full (B,H,W,C) prediction tensor per floor,
    which is exactly the AVIRIS-scale OOM pattern the rest of this module was
    already rewritten to avoid (see the module-level "Chunked model calls"
    docstring above) — never repeat that here.
    """
    B, H, W, C = x.shape
    mean_spec = stats["mean_spectrum"].to(device)

    def score_constant_per_patch(pred_fn) -> dict:
        """pred_fn(i0, i1, scene_slice) -> (chunk, H, W, C) prediction tensor."""
        acc = ChunkedMetricAccumulator(min_energy)
        for i in range(0, B, PROBE_BATCH):
            j = min(i + PROBE_BATCH, B)
            xb = x[i:j]
            pred = pred_fn(i, j)
            acc.update(xb, pred)
        return acc.result()

    zero_rate = {}

    # mean_global: the single train-mean spectrum, broadcast to every pixel.
    def _global(i, j):
        b = j - i
        return mean_spec.view(1, 1, 1, C).expand(b, H, W, C)
    zero_rate["mean_global"] = score_constant_per_patch(_global)

    # mean_fold: identical to mean_global for a single train/val/test split —
    # kept as a distinct, explicitly-aliased entry so the YAML's historical
    # `baselines` list stays meaningful rather than silently dropping a name.
    zero_rate["mean_fold"] = dict(zero_rate["mean_global"], alias_of="mean_global")

    # mean_region: the train-scene mean spectrum whose name matches this test
    # patch's scene, falling back to mean_global when the scene never
    # appeared in the (capped) training sample.
    n_fallback = 0
    scene_means = stats["scene_means"]
    def _region(i, j):
        b = j - i
        rows = []
        for k in range(i, j):
            sc = scenes[k] if k < len(scenes) else None
            m = scene_means.get(sc) if sc is not None else None
            if m is None:
                nonlocal n_fallback
                n_fallback += 1
                m = stats["mean_spectrum"]
            rows.append(m)
        return torch.stack(rows).to(device).view(b, 1, 1, C).expand(b, H, W, C)
    zero_rate["mean_region"] = score_constant_per_patch(_region)
    test_scenes = set(scenes) if scenes else set()
    train_scenes = set(stats.get("train_scenes", []))
    overlap = (len(test_scenes & train_scenes) / len(test_scenes)
              if test_scenes else float("nan"))
    zero_rate["mean_region"]["n_region_fallback"] = n_fallback
    zero_rate["mean_region"]["region_scene_overlap"] = overlap

    best_zero_rate = _nanmax_dict(
        [zero_rate["mean_global"], zero_rate["mean_fold"], zero_rate["mean_region"]],
        ["psnr", "ssim"])
    best_zero_rate.update(_nanmin_dict(
        [zero_rate["mean_global"], zero_rate["mean_fold"], zero_rate["mean_region"]],
        ["sam", "sam_valid"]))

    # mean_patch: the TEST patch's own spatial mean spectrum. Rate-aware (a
    # C-float-per-patch predictor), reported on its own axis — never mixed
    # into best_zero_rate_floor. Expected, not alarming, that a trained 64:1
    # model can lose to this; that fact is exactly what miscalibrated the old
    # pass/fail gate.
    def _patch_mean(i, j):
        xb = x[i:j]
        return xb.mean(dim=(1, 2), keepdim=True).expand_as(xb)
    mean_patch = score_constant_per_patch(_patch_mean)
    mean_patch["note"] = ("uses the TEST patch's own spatial-mean spectrum — a "
                          "C-float-per-patch predictor, not a zero-rate floor")

    # identity oracle: SAM's own epsilon means a perfect copy does not score 0.
    identity_oracle = metrics(x, x, min_energy)

    # random null: n_random_draws random predictions on a small sub-sample,
    # for a percentile check. Only the metrics in random_null_metrics are
    # computed (SSIM of white noise costs the most time for a ~0 either way).
    p1 = cfg["p1_trivial_floors"]
    n_draws = p1["n_random_draws"]
    n_sub = min(p1.get("random_null_patches", 8), B)
    x_sub = x[:n_sub]
    want = set(p1.get("random_null_metrics", ["psnr", "sam", "sam_valid"]))
    g = torch.Generator(device="cpu").manual_seed(cfg["sampling"]["seed"] + 7)

    def _random_arrays(kind: str) -> dict:
        arrs = {k: np.empty(n_draws, dtype=np.float64) for k in want}
        for d in range(n_draws):
            if kind == "uniform":
                pred = torch.rand(x_sub.shape, generator=g).to(device)
            else:
                pred = torch.randn(x_sub.shape, generator=g).to(device).clamp(0, 1)
            m = metrics(x_sub, pred, min_energy)
            for k in want:
                arrs[k][d] = m[k]
        best = {}
        if "psnr" in want:
            best["psnr"] = float(np.nanmax(arrs["psnr"]))
        if "sam" in want:
            best["sam"] = float(np.nanmin(arrs["sam"]))
        if "sam_valid" in want:
            best["sam_valid"] = float(np.nanmin(arrs["sam_valid"]))
        return {"mean": {k: float(np.nanmean(v)) for k, v in arrs.items()},
               "best": best, "arrays": {k: v.tolist() for k, v in arrs.items()}}

    random_res = {"random_uniform": _random_arrays("uniform"),
                  "random_normal": _random_arrays("normal")}

    return {
        "zero_rate": zero_rate,
        "best_zero_rate_floor": best_zero_rate,
        "mean_patch": mean_patch,
        "identity_oracle": identity_oracle,
        "random": random_res,
        "random_null_patches": n_sub,
        "train_stats": {"n_used": stats["n_used"],
                        "train_scenes": stats.get("train_scenes", [])},
    }


def _floors_signature(dataset: str, split: str, cfg: dict, x: torch.Tensor) -> dict:
    p1 = cfg["p1_trivial_floors"]
    return {
        "dataset": dataset, "split": split,
        "sampling_seed": cfg["sampling"]["seed"],
        "max_patches": cfg["sampling"]["max_patches"],
        "n_patches": int(x.shape[0]), "C": int(x.shape[-1]),
        "n_random_draws": p1["n_random_draws"],
        "train_stat_patches": p1.get("train_stat_patches", 512),
        "reinstated_on": p1.get("reinstated_on"),
    }


def load_or_build_floors(dataset: str, split: str, cfg: dict, x: torch.Tensor,
                         scenes: list[str], stats: dict, args, device) -> dict:
    """
    P1's floors depend only on (dataset, split, sampling seed, cap) — never on
    the checkpoint/model/seed being evaluated — so cache them to disk instead
    of recomputing the 1000-draw random null once per cell. A 3-dataset x
    2-checkpoint-seed sweep then pays the cost 3 times instead of 6+.

    inference/verdict.py:load_cells() already skips any file named
    `floors_*` (it is not a cell result), so this cache lives safely inside
    the same --out-dir.
    """
    sig = _floors_signature(dataset, split, cfg, x)
    floors_dir = Path(args.floors_dir or args.out_dir)
    floors_dir.mkdir(parents=True, exist_ok=True)
    cache_path = floors_dir / f"floors_{dataset}_{split}.json"

    if cache_path.is_file() and not getattr(args, "recompute_floors", False):
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("signature") == sig:
                print(f"  P1 floors: reused cache ({cache_path})")
                return cached["floors"]
        except (json.JSONDecodeError, KeyError):
            pass

    if stats.get("error"):
        floors = {"error": f"train statistics unavailable: {stats['error']}"}
    else:
        eps = cfg["p1_trivial_floors"]["sam_valid_min_energy"]
        floors = p1_shared_floors(x, scenes, stats, cfg, eps, device)
        n_draws = cfg["p1_trivial_floors"]["n_random_draws"]
        n_sub = floors.get("random_null_patches", 0)
        print(f"  P1 floors: built ({n_draws} draws on {n_sub} patches)")

    cache_path.write_text(json.dumps({"signature": sig, "floors": floors}, indent=1))
    return floors


def p1_report(model_recon: dict, model_recon_sub: dict, floors: dict, cfg: dict) -> dict:
    """
    Pure arithmetic against the shared floors — no model calls here, so a
    failure can only come from `floors` itself carrying an error (propagated
    through) or a malformed floors dict.
    """
    if floors.get("error"):
        return {"error": f"floors unavailable: {floors['error']}"}

    best = floors["best_zero_rate_floor"]
    mp = floors["mean_patch"]
    oracle = floors["identity_oracle"]
    p1 = cfg["p1_trivial_floors"]

    def lift(vs: dict) -> dict:
        return {
            "psnr_db": model_recon["psnr"] - vs["psnr"],
            "ssim_absolute": model_recon["ssim"] - vs["ssim"],
            "sam_relative": (vs["sam"] - model_recon["sam"]) / max(vs["sam"], 1e-12),
            "sam_valid_relative": ((vs["sam_valid"] - model_recon["sam_valid"])
                                   / max(vs["sam_valid"], 1e-12)
                                   if not math.isnan(vs.get("sam_valid", float("nan")))
                                   else float("nan")),
        }

    denom = oracle["psnr"] - best["psnr"]
    headroom = ((model_recon["psnr"] - best["psnr"]) / denom
               if denom > 1e-9 else float("nan"))

    rand = floors["random"]
    n_sub = floors["random_null_patches"]

    def percentile(kind: str, key: str, better_is_lower: bool) -> float:
        arr = np.asarray(rand[kind]["arrays"].get(key, []), dtype=np.float64)
        if arr.size == 0 or key not in model_recon_sub:
            return float("nan")
        v = model_recon_sub[key]
        return float(np.mean(arr <= v) if better_is_lower else np.mean(arr >= v))

    rand_pct = {
        "psnr_vs_uniform": percentile("random_uniform", "psnr", better_is_lower=False),
        "sam_valid_vs_uniform": percentile("random_uniform", "sam_valid", better_is_lower=True),
        "psnr_vs_normal": percentile("random_normal", "psnr", better_is_lower=False),
        "sam_valid_vs_normal": percentile("random_normal", "sam_valid", better_is_lower=True),
    }

    return {
        "reporting": "diagnostic_only",
        "best_zero_rate_floor": best,
        "mean_patch": {k: v for k, v in mp.items() if k != "note"},
        "identity_oracle": oracle,
        "model": model_recon,
        "lift_over_zero_rate": lift(best),
        "lift_over_mean_patch": lift(mp),
        "headroom_captured_psnr": headroom,
        "random_percentile": rand_pct,
        "random_pct_computed_on_n": n_sub,
        "registered_thresholds_not_enforced": p1["min_lift_over_best_floor"],
    }


# ---------------------------------------------------------------------------
# P5 — Spectral band-masking inpainting (reinstated 2026-09-19, diagnostic only)
# ---------------------------------------------------------------------------

def p5_spectral_inpainting(model, x: torch.Tensor, stats: dict, cfg: dict,
                           min_energy: float, max_patches: int | None) -> dict:
    """
    Zero a contiguous 10%-of-bands block at 5 positions along the spectrum,
    run the FULL patch through model.reconstruct(), and score reconstruction
    quality ONLY on the masked bands — against filling the same bands with the
    band-wise training-set mean (and, as a lower-bound reference, a zero
    fill/pass-through null).

    Chunked per position AND per PROBE_BATCH, mirroring the rest of this
    module: the deleted 2026-08-21 version masked and forwarded the WHOLE
    `x` at once, which is exactly the full-tensor OOM pattern P2/P3/P4 were
    already rewritten to avoid.

    The models were never trained on masked input, so this measures an
    implicit spectral prior under distribution shift, not trained inpainting.
    """
    if stats.get("error"):
        return {"error": f"train statistics unavailable: {stats['error']}"}

    p = cfg["p5_spectral_inpainting"]
    B_full, H, W, C = x.shape
    cap = min(max_patches or p.get("max_patches", 128), B_full)
    xs = x[:cap]
    B = xs.shape[0]

    k = max(1, round(p["mask_fraction"] * C))
    n_pos = p["n_mask_positions"]
    starts = np.linspace(0, max(C - k, 0), n_pos).round().astype(int)
    mean_spec = stats["mean_spectrum"].to(xs.device)
    fill_value = p.get("mask_fill_value", 0.0)

    per_position = []
    for s0 in starts:
        s0 = int(s0)
        sl = slice(s0, s0 + k)
        acc_model = ChunkedMetricAccumulator(min_energy)
        acc_mean = ChunkedMetricAccumulator(min_energy)
        acc_zero = ChunkedMetricAccumulator(min_energy)
        with torch.no_grad():
            for i in range(0, B, PROBE_BATCH):
                xb = xs[i:i + PROBE_BATCH]
                xm = xb.clone()
                xm[..., sl] = fill_value
                rb = model.reconstruct(xm)
                tgt = xb[..., sl]
                acc_model.update(tgt, rb[..., sl])
                mean_fill = mean_spec[sl].view(1, 1, 1, k).expand_as(tgt)
                acc_mean.update(tgt, mean_fill)
                acc_zero.update(tgt, torch.zeros_like(tgt))
                del rb, xm
        per_position.append({
            "start": s0, "k": k,
            "model": acc_model.result(), "meanfill": acc_mean.result(),
            "zerofill": acc_zero.result(),
        })

    def avg(field: str, metric: str) -> float:
        vals = [pos[field][metric] for pos in per_position]
        return float(np.nanmean(vals))

    mse_model = avg("model", "mse")
    mse_mean = avg("meanfill", "mse")
    mse_zero = avg("zerofill", "mse")
    psnr_model = avg("model", "psnr")
    psnr_mean = avg("meanfill", "psnr")
    psnr_zero = avg("zerofill", "psnr")

    return {
        "reporting": "diagnostic_only",
        "mask_kind": "contiguous_band_block",
        "mask_bands": k, "mask_fraction_actual": k / C,
        "n_positions": len(starts), "positions": starts.tolist(),
        "n_patches": B,
        "per_position": per_position,
        "masked_mse_model": mse_model, "masked_mse_meanfill": mse_mean,
        "masked_mse_zerofill": mse_zero,
        "masked_psnr_model": psnr_model, "masked_psnr_meanfill": psnr_mean,
        "masked_psnr_zerofill": psnr_zero,
        # Averaging PSNR over positions is a mean-of-logs (defensible: the
        # positions are separate experiments, not chunks of one) — also report
        # the pooled-MSE version so the two can be sanity-checked against
        # each other.
        "masked_psnr_from_pooled_mse": 10.0 * math.log10(1.0 / max(mse_model, 1e-12)),
        "masked_sam_model": avg("model", "sam"),
        "masked_sam_meanfill": avg("meanfill", "sam"),
        "masked_sam_valid_model": avg("model", "sam_valid"),
        "masked_sam_valid_meanfill": avg("meanfill", "sam_valid"),
        "masked_ssim_model": avg("model", "ssim"),
        "relative_gain": (mse_mean - mse_model) / max(mse_mean, 1e-12),
        "psnr_gain_db": psnr_model - psnr_mean,
        # ~1.0 => the model reproduced the zeros it was handed (pure
        # pass-through, no spectral prior); << 1.0 => it filled the gap from
        # something it learned about spectral shape.
        "passthrough_index": mse_model / max(mse_zero, 1e-12),
        "registered_threshold_not_enforced": p["min_relative_gain_over_baseline"],
    }


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
             x, scenes, device, floors: dict, stats: dict) -> dict:
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

    # P1 — pure arithmetic against the shared per-dataset floors; no model
    # calls, so this is cheap and runs before the empty_cache() calls below
    # matter. Isolated: a bug here must not cost P2/P3/P4/P5 for this cell.
    if getattr(args, "skip_p1", False):
        res["P1_trivial_floors"] = {"skipped": True}
    else:
        n_sub = cfg["p1_trivial_floors"].get("random_null_patches", 8)
        n_sub = min(n_sub, xd.shape[0])
        res["P1_trivial_floors"] = _isolated(
            "P1_trivial_floors",
            lambda: p1_report(recon_m, metrics(xd[:n_sub], recon[:n_sub], eps),
                              floors, cfg))

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

    # Free the full (B,H,W,C) reconstruction before P5's extra forward passes
    # — on AVIRIS at the sampling cap this tensor is multiple GiB, and it has
    # been live since the reconstruct() call above.
    del recon
    torch.cuda.empty_cache()

    # P5 — n_mask_positions extra forward passes; the expensive new probe.
    # Isolated the same way as P1.
    if getattr(args, "skip_p5", False):
        res["P5_spectral_inpainting"] = {"skipped": True}
    else:
        res["P5_spectral_inpainting"] = _isolated(
            "P5_spectral_inpainting",
            lambda: p5_spectral_inpainting(model, xd, stats, cfg, eps,
                                           args.p5_max_patches))

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
                   help="Override P1's random-null draw count (reinstated "
                        "2026-09-19, diagnostic only). Default: the "
                        "preregistered n_random_draws.")
    p.add_argument("--probe-batch", type=int, default=None,
                   help="Chunk size for model forwards inside probes "
                        "(memory only; does not change results).")
    p.add_argument("--p5-max-patches", type=int, default=None,
                   help="Cap on patches scored by P5 (band-masking), separate "
                        "from --max-patches: P5 costs n_mask_positions extra "
                        "full forward passes per cell. Default: the "
                        "preregistered p5_spectral_inpainting.max_patches.")
    p.add_argument("--recompute-floors", action="store_true",
                   help="Ignore any cached floors_<DS>_<split>.json even if "
                        "its signature matches, and rebuild P1's floors.")
    p.add_argument("--floors-dir", default=None,
                   help="Where floors_<DS>_<split>.json is cached/read "
                        "(default: --out-dir).")
    p.add_argument("--skip-p1", action="store_true",
                   help="Escape hatch: skip P1 entirely for this run.")
    p.add_argument("--skip-p5", action="store_true",
                   help="Escape hatch: skip P5 entirely for this run.")
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
    if args.n_random_draws is not None:
        cfg["p1_trivial_floors"]["n_random_draws"] = args.n_random_draws
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

    # P1/P5 shared setup — computed ONCE per dataset (not per cell), so
    # tomorrow's 2-seed sweep pays the 1000-draw random null and the
    # train-split scan once per dataset, not once per checkpoint. A failure
    # here must not abort the dataset's cells: it degrades every cell's
    # P1/P5 to an error record instead (see load_or_build_floors /
    # p1_report / p5_spectral_inpainting's own `stats.get("error")` guards).
    if args.skip_p1 and args.skip_p5:
        stats, floors = {"skipped": True}, {"skipped": True}
    else:
        try:
            stats = train_band_statistics(args.dataset, cfg, args.packed_root, args.data_root)
            print(f"  train statistics: {stats['n_used']} patches, "
                  f"{len(stats['train_scenes'])} train scenes")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"  WARNING: train statistics failed ({e}) — P1/P5 will report NaN")
            stats = {"error": str(e), "traceback": tb}

        if args.skip_p1:
            floors = {"skipped": True}
        else:
            xd_for_floors = x.to(device)
            try:
                floors = load_or_build_floors(args.dataset, split, cfg,
                                              xd_for_floors, scenes, stats, args, device)
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"  WARNING: P1 floors failed ({e}) — P1 will report NaN")
                floors = {"error": str(e), "traceback": tb}
            del xd_for_floors
            torch.cuda.empty_cache()

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
            res = run_cell(m, args.dataset, l, args, cfg, x, scenes, device,
                           floors, stats)
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
        p1r = res.get("P1_trivial_floors", {}) or {}
        p5r = res.get("P5_spectral_inpainting", {}) or {}
        p1_lift = p1r.get("lift_over_zero_rate", {}).get("psnr_db", float("nan"))
        p5_gain = p5r.get("relative_gain", float("nan"))
        print(f"  {m:<24} {l:<9} "
              f"{'COLLAPSED' if res['collapsed'] else 'ok':<10} "
              f"rate={p2['latent_elements']:>7,} ({p2['deviation_pct']:+.1f}%) "
              f"active={p3['active_unit_fraction']:.2f} "
              f"swapDSAM={p3['latent_swap_delta']:.3f} "
              f"SRI={p4['sri']:+.3f} "
              f"SAMv={res['reconstruction']['sam_valid']:.4f} "
              f"P1lift={p1_lift:+.2f}dB "
              f"P5gain={p5_gain:+.3f}")
    print(f"\nwrote {len(cells)} result file(s) to {out_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
