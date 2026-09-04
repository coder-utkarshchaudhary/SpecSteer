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
    if model_name == "vae-our" and p.get("report_per_branch_mse", True):
        rf, rs, rp, *_ = batched_forward(model, x)
        out["per_branch_mse"] = {
            "mse_final": float(F.mse_loss(rf, x)),
            "mse_spatial": float(F.mse_loss(rs, x)),
            "mse_spectral": float(F.mse_loss(rp, x)),
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
        "P2_latent_budget": p2_latent_budget(model, xd, model_name, cfg),
        "P3_collapse": p3_collapse(model, xd, cfg, eps),
        "P4_spatial_reliance": p4_spatial_reliance(model, xd, model_name, cfg, eps),
    }
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"diagnostics | {args.dataset} | split={split} | preregistered {cfg['registered_on']}")
    x, scenes = load_patches(args.dataset, split, cfg, args.packed_root, args.data_root)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for m, l in cells:
        res = run_cell(m, args.dataset, l, args, cfg, x, scenes, device)
        name = checkpoint_name(m, l, seed=args.seed, select=args.select).replace(".pt", "")
        (out_dir / f"{args.dataset}__{name}.json").write_text(json.dumps(res, indent=1))
        if res.get("error"):
            print(f"  {m:<24} {l:<9} MISSING  ({res['error']})")
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
