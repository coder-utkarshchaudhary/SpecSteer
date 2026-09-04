"""
inference/inference.py
----------------------
Model-agnostic inference / evaluation for the HSI VAE ablation study.

Given --model, --dataset, and --loss, this loads the matching checkpoint from
<ckpt-dir>/<DATASET>/<name>.pt, runs the model's ``reconstruct`` over the test
split, and reports reconstruction quality: MSE, SAM (radians), PSNR, SSIM.

Run from the repo root with PYTHONPATH set:
    PYTHONPATH=. python inference/inference.py --model vae-our --dataset IIRS --loss physics
    python inference/inference.py --help
"""

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch

from modules.losses import spectral_angle_mapper_loss
from modules.registry import MODEL_NAMES, build_model, resolve_checkpoint
from utils.config import DATASETS, apply_dataset, settings
from utils.hyperparams import apply_cli_overrides, apply_hyperparams, load_hyperparams
from utils.logging_setup import get_run_logger, timestamp
from utils.training.dataloader import build_dataloader


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------
# These now live in modules/metrics.py and are re-exported here so the existing
# call sites (and inference/downstream.py, inference/probes.py, which import
# them from this module) keep working.
#
# NOTE — SSIM CHANGED. This file used to define a GLOBAL single-scale SSIM: one
# mean and one variance per sample over the whole flattened cube, no windowing.
# The notebooks meanwhile used an 11x11 Gaussian-windowed per-band SSIM. The two
# were never comparable. modules/metrics.py keeps the windowed one, so SSIM
# values reported before this change should not be compared against values
# reported after it.
from modules.metrics import (  # noqa: E402,F401
    compute_mse,
    compute_psnr,
    compute_ssim,
)


def compute_psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    """PSNR (dB) from an already-pooled MSE. data_range matches modules/metrics.py."""
    return 10.0 * math.log10(data_range ** 2 / max(mse, 1e-12))


def _sam_valid_min_energy(default: float = 1.0e-8) -> float:
    """
    The valid-pixel energy threshold, from inference/preregistration.yaml
    (p1_trivial_floors.sam_valid_min_energy) — the SAME epsilon probes.py uses,
    so the headline table and the probe diagnostics agree on what a valid pixel
    is. Falls back to the preregistered default if the YAML is unreadable.
    """
    try:
        import yaml
        cfg = yaml.safe_load(
            (Path(__file__).parent / "preregistration.yaml").read_text())
        return float(cfg["p1_trivial_floors"]["sam_valid_min_energy"])
    except Exception:
        return default


def sam_valid_sums(x: torch.Tensor, recon: torch.Tensor,
                   min_energy: float) -> tuple[float, int, int]:
    """
    Batch accumulator for pi/2-excluded SAM (same definition as
    inference/probes.py:sam_valid, pooled over the whole split).

    SAM normalises by sqrt(sum(x^2) + 1e-8); a pixel with spectral energy far
    below that epsilon contributes exactly pi/2 whatever the model predicts
    (CRIMS: ~24% of pixels -> a hard raw-SAM floor of ~0.38 rad unrelated to
    model quality). Restricting the mean to valid pixels is what makes SAM
    comparable across datasets.

    Returns (sum of angles over valid pixels, n valid pixels, n total pixels).
    """
    energy = (x ** 2).sum(dim=-1)
    mask = energy >= min_energy
    dot = (x * recon).sum(dim=-1)
    nt = torch.sqrt((x ** 2).sum(dim=-1) + 1e-8)
    np_ = torch.sqrt((recon ** 2).sum(dim=-1) + 1e-8)
    cos = torch.clamp(dot / (nt * np_ + 1e-8), -1 + 1e-8, 1 - 1e-8)
    angles = torch.acos(cos)
    return (float(angles[mask].sum()), int(mask.sum()), int(mask.numel()))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _strip_module_prefix(state_dict):
    """Strip a leading 'module.' from keys (DataParallel-trained checkpoints)."""
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k[len("module."):] if k.startswith("module.") else k] = v
    return cleaned


def load_model(model_name, ckpt_file, device):
    """Build the model, materialize lazy layers, and load its weights."""
    model = build_model(model_name).to(device)

    # Materialize LazyLinear layers before loading state (dims come from settings).
    dummy = torch.randn(
        2, settings.input_height, settings.input_width, settings.input_channels,
        device=device,
    )
    with torch.no_grad():
        model(dummy)

    ckpt = torch.load(ckpt_file, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(_strip_module_prefix(state))
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained HSI VAE checkpoint on the test split."
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--loss", default="physics", choices=["standard", "physics"])
    parser.add_argument("--packed-root", default=None,
                        help="Override the packed-shard dir (data/packed/<DS>).")
    parser.add_argument("--data-root", default=None,
                        help="Override the dataset's processed root.")
    parser.add_argument("--ckpt-dir", default="model",
                        help="Checkpoint root (per-dataset subfolders).")
    parser.add_argument("--seed", type=int, default=None,
                    help="Which training seed's checkpoint to evaluate. Omit when only one seed exists; required once several do, since picking implicitly would make the result depend on file order.")
    parser.add_argument("--select", choices=("sam", "mse"), default="sam",
                    help="Which checkpoint to load: the epoch selected on best val SAM (default) or on best val reconstruction MSE. Every cell writes both; a comparison must read the SAME criterion for every model.")
    parser.add_argument("--ckpt", default=None,
                        help="Explicit checkpoint path (overrides --ckpt-dir/--model/--loss).")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--num-workers", type=int, default=settings.num_workers)
    parser.add_argument("--out-json", default=None,
                        help="Optional path to dump the metrics as JSON for aggregation.")
    parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                        help="One-off Settings override, repeatable — MUST match "
                             "the --set values the checkpoint was trained with "
                             "(capacity-point runs), or the rebuilt model's "
                             "shapes will not match the state dict.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # verify=True cross-checks the configured band count against what is
    # actually on disk, so a mismatch fails here with both numbers rather
    # than as an opaque assert inside SpectralBranch.forward.
    apply_dataset(args.dataset, verify=True, processed_root=args.data_root)
    # The per-dataset YAML sets the latent-rate and capacity knobs the model was
    # TRAINED with. Without this the model is rebuilt at the Settings dataclass
    # defaults and load_state_dict fails on a size mismatch for every cell.
    # Mirrors train/train.py. --set re-applies any one-off overrides the
    # checkpoint was trained with (capacity points).
    apply_hyperparams(settings, load_hyperparams(args.dataset))
    overrides = apply_cli_overrides(settings, args.set)
    if overrides:
        print(f"--set overrides active: {overrides}")

    logger = get_run_logger(
        "inference", args.model, args.dataset, loss=args.loss, ts=timestamp(),
    )

    ckpt_file = (
        Path(args.ckpt) if args.ckpt
        else resolve_checkpoint(args.ckpt_dir, args.dataset, args.model, args.loss,
                                seed=args.seed, select=args.select)
    )
    if not ckpt_file.exists():
        logger.error(f"Checkpoint not found: {ckpt_file}")
        raise SystemExit(f"Checkpoint not found: {ckpt_file}")

    logger.info("==============================================")
    logger.info(f"  model    : {args.model}")
    logger.info(f"  dataset  : {args.dataset} (C={settings.input_channels})")
    logger.info(f"  loss     : {args.loss}")
    logger.info(f"  ckpt     : {ckpt_file}")
    logger.info(f"  split    : {args.split}")
    logger.info(f"  device   : {device}")
    logger.info("==============================================")

    model, ckpt_meta = load_model(args.model, ckpt_file, device)

    loader = build_dataloader(
        args.dataset, args.split,
        processed_root=args.data_root,
        packed_root=args.packed_root,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    # Sample-weighted, not batch-count-weighted: the test split runs with
    # drop_last=False, so a short final batch (CRIMS: 369 patches at batch 16 ->
    # a last batch of 1) would otherwise carry the same weight as a full one.
    # MSE/SAM are means over elements/pixels, so weighting each batch mean by its
    # sample count and dividing by the total recovers the split-wide mean; PSNR
    # is derived once from the pooled MSE rather than averaged in dB.
    mse_wsum = sam_wsum = ssim_wsum = 0.0
    sam_valid_sum, n_valid_px, n_total_px = 0.0, 0, 0
    min_energy = _sam_valid_min_energy()
    n_samples = 0
    n_batches = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            b = x.shape[0]
            recon = model.reconstruct(x)
            mse_wsum += compute_mse(x, recon) * b
            sam_wsum += spectral_angle_mapper_loss(x, recon).item() * b
            ssim_wsum += compute_ssim(x, recon) * b
            sv_sum, sv_n, sv_total = sam_valid_sums(x, recon, min_energy)
            sam_valid_sum += sv_sum
            n_valid_px += sv_n
            n_total_px += sv_total
            n_samples += b
            n_batches += 1

    n_samples = max(n_samples, 1)
    mse = mse_wsum / n_samples
    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "loss": args.loss,
        "seed": args.seed,
        "select": args.select,
        "ckpt": str(ckpt_file),
        "split": args.split,
        "n_batches": n_batches,
        "n_samples": len(loader.dataset),
        "mse": mse,
        "sam_rad": sam_wsum / n_samples,
        # pi/2-excluded SAM — the cross-dataset-comparable spectral number
        # (definition shared with inference/probes.py). Any ranking on CRIMS
        # MUST use this, never raw sam_rad.
        "sam_valid": (sam_valid_sum / n_valid_px) if n_valid_px else float("nan"),
        "valid_pixel_frac": (n_valid_px / n_total_px) if n_total_px else float("nan"),
        "psnr": float(compute_psnr_from_mse(mse)),
        "ssim": ssim_wsum / n_samples,
        "trained_epochs": ckpt_meta.get("epoch"),
        "batch_size_trained": ckpt_meta.get("batch_size"),
        "platform_trained": ckpt_meta.get("platform"),
        "lambda_physics": ckpt_meta.get("lambda_physics"),
        "beta": ckpt_meta.get("beta"),
    }
    logger.info(f"Results over {metrics['n_samples']} {args.split} patches ({n_batches} batches):")
    logger.info(f"  MSE  : {metrics['mse']:.6f}")
    logger.info(f"  SAM  : {metrics['sam_rad']:.6f} rad")
    logger.info(f"  SAMv : {metrics['sam_valid']:.6f} rad "
                f"(pi/2-excluded; {100 * metrics['valid_pixel_frac']:.1f}% pixels valid)")
    logger.info(f"  PSNR : {metrics['psnr']:.4f} dB")
    logger.info(f"  SSIM : {metrics['ssim']:.4f}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Wrote metrics JSON to {out_path}")


if __name__ == "__main__":
    main()
