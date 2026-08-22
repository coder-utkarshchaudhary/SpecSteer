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
from collections import OrderedDict
from pathlib import Path

import torch

from modules.losses import spectral_angle_mapper_loss
from modules.registry import MODEL_NAMES, build_model, resolve_checkpoint
from utils.config import DATASETS, apply_dataset, settings
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # verify=True cross-checks the configured band count against what is
    # actually on disk, so a mismatch fails here with both numbers rather
    # than as an opaque assert inside SpectralBranch.forward.
    apply_dataset(args.dataset, verify=True, processed_root=args.data_root)

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

    model, _ = load_model(args.model, ckpt_file, device)

    loader = build_dataloader(
        args.dataset, args.split,
        processed_root=args.data_root,
        packed_root=args.packed_root,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    mse_sum = sam_sum = psnr_sum = ssim_sum = 0.0
    n = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            recon = model.reconstruct(x)
            mse_sum += compute_mse(x, recon)
            sam_sum += spectral_angle_mapper_loss(x, recon).item()
            psnr_sum += compute_psnr(x, recon)
            ssim_sum += compute_ssim(x, recon)
            n += 1

    n = max(n, 1)
    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "loss": args.loss,
        "ckpt": str(ckpt_file),
        "split": args.split,
        "n_batches": n,
        "n_samples": len(loader.dataset),
        "mse": mse_sum / n,
        "sam_rad": sam_sum / n,
        "psnr": psnr_sum / n,
        "ssim": ssim_sum / n,
    }
    logger.info(f"Results over {metrics['n_samples']} {args.split} patches ({n} batches):")
    logger.info(f"  MSE  : {metrics['mse']:.6f}")
    logger.info(f"  SAM  : {metrics['sam_rad']:.6f} rad")
    logger.info(f"  PSNR : {metrics['psnr']:.4f} dB")
    logger.info(f"  SSIM : {metrics['ssim']:.4f}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Wrote metrics JSON to {out_path}")


if __name__ == "__main__":
    main()
