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
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from modules.losses import spectral_angle_mapper_loss
from modules.registry import MODEL_NAMES, build_model, checkpoint_name
from utils.config import DATASETS, apply_dataset, settings
from utils.training.dataloader import build_dataloader


# ---------------------------------------------------------------------------
# Reconstruction metrics (batch tensors are channels-last (B, H, W, C), [0, 1])
# ---------------------------------------------------------------------------

def compute_mse(x, recon):
    return torch.mean((x - recon) ** 2).item()


def compute_psnr(x, recon, max_val=1.0):
    mse = torch.mean((x - recon) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / mse)


def compute_ssim(x, recon, max_val=1.0):
    """
    Mean global SSIM over the batch (single-scale, per-sample, all bands pooled).
    Lightweight port of the notebook helper — no windowing, sufficient as a
    comparative reconstruction metric across the ablation.
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    scores = []
    for i in range(x.shape[0]):
        a = x[i].reshape(-1)
        b = recon[i].reshape(-1)
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(unbiased=False), b.var(unbiased=False)
        cov = ((a - mu_a) * (b - mu_b)).mean()
        ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
            (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
        )
        scores.append(ssim.item())
    return float(np.mean(scores))


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
    parser.add_argument("--data-root", default=None,
                        help="Override the dataset's processed root.")
    parser.add_argument("--ckpt-dir", default="model",
                        help="Checkpoint root (per-dataset subfolders).")
    parser.add_argument("--ckpt", default=None,
                        help="Explicit checkpoint path (overrides --ckpt-dir/--model/--loss).")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--num-workers", type=int, default=settings.num_workers)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    apply_dataset(args.dataset)

    ckpt_file = (
        Path(args.ckpt) if args.ckpt
        else Path(args.ckpt_dir) / args.dataset / checkpoint_name(args.model, args.loss)
    )
    if not ckpt_file.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_file}")

    print("==============================================")
    print(f"  model    : {args.model}")
    print(f"  dataset  : {args.dataset} (C={settings.input_channels})")
    print(f"  loss     : {args.loss}")
    print(f"  ckpt     : {ckpt_file}")
    print(f"  split    : {args.split}")
    print(f"  device   : {device}")
    print("==============================================")

    model, _ = load_model(args.model, ckpt_file, device)

    loader = build_dataloader(
        args.dataset, args.split,
        processed_root=args.data_root,
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
    print(f"\nResults over {len(loader.dataset)} {args.split} patches ({n} batches):")
    print(f"  MSE  : {mse_sum / n:.6f}")
    print(f"  SAM  : {sam_sum / n:.6f} rad")
    print(f"  PSNR : {psnr_sum / n:.4f} dB")
    print(f"  SSIM : {ssim_sum / n:.4f}")


if __name__ == "__main__":
    main()
