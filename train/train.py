"""
train/train.py
--------------
Model-agnostic training entrypoint for the HSI VAE ablation study.

The model is selected via --model and resolved through modules/registry.py, so
this script never branches on model type: it just calls the model's
``loss_terms`` each step. Two loss regimes are supported via --loss:

    standard : model's native VAE loss  (MSE + beta * KLD)
    physics  : native + lambda_physics * SAM (Spectral Angle Mapper)

"vae-our" is physics-informed by design (SAM intrinsic) and is therefore
physics-only — passing --loss standard for it is rejected.

Run from the repo root with PYTHONPATH set (scripts/train.sh does this):
    PYTHONPATH=. python train/train.py --model vae-our --dataset IIRS --loss physics
    python train/train.py --help

Checkpoints are written to <ckpt-dir>/<DATASET>/<name>.pt where <name> is:
    vae-our            -> vae-our.pt
    <model>            -> <model>_<loss>.pt   (e.g. vae-standard_physics.pt)
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from modules.registry import (
    MODEL_NAMES,
    PHYSICS_ONLY,
    build_model,
    checkpoint_name,
)
from utils.config import DATASETS, apply_dataset, settings
from utils.training.dataloader import build_dataloader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_vae(
    model,
    dataloader,
    epochs,
    device,
    use_physics,
    lr=1e-4,
    beta=1e-3,
    lambda_physics=0.3,
    val_dataloader=None,
    ckpt_path=None,
    ckpt_meta=None,
):
    """
    Generic VAE training loop driven by ``model.loss_terms``.

    Args:
        model          : any model implementing the registry contract
        dataloader     : training DataLoader yielding (B, H, W, C) tensors
        epochs         : number of epochs
        device         : torch.device
        use_physics    : add the SAM term to the loss
        lr             : AdamW learning rate
        beta           : KL weight
        lambda_physics : SAM weight (ignored when use_physics is False)
        val_dataloader : optional validation DataLoader
        ckpt_path      : optional Path to save best_model checkpoint
        ckpt_meta      : dict merged into every saved checkpoint (model/dataset/loss_type)
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    ckpt_meta = ckpt_meta or {}
    best_val_loss = math.inf

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = train_mse = train_sam = train_kld = 0.0

        for x_batch in dataloader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            terms = model.loss_terms(x_batch, beta=beta, lambda_physics=lambda_physics,
                                     use_physics=use_physics)
            loss = terms["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mse += terms["mse"].item()
            train_sam += terms["sam"].item()
            train_kld += terms["kld"].item()

        n_train = max(len(dataloader), 1)
        train_loss /= n_train
        train_mse /= n_train
        train_sam /= n_train
        train_kld /= n_train
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validation ----
        val_loss = val_mse = val_sam = val_kld = 0.0
        if val_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for x_val in val_dataloader:
                    x_val = x_val.to(device)
                    terms = model.loss_terms(x_val, beta=beta, lambda_physics=lambda_physics,
                                             use_physics=use_physics)
                    val_loss += terms["loss"].item()
                    val_mse += terms["mse"].item()
                    val_sam += terms["sam"].item()
                    val_kld += terms["kld"].item()

            n_val = max(len(val_dataloader), 1)
            val_loss /= n_val
            val_mse /= n_val
            val_sam /= n_val
            val_kld /= n_val

        scheduler.step()

        # ---- Console logging ----
        val_str = (
            f" | Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} "
            f"| Val SAM: {val_sam:.4f} | Val KLD: {val_kld:.4f}"
            if val_dataloader is not None else ""
        )
        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | "
            f"SAM: {train_sam:.4f} | KLD: {train_kld:.4f} | "
            f"LR: {current_lr:.2e}"
            + val_str
        )

        # ---- Checkpoint (best by monitored loss) ----
        if ckpt_path is not None:
            monitor = val_loss if val_dataloader is not None else train_loss
            if monitor < best_val_loss:
                best_val_loss = monitor
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": monitor,
                        **ckpt_meta,
                    },
                    ckpt_path,
                )

    if ckpt_path is not None:
        print(f"Best checkpoint saved to: {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an HSI VAE (model-agnostic) for the ablation study."
    )

    # Ablation selectors
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES),
                        help="Which model architecture to train.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                        help="Which dataset to train on (sets band count).")
    parser.add_argument("--loss", default="physics", choices=["standard", "physics"],
                        help="Loss regime: standard (MSE+KLD) or physics (+SAM). "
                             "vae-our is physics-only.")

    # Data
    parser.add_argument("--data-root", default=None,
                        help="Override the dataset's processed root "
                             "(default: utils.config.DATASETS[<dataset>]['processed_root']).")
    parser.add_argument("--num-workers", type=int, default=settings.num_workers)

    # Hyper-parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-3, help="KL divergence weight")
    parser.add_argument("--lambda-physics", type=float, default=0.3, help="SAM loss weight")
    parser.add_argument("--seed", type=int, default=42)

    # Checkpointing
    parser.add_argument("--ckpt-dir", default="model",
                        help="Root directory for checkpoints (per-dataset subfolders).")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Guard: physics-only models cannot run a "standard" loss regime.
    if args.model in PHYSICS_ONLY and args.loss == "standard":
        raise SystemExit(
            f"Model '{args.model}' is physics-only (SAM intrinsic); "
            f"use --loss physics."
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_physics = args.loss == "physics"

    # Configure the global settings for this dataset's band count *before*
    # building the model or dataloaders.
    apply_dataset(args.dataset)

    print("==============================================")
    print(f"  model    : {args.model}")
    print(f"  dataset  : {args.dataset} (C={settings.input_channels})")
    print(f"  loss     : {args.loss}")
    print(f"  device   : {device}")
    print("==============================================")

    # Dataloaders
    train_loader = build_dataloader(
        args.dataset, "train",
        processed_root=args.data_root,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        args.dataset, "valid",
        processed_root=args.data_root,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = build_model(args.model).to(device)

    # Materialize LazyLinear layers with a single dummy forward before training.
    dummy = torch.randn(
        2, settings.input_height, settings.input_width, settings.input_channels,
        device=device,
    )
    with torch.no_grad():
        model(dummy)

    # Checkpoint destination
    ckpt_path = Path(args.ckpt_dir) / args.dataset / checkpoint_name(args.model, args.loss)
    ckpt_meta = {"model": args.model, "dataset": args.dataset, "loss_type": args.loss}

    train_vae(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        device=device,
        use_physics=use_physics,
        lr=args.lr,
        beta=args.beta,
        lambda_physics=args.lambda_physics,
        val_dataloader=val_loader,
        ckpt_path=ckpt_path,
        ckpt_meta=ckpt_meta,
    )


if __name__ == "__main__":
    main()
