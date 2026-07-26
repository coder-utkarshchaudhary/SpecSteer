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
import logging
import math
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from modules.registry import (
    MODEL_NAMES,
    PHYSICS_ONLY,
    build_model,
    checkpoint_name,
)
from utils.config import DATASETS, apply_dataset, settings
from utils.hyperparams import apply_hyperparams, load_hyperparams
from utils.logging_setup import get_run_logger, timestamp
from utils.notify import RunNotifier
from utils.training.dataloader import build_dataloader


# ---------------------------------------------------------------------------
# nn.DataParallel adapter
# ---------------------------------------------------------------------------

class _LossTermsAdapter(nn.Module):
    """Route DP's ``forward()`` to the inner model's ``loss_terms(...)``.

    ``nn.DataParallel`` only splits/gathers the module's ``forward`` call, so
    we expose ``loss_terms`` through ``forward`` and unsqueeze each scalar term
    to a length-1 first-dim tensor. DP concatenates those along dim 0 (one row
    per GPU); the training loop takes ``.mean()`` afterwards to reduce.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x, beta, lambda_physics, use_physics):
        terms = self.inner.loss_terms(
            x, beta=beta, lambda_physics=lambda_physics, use_physics=use_physics,
        )
        return {k: v.unsqueeze(0) for k, v in terms.items()}


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying HSI-VAE model regardless of DP / adapter wrapping."""
    if isinstance(model, nn.DataParallel):
        model = model.module
    if isinstance(model, _LossTermsAdapter):
        model = model.inner
    return model


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
    weight_decay=1e-5,
    patience=7,
    val_dataloader=None,
    ckpt_path=None,
    ckpt_meta=None,
    logger=None,
    notifier=None,
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
        weight_decay   : AdamW weight decay
        patience       : early stopping patience (epochs w/o val_loss improvement); disabled if val_dataloader is None
        val_dataloader : optional validation DataLoader
        ckpt_path      : optional Path to save best_model checkpoint
        ckpt_meta      : dict merged into every saved checkpoint (model/dataset/loss_type)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    ckpt_meta = ckpt_meta or {}
    best_val_loss = math.inf
    no_improve_epochs = 0

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = train_mse = train_sam = train_kld = 0.0

        train_bar = tqdm(
            dataloader,
            desc=f"epoch {epoch}/{epochs} train",
            leave=False,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for i, x_batch in enumerate(train_bar, start=1):
            x_batch = x_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                terms = model(x_batch, beta, lambda_physics, use_physics)
                loss = terms["loss"].mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_mse += terms["mse"].mean().item()
            train_sam += terms["sam"].mean().item()
            train_kld += terms["kld"].mean().item()

            train_bar.set_postfix(loss=f"{train_loss / i:.4f}",
                                  mse=f"{train_mse / i:.4f}",
                                  sam=f"{train_sam / i:.4f}")
        train_bar.close()

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
            val_bar = tqdm(
                val_dataloader,
                desc=f"epoch {epoch}/{epochs} val  ",
                leave=False,
                file=sys.stdout,
                dynamic_ncols=True,
            )
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                for j, x_val in enumerate(val_bar, start=1):
                    x_val = x_val.to(device, non_blocking=True)
                    terms = model(x_val, beta, lambda_physics, use_physics)
                    val_loss += terms["loss"].mean().item()
                    val_mse += terms["mse"].mean().item()
                    val_sam += terms["sam"].mean().item()
                    val_kld += terms["kld"].mean().item()
                    val_bar.set_postfix(loss=f"{val_loss / j:.4f}",
                                        mse=f"{val_mse / j:.4f}",
                                        sam=f"{val_sam / j:.4f}")
            val_bar.close()

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
        logger.info(
            f"Epoch [{epoch}/{epochs}] | "
            f"Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | "
            f"SAM: {train_sam:.4f} | KLD: {train_kld:.4f} | "
            f"LR: {current_lr:.2e}"
            + val_str
        )

        if notifier is not None:
            notifier.record_epoch(epoch, {
                "train_loss": train_loss,
                "train_mse": train_mse,
                "train_sam": train_sam,
                "train_kld": train_kld,
                "val_loss": val_loss if val_dataloader is not None else None,
                "val_mse": val_mse if val_dataloader is not None else None,
                "val_sam": val_sam if val_dataloader is not None else None,
                "val_kld": val_kld if val_dataloader is not None else None,
            })

        # ---- Checkpoint (best by monitored loss) + early stopping ----
        monitor = val_loss if val_dataloader is not None else train_loss
        improved = monitor < best_val_loss
        if improved:
            best_val_loss = monitor
            no_improve_epochs = 0
            if ckpt_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": _unwrap(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": monitor,
                        **ckpt_meta,
                    },
                    ckpt_path,
                )
                logger.info(f"Epoch {epoch}: new best {monitor:.6f} — checkpoint saved")
            if notifier is not None:
                notifier.mark_best(epoch, monitor)
        else:
            no_improve_epochs += 1

        # Early stopping only when a validation set is available.
        if val_dataloader is not None and no_improve_epochs >= patience:
            logger.info(f"Early stopping at epoch {epoch}: no val_loss improvement for {patience} epochs.")
            return "early_stop"

    if ckpt_path is not None:
        logger.info(f"Best checkpoint saved to: {ckpt_path}")
    return "ok"


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
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Falls back to YAML then Settings.num_workers.")

    # Hyper-parameters — defaults resolve as: CLI > per-dataset YAML > hard fallback.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None, help="KL divergence weight")
    parser.add_argument("--lambda-physics", type=float, default=None, help="SAM loss weight")
    parser.add_argument("--seed", type=int, default=None)

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

def _resolve(cli_val, yaml_dict, key, fallback):
    """CLI > YAML > hard fallback."""
    if cli_val is not None:
        return cli_val
    return yaml_dict.get(key, fallback)


def main():
    args = parse_args()

    # Guard: physics-only models cannot run a "standard" loss regime.
    if args.model in PHYSICS_ONLY and args.loss == "standard":
        raise SystemExit(
            f"Model '{args.model}' is physics-only (SAM intrinsic); "
            f"use --loss physics."
        )

    # Configure the global settings for this dataset's band count *before*
    # loading the YAML (YAML may then override Settings fields like
    # batch_size and the per-dataset baseline widths).
    apply_dataset(args.dataset)
    hp = load_hyperparams(args.dataset)
    apply_hyperparams(settings, hp)

    # Resolve optimization hyperparams with CLI > YAML > hard-fallback precedence.
    epochs = _resolve(args.epochs, hp, "epochs", 100)
    batch_size = _resolve(args.batch_size, hp, "batch_size", settings.batch_size)
    num_workers = _resolve(args.num_workers, hp, "num_workers", settings.num_workers)
    lr = _resolve(args.lr, hp, "lr", 1e-4)
    beta = _resolve(args.beta, hp, "beta", 1e-3)
    lambda_physics = _resolve(args.lambda_physics, hp, "lambda_physics", 0.3)
    seed = _resolve(args.seed, hp, "seed", 42)
    weight_decay = hp.get("weight_decay", 1e-5)
    patience = hp.get("early_stopping_patience", 7)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_physics = args.loss == "physics"

    logger = get_run_logger(
        "train", args.model, args.dataset, loss=args.loss, ts=timestamp(),
    )

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

    logger.info("==============================================")
    logger.info(f"  model     : {args.model}")
    logger.info(f"  dataset   : {args.dataset} (C={settings.input_channels})")
    logger.info(f"  loss      : {args.loss}")
    logger.info(f"  device    : {device}  gpus: {n_gpus}")
    logger.info(f"  epochs    : {epochs}  batch_size: {batch_size}  workers: {num_workers}")
    logger.info(f"  lr        : {lr}  beta: {beta}  lambda_physics: {lambda_physics}")
    logger.info(f"  wd        : {weight_decay}  patience: {patience}  seed: {seed}")
    logger.info("==============================================")

    # Dataloaders
    train_loader = build_dataloader(
        args.dataset, "train",
        processed_root=args.data_root,
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = build_dataloader(
        args.dataset, "valid",
        processed_root=args.data_root,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    raw_model = build_model(args.model).to(device)

    # Materialize LazyLinear layers with a single dummy forward before training.
    # Kept in fp32 so lazy weights are initialised in fp32; autocast during
    # training then casts activations to fp16 without touching the params.
    dummy = torch.randn(
        2, settings.input_height, settings.input_width, settings.input_channels,
        device=device,
    )
    with torch.no_grad():
        raw_model(dummy)

    # Wrap so nn.DataParallel can split loss_terms across GPUs when available.
    model = _LossTermsAdapter(raw_model)
    if n_gpus > 1:
        logger.info(f"Wrapping in nn.DataParallel across {n_gpus} GPUs "
                    f"(per-GPU batch: {batch_size // n_gpus}).")
        model = nn.DataParallel(model)
    model.to(device)

    # Checkpoint destination
    ckpt_path = Path(args.ckpt_dir) / args.dataset / checkpoint_name(args.model, args.loss)
    ckpt_meta = {"model": args.model, "dataset": args.dataset, "loss_type": args.loss}

    notifier = RunNotifier(
        model=args.model,
        dataset=args.dataset,
        loss=args.loss,
        epochs_planned=epochs,
    )

    try:
        status = train_vae(
            model=model,
            dataloader=train_loader,
            epochs=epochs,
            device=device,
            use_physics=use_physics,
            lr=lr,
            beta=beta,
            lambda_physics=lambda_physics,
            weight_decay=weight_decay,
            patience=patience,
            val_dataloader=val_loader,
            ckpt_path=ckpt_path,
            ckpt_meta=ckpt_meta,
            logger=logger,
            notifier=notifier,
        )
    except BaseException as e:
        tb = traceback.format_exc()
        logger.exception("Training run failed with an exception.")
        notifier.flush_run("fail", extra=f"{type(e).__name__}: {e}\n\n{tb}")
        raise

    notifier.flush_run(status or "ok", extra=f"ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
