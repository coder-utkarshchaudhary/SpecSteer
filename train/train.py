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

Perf features (math-preserving):
    * BF16 autocast on Ampere+ GPUs (fp16 + GradScaler elsewhere)
    * channels_last / channels_last_3d memory format
    * On-GPU metric accumulation; a single .item() sync per epoch
    * Optional torch.compile(model, mode="reduce-overhead") via --compile
    * DP only when --allow-multi-gpu is passed AND >1 GPU visible
"""

import argparse
import logging
import math
import os
import random
import sys
import time
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
from utils.config import DATASETS, apply_dataset, settings, verify_channels
from utils.hyperparams import apply_hyperparams, load_hyperparams
from utils.logging_setup import (
    get_run_logger,
    log_tensor,
    tail_log,
    timestamp,
)
from utils.notify import RunNotifier
from utils.training.dataloader import build_dataloader

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore
    _HAS_WANDB = False


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
    """Return the underlying HSI-VAE model regardless of DP / adapter / compile wrapping."""
    if isinstance(model, nn.DataParallel):
        model = model.module
    # torch.compile wraps modules in OptimizedModule with the original at ._orig_mod.
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if isinstance(model, _LossTermsAdapter):
        model = model.inner
    # If _orig_mod pointed at the adapter, unwrap once more.
    if isinstance(model, _LossTermsAdapter):
        model = model.inner
    return model


# ---------------------------------------------------------------------------
# Autocast / memory-format helpers
# ---------------------------------------------------------------------------

def _pick_amp_dtype(device: torch.device, logger: logging.Logger) -> torch.dtype:
    """BF16 on Ampere+ (SM8.0+); FP16 elsewhere on CUDA; FP32 on CPU."""
    if device.type != "cuda":
        return torch.float32
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        major = 0
    if major >= 8:
        logger.info("AMP: bfloat16 (Ampere+, no GradScaler)")
        return torch.bfloat16
    logger.info("AMP: float16 (pre-Ampere, GradScaler enabled)")
    return torch.float16


def _memory_format_for(model_name: str) -> torch.memory_format:
    """channels_last_3d for vae-3d-spatio-spectral, channels_last for the rest."""
    if model_name == "vae-3d-spatio-spectral":
        return torch.channels_last_3d
    return torch.channels_last


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
    amp_dtype: torch.dtype = torch.float32,
    tqdm_every: int = 20,
    wandb_run=None,
):
    """Generic VAE training loop driven by ``model.loss_terms``."""
    if logger is None:
        logger = logging.getLogger(__name__)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    ckpt_meta = ckpt_meta or {}
    best_val_loss = math.inf
    no_improve_epochs = 0

    use_amp = (device.type == "cuda")
    # BF16 doesn't need the scaler; FP16 does.
    needs_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_wall_starts = []

    for epoch in range(1, epochs + 1):
        # Flip DEBUG -> INFO at the epoch boundary.
        if hasattr(logger, "set_epoch"):
            logger.set_epoch(epoch - 1)  # epoch is 1-indexed; policy is 0-indexed
        epoch_t0 = time.time()
        epoch_wall_starts.append(epoch_t0)

        # ---- Training ----
        model.train()
        # Accumulate on-GPU to avoid per-batch CUDA syncs.
        run_loss = torch.zeros((), device=device)
        run_mse = torch.zeros((), device=device)
        run_sam = torch.zeros((), device=device)
        run_kld = torch.zeros((), device=device)

        train_bar = tqdm(
            dataloader,
            desc=f"epoch {epoch}/{epochs} train",
            leave=False,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for i, x_batch in enumerate(train_bar, start=1):
            x_batch = x_batch.to(device, non_blocking=True)
            if epoch == 1 and i == 1:
                log_tensor(logger, "x_batch", x_batch)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                terms = model(x_batch, beta, lambda_physics, use_physics)
                loss = terms["loss"].mean()

            if needs_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                run_loss += loss.detach()
                run_mse += terms["mse"].detach().mean()
                run_sam += terms["sam"].detach().mean()
                run_kld += terms["kld"].detach().mean()

            if i % tqdm_every == 0:
                # One sync per tqdm_every batches, not every batch.
                train_bar.set_postfix(
                    loss=f"{(run_loss / i).item():.4f}",
                    mse=f"{(run_mse / i).item():.4f}",
                    sam=f"{(run_sam / i).item():.4f}",
                )
        train_bar.close()

        n_train = max(len(dataloader), 1)
        train_loss = (run_loss / n_train).item()
        train_mse = (run_mse / n_train).item()
        train_sam = (run_sam / n_train).item()
        train_kld = (run_kld / n_train).item()
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validation ----
        val_loss = val_mse = val_sam = val_kld = 0.0
        if val_dataloader is not None:
            model.eval()
            v_loss = torch.zeros((), device=device)
            v_mse = torch.zeros((), device=device)
            v_sam = torch.zeros((), device=device)
            v_kld = torch.zeros((), device=device)
            val_bar = tqdm(
                val_dataloader,
                desc=f"epoch {epoch}/{epochs} val  ",
                leave=False,
                file=sys.stdout,
                dynamic_ncols=True,
            )
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                for j, x_val in enumerate(val_bar, start=1):
                    x_val = x_val.to(device, non_blocking=True)
                    terms = model(x_val, beta, lambda_physics, use_physics)
                    v_loss += terms["loss"].mean()
                    v_mse += terms["mse"].mean()
                    v_sam += terms["sam"].mean()
                    v_kld += terms["kld"].mean()
                    if j % tqdm_every == 0:
                        val_bar.set_postfix(
                            loss=f"{(v_loss / j).item():.4f}",
                            mse=f"{(v_mse / j).item():.4f}",
                            sam=f"{(v_sam / j).item():.4f}",
                        )
            val_bar.close()

            n_val = max(len(val_dataloader), 1)
            val_loss = (v_loss / n_val).item()
            val_mse = (v_mse / n_val).item()
            val_sam = (v_sam / n_val).item()
            val_kld = (v_kld / n_val).item()

        scheduler.step()

        # ---- Console logging ----
        val_str = (
            f" | Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} "
            f"| Val SAM: {val_sam:.4f} | Val KLD: {val_kld:.4f}"
            if val_dataloader is not None else ""
        )
        epoch_wall = time.time() - epoch_t0
        logger.info(
            f"Epoch [{epoch}/{epochs}] | "
            f"Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | "
            f"SAM: {train_sam:.4f} | KLD: {train_kld:.4f} | "
            f"LR: {current_lr:.2e} | wall {epoch_wall:.1f}s"
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

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/mse": train_mse,
                "train/sam": train_sam,
                "train/kld": train_kld,
                "train/lr": current_lr,
                "train/epoch_wall_s": epoch_wall,
                **(
                    {
                        "val/loss": val_loss,
                        "val/mse": val_mse,
                        "val/sam": val_sam,
                        "val/kld": val_kld,
                    }
                    if val_dataloader is not None
                    else {}
                ),
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

        # ---- Heartbeat (every log_every epochs) ----
        if notifier is not None:
            # Estimate ETA from the mean epoch wall time so far.
            recent = epoch_wall_starts[-min(len(epoch_wall_starts), 5):]
            elapsed = time.time() - recent[0]
            n_seen = len(recent)
            mean_epoch_s = elapsed / max(n_seen, 1)
            eta = mean_epoch_s * (epochs - epoch)
            notifier.heartbeat(epoch, eta_seconds=eta)

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
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--loss", default="physics", choices=["standard", "physics"])

    # Data
    parser.add_argument("--data-root", default=None,
                        help="Legacy per-patch tree (forces the legacy backend). "
                             "Omit to use the packed fp16 shards in data/packed/.")
    parser.add_argument("--packed-root", default=None,
                        help="Override the packed-shard directory for this dataset.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-ram", action="store_true",
                        help="Load the whole packed split into RAM (5-25 GB per "
                             "split post-cap). Removes disk from the loop entirely.")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="Further cap training patches without re-packing. "
                             "pack.py already applied settings.train_patch_cap.")

    # Hyper-parameters
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--lambda-physics", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Checkpointing
    parser.add_argument("--ckpt-dir", default="model")

    # Perf toggles
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile(model, mode='reduce-overhead'). "
                             "Skipped automatically for vae-3d-spatio-spectral.")
    parser.add_argument("--allow-multi-gpu", action="store_true",
                        help="Wrap in nn.DataParallel if >1 GPU is visible. "
                             "Default (off) keeps things single-GPU on HPC.")

    # W&B
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="hsi-pi-vae")
    parser.add_argument("--wandb-entity", default=None)

    # Debug logging
    parser.add_argument("--debug-epochs", type=int, default=3,
                        help="Number of leading epochs to log at DEBUG level.")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve(cli_val, yaml_dict, key, fallback):
    if cli_val is not None:
        return cli_val
    return yaml_dict.get(key, fallback)


def main():
    args = parse_args()

    if args.model in PHYSICS_ONLY and args.loss == "standard":
        raise SystemExit(
            f"Model '{args.model}' is physics-only (SAM intrinsic); "
            f"use --loss physics."
        )

    apply_dataset(args.dataset)
    hp = load_hyperparams(args.dataset)
    apply_hyperparams(settings, hp)

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
        debug_epochs=args.debug_epochs,
    )

    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    gpu_name = torch.cuda.get_device_name(0) if n_gpus else "cpu"

    logger.info("==============================================")
    logger.info(f"  model         : {args.model}")
    logger.info(f"  dataset       : {args.dataset} (C={settings.input_channels})")
    logger.info(f"  loss          : {args.loss}")
    logger.info(f"  device        : {device}  gpus: {n_gpus} ({gpu_name})")
    logger.info(f"  epochs        : {epochs}  batch_size: {batch_size}  workers: {num_workers}")
    logger.info(f"  lr            : {lr}  beta: {beta}  lambda_physics: {lambda_physics}")
    logger.info(f"  wd            : {weight_decay}  patience: {patience}  seed: {seed}")
    logger.info(f"  compile       : {args.compile}  allow_multi_gpu: {args.allow_multi_gpu}")
    logger.info(f"  debug_epochs  : {args.debug_epochs}")
    logger.info("==============================================")

    amp_dtype = _pick_amp_dtype(device, logger)

    # Checkpoint destination
    ckpt_path = Path(args.ckpt_dir) / args.dataset / checkpoint_name(args.model, args.loss)
    ckpt_meta = {"model": args.model, "dataset": args.dataset, "loss_type": args.loss}

    # ---- Notifier FIRST -------------------------------------------------
    # Everything below here — dataloader construction, band-count verification,
    # model build, the LazyLinear dummy forward — can throw, and all of it used
    # to sit ABOVE this point. A failure there produced zero Telegram output and
    # was not caught by the [FAIL] handler: that is exactly why all seven CRIMS
    # slots vanished silently instead of reporting FileNotFoundError. The
    # notifier now exists, and the try block covers setup, before anything
    # touches disk.
    notifier = RunNotifier(
        model=args.model,
        dataset=args.dataset,
        loss=args.loss,
        epochs_planned=epochs,
    )

    resolved_cfg = {
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "beta": beta,
        "lambda_physics": lambda_physics,
        "seed": seed,
        "weight_decay": weight_decay,
        "patience": patience,
        "gpu": gpu_name,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "compile": bool(args.compile),
        "allow_multi_gpu": bool(args.allow_multi_gpu),
        "cache_ram": bool(args.cache_ram),
    }
    notifier.send_start(resolved_cfg)

    status = "fail"
    wandb_run = None
    try:
        # ---- Band-count sanity check ----
        # Fails loudly here, with the dataset name and both numbers, rather than
        # 500 frames deep as an opaque assert inside SpectralBranch.forward.
        verify_channels(args.dataset, args.data_root)

        # ---- Dataloaders ----
        train_loader = build_dataloader(
            args.dataset, "train",
            processed_root=args.data_root,
            packed_root=args.packed_root,
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            cache_ram=args.cache_ram, limit=args.limit_train,
        )
        val_loader = build_dataloader(
            args.dataset, "valid",
            processed_root=args.data_root,
            packed_root=args.packed_root,
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            cache_ram=args.cache_ram,
        )
        # Log the resolved storage backend explicitly. Which one was used had to
        # be inferred from epoch timings after the last grid; it never should be.
        def _describe(loader):
            ds = loader.dataset
            kind = type(ds).__name__
            if kind == "PackedPatchDataset":
                return (f"packed{'+RAM' if getattr(ds, 'cached', False) else ' (memmap)'} "
                        f"{ds.npy_path}")
            return f"legacy per-patch tree ({len(getattr(ds, 'patch_files', []))} files) — SLOW"

        logger.info(f"Data backend  : {_describe(train_loader)}")
        logger.info(f"Train patches : {len(train_loader.dataset)} "
                    f"| Val patches: {len(val_loader.dataset)}")
        logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        # ---- Model ----
        raw_model = build_model(args.model).to(device)

        # Materialize LazyLinear layers with a single dummy forward before training.
        dummy = torch.randn(
            2, settings.input_height, settings.input_width, settings.input_channels,
            device=device,
        )
        with torch.no_grad():
            raw_model(dummy)

        n_params = sum(p.numel() for p in raw_model.parameters())
        logger.info(f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

        model = _LossTermsAdapter(raw_model)

        # Optional torch.compile — the training loop invokes model.forward, which
        # routes into loss_terms via _LossTermsAdapter. Compiling the adapter
        # means the compiled graph actually covers the hot path.
        # Skipped for vae-3d-spatio-spectral where Dynamo frequently trips on the
        # 3D transposed convs.
        if args.compile:
            if args.model == "vae-3d-spatio-spectral":
                logger.info("--compile requested but skipped for vae-3d-spatio-spectral")
            else:
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("torch.compile: enabled (mode=reduce-overhead)")
                except Exception as e:
                    logger.warning("torch.compile failed to enable, continuing eager: %s", e)

        if n_gpus > 1 and args.allow_multi_gpu:
            logger.info(f"Wrapping in nn.DataParallel across {n_gpus} GPUs "
                        f"(per-GPU batch: {batch_size // n_gpus}).")
            model = nn.DataParallel(model)
        elif n_gpus > 1:
            logger.info(f"{n_gpus} GPUs visible; keeping single-GPU. Pass --allow-multi-gpu to enable DP.")

        model.to(device)

        # Ampere+ perf: TF32 for the FP32 matmul paths that AMP doesn't cover
        # (linear reparameterization, norm layers). Preserves math semantics
        # closely enough for training but noticeably faster on Ampere/Hopper.
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Memory-format hint. cuDNN uses channels-last kernels on Ampere+ when
        # the model's parameters and inputs are both in channels_last. Safe no-op
        # when the model's convs don't benefit (registry contract is 4D input).
        mem_fmt = _memory_format_for(args.model)
        try:
            model.to(memory_format=mem_fmt)
            logger.debug("memory_format applied: %s", mem_fmt)
        except Exception as e:
            logger.debug("memory_format apply failed, continuing default: %s", e)

        # ---- Weights & Biases ----
        if _HAS_WANDB and not args.no_wandb:
            try:
                wandb_mode = os.environ.get("WANDB_MODE", "online")
                run_name = f"{args.model}__{args.loss}__{args.dataset}"
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    mode=wandb_mode,
                    config={**resolved_cfg, "n_params": n_params},
                    tags=[args.model, args.dataset, args.loss],
                    reinit=True,
                )
                logger.info(f"wandb: initialised (mode={wandb_mode}, run={run_name})")
            except Exception as e:
                logger.warning("wandb init failed, continuing without: %s", e)
                wandb_run = None
        elif args.no_wandb:
            logger.info("wandb: disabled by --no-wandb")
        else:
            logger.info("wandb: not installed; skipping")

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
            amp_dtype=amp_dtype,
            wandb_run=wandb_run,
        )
    except BaseException as e:
        tb = traceback.format_exc()
        logger.exception("Training run failed with an exception.")
        tail = tail_log(logger, n_lines=60)
        extra = f"{type(e).__name__}: {e}\n\n--- last {60} log lines ---\n{tail}\n\n{tb}"
        notifier.flush_run("fail", extra=extra)
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                pass
        raise

    notifier.flush_run(status or "ok", extra=f"ckpt: {ckpt_path}")
    if wandb_run is not None:
        try:
            wandb_run.finish(exit_code=0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
