"""
train/train.py
---------------
Training script for the Dual-Stream Physics-Informed VAE on IIRS HSI patches.

Run from the repo root with PYTHONPATH set:
    PYTHONPATH=. python train/train.py [args]
    # or use: scripts/train.sh

Usage:
    python train/train.py --help

Quick start (after running scripts/preprocess.sh):
    python train/train.py --epochs 100 --wandb-project hsi-pi-vae

Checkpoints are saved to --ckpt-dir (default: checkpoints/).
One-time W&B setup: run `wandb login` before your first training run.
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from modules.SpatialBranch import SpatialEncoderDecoder
from modules.SpectralBranch import SpectralEncoderDecoder
from utils.config import settings


# ---------------------------------------------------------------------------
# Physics-informed loss functions
# ---------------------------------------------------------------------------

def spectral_angle_mapper_loss(y_true, y_pred):
    """
    Computes the SAM between the ground truth and reconstructed HSI cubes.
    y_true, y_pred: (B, H, W, C)
    """
    # Dot product along the wavelength dimension
    dot_product = torch.sum(y_true * y_pred, dim=-1)

    # L2 Norms
    norm_true = torch.linalg.norm(y_true, dim=-1)
    norm_pred = torch.linalg.norm(y_pred, dim=-1)

    # Cosine similarity with epsilon to prevent division by zero and NaN gradients
    cos_sim = dot_product / (norm_true * norm_pred + 1e-8)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-8, 1.0 - 1e-8)

    # SAM is the average arccos across all pixels
    sam = torch.acos(cos_sim)
    return torch.mean(sam)


def kl_divergence(mu, logvar):
    """ Standard KL Divergence pushing the manifold to N(0, I) """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# ---------------------------------------------------------------------------
# Unified Dual-Stream PI-VAE
# ---------------------------------------------------------------------------

class HSI_DualStream_PI_VAE(nn.Module):
    def __init__(self, conv_output_c, conv_output_h, conv_output_w):
        super().__init__()

        # Initialize isolated branches
        self.spatial_stream = SpatialEncoderDecoder(conv_output_c, conv_output_h, conv_output_w)
        self.spectral_stream = SpectralEncoderDecoder()

        # 1×1 linear fusion: (B, H, W, 2C) → (B, H, W, C)
        self.fusion_layer = nn.Linear(settings.input_channels * 2, settings.input_channels)

    def reparameterize(self, z_features):
        """
        Chunks the encoder output into mu and logvar, then samples z.

        Spatial:  z_features (B, 2*latent_dim)        → mu/logvar (B, latent_dim)
        Spectral: z_features (B, 2*spectral_latent, H, W) → mu/logvar (B, spectral_latent, H, W)
        """
        mu, logvar = torch.chunk(z_features, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, x):
        # --- Spatial Stream ---
        spatial_features = self.spatial_stream.encoder(x)   # (B, 2*latent_dim)
        z_s, mu_s, logvar_s = self.reparameterize(spatial_features)
        recon_s = self.spatial_stream.decoder(z_s)          # (B, H, W, C)

        # --- Spectral Stream ---
        spectral_features = self.spectral_stream.encoder(x) # (B, 2*spectral_latent, H, W)
        z_p, mu_p, logvar_p = self.reparameterize(spectral_features)
        recon_p = self.spectral_stream.decoder(z_p)         # (B, H, W, C)

        # --- Late Fusion ---
        combined = torch.cat([recon_s, recon_p], dim=-1)    # (B, H, W, 2C)
        recon_final = self.fusion_layer(combined)            # (B, H, W, C)

        return recon_final, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p


# ---------------------------------------------------------------------------
# Loss computation helper (shared between train and val loops)
# ---------------------------------------------------------------------------

def _compute_losses(model, x_batch, mse_loss_fn, beta, lambda_physics):
    """
    Run one forward pass and return all loss components.

    Returns:
        loss         : total scalar loss (backprop target)
        total_mse    : combined MSE term
        sam_loss     : SAM physics loss
        total_kld    : combined KL divergence
    """
    recon_final, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p = model(x_batch)

    # 1. Reconstruction loss (MSE) — final + auxiliary branches
    mse_final    = mse_loss_fn(recon_final, x_batch)
    mse_spatial  = mse_loss_fn(recon_s, x_batch)
    mse_spectral = mse_loss_fn(recon_p, x_batch)
    total_mse = mse_final + 0.5 * mse_spatial + 0.5 * mse_spectral

    # 2. Physics prior loss (SAM) on the fused reconstruction
    sam_loss = spectral_angle_mapper_loss(x_batch, recon_final)

    # 3. KL Divergence — normalised by batch size
    B = x_batch.shape[0]
    kld_s = kl_divergence(mu_s, logvar_s) / B
    kld_p = kl_divergence(mu_p, logvar_p) / B
    total_kld = kld_s + kld_p

    # Total: ELBO + physics steering
    loss = total_mse + (beta * total_kld) + (lambda_physics * sam_loss)

    return loss, total_mse, sam_loss, total_kld, recon_final


# ---------------------------------------------------------------------------
# W&B image logging helpers
# ---------------------------------------------------------------------------

def _band_to_wandb_image(tensor_hwc, band_idx=settings.norm_band_idx, caption=""):
    """
    Extract one band from a (H, W, C) tensor, normalise to [0,1] and return
    a wandb.Image.  Requires wandb to be imported by the caller.
    """
    import wandb
    band = tensor_hwc[:, :, band_idx].detach().cpu().float().numpy()
    lo, hi = band.min(), band.max()
    band = (band - lo) / (hi - lo + 1e-8)
    return wandb.Image(band, caption=caption)


def _log_reconstructions(wandb, model, val_loader, device, epoch, n_samples=4):
    """
    Log side-by-side original/reconstruction pairs to W&B for the first
    *n_samples* items in the first validation batch.
    """
    model.eval()
    with torch.no_grad():
        x_val = next(iter(val_loader)).to(device)
        x_sample = x_val[:n_samples]
        recon, *_ = model(x_sample)

    images = []
    for i in range(x_sample.shape[0]):
        images.append(_band_to_wandb_image(x_sample[i], caption=f"orig_{i}"))
        images.append(_band_to_wandb_image(recon[i], caption=f"recon_{i}"))

    wandb.log({"reconstructions": images}, step=epoch)
    model.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_hsi_vae(
    model,
    dataloader,
    epochs,
    device,
    lr=1e-4,
    beta=0.001,
    lambda_physics=0.5,
    val_dataloader=None,
    ckpt_dir=None,
    use_wandb=False,
    log_recon_every=10,
):
    """
    Main training loop.

    Args:
        model           : HSI_DualStream_PI_VAE instance
        dataloader      : DataLoader for the training split
        epochs          : number of training epochs
        device          : torch.device
        lr              : AdamW learning rate
        beta            : KL divergence weight
        lambda_physics  : SAM loss weight
        val_dataloader  : optional DataLoader for the validation split
        ckpt_dir        : if given, save checkpoints here
        use_wandb       : whether to log to Weights & Biases
        log_recon_every : log reconstruction images every N epochs
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss_fn = nn.MSELoss()

    model.to(device)

    best_val_loss = math.inf
    ckpt_path = Path(ckpt_dir) if ckpt_dir else None
    if ckpt_path:
        ckpt_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_mse = train_sam = train_kld = 0.0

        for x_batch in dataloader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            loss, mse, sam, kld, _ = _compute_losses(
                model, x_batch, mse_loss_fn, beta, lambda_physics
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mse  += mse.item()
            train_sam  += sam.item()
            train_kld  += kld.item()

        n_train = len(dataloader)
        train_loss /= n_train
        train_mse  /= n_train
        train_sam  /= n_train
        train_kld  /= n_train
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validation ----
        val_loss = val_mse = val_sam = val_kld = 0.0
        if val_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for x_val in val_dataloader:
                    x_val = x_val.to(device)
                    vloss, vmse, vsam, vkld, _ = _compute_losses(
                        model, x_val, mse_loss_fn, beta, lambda_physics
                    )
                    val_loss += vloss.item()
                    val_mse  += vmse.item()
                    val_sam  += vsam.item()
                    val_kld  += vkld.item()

            n_val = len(val_dataloader)
            val_loss /= n_val
            val_mse  /= n_val
            val_sam  /= n_val
            val_kld  /= n_val

        scheduler.step()

        # ---- Console logging ----
        val_str = (
            f" | Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} "
            f"| Val SAM: {val_sam:.4f} | Val KLD: {val_kld:.4f}"
            if val_dataloader else ""
        )
        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | "
            f"SAM: {train_sam:.4f} | KLD: {train_kld:.4f} | "
            f"LR: {current_lr:.2e}"
            + val_str
        )

        # ---- W&B logging ----
        if use_wandb:
            import wandb
            log_dict = {
                "train/loss": train_loss,
                "train/mse":  train_mse,
                "train/sam":  train_sam,
                "train/kld":  train_kld,
                "train/lr":   current_lr,
            }
            if val_dataloader is not None:
                log_dict.update({
                    "val/loss": val_loss,
                    "val/mse":  val_mse,
                    "val/sam":  val_sam,
                    "val/kld":  val_kld,
                })
            wandb.log(log_dict, step=epoch)

            if val_dataloader is not None and epoch % log_recon_every == 0:
                _log_reconstructions(wandb, model, val_dataloader, device, epoch)

        # ---- Checkpoint saving ----
        if ckpt_path:
            # Save best model by validation loss (or train loss if no val loader)
            monitor = val_loss if val_dataloader else train_loss
            if monitor < best_val_loss:
                best_val_loss = monitor
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": monitor,
                    },
                    ckpt_path / "best_model.pt",
                )

            # Periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_loss,
                    },
                    ckpt_path / f"epoch_{epoch:04d}.pt",
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Dual-Stream Physics-Informed VAE on IIRS HSI patches."
    )

    # Data
    parser.add_argument(
        "--data-root",
        default=settings.data_processed_root,
        help=f"Processed data root from slice.py (default: {settings.data_processed_root})",
    )
    parser.add_argument(
        "--num-workers", type=int, default=settings.num_workers,
        help=f"DataLoader worker processes (default: {settings.num_workers})",
    )

    # Training hyper-parameters
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--batch-size",      type=int,   default=settings.batch_size)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--beta",            type=float, default=0.001,
                        help="KL divergence weight")
    parser.add_argument("--lambda-physics",  type=float, default=0.5,
                        help="SAM loss weight")
    parser.add_argument("--log-recon-every", type=int,   default=10,
                        help="Log reconstruction images to W&B every N epochs")

    # Checkpointing
    parser.add_argument("--ckpt-dir", default="checkpoints",
                        help="Directory to save model checkpoints")

    # W&B
    parser.add_argument("--wandb-project", default="hsi-pi-vae")
    parser.add_argument("--wandb-entity",  default=None,
                        help="W&B entity (team or username). "
                             "Defaults to the account used during `wandb login`.")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    CONV_OUT_C = settings.conv_output_c
    CONV_OUT_H = settings.conv_output_h
    CONV_OUT_W = settings.conv_output_w

    model = HSI_DualStream_PI_VAE(
        conv_output_c=CONV_OUT_C,
        conv_output_h=CONV_OUT_H,
        conv_output_w=CONV_OUT_W,
    )

    # Build dataloaders from real processed patches
    from utils.training.dataloader import build_dataloader

    train_loader = build_dataloader(
        args.data_root, "train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        args.data_root, "valid",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Initialise W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run_config = {
            # Model dimensions
            "input_height":            settings.input_height,
            "input_width":             settings.input_width,
            "input_channels":          settings.input_channels,
            "reduced_dims":            settings.reduced_dims,
            "latent_dim":              settings.latent_dim,
            "n_2D_conv_blocks":        settings.n_2D_conv_blocks,
            "spectral_n_1D_conv_blocks": settings.spectral_n_1D_conv_blocks,
            "spectral_latent_dim":     settings.spectral_latent_dim,
            # Training
            "epochs":                  args.epochs,
            "batch_size":              args.batch_size,
            "lr":                      args.lr,
            "beta":                    args.beta,
            "lambda_physics":          args.lambda_physics,
            # Data
            "patch_size":              settings.patch_size,
            "patch_stride":            settings.patch_stride,
            "band_start":              settings.band_start,
            "band_end":                settings.band_end,
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=run_config,
        )
        wandb.watch(model, log="gradients", log_freq=100)

    # Train
    train_hsi_vae(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        beta=args.beta,
        lambda_physics=args.lambda_physics,
        val_dataloader=val_loader,
        ckpt_dir=args.ckpt_dir,
        use_wandb=use_wandb,
        log_recon_every=args.log_recon_every,
    )

    if use_wandb:
        wandb.finish()
