"""
modules/losses.py
-----------------
Shared, model-agnostic loss primitives for the HSI VAE ablation study.

Every model's ``loss_terms`` (see modules/vae_*.py) is built from these two
functions so the loss formulation is identical across the ablation grid:

    standard : MSE + beta * KLD
    physics  : MSE + beta * KLD + lambda_physics * SAM

These are the *finalized* forms used in the reference notebooks (docs/*.ipynb),
up to the "Train" markdown section:
  - SAM uses a sqrt(sum(x^2) + eps) norm and is clamped before arccos to keep
    gradients finite.
  - KL divergence uses the mean form (batch-normalized) for comparability across
    models with very different latent sizes.
"""

import torch


def spectral_angle_mapper_loss(y_true, y_pred):
    """
    Spectral Angle Mapper (SAM) — the differentiable physics prior.

    Measures the mean angle (radians) between the true and reconstructed
    per-pixel spectra. Invariant to per-pixel scaling, so it steers the model
    toward physically-consistent spectral *shape*.

    Args:
        y_true, y_pred : (B, H, W, C) tensors (channels-last; wavelength is dim=-1)

    Returns:
        Scalar mean spectral angle.
    """
    dot_product = torch.sum(y_true * y_pred, dim=-1)
    norm_true = torch.sqrt(torch.sum(y_true ** 2, dim=-1) + 1e-8)
    norm_pred = torch.sqrt(torch.sum(y_pred ** 2, dim=-1) + 1e-8)

    cos_sim = dot_product / (norm_true * norm_pred + 1e-8)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-8, 1.0 - 1e-8)

    return torch.mean(torch.acos(cos_sim))


def kl_divergence(mu, logvar):
    """
    KL divergence to the standard normal prior, mean-reduced.

    Args:
        mu, logvar : latent parameters of any shape (the mean reduces over all
                     elements, so latent size does not bias the magnitude).

    Returns:
        Scalar KL divergence.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
