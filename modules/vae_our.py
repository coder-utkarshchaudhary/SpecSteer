"""
modules/vae_our.py
------------------
"vae-our" — the dual-stream, physics-informed VAE (HSI_DualStream_PI_VAE).

Two independent VAE streams each reconstruct the full cube:
  - Spatial stream  : per-pixel 1x1 spectral reduction -> 2D conv bottleneck ->
                      global per-patch latent.
  - Spectral stream : per-pixel 1D spectral conv -> spatially-resolved latent map.
A learned linear layer late-fuses the two reconstructions (sigmoid output head).

This is the finalized notebook model (docs/*.ipynb, up to the Train section):
  - reparameterize clamps logvar to [-30, 20] before exp (numerical stability),
  - the fused reconstruction passes through a sigmoid (inputs are max-normalized
    to [0, 1] by the dataloader),
  - SAM is intrinsic to this model, so it is always physics-informed
    (``PHYSICS_ONLY`` in modules/registry.py).

I/O convention: channels-last (B, H, W, C) throughout.
"""

import torch
import torch.nn as nn

from modules.SpatialBranch import SpatialEncoderDecoder
from modules.SpectralBranch import SpectralEncoderDecoder
from modules.losses import spectral_angle_mapper_loss, kl_divergence
from utils.config import settings


class HSI_DualStream_PI_VAE(nn.Module):
    """Dual-stream late-fusion physics-informed VAE ("vae-our")."""

    def __init__(self, conv_output_c=None, conv_output_h=None, conv_output_w=None):
        super().__init__()

        # Default derived conv dims from the (dataset-configured) global settings.
        conv_output_c = conv_output_c if conv_output_c is not None else settings.conv_output_c
        conv_output_h = conv_output_h if conv_output_h is not None else settings.conv_output_h
        conv_output_w = conv_output_w if conv_output_w is not None else settings.conv_output_w

        self.spatial_stream = SpatialEncoderDecoder(conv_output_c, conv_output_h, conv_output_w)
        self.spectral_stream = SpectralEncoderDecoder()

        # Late fusion: (B, H, W, 2C) -> (B, H, W, C)
        self.fusion_layer = nn.Linear(settings.input_channels * 2, settings.input_channels)

        self.mse_loss_fn = nn.MSELoss()

    def reparameterize(self, z_features):
        """
        Chunk encoder output into mu/logvar (on dim=1) and sample z.

        Spatial:  (B, 2*latent_dim)                 -> (B, latent_dim)
        Spectral: (B, 2*spectral_latent, H, W)       -> (B, spectral_latent, H, W)
        """
        mu, logvar = torch.chunk(z_features, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, x):
        # --- Spatial stream ---
        spatial_features = self.spatial_stream.encoder(x)     # (B, 2*latent_dim)
        z_s, mu_s, logvar_s = self.reparameterize(spatial_features)
        recon_s = self.spatial_stream.decoder(z_s)            # (B, H, W, C)

        # --- Spectral stream ---
        spectral_features = self.spectral_stream.encoder(x)   # (B, 2*spectral_latent, H, W)
        z_p, mu_p, logvar_p = self.reparameterize(spectral_features)
        recon_p = self.spectral_stream.decoder(z_p)           # (B, H, W, C)

        # --- Late fusion ---
        combined = torch.cat([recon_s, recon_p], dim=-1)      # (B, H, W, 2C)
        recon_final = torch.sigmoid(self.fusion_layer(combined))   # (B, H, W, C)

        return recon_final, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p

    def loss_terms(self, x, beta=1e-3, lambda_physics=0.3, use_physics=True):
        """
        Compute all loss components for one batch.

        vae-our is physics-informed by design; ``use_physics`` is accepted for a
        uniform interface but defaults to True and is expected to stay True (the
        registry marks this model PHYSICS_ONLY).

        Returns:
            dict(loss=, mse=, kld=, sam=) — all scalar tensors.
        """
        recon_final, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p = self(x)

        # Multi-branch reconstruction MSE.
        mse_final = self.mse_loss_fn(recon_final, x)
        mse_spatial = self.mse_loss_fn(recon_s, x)
        mse_spectral = self.mse_loss_fn(recon_p, x)
        total_mse = mse_final + 0.5 * mse_spatial + 0.5 * mse_spectral

        # Combined KL (mean-form primitives; already batch-normalized).
        total_kld = kl_divergence(mu_s, logvar_s) + kl_divergence(mu_p, logvar_p)

        # Physics prior (SAM) on the fused reconstruction.
        sam = spectral_angle_mapper_loss(x, recon_final)

        loss = total_mse + beta * total_kld
        if use_physics:
            loss = loss + lambda_physics * sam

        return {"loss": loss, "mse": total_mse, "kld": total_kld, "sam": sam}

    @torch.no_grad()
    def reconstruct(self, x):
        """Return the fused reconstruction (B, H, W, C) for inference."""
        recon_final, *_ = self(x)
        return recon_final

    # ------------------------------------------------------------------
    # Downstream-experiment contract (inference/downstream.py)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_latents(self, x):
        """
        Deterministic (mu) latents of both streams, for noise-injection and
        interpolation experiments:

            [ z_spatial  (B, latent_dim),
              z_spectral (B, spectral_latent_dim, H, W) ]
        """
        spatial_features = self.spatial_stream.encoder(x)
        mu_s, _ = torch.chunk(spatial_features, 2, dim=1)
        spectral_features = self.spectral_stream.encoder(x)
        mu_p, _ = torch.chunk(spectral_features, 2, dim=1)
        return [mu_s, mu_p]

    @torch.no_grad()
    def decode_latents(self, latents):
        """
        [ z_spatial, z_spectral ] -> fused reconstruction (B, H, W, C).

        Mirrors forward()'s late fusion so a perturbed/interpolated latent pair
        maps back through the exact reconstruction path used in training.
        """
        z_s, z_p = latents
        recon_s = self.spatial_stream.decoder(z_s)
        recon_p = self.spectral_stream.decoder(z_p)
        combined = torch.cat([recon_s, recon_p], dim=-1)
        return torch.sigmoid(self.fusion_layer(combined))
