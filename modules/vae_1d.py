"""
modules/vae_1d.py
-----------------
"vae-1d-pixelwise" — a purely per-pixel 1D spectral VAE (no spatial mixing);
each pixel's spectrum is encoded/decoded independently.

Baseline C — the "Unmixing" standard.
    Inspiration: Blind Hyperspectral Unmixing Using Deep Autoencoders
                 (Su et al., 2019).

Every pixel spectrum (a length-C vector) is pushed through a shared MLP
encoder → latent → MLP decoder, with the (B, H, W) grid folded entirely into
the batch dimension (batch = B*H*W, features = C). The 2D spatial grid is never
seen by the network, so it excels at preserving per-pixel chemistry (very low
SAM) but has no neighbourhood context to denoise/regularize corrupted pixels
(low PSNR/SSIM) — the hypothesised failure mode.

It satisfies the model-agnostic contract used by train/train.py and
inference/inference.py (see modules/vae_our.py for the reference):

    forward(x)                                  # x: (B, H, W, C)
    loss_terms(x, beta, lambda_physics, use_physics) -> dict(loss, mse, kld, sam)
    reconstruct(x) -> (B, H, W, C)

plus the downstream-experiment contract (inference/downstream.py):

    encode_latents(x) -> [ (B, H, W, Z) ]       # deterministic (mu) latents
    decode_latents([z]) -> (B, H, W, C)

Loss (built from modules/losses.py):
    standard : mse + beta * kld
    physics  : mse + beta * kld + lambda_physics * sam

I/O convention: channels-last (B, H, W, C) throughout.
"""

import torch
import torch.nn as nn

from modules.losses import spectral_angle_mapper_loss, kl_divergence
from utils.config import settings


# Per-pixel MLP widths and latent size. Kept module-local (not in the shared
# Settings) so this baseline is fully self-contained; only the band count is
# read from settings at build time.
HIDDEN_DIMS = (256, 128)
LATENT_DIM = 32


class VAE_1D_Pixelwise(nn.Module):
    """Per-pixel 1D spectral VAE ("vae-1d-pixelwise"), Baseline C."""

    def __init__(self):
        super().__init__()
        c = settings.input_channels
        self.latent_dim = LATENT_DIM

        # ---- Encoder MLP: C -> hidden... -> 2*Z (mu || logvar) ----
        enc_layers = []
        in_f = c
        for h in HIDDEN_DIMS:
            enc_layers += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        enc_layers.append(nn.Linear(in_f, 2 * LATENT_DIM))
        self.encoder = nn.Sequential(*enc_layers)

        # ---- Decoder MLP: Z -> hidden(reversed)... -> C ----
        dec_layers = []
        in_f = LATENT_DIM
        for h in reversed(HIDDEN_DIMS):
            dec_layers += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        dec_layers.append(nn.Linear(in_f, c))
        self.decoder = nn.Sequential(*dec_layers)

        self.mse_loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    @staticmethod
    def reparameterize(params):
        """(N, 2Z) -> z, mu, logvar each (N, Z)."""
        mu, logvar = torch.chunk(params, 2, dim=-1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar

    def forward(self, x):
        """
        x: (B, H, W, C) -> recon (B, H, W, C), mu (B,H,W,Z), logvar (B,H,W,Z).
        """
        b, h, w, c = x.shape
        flat = x.reshape(b * h * w, c)                # (N, C); N = B*H*W

        params = self.encoder(flat)                   # (N, 2Z)
        z, mu, logvar = self.reparameterize(params)   # (N, Z) each
        recon = torch.sigmoid(self.decoder(z))        # (N, C)

        recon = recon.reshape(b, h, w, c)
        mu = mu.reshape(b, h, w, self.latent_dim)
        logvar = logvar.reshape(b, h, w, self.latent_dim)
        return recon, mu, logvar

    def loss_terms(self, x, beta=1e-3, lambda_physics=0.3, use_physics=False):
        recon, mu, logvar = self(x)
        mse = self.mse_loss_fn(recon, x)
        kld = kl_divergence(mu, logvar)
        sam = spectral_angle_mapper_loss(x, recon)
        loss = mse + beta * kld
        if use_physics:
            loss = loss + lambda_physics * sam
        return {"loss": loss, "mse": mse, "kld": kld, "sam": sam}

    @torch.no_grad()
    def reconstruct(self, x):
        recon, *_ = self(x)
        return recon

    # ------------------------------------------------------------------
    # Downstream-experiment contract (inference/downstream.py)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_latents(self, x):
        """Deterministic per-pixel latents (mu): [ (B, H, W, Z) ]."""
        b, h, w, c = x.shape
        params = self.encoder(x.reshape(b * h * w, c))
        mu, _ = torch.chunk(params, 2, dim=-1)
        return [mu.reshape(b, h, w, self.latent_dim)]

    @torch.no_grad()
    def decode_latents(self, latents):
        """[ (B, H, W, Z) ] -> recon (B, H, W, C)."""
        z = latents[0]
        b, h, w, _ = z.shape
        recon = torch.sigmoid(self.decoder(z.reshape(b * h * w, self.latent_dim)))
        return recon.reshape(b, h, w, settings.input_channels)
