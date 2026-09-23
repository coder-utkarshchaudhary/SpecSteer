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


class VAE_1D_Pixelwise(nn.Module):
    """Per-pixel 1D spectral VAE ("vae-1d-pixelwise"), Baseline C.

    Capacity knobs (`vae_1d_hidden_dims`, `vae_1d_latent_dim`) are read from
    `settings` at build time so a per-dataset hyperparam YAML can match this
    baseline's param count to vae-our at each dataset.
    """

    def __init__(self):
        super().__init__()
        c = settings.input_channels
        hidden_dims = tuple(settings.vae_1d_hidden_dims)
        latent_dim = settings.vae_1d_latent_dim
        self.latent_dim = latent_dim

        # ---- Encoder MLP: C -> hidden... -> 2*Z (mu || logvar) ----
        # Wide MLP hidden dims (up to ~7k for CRIMS, ~4k for IIRS) with plain
        # ReLU risk dead-neuron collapse: too many units get pushed to always-
        # zero early and never recover, degrading the baseline's true capacity.
        # LayerNorm keeps pre-activation scale bounded per pixel spectrum, and
        # GELU is smooth (no zero-gradient half-space) so units cannot die.
        enc_layers = []
        in_f = c
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_f, h), nn.LayerNorm(h), nn.GELU()]
            in_f = h
        enc_layers.append(nn.Linear(in_f, 2 * latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ---- Decoder MLP: Z -> hidden(reversed)... -> C ----
        dec_layers = []
        in_f = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_f, h), nn.LayerNorm(h), nn.GELU()]
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

        chunk_size = 16384
        if flat.shape[0] <= chunk_size:
            params = self.encoder(flat)
            z, mu, logvar = self.reparameterize(params)
            recon = torch.sigmoid(self.decoder(z))
        else:
            p_list, z_list, mu_list, logvar_list, r_list = [], [], [], [], []
            for i in range(0, flat.shape[0], chunk_size):
                fb = flat[i:i + chunk_size]
                pb = self.encoder(fb)
                zb, mub, logvarb = self.reparameterize(pb)
                rb = torch.sigmoid(self.decoder(zb))
                z_list.append(zb)
                mu_list.append(mub)
                logvar_list.append(logvarb)
                r_list.append(rb)
            z = torch.cat(z_list, dim=0)
            mu = torch.cat(mu_list, dim=0)
            logvar = torch.cat(logvar_list, dim=0)
            recon = torch.cat(r_list, dim=0)

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
        # `mse_final` is the RECONSTRUCTION MSE, the quantity that is comparable
        # across every model in the grid. For a single-stream model it equals
        # `mse`; for vae-our it does not (see modules/vae_our.py). `recon` rides
        # along so train.py can compute PSNR/SSIM without a second forward pass.
        return {"loss": loss, "mse": mse, "mse_final": mse,
                "kld": kld, "sam": sam, "recon": recon}

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
        flat = x.reshape(b * h * w, c)
        chunk_size = 16384
        if flat.shape[0] <= chunk_size:
            params = self.encoder(flat)
        else:
            params = torch.cat([self.encoder(flat[i:i + chunk_size])
                                for i in range(0, flat.shape[0], chunk_size)], dim=0)
        mu, _ = torch.chunk(params, 2, dim=-1)
        return [mu.reshape(b, h, w, self.latent_dim)]

    @torch.no_grad()
    def decode_latents(self, latents):
        """[ (B, H, W, Z) ] -> recon (B, H, W, C)."""
        z = latents[0]
        b, h, w, _ = z.shape
        flat = z.reshape(b * h * w, self.latent_dim)
        chunk_size = 16384
        if flat.shape[0] <= chunk_size:
            recon = torch.sigmoid(self.decoder(flat))
        else:
            recon = torch.cat([torch.sigmoid(self.decoder(flat[i:i + chunk_size]))
                               for i in range(0, flat.shape[0], chunk_size)], dim=0)
        return recon.reshape(b, h, w, settings.input_channels)
