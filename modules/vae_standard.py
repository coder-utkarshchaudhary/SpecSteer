"""
modules/vae_standard.py
-----------------------
"vae-standard" — a standard single-stream 2D-convolution VAE baseline.

Baseline A — the "GenAI" standard.
    Inspiration: High-Resolution Image Synthesis with Latent Diffusion Models
                 (Rombach et al., 2022) — the Stable Diffusion AutoencoderKL,
                 modified only to accept C input channels instead of 3.

The hyperspectral cube is treated like a "very thick RGB image": purely 2D
convolutions downsample it to a compact spatial latent grid, then transpose
convolutions upsample it back. Because 2D convs mix neighbouring pixels, the
per-pixel chemical signature is blurred in latent space (an ice pixel and a
rock pixel get averaged) — the hypothesised failure mode (high PSNR but poor
SAM), which the dual-stream vae-our is designed to fix.

It satisfies the model-agnostic contract used by train/train.py and
inference/inference.py (see modules/vae_our.py for the reference):

    forward(x)                                  # x: (B, H, W, C)
    loss_terms(x, beta, lambda_physics, use_physics) -> dict(loss, mse, kld, sam)
    reconstruct(x) -> (B, H, W, C)

plus the downstream-experiment contract (inference/downstream.py):

    encode_latents(x) -> [ (B, Zc, h', w') ]    # deterministic (mu) latents
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


class VAE_Standard(nn.Module):
    """Single-stream 2D-conv VAE baseline ("vae-standard"), Baseline A.

    Capacity knobs (`vae_standard_base_ch`, `vae_standard_n_down`,
    `vae_standard_latent_ch`) are read from `settings` at build time so a
    per-dataset hyperparam YAML can match this baseline's param count to
    vae-our at each dataset.
    """

    def __init__(self):
        super().__init__()
        c = settings.input_channels
        base_ch = settings.vae_standard_base_ch
        n_down = settings.vae_standard_n_down
        latent_ch = settings.vae_standard_latent_ch
        self.latent_ch = latent_ch

        # ---- Encoder: (B, C, H, W) -> (B, 2*latent_ch, H/2^N, W/2^N) ----
        enc = [nn.Conv2d(c, base_ch, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        in_c = base_ch
        for _ in range(n_down):
            out_c = in_c * 2
            enc += [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1), nn.ReLU()]
            in_c = out_c
        # Project to mu||logvar latent map (1x1 conv, no activation).
        enc.append(nn.Conv2d(in_c, 2 * latent_ch, kernel_size=1))
        self.encoder = nn.Sequential(*enc)

        # ---- Decoder: (B, latent_ch, H/2^N, W/2^N) -> (B, C, H, W) ----
        # Mirror the encoder's channel schedule (in_c is the deepest width).
        dec = [nn.Conv2d(latent_ch, in_c, kernel_size=1), nn.ReLU()]
        for _ in range(n_down):
            out_c = in_c // 2
            dec += [nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1), nn.ReLU()]
            in_c = out_c
        # in_c is now base_ch; final 3x3 conv back to C bands.
        dec.append(nn.Conv2d(in_c, c, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*dec)

        self.mse_loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    @staticmethod
    def reparameterize(params):
        """(B, 2Zc, h, w) -> z, mu, logvar each (B, Zc, h, w) (chunk on channels)."""
        mu, logvar = torch.chunk(params, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar

    def forward(self, x):
        """x: (B, H, W, C) -> recon (B, H, W, C), mu, logvar (B, Zc, h', w')."""
        x = x.permute(0, 3, 1, 2)                     # (B, C, H, W)
        params = self.encoder(x)                      # (B, 2Zc, h', w')
        z, mu, logvar = self.reparameterize(params)   # (B, Zc, h', w')
        recon = torch.sigmoid(self.decoder(z))        # (B, C, H, W)
        recon = recon.permute(0, 2, 3, 1)             # (B, H, W, C)
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
        """Deterministic latent map (mu): [ (B, Zc, h', w') ]."""
        params = self.encoder(x.permute(0, 3, 1, 2))
        mu, _ = torch.chunk(params, 2, dim=1)
        return [mu]

    @torch.no_grad()
    def decode_latents(self, latents):
        """[ (B, Zc, h', w') ] -> recon (B, H, W, C)."""
        recon = torch.sigmoid(self.decoder(latents[0]))
        return recon.permute(0, 2, 3, 1)
