"""
modules/vae_3d.py
-----------------
"vae-3d-spatio-spectral" — a 3D-convolution VAE that jointly models spatial and
spectral structure by treating each patch as a single-channel volume.

Baseline B — the "Traditional HSI" standard.
    Inspiration: 3D Convolutional Neural Networks for Hyperspectral Image
                 Classification (Chen et al., 2016) / 3D-CAE for Hyperspectral
                 Unmixing (Palsson et al., 2018).

The (H, W, C) patch is lifted to a single-channel volume (B, 1, C, H, W) and
processed entirely with Conv3d / ConvTranspose3d. Each 3D kernel spans a local
(depth × height × width) block, so it inevitably averages neighbouring spectral
bands together with neighbouring spatial pixels. It is also the most
parameter-heavy of the ablation and prone to posterior collapse when squeezed
into a small latent — the hypothesised failure modes.

Design choice for robustness: the spatial dims (H, W) are downsampled by strided
3D convs, but the spectral depth is preserved (stride 1, padding 1) so the
encode→decode round-trip is exact for *any* band count (IIRS=256, M3=84,
AVIRIS=424) without per-dataset spectral arithmetic.

It satisfies the model-agnostic contract used by train/train.py and
inference/inference.py (see modules/vae_our.py for the reference):

    forward(x)                                  # x: (B, H, W, C)
    loss_terms(x, beta, lambda_physics, use_physics) -> dict(loss, mse, kld, sam)
    reconstruct(x) -> (B, H, W, C)

plus the downstream-experiment contract (inference/downstream.py):

    encode_latents(x) -> [ (B, Zc, C, h', w') ]  # deterministic (mu) latents
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


# Capacity. 3D convs are expensive, so BASE is kept small. With H=W=64 and
# N_DOWN=3 the spatial grid is 8x8; the spectral depth C is preserved end to end.
BASE_CH = 16
N_DOWN = 3
LATENT_CH = 8

# Kernel/stride tuples are (depth, height, width): depth (spectral) uses stride 1
# so C is preserved; H/W use stride 2 to down/upsample by a factor of 2 per block.
_DOWN_K = (3, 4, 4)
_DOWN_S = (1, 2, 2)
_DOWN_P = (1, 1, 1)


class VAE_3D_SpatioSpectral(nn.Module):
    """3D spatio-spectral VAE ("vae-3d-spatio-spectral"), Baseline B."""

    def __init__(self):
        super().__init__()
        self.latent_ch = LATENT_CH

        # ---- Encoder: (B, 1, C, H, W) -> (B, 2*LATENT_CH, C, h', w') ----
        enc = [nn.Conv3d(1, BASE_CH, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        in_c = BASE_CH
        for _ in range(N_DOWN):
            out_c = in_c * 2
            enc += [nn.Conv3d(in_c, out_c, kernel_size=_DOWN_K, stride=_DOWN_S, padding=_DOWN_P),
                    nn.ReLU()]
            in_c = out_c
        enc.append(nn.Conv3d(in_c, 2 * LATENT_CH, kernel_size=1))
        self.encoder = nn.Sequential(*enc)

        # ---- Decoder: (B, LATENT_CH, C, h', w') -> (B, 1, C, H, W) ----
        dec = [nn.Conv3d(LATENT_CH, in_c, kernel_size=1), nn.ReLU()]
        for _ in range(N_DOWN):
            out_c = in_c // 2
            dec += [nn.ConvTranspose3d(in_c, out_c, kernel_size=_DOWN_K, stride=_DOWN_S,
                                       padding=_DOWN_P), nn.ReLU()]
            in_c = out_c
        # in_c is now BASE_CH; final 3x3x3 conv back to a single-channel volume.
        dec.append(nn.Conv3d(in_c, 1, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*dec)

        self.mse_loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    @staticmethod
    def reparameterize(params):
        """(B, 2Zc, C, h, w) -> z, mu, logvar each (B, Zc, C, h, w)."""
        mu, logvar = torch.chunk(params, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar

    def forward(self, x):
        """x: (B, H, W, C) -> recon (B, H, W, C), mu, logvar (B, Zc, C, h', w')."""
        # (B, H, W, C) -> (B, 1, C, H, W): spectral is the volume's depth axis.
        vol = x.permute(0, 3, 1, 2).unsqueeze(1)      # (B, 1, C, H, W)
        params = self.encoder(vol)                    # (B, 2Zc, C, h', w')
        z, mu, logvar = self.reparameterize(params)   # (B, Zc, C, h', w')
        recon = torch.sigmoid(self.decoder(z))        # (B, 1, C, H, W)
        recon = recon.squeeze(1).permute(0, 2, 3, 1)  # (B, H, W, C)
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
        """Deterministic latent volume (mu): [ (B, Zc, C, h', w') ]."""
        vol = x.permute(0, 3, 1, 2).unsqueeze(1)
        params = self.encoder(vol)
        mu, _ = torch.chunk(params, 2, dim=1)
        return [mu]

    @torch.no_grad()
    def decode_latents(self, latents):
        """[ (B, Zc, C, h', w') ] -> recon (B, H, W, C)."""
        recon = torch.sigmoid(self.decoder(latents[0]))
        return recon.squeeze(1).permute(0, 2, 3, 1)
