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

All three axes are downsampled by strided 3D convs (k=4, s=2, p=1 — exact
halving/doubling). The depth axis is zero-padded up to a multiple of 2**n_down
before the encoder and cropped back afterwards, so the encode→decode round-trip
is exact for *any* band count (IIRS=256, M3=84→88, AVIRIS=424, CRIMS=456)
without per-dataset spectral arithmetic.

It satisfies the model-agnostic contract used by train/train.py and
inference/inference.py (see modules/vae_our.py for the reference):

    forward(x)                                  # x: (B, H, W, C)
    loss_terms(x, beta, lambda_physics, use_physics) -> dict(loss, mse, kld, sam)
    reconstruct(x) -> (B, H, W, C)

plus the downstream-experiment contract (inference/downstream.py):

    encode_latents(x) -> [ (B, Zc, d, h', w') ]  # deterministic (mu) latents
    decode_latents([z]) -> (B, H, W, C)          # d = C_padded / 2**n_down

Loss (built from modules/losses.py):
    standard : mse + beta * kld
    physics  : mse + beta * kld + lambda_physics * sam

I/O convention: channels-last (B, H, W, C) throughout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.losses import spectral_angle_mapper_loss, kl_divergence
from utils.config import settings


# Kernel/stride/padding tuples are (depth, height, width). All three axes use
# k=4, s=2, p=1 — the same exact-halving/doubling convention SpectralBranch uses:
#     Conv3d:          L_out = L_in // 2
#     ConvTranspose3d: L_out = 2 * L_in
#
# The depth (spectral) axis was previously stride 1, which kept every block at
# full spectral resolution and made this the heaviest model in the ablation by
# far (7,576 GMAC/sample at AVIRIS; the decoder's final ConvTranspose3d alone was
# ~500 G of it). Striding depth cuts that ~8x and, if anything, reinforces this
# baseline's stated hypothesis — that 3D kernels inevitably average neighbouring
# bands together with neighbouring pixels.
#
# k=4 rather than 3 on the depth axis is REQUIRED, not cosmetic: with k=3, s=2,
# p=1 the forward gives ceil(L/2) but the transpose gives 2L-1, which compounds
# over n_down blocks (AVIRIS: 53 -> 105 -> 209 -> 417, short of 424) so no crop
# can recover the original band count.
_DOWN_K = (4, 4, 4)
_DOWN_S = (2, 2, 2)
_DOWN_P = (1, 1, 1)


class VAE_3D_SpatioSpectral(nn.Module):
    """3D spatio-spectral VAE ("vae-3d-spatio-spectral"), Baseline B.

    Capacity knobs (`vae_3d_base_ch`, `vae_3d_n_down`, `vae_3d_latent_ch`) are
    read from `settings` at build time so a per-dataset hyperparam YAML can
    match this baseline's param count to vae-our at each dataset.
    """

    def __init__(self):
        super().__init__()
        base_ch = settings.vae_3d_base_ch
        n_down = settings.vae_3d_n_down
        latent_ch = settings.vae_3d_latent_ch
        self.latent_ch = latent_ch
        self.n_down = n_down

        # Depth must be a multiple of 2**n_down for the strided round-trip to be
        # exact. Captured at build time (not read from the global at forward
        # time) so a later apply_dataset() can't silently desync a live model.
        self.input_channels = settings.input_channels
        self._depth_mult = 2 ** n_down
        self._depth_padded = (
            -(-self.input_channels // self._depth_mult) * self._depth_mult
        )
        # IIRS 256 / AVIRIS 424 / CRIMS 456 are already multiples of 8; only
        # M3 (84 -> 88) actually pads.
        self._depth_pad = self._depth_padded - self.input_channels

        # ---- Encoder: (B, 1, C, H, W) -> (B, 2*latent_ch, C, h', w') ----
        enc = [nn.Conv3d(1, base_ch, kernel_size=3, stride=1, padding=1), nn.ReLU()]
        in_c = base_ch
        for _ in range(n_down):
            out_c = in_c * 2
            enc += [nn.Conv3d(in_c, out_c, kernel_size=_DOWN_K, stride=_DOWN_S, padding=_DOWN_P),
                    nn.ReLU()]
            in_c = out_c
        enc.append(nn.Conv3d(in_c, 2 * latent_ch, kernel_size=1))
        self.encoder = nn.Sequential(*enc)

        # ---- Decoder: (B, latent_ch, C, h', w') -> (B, 1, C, H, W) ----
        dec = [nn.Conv3d(latent_ch, in_c, kernel_size=1), nn.ReLU()]
        for _ in range(n_down):
            out_c = in_c // 2
            dec += [nn.ConvTranspose3d(in_c, out_c, kernel_size=_DOWN_K, stride=_DOWN_S,
                                       padding=_DOWN_P), nn.ReLU()]
            in_c = out_c
        # in_c is now base_ch; final 3x3x3 conv back to a single-channel volume.
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

    def _to_volume(self, x):
        """(B, H, W, C) -> (B, 1, C_padded, H, W), replicating the edge band."""
        vol = x.permute(0, 3, 1, 2).unsqueeze(1)      # (B, 1, C, H, W)
        if self._depth_pad:
            # Pad the depth axis only. F.pad's last-axis-first ordering means the
            # depth pair sits third: (W_l, W_r, H_l, H_r, D_l, D_r).
            vol = F.pad(vol, (0, 0, 0, 0, 0, self._depth_pad), mode="replicate")
        return vol

    def _from_volume(self, vol):
        """(B, 1, C_padded, H, W) -> (B, H, W, C), cropping the depth padding."""
        if self._depth_pad:
            vol = vol[:, :, : self.input_channels]
        return vol.squeeze(1).permute(0, 2, 3, 1)     # (B, H, W, C)

    def forward(self, x):
        """x: (B, H, W, C) -> recon (B, H, W, C), mu, logvar (B, Zc, d, h', w')."""
        # (B, H, W, C) -> (B, 1, C, H, W): spectral is the volume's depth axis.
        vol = self._to_volume(x)                      # (B, 1, C_pad, H, W)
        params = self.encoder(vol)                    # (B, 2Zc, d, h', w')
        z, mu, logvar = self.reparameterize(params)   # (B, Zc, d, h', w')
        recon = torch.sigmoid(self.decoder(z))        # (B, 1, C_pad, H, W)
        recon = self._from_volume(recon)              # (B, H, W, C)
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
        """Deterministic latent volume (mu): [ (B, Zc, C/2**n_down, h', w') ]."""
        params = self.encoder(self._to_volume(x))
        mu, _ = torch.chunk(params, 2, dim=1)
        return [mu]

    @torch.no_grad()
    def decode_latents(self, latents):
        """[ (B, Zc, C/2**n_down, h', w') ] -> recon (B, H, W, C)."""
        recon = torch.sigmoid(self.decoder(latents[0]))
        return self._from_volume(recon)
