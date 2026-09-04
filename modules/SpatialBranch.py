"""
modules/SpatialBranch.py
------------------------
Spatial stream of vae-our (PRISM): per-pixel spectral reduction -> stride-2
conv pyramid -> 8x8 spatial GRID latent -> mirrored transpose-conv decoder.

ARCHITECTURE CHANGE (Iteration 1, docs/new_plan.md, 2026-09-04)
===============================================================
The previous encoder continued the pyramid to 4x4 and then went
`Flatten(8192) -> LazyLinear(2*256)`: a single global 256-vector for the whole
64x64 patch — a ~4096:1 spatial squeeze with no spatial addressing. That global
bottleneck was the ranked root cause of the IIRS fidelity gap (PSNR -5.6 dB /
SSIM -0.11 vs vae-3d) and the CRIMS SSIM collapse (0.56): all spatial texture
had to be hallucinated back from one vector.

This version stops the stride-2 pyramid at the 8x8 grid (3 blocks, not 4) and
projects channels with a 1x1 conv, so the latent keeps spatial structure:

    z_s : (B, 2 * vae_our_spatial_latent_ch, 8, 8)    # mu/logvar on dim=1

like vae-standard's 8x8 latent map, while the spectral stream keeps the
per-pixel spectral-shape channel. The decoder starts from (d_s, 8, 8) via a
1x1 expansion and mirrors the pyramid with ConvTranspose2d blocks. The two
dense layers this removes (Flatten->LazyLinear ~4.2M, Linear 256->8192 ~2.1M)
were ~6M params spent compressing/expanding one vector.

`SpatialEncoderDecoder.reparameterize` chunks on dim=1, which works unchanged
for the 4-D latent.
"""

from torch import nn
from torch.nn import Conv1d, Conv2d
import torch

from utils.config import settings


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1D_block = Conv1d(
            in_channels=1,
            out_channels=settings.reduced_dims,
            kernel_size=settings.input_channels
        )

        in_c = settings.reduced_dims
        out_c = 2 * in_c

        conv2d_layers = []
        for _ in range(settings.n_2D_conv_blocks):
            layer = Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=settings.conv2D_kernel_size,
                stride=2,
                padding=1
            )
            conv2d_layers.append(layer)
            conv2d_layers.append(nn.ReLU())
            in_c = out_c
            out_c *= 2

        self.conv2D_block = nn.Sequential(*conv2d_layers)

        # 1x1 projection onto the grid latent. Emits 2*d_s channels so
        # reparameterize's chunk(2, dim=1) yields (B, d_s, 8, 8) for mu and
        # logvar respectively. Replaces Flatten -> LazyLinear (the global
        # bottleneck this iteration removes).
        self.latent_proj = Conv2d(
            in_channels=in_c,
            out_channels=2 * settings.vae_our_spatial_latent_ch,
            kernel_size=1,
        )

    def forward(self, x):
        """
            x: (B, H, W, C)
            returns:
                (B, 2*vae_our_spatial_latent_ch, G, G) with G = H / 2^n_blocks
                — mu and logvar concatenated along dim=1
                  (caller should pass to reparameterize)
        """
        batch, h, w, c = x.shape
        assert h == settings.input_height, f"SPATIAL ENCODER: Mismatch in height. Expected: {settings.input_height} found: {h}."
        assert w == settings.input_width, f"SPATIAL ENCODER: Mismatch in width. Expected: {settings.input_width} found: {w}."
        assert c == settings.input_channels, f"SPATIAL ENCODER: Mismatch in channels. Expected: {settings.input_channels} found: {c}."

        x = x.reshape(batch * h * w, 1, c)

        x = self.conv1D_block(x)
        # (B*H*W, reduced_dim, 1)

        x = x.squeeze(-1)
        # (B*H*W, reduced_dim)

        x = x.reshape(batch, h, w, settings.reduced_dims)

        x = x.permute(0, 3, 1, 2)
        # (B, reduced_dim, H, W)

        x = self.conv2D_block(x)
        # (B, conv_output_c, conv_output_h, conv_output_w) = (B, 8r, 8, 8)

        x = self.latent_proj(x)
        # (B, 2*d_s, 8, 8)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid_c = settings.conv_output_c   # 8r — mirrors the encoder's top
        self.grid_h = settings.conv_output_h
        self.grid_w = settings.conv_output_w

        # 1x1 expansion from the sampled grid latent back to the pyramid width.
        # Replaces the old Linear(latent_dim -> 8192).
        self.latent_expand = Conv2d(
            in_channels=settings.vae_our_spatial_latent_ch,
            out_channels=self.grid_c,
            kernel_size=1,
        )

        in_c = self.grid_c

        layers = []
        for _ in range(settings.n_2D_conv_blocks):
            out_c = in_c // 2
            layer = nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )

            layers.append(layer)
            layers.append(nn.ReLU())

            in_c = out_c

        self.transposeconv2D_block = nn.Sequential(*layers)

        self.conv1D_block = Conv1d(
            in_channels=1,
            out_channels=settings.input_channels,
            kernel_size=settings.reduced_dims
        )

    def forward(self, z):
        """
        z: (B, vae_our_spatial_latent_ch, G, G)
        returns:
            (B, H, W, Bands)
        """
        B = z.shape[0]
        assert z.shape[1] == settings.vae_our_spatial_latent_ch, f"SPATIAL DECODER: Mismatch in latent channels. Expected {settings.vae_our_spatial_latent_ch}, found {z.shape[1]}"
        assert z.shape[2] == self.grid_h and z.shape[3] == self.grid_w, f"SPATIAL DECODER: Mismatch in latent grid. Expected {self.grid_h}x{self.grid_w}, found {z.shape[2]}x{z.shape[3]}"

        x = self.latent_expand(z)
        # (B, 8r, 8, 8)

        x = self.transposeconv2D_block(x)
        # (B, reduced_dim, H, W)

        assert x.shape[1] == settings.reduced_dims, f"SPATIAL DECODER: Mismatch in channels. Expected reduced_dim={settings.reduced_dims}, found {x.shape[1]}"
        assert x.shape[2] == settings.input_height, f"SPATIAL DECODER: Mismatch in height. Expected height={settings.input_height}, found {x.shape[2]}"
        assert x.shape[3] == settings.input_width, f"SPATIAL DECODER: Mismatch in width. Expected width={settings.input_width}, found {x.shape[3]}"

        H, W = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        # (B, H, W, reduced_dim)

        x = x.reshape(
            B * H * W,
            1,
            settings.reduced_dims
        )

        x = self.conv1D_block(x)
        # (B*H*W, Bands, 1)

        x = x.squeeze(-1)
        # (B*H*W, Bands)

        x = x.reshape(B, settings.input_height, settings.input_width, settings.input_channels)
        # (B, H, W, C)

        return x


class SpatialEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    @staticmethod
    def reparameterize(z_map):
        """
        Split encoder output into mu/logvar and sample z via the
        reparameterization trick.

        Args:
            z_map : (B, 2*d_s, G, G) — raw encoder output

        Returns:
            z      : (B, d_s, G, G) — sampled latent
            mu     : (B, d_s, G, G)
            logvar : (B, d_s, G, G)
        """
        mu, logvar = torch.chunk(z_map, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, x):
        """
        x: (B, H, W, C)
        returns:
            z             : (B, d_s, G, G)   — sampled grid latent
            mu            : (B, d_s, G, G)
            logvar        : (B, d_s, G, G)
            reconstruction: (B, H, W, C)

        Note: HSI_DualStream_PI_VAE in train.py calls .encoder and .decoder
        directly (with its own shared reparameterize).  This standalone forward
        is provided for single-branch inference and LDM Phase-2 encoding.
        """
        z_map = self.encoder(x)                          # (B, 2*d_s, G, G)
        z, mu, logvar = self.reparameterize(z_map)       # (B, d_s, G, G) each
        reconstruction = self.decoder(z)                 # (B, H, W, C)

        return z, mu, logvar, reconstruction
