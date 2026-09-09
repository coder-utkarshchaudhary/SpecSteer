"""
modules/vae_our_variants.py
---------------------------
Contains custom high-yield variants of the dual-stream physics-informed VAE ("vae-our").

Implemented Variants:
  1. `vae-our-nl` : Spatially-Adaptive Non-Linear 2D Spectral Projection (PRISM-NL)
     Replaces 1D convolutions in the spatial branch with 2D convolutions to exploit
     local spatial-spectral correlation and avoid linear subspace bottlenecks.
  2. `vae-our-specvit` : Spectral Vision Transformer branch (PRISM-SpecViT)
     Replaces 1D convolutions in the spectral branch with a lightweight per-pixel
     Transformer Encoder along the spectral axis to capture non-local physical wavelength absorption lines.
  3. `vae-our-nl-specvit` : Combines both Non-Linear 2D Projection and Spectral Transformer.
"""

import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, Conv1d, ConvTranspose1d

from utils.config import settings
from modules.losses import spectral_angle_mapper_loss, kl_divergence
from modules.vae_our import HSI_DualStream_PI_VAE, AdaptiveGatedFusion, LinearFusion


# ===========================================================================
# 1. Spatially-Adaptive Non-Linear Projection (NL Variant for Spatial Stream)
# ===========================================================================

class SpatialEncoderNL(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x3 2D Convolution instead of 1D pixel-wise projection
        self.conv2D_input = Conv2d(
            in_channels=settings.input_channels,
            out_channels=settings.reduced_dims,
            kernel_size=3,
            padding=1
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

        self.latent_proj = Conv2d(
            in_channels=in_c,
            out_channels=2 * settings.vae_our_spatial_latent_ch,
            kernel_size=1,
        )

    def forward(self, x):
        # x: (B, H, W, C)
        x_perm = x.permute(0, 3, 1, 2)                      # (B, C, H, W)
        x_proj = self.conv2D_input(x_perm)                  # (B, reduced_dims, H, W)
        x_features = self.conv2D_block(x_proj)              # (B, conv_output_c, G, G)
        x_latent = self.latent_proj(x_features)             # (B, 2*d_s, G, G)
        return x_latent


class SpatialDecoderNL(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid_c = settings.conv_output_c
        self.grid_h = settings.conv_output_h
        self.grid_w = settings.conv_output_w

        self.latent_expand = Conv2d(
            in_channels=settings.vae_our_spatial_latent_ch,
            out_channels=self.grid_c,
            kernel_size=1,
        )

        in_c = self.grid_c

        layers = []
        for _ in range(settings.n_2D_conv_blocks):
            out_c = in_c // 2
            layer = ConvTranspose2d(
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

        # 3x3 2D Convolution instead of 1D pixel-wise reconstruction
        self.conv2D_output = Conv2d(
            in_channels=settings.reduced_dims,
            out_channels=settings.input_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, z):
        x = self.latent_expand(z)                           # (B, grid_c, G, G)
        x = self.transposeconv2D_block(x)                   # (B, reduced_dims, H, W)
        x = self.conv2D_output(x)                           # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)                           # (B, H, W, C)
        return torch.sigmoid(x)


class SpatialEncoderDecoderNL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpatialEncoderNL()
        self.decoder = SpatialDecoderNL()

    @staticmethod
    def reparameterize(z_map):
        mu, logvar = torch.chunk(z_map, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


# ===========================================================================
# 2. Spectral Vision Transformer (SpecViT Variant for Spectral Stream)
# ===========================================================================

class SpectralViTEncoder(nn.Module):
    """
    Spectral Vision Transformer (SpecViT) Encoder.
    Replaces 1D convolutions with a self-attention mechanism over the spectral bands.
    Processes each pixel independently (folded into batch) but captures global, non-local
    band interactions (such as specific chemical absorption dips).
    """
    def __init__(self, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        # Position embeddings along the spectral axis of length C
        self.pos_embed = nn.Parameter(torch.zeros(settings.input_channels, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Mapping token sequence back to latent space parameterisation
        self.linear = nn.Linear(settings.input_channels * d_model, 2 * settings.spectral_latent_dim)
        
    def forward(self, x):
        batch, h, w, c = x.shape
        x = x.reshape(batch * h * w, 1, c)         # (B*H*W, 1, C)
        
        x = self.input_proj(x)                     # (B*H*W, d_model, C)
        x = x.transpose(1, 2)                      # (B*H*W, C, d_model)
        x = x + self.pos_embed.unsqueeze(0)        # Broadcast position embeddings
        
        x = self.transformer(x)                    # (B*H*W, C, d_model)
        x = x.reshape(batch * h * w, -1)           # (B*H*W, C * d_model)
        x = self.linear(x)                         # (B*H*W, 2 * spectral_latent_dim)
        
        x = x.reshape(batch, h, w, 2 * settings.spectral_latent_dim)
        return x.permute(0, 3, 1, 2)               # (B, 2*spectral_latent_dim, H, W)


class SpectralViTDecoder(nn.Module):
    """
    Spectral Vision Transformer (SpecViT) Decoder.
    Reconstructs the spectrum of length C using self-attention.
    """
    def __init__(self, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(
            settings.spectral_latent_dim,
            settings.input_channels * d_model
        )
        # Position embeddings along the spectral axis of length C
        self.pos_embed = nn.Parameter(torch.zeros(settings.input_channels, d_model))
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Conv1d(d_model, 1, kernel_size=3, padding=1)
        
    def forward(self, z):
        batch, c, h, w = z.shape
        x = z.permute(0, 2, 3, 1).reshape(batch * h * w, c)   # (B*H*W, spectral_latent_dim)
        
        x = self.linear(x)                                    # (B*H*W, C * d_model)
        x = x.reshape(batch * h * w, settings.input_channels, self.d_model) # (B*H*W, C, d_model)
        x = x + self.pos_embed.unsqueeze(0)
        
        x = self.transformer(x)                              # (B*H*W, C, d_model)
        x = x.transpose(1, 2)                                # (B*H*W, d_model, C)
        x = self.output_proj(x)                              # (B*H*W, 1, C)
        x = x.squeeze(1)                                     # (B*H*W, C)
        
        return torch.sigmoid(x.reshape(batch, h, w, settings.input_channels))


class SpectralEncoderDecoderSpecViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpectralViTEncoder()
        self.decoder = SpectralViTDecoder()

    @staticmethod
    def reparameterize(z_map):
        mu, logvar = torch.chunk(z_map, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


# ===========================================================================
# 3. Custom Model Declarations (Inherit from HSI_DualStream_PI_VAE)
# ===========================================================================

class HSI_DualStream_PI_VAE_NL(HSI_DualStream_PI_VAE):
    """PRISM-NL: Spatially-Adaptive Non-Linear Spectral Projection."""
    def __init__(self):
        super().__init__()
        self.spatial_stream = SpatialEncoderDecoderNL()


class HSI_DualStream_PI_VAE_SpecViT(HSI_DualStream_PI_VAE):
    """PRISM-SpecViT: Spectral Vision Transformer Stream."""
    def __init__(self):
        super().__init__()
        self.spectral_stream = SpectralEncoderDecoderSpecViT()


class HSI_DualStream_PI_VAE_NL_SpecViT(HSI_DualStream_PI_VAE):
    """PRISM-Hybrid: Spatially-Adaptive NL + Spectral ViT Streams."""
    def __init__(self):
        super().__init__()
        self.spatial_stream = SpatialEncoderDecoderNL()
        self.spectral_stream = SpectralEncoderDecoderSpecViT()


# ===========================================================================
# 4. Dynamic Registry Registration
# ===========================================================================

from modules.registry import MODELS, PHYSICS_ONLY
import modules.registry

MODELS["vae-our-nl"] = HSI_DualStream_PI_VAE_NL
MODELS["vae-our-specvit"] = HSI_DualStream_PI_VAE_SpecViT
MODELS["vae-our-nl-specvit"] = HSI_DualStream_PI_VAE_NL_SpecViT

PHYSICS_ONLY.add("vae-our-nl")
PHYSICS_ONLY.add("vae-our-specvit")
PHYSICS_ONLY.add("vae-our-nl-specvit")

# Re-build MODEL_NAMES tuple in modules.registry so subsequent imports see updated model options
modules.registry.MODEL_NAMES = tuple(MODELS.keys())
