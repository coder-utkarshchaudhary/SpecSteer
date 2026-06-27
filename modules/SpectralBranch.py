from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
import torch

from utils.config import settings

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        in_c = 1
        out_c = settings.input_channels

        conv1d_layers = []
        for _ in range(settings.spectral_n_1D_conv_blocks):
            layer = Conv1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=settings.spectral_conv1D_kernel_size,
                stride=2,
                padding=1
            )
            conv1d_layers.append(layer)
            conv1d_layers.append(nn.ReLU())
            in_c = out_c
            out_c *= 2
            
        self.conv1D_block = nn.Sequential(*conv1d_layers)
        
        self.flatten = nn.Flatten()

        # FIX: emit 2*spectral_latent_dim so reparameterize's chunk(2, dim=1)
        # yields (B, spectral_latent_dim, H, W) for mu and logvar respectively.
        self.linear = nn.LazyLinear(2 * settings.spectral_latent_dim)

    def forward(self, x):
        """
        x: (B, H, W, C)
        returns:
            (B, 2*spectral_latent_dim, H, W)  — mu and logvar concatenated along the channel dim (dim=1). Caller should pass to reparameterize.
        """
        batch, h, w, c = x.shape
        assert h == settings.input_height, f"SPECTRAL ENCODER: Mismatch in height. Expected: {settings.input_height} found: {h}."
        assert w == settings.input_width, f"SPECTRAL ENCODER: Mismatch in width. Expected: {settings.input_width} found: {w}."
        assert c == settings.input_channels, f"SPECTRAL ENCODER: Mismatch in channels. Expected: {settings.input_channels} found: {c}."

        # Fold spatial dimensions into batch dimension to isolate spectral sequences
        x = x.reshape(batch * h * w, 1, c)
        # (B*H*W, 1, input_channels)

        x = self.conv1D_block(x)
        # (B*H*W, final_conv_c, final_conv_l)

        x = self.flatten(x)
        # (B*H*W, final_conv_c * final_conv_l)

        x = self.linear(x)
        # (B*H*W, 2*spectral_latent_dim)

        x = x.reshape(batch, h, w, 2 * settings.spectral_latent_dim)
        # (B, H, W, 2*spectral_latent_dim)

        # Permute to standard PyTorch format (channels first)
        x = x.permute(0, 3, 1, 2)
        # (B, 2*spectral_latent_dim, H, W)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # FIX: in_features = spectral_latent_dim (the reparameterized z dim,
        # *not* 2x) because decoder receives the sampled z after chunk.
        self.linear = nn.Linear(
            in_features=settings.spectral_latent_dim,
            out_features=settings.spectral_linear_expansion_dim
        )

        # Expected dimensions required to reshape the linear output back into a 1D sequence map
        self.trans_in_c = settings.spectral_transpose_c
        self.trans_in_l = settings.spectral_transpose_l

        in_c = self.trans_in_c
        
        layers = []
        for i in range(settings.spectral_n_1D_conv_blocks):
            # The final layer must output a single channel (the reconstructed 1D sequence)
            is_last = i == (settings.spectral_n_1D_conv_blocks - 1)
            out_c = 1 if is_last else in_c // 2
            
            layer = ConvTranspose1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=settings.spectral_conv1D_kernel_size,
                stride=2,
                padding=1
            )
            layers.append(layer)
            if not is_last:
                layers.append(nn.ReLU())
                
            in_c = out_c

        self.transposeconv1D_block = nn.Sequential(*layers)

    def forward(self, z):
        """
        z: (B, spectral_latent_dim, H, W)
        returns:
            (B, H, W, input_channels)
        """
        batch, c, h, w = z.shape
        assert c == settings.spectral_latent_dim, f"SPECTRAL DECODER: Mismatch in channels. Expected: {settings.spectral_latent_dim} found: {c}."
        assert h == settings.input_height, f"SPECTRAL DECODER: Mismatch in height. Expected: {settings.input_height} found: {h}."
        assert w == settings.input_width, f"SPECTRAL DECODER: Mismatch in width. Expected: {settings.input_width} found: {w}."

        # Permute and fold spatial dimensions
        x = z.permute(0, 2, 3, 1)
        # (B, H, W, spectral_latent_dim)
        
        x = x.reshape(batch * h * w, settings.spectral_latent_dim)
        # (B*H*W, spectral_latent_dim)

        x = self.linear(x)
        # (B*H*W, spectral_linear_expansion_dim)

        x = x.view(batch * h * w, self.trans_in_c, self.trans_in_l)
        # (B*H*W, spectral_transpose_c, spectral_transpose_l)

        x = self.transposeconv1D_block(x)
        # (B*H*W, 1, output_sequence_length)
        
        assert x.shape[2] == settings.input_channels, f"SPECTRAL DECODER: Sequence reconstruction length mismatch. Expected: {settings.input_channels} found: {x.shape[2]}."

        x = x.squeeze(1)
        # (B*H*W, input_channels)

        # Unfold spatial dimensions back to the original geometry
        x = x.reshape(batch, h, w, settings.input_channels)
        # (B, H, W, input_channels)
        
        return x

class SpectralEncoderDecoder(nn.Module):
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
            z_map : (B, 2*spectral_latent_dim, H, W) — raw encoder output

        Returns:
            z      : (B, spectral_latent_dim, H, W)
            mu     : (B, spectral_latent_dim, H, W)
            logvar : (B, spectral_latent_dim, H, W)
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
            z             : (B, spectral_latent_dim, H, W)
            mu            : (B, spectral_latent_dim, H, W)
            logvar        : (B, spectral_latent_dim, H, W)
            reconstruction: (B, H, W, input_channels)

        Note: HSI_DualStream_PI_VAE in train.py calls .encoder and .decoder
        directly (with its own shared reparameterize).  This standalone forward
        is provided for single-branch inference and LDM Phase-2 encoding.
        """
        z_map = self.encoder(x)                          # (B, 2*spectral_latent_dim, H, W)
        z, mu, logvar = self.reparameterize(z_map)       # (B, spectral_latent_dim, H, W)
        reconstruction = self.decoder(z)                 # (B, H, W, input_channels)

        return z, mu, logvar, reconstruction