import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from modules.SpatialBranch import SpatialEncoderDecoder
from modules.SpectralBranch import SpectralEncoderDecoder
from utils.config import settings

"""
1. PHYSICS LOSS: Differentiable Spectral Angle Mapper
"""
def spectral_angle_mapper_loss(y_true, y_pred):
    """
    Computes the SAM between the ground truth and reconstructed HSI cubes.
    y_true, y_pred: (B, H, W, C)
    """
    # Dot product along the wavelength dimension
    dot_product = torch.sum(y_true * y_pred, dim=-1)
    
    # L2 Norms
    norm_true = torch.linalg.norm(y_true, dim=-1)
    norm_pred = torch.linalg.norm(y_pred, dim=-1)
    
    # Cosine similarity with epsilon to prevent division by zero and NaN gradients
    cos_sim = dot_product / (norm_true * norm_pred + 1e-8)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-8, 1.0 - 1e-8)
    
    # SAM is the average arccos across all pixels
    sam = torch.acos(cos_sim)
    return torch.mean(sam)

def kl_divergence(mu, logvar):
    """ Standard KL Divergence pushing the manifold to N(0, I) """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# ---------------------------------------------------------
# 2. UNIFIED DUAL-STREAM PI-VAE ARCHITECTURE
# ---------------------------------------------------------
class HSI_DualStream_PI_VAE(nn.Module):
    def __init__(self, conv_output_c, conv_output_h, conv_output_w):
        super().__init__()
        
        # Initialize isolated branches
        self.spatial_stream = SpatialEncoderDecoder(conv_output_c, conv_output_h, conv_output_w)
        self.spectral_stream = SpectralEncoderDecoder()

        # Simple 1x1 Conv Fusion to merge the final reconstructed cubes
        # Maps concatenated (C + C) back to C
        self.fusion_layer = nn.Linear(settings.input_channels * 2, settings.input_channels)

    def reparameterize(self, z_features):
        """ Chunks the encoder output into mu and logvar, then samples z. """
        # spatial_features: (B, 512) -> mu: (B, 256), logvar: (B, 256)
        # spectral_features: (B, 2048, H, W) -> mu: (B, 1024, H, W), logvar: (B, 1024, H, W)
        mu, logvar = torch.chunk(z_features, 2, dim=1)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, x):
        # --- Spatial Stream ---
        spatial_features = self.spatial_stream.encoder(x)
        z_s, mu_s, logvar_s = self.reparameterize(spatial_features)
        recon_s = self.spatial_stream.decoder(z_s)

        # --- Spectral Stream ---
        spectral_features = self.spectral_stream.encoder(x)
        z_p, mu_p, logvar_p = self.reparameterize(spectral_features)
        recon_p = self.spectral_stream.decoder(z_p)

        # --- Late Fusion ---
        # recon_s and recon_p are both (B, H, W, C)
        combined = torch.cat([recon_s, recon_p], dim=-1) # (B, H, W, 2C)
        recon_final = self.fusion_layer(combined)        # (B, H, W, C)

        return recon_final, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p

# ---------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------
def train_hsi_vae(model, dataloader, epochs, device, lr=1e-4, beta=0.001, lambda_physics=0.5):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mse_loss_fn = nn.MSELoss()

    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, x_batch in enumerate(dataloader):
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            recon_final, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p = model(x_batch)

            # 1. Standard Reconstruction Loss (MSE)
            mse_final = mse_loss_fn(recon_final, x_batch)
            mse_spatial = mse_loss_fn(recon_s, x_batch)
            mse_spectral = mse_loss_fn(recon_p, x_batch)
            total_mse = mse_final + 0.5 * mse_spatial + 0.5 * mse_spectral

            # 2. Physics Prior Loss (SAM)
            sam_loss = spectral_angle_mapper_loss(x_batch, recon_final)

            # 3. KL Divergence (Latent Regularization)
            # Normalize KL by batch size
            B = x_batch.shape[0]
            kld_s = kl_divergence(mu_s, logvar_s) / B
            kld_p = kl_divergence(mu_p, logvar_p) / B
            total_kld = kld_s + kld_p

            # Total Loss = ELBO + Physics Steering
            loss = total_mse + (beta * total_kld) + (lambda_physics * sam_loss)

            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion from SAM arccos domain
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss/len(dataloader):.4f} | "
              f"MSE: {total_mse.item():.4f} | SAM: {sam_loss.item():.4f} | KLD: {total_kld.item():.4f}")

if __name__ == "__main__":
    # Example Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate these based on your settings.n_2D_conv_blocks downsampling
    CONV_OUT_C = settings.reduced_dims * (2 ** settings.n_2D_conv_blocks)
    CONV_OUT_H = settings.input_height // (2 ** settings.n_2D_conv_blocks)
    CONV_OUT_W = settings.input_width // (2 ** settings.n_2D_conv_blocks)

    model = HSI_DualStream_PI_VAE(
        conv_output_c=CONV_OUT_C,
        conv_output_h=CONV_OUT_H,
        conv_output_w=CONV_OUT_W
    )

    # Placeholder dataloader 
    # dataset = Chandrayaan2Dataset(...)
    # dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True)
    
    # For testing compilation:
    dummy_data = torch.randn(8, settings.input_height, settings.input_width, settings.input_channels)
    dummy_loader = [(dummy_data)]

    train_hsi_vae(
        model=model,
        dataloader=dummy_loader,
        epochs=100,
        device=device,
        beta=0.005,           # KL Divergence weight
        lambda_physics=1.0    # Manifold Steering weight
    )