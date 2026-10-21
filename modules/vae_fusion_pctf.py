"""
modules/vae_fusion_pctf.py
--------------------------
Physics-Constrained Trust-Region Fusion (PCTF) and Fusion Ablation Suite
for Dual-Stream HSI VAE (PRISM).

Implements the five fusion mechanisms for the methodological ablation:
  1. `vae-fusion-fixed` : Fixed-Weight Fusion (alpha = 0.5, parameter-free convex blend)
  2. `vae-fusion-unconstrained` : Learned Unconstrained Fusion (Gate trained without SAM penalty)
  3. `vae-fusion-sam-penalty` : Learned + SAM Penalty (Standard PRISM regularized fusion)
  4. `vae-fusion-constrained` : Physics-Constrained Trust-Region Fusion (PCTF)
     Formulation: min_{alpha_{ij}} D(x_f, x)  s.t.  SAM(x_{ij}, x_{f,ij}) <= epsilon
     Uses spatial reconstruction as much as possible within a physically admissible spectral trust region.
  5. `vae-fusion-uncertainty` : Uncertainty-Aware Constrained Fusion (U-PCTF)
     Formulation: min_{alpha_{ij}} D(x_f, x)  s.t.  SAM(x_{ij}, x_{f,ij}) <= epsilon_{ij}
     where epsilon_{ij} = f(spectral SNR, ||x_s - x_p||, latent uncertainty).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import settings
from modules.losses import kl_divergence, spectral_angle_mapper_loss
from modules.vae_our import HSI_DualStream_PI_VAE, SpectralEncoderDecoder
from modules.vae_our_variants import SpatialEncoderDecoderNL


# ---------------------------------------------------------------------------
# Numerical Helpers
# ---------------------------------------------------------------------------

def _pixelwise_sam(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute per-pixel spectral angle mapper (SAM) in radians.
    Inputs:
        x1, x2 : (B, H, W, C) channels-last reflectance tensors in [0, 1]
    Returns:
        (B, H, W, 1) spectral angle in radians per pixel
    """
    dot = torch.sum(x1 * x2, dim=-1, keepdim=True)
    norm1 = torch.norm(x1, p=2, dim=-1, keepdim=True)
    norm2 = torch.norm(x2, p=2, dim=-1, keepdim=True)
    denom = norm1 * norm2 + eps
    cos_theta = torch.clamp(dot / denom, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos_theta)


# ===========================================================================
# 1. Fusion Module Implementations
# ===========================================================================

class FixedWeightFusionModule(nn.Module):
    """
    Experiment 1: Fixed-Weight Fusion.
    Parameter-free equal convex blend:
        x_f = alpha * x_s + (1 - alpha) * x_p  (alpha = 0.5)
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, recon_s, recon_p, target_x=None, mu_s=None, logvar_s=None, mu_p=None, logvar_p=None):
        fused = self.alpha * recon_s + (1.0 - self.alpha) * recon_p
        alpha_map = torch.full_like(recon_s[..., :1], self.alpha)
        return fused, alpha_map, None


class LearnedUnconstrainedFusionModule(nn.Module):
    """
    Experiment 2: Learned Unconstrained Fusion.
    Learned gating network optimizing reconstruction MSE without SAM constraint.
    """

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, recon_s, recon_p, target_x=None, mu_s=None, logvar_s=None, mu_p=None, logvar_p=None):
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1)))
        fused = alpha * h_s + (1.0 - alpha) * h_p
        return fused.permute(0, 2, 3, 1), alpha.permute(0, 2, 3, 1), None


class LearnedSAMPenaltyFusionModule(nn.Module):
    """
    Experiment 3: Learned + SAM Penalty (Current PRISM Baseline).
    Learned gating network regularized by scalar SAM penalty in the objective.
    """

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, recon_s, recon_p, target_x=None, mu_s=None, logvar_s=None, mu_p=None, logvar_p=None):
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1)))
        fused = alpha * h_s + (1.0 - alpha) * h_p
        return fused.permute(0, 2, 3, 1), alpha.permute(0, 2, 3, 1), None


class PhysicsConstrainedTrustRegionFusionModule(nn.Module):
    """
    Experiment 4: Physics-Constrained Trust-Region Fusion (PCTF).
    Formulation:
        min_{alpha_{ij}} D(x_f, x)  s.t.  SAM(x_{ij}, x_{f,ij}) <= epsilon

    Spatial context is utilized as much as possible up to the boundary of
    the physically admissible spectral trust region (epsilon). If the candidate
    spatial reliance breaches the trust region, alpha is projected back towards
    the physical spectral branch.
    """

    def __init__(self, channels: int, hidden: int = 64, epsilon: float = 0.08):
        super().__init__()
        self.epsilon = float(epsilon)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, recon_s, recon_p, target_x=None, mu_s=None, logvar_s=None, mu_p=None, logvar_p=None):
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha_cand = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1))).permute(0, 2, 3, 1)

        # Candidate fusion using proposed spatial weight
        recon_cand = alpha_cand * recon_s + (1.0 - alpha_cand) * recon_p

        # Ground-truth input target if present, else physical spectral stream reference
        ref = target_x if target_x is not None else recon_p
        sam_cand = _pixelwise_sam(recon_cand, ref)  # (B, H, W, 1)

        # Trust-region contraction factor:
        # If sam_cand <= epsilon: factor = 1.0 (unconstrained spatial reliance)
        # If sam_cand > epsilon: factor = epsilon / sam_cand (contracts alpha towards 0)
        eps_t = self.epsilon
        factor = torch.clamp(eps_t / (sam_cand + 1e-7), max=1.0)
        alpha_proj = alpha_cand * factor

        fused = alpha_proj * recon_s + (1.0 - alpha_proj) * recon_p

        # Barrier violation penalty: ReLU(SAM - epsilon)^2
        violation = torch.relu(sam_cand - eps_t)
        barrier_loss = torch.mean(violation ** 2)

        return fused, alpha_proj, barrier_loss


class UncertaintyAwareConstrainedFusionModule(nn.Module):
    """
    Experiment 5: Uncertainty-Aware Constrained Fusion (U-PCTF).
    Formulation:
        min_{alpha_{ij}} D(x_f, x)  s.t.  SAM(x_{ij}, x_{f,ij}) <= epsilon_{ij}
        where epsilon_{ij} = f(spectral SNR, ||x_s - x_p||, latent uncertainty).

    Dynamically contracts the spectral trust boundary in ambiguous/high-discrepancy
    regions to guard against spatial hallucination, while relaxing the boundary
    in high-confidence regions to maximize spatial texture.
    """

    def __init__(self, channels: int, hidden: int = 64, eps_min: float = 0.03, eps_max: float = 0.12):
        super().__init__()
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        # Trust boundary predictor from local discrepancy, SNR proxy, and variance
        self.trust_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, recon_s, recon_p, target_x=None, mu_s=None, logvar_s=None, mu_p=None, logvar_p=None):
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha_cand = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1))).permute(0, 2, 3, 1)

        # 1. Inter-stream spectral discrepancy ||x_s - x_p||_2
        discrepancy = torch.norm(recon_s - recon_p, p=2, dim=-1, keepdim=True)  # (B, H, W, 1)
        mean_disc = torch.mean(discrepancy, dim=(1, 2), keepdim=True) + 1e-6
        norm_disc = torch.clamp(discrepancy / mean_disc, 0.0, 5.0) / 5.0

        # 2. Spectral SNR proxy (mean / std along band axis)
        spec_std = torch.std(recon_p, dim=-1, keepdim=True) + 1e-6
        spec_mean = torch.mean(recon_p, dim=-1, keepdim=True)
        norm_snr = torch.clamp(spec_mean / spec_std, 0.0, 10.0) / 10.0

        # 3. Spatial latent uncertainty map
        if logvar_s is not None:
            var_s = torch.exp(0.5 * torch.clamp(logvar_s, -10.0, 10.0))
            var_map = torch.mean(var_s, dim=1, keepdim=True)  # (B, 1, G, G)
            var_map = F.interpolate(var_map, size=(recon_s.shape[1], recon_s.shape[2]), mode="bilinear", align_corners=False)
            norm_unc = torch.clamp(var_map.permute(0, 2, 3, 1), 0.0, 5.0) / 5.0
        else:
            norm_unc = torch.zeros_like(norm_disc)

        feat = torch.cat([norm_disc, norm_snr, norm_unc], dim=-1).permute(0, 3, 1, 2)
        # trust_scale in [0, 1]: 1 = high confidence (loose constraint), 0 = high uncertainty (tight constraint)
        confidence = self.trust_net(feat).permute(0, 2, 3, 1)
        epsilon_ij = self.eps_min + (self.eps_max - self.eps_min) * confidence  # (B, H, W, 1)

        # Candidate fusion and spectral check
        recon_cand = alpha_cand * recon_s + (1.0 - alpha_cand) * recon_p
        ref = target_x if target_x is not None else recon_p
        sam_cand = _pixelwise_sam(recon_cand, ref)

        # Contraction factor bounded by adaptive trust radius
        factor = torch.clamp(epsilon_ij / (sam_cand + 1e-7), max=1.0)
        alpha_proj = alpha_cand * factor

        fused = alpha_proj * recon_s + (1.0 - alpha_proj) * recon_p

        # Barrier loss on trust-boundary violations
        violation = torch.relu(sam_cand - epsilon_ij)
        barrier_loss = torch.mean(violation ** 2)

        return fused, alpha_proj, barrier_loss


# ===========================================================================
# 2. Base Dual-Stream Architecture for Fusion Ablation
# ===========================================================================

class HSI_DualStream_Fusion_Base(HSI_DualStream_PI_VAE):
    """
    Common Dual-Stream base model for the fusion ablation experiments.
    Equipped with PRISM-NL's spatial stream and standard spectral stream.
    Subclasses only specify their fusion module and loss computation regime.
    """

    def __init__(self, fusion_module: nn.Module, loss_mode: str = "sam_penalty"):
        super().__init__()
        # State-of-the-art Non-Linear spatial stream
        self.spatial_stream = SpatialEncoderDecoderNL()
        self.spectral_stream = SpectralEncoderDecoder()
        self.fusion = fusion_module
        self.loss_mode = loss_mode  # "fixed", "unconstrained", "sam_penalty", "constrained", "uncertainty"

    def forward(self, x, zero_spatial: bool = False):
        spatial_features = self.spatial_stream.encoder(x)
        if zero_spatial:
            spatial_features = torch.zeros_like(spatial_features)
        z_s, mu_s, logvar_s = self.reparameterize(spatial_features)
        recon_s = self.spatial_stream.decoder(z_s)

        spectral_features = self.spectral_stream.encoder(x)
        z_p, mu_p, logvar_p = self.reparameterize(spectral_features)
        recon_p = self.spectral_stream.decoder(z_p)

        fused, alpha, barrier = self.fusion(
            recon_s, recon_p, target_x=x,
            mu_s=mu_s, logvar_s=logvar_s, mu_p=mu_p, logvar_p=logvar_p,
        )

        return fused, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p, alpha, barrier

    def loss_terms(self, x, beta: float = 1e-3, lambda_physics: float = 0.3, use_physics: bool = True):
        fused, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p, alpha, barrier = self(x)

        # Standard multi-branch weighted MSE (5:1:1 convention)
        mse_final = self.mse_loss_fn(fused, x)
        mse_spatial = self.mse_loss_fn(recon_s, x)
        mse_spectral = self.mse_loss_fn(recon_p, x)
        w_aux = settings.vae_our_aux_mse_weight
        denom = 0.5 + 2.0 * w_aux
        total_mse = (0.5 * mse_final + w_aux * mse_spatial + w_aux * mse_spectral) / denom

        # Latent KL divergence
        total_kld = 0.5 * (kl_divergence(mu_s, logvar_s) + kl_divergence(mu_p, logvar_p))

        # Overall Spectral Angle Mapper on final reconstruction
        sam = spectral_angle_mapper_loss(x, fused)

        loss = total_mse + beta * total_kld

        # Mode-dependent loss construction
        if self.loss_mode == "unconstrained":
            # Exp 2: Pure ELBO without SAM penalty
            pass
        elif self.loss_mode == "sam_penalty":
            # Exp 3: Standard PRISM scalar SAM penalty
            if use_physics:
                loss = loss + lambda_physics * sam
        elif self.loss_mode in ("constrained", "uncertainty"):
            # Exp 4 & 5: Trust-Region Barrier optimization (min D s.t. SAM <= eps)
            if barrier is not None:
                # Barrier loss drives convergence onto the physical manifold
                loss = loss + lambda_physics * barrier
            elif use_physics:
                loss = loss + lambda_physics * sam
        elif self.loss_mode == "fixed":
            # Exp 1: Fixed weight fusion with standard physics regularizer
            if use_physics:
                loss = loss + lambda_physics * sam

        return {
            "loss": loss,
            "mse": total_mse,
            "mse_final": mse_final,
            "mse_spatial": mse_spatial,
            "mse_spectral": mse_spectral,
            "kld": total_kld,
            "sam": sam,
            "recon": fused,
        }

    @torch.no_grad()
    def reconstruct(self, x, zero_spatial: bool = False):
        fused, *_ = self(x, zero_spatial=zero_spatial)
        return fused

    @torch.no_grad()
    def decode_latents(self, latents, zero_spatial: bool = False):
        z_s, z_p = latents
        if zero_spatial:
            z_s = torch.zeros_like(z_s)
        recon_s = self.spatial_stream.decoder(z_s)
        recon_p = self.spectral_stream.decoder(z_p)
        fused, _, _ = self.fusion(recon_s, recon_p, target_x=None)
        return fused


# ===========================================================================
# 3. Five Model Registrations
# ===========================================================================

class HSI_DualStream_Fusion_Fixed(HSI_DualStream_Fusion_Base):
    """Experiment 1: Fixed-Weight Fusion (alpha = 0.5)."""
    def __init__(self):
        super().__init__(FixedWeightFusionModule(alpha=0.5), loss_mode="fixed")


class HSI_DualStream_Fusion_Unconstrained(HSI_DualStream_Fusion_Base):
    """Experiment 2: Learned Unconstrained Fusion (No SAM penalty)."""
    def __init__(self):
        super().__init__(
            LearnedUnconstrainedFusionModule(settings.input_channels, settings.vae_our_fusion_hidden),
            loss_mode="unconstrained",
        )


class HSI_DualStream_Fusion_SAMPenalty(HSI_DualStream_Fusion_Base):
    """Experiment 3: Learned + SAM Penalty (Current PRISM Regularized Fusion)."""
    def __init__(self):
        super().__init__(
            LearnedSAMPenaltyFusionModule(settings.input_channels, settings.vae_our_fusion_hidden),
            loss_mode="sam_penalty",
        )


class HSI_DualStream_Fusion_Constrained(HSI_DualStream_Fusion_Base):
    """Experiment 4: Physics-Constrained Trust-Region Fusion (PCTF)."""
    def __init__(self):
        super().__init__(
            PhysicsConstrainedTrustRegionFusionModule(settings.input_channels, settings.vae_our_fusion_hidden, epsilon=0.08),
            loss_mode="constrained",
        )


class HSI_DualStream_Fusion_Uncertainty(HSI_DualStream_Fusion_Base):
    """Experiment 5: Uncertainty-Aware Constrained Fusion (U-PCTF)."""
    def __init__(self):
        super().__init__(
            UncertaintyAwareConstrainedFusionModule(settings.input_channels, settings.vae_our_fusion_hidden, eps_min=0.03, eps_max=0.12),
            loss_mode="uncertainty",
        )
