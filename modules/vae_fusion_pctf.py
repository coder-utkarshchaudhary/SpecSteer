"""
modules/vae_fusion_pctf.py
--------------------------
Physics-Constrained Trust-Region Fusion (PCTF) and Fusion Ablation Suite
for Dual-Stream HSI VAE (PRISM).

Implements the refined fusion mechanisms based on constrained-optimization theory:
  1. `vae-fusion-fixed` : Fixed-Weight Fusion (alpha = 0.5 per pixel, parameter-free)
  2. `vae-fusion-unconstrained` : Learned Unconstrained Fusion (alpha = Gate(h_s, h_p), no SAM penalty)
  3. `vae-fusion-sam-penalty` : Learned + SAM Penalty (Current PRISM baseline: L_ELBO + lambda * L_SAM)
  4. `vae-fusion-constrained` : Primal-Dual Constrained Optimization (PCTF)
     Formulation: min_{theta, phi, alpha} L_ELBO  s.t.  E[SAM(x, x_f)] <= epsilon
     Lagrange multiplier lambda is dynamically updated via dual gradient ascent during training.
  5. `vae-fusion-uncertainty` : Explicit Uncertainty-Aware Constrained Fusion (U-PCTF)
     Tolerance epsilon_i = f(u_i, d_i, s_i) where:
       * u_i : spatial latent uncertainty (sigma_s^2)
       * d_i : branch disagreement SAM(x_s, x_p)
       * s_i : spectral signal-to-noise ratio
     High discrepancy and uncertainty tighten the spectral trust region (smaller epsilon_i).

Key Architectural Invariants:
  - Single fusion weight per spatial pixel: alpha is shape (B, H, W, 1) shared across
    all spectral bands to prevent independent inter-band mixing and spectral distortion.
  - Training vs Inference separation: Training enforces E[SAM(x, x_f)] <= epsilon with
    respect to ground truth x. Inference operates ground-truth-free, utilizing the learned
    gate and internal stream consistency SAM(x_s, x_p).
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
# 1. Fusion Modules with Single Weight Per Pixel (Shared Across Bands)
# ===========================================================================

class FixedWeightFusionModule(nn.Module):
    """
    Experiment 1: Fixed-Weight Fusion.
    Scalar alpha = 0.5 shared across all pixels and bands:
        x_f = 0.5 * x_s + 0.5 * x_p
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, recon_s, recon_p, training: bool = True):
        fused = self.alpha * recon_s + (1.0 - self.alpha) * recon_p
        alpha_map = torch.full_like(recon_s[..., :1], self.alpha)
        return fused, alpha_map


class LearnedUnconstrainedFusionModule(nn.Module):
    """
    Experiment 2: Learned Unconstrained Fusion.
    Learned 1x1 conv gate predicting a single scalar alpha_ij per spatial pixel,
    shared across all spectral bands:
        alpha = sigmoid( Gate( [h_s, h_p] ) )  # (B, H, W, 1)
        x_f = alpha * x_s + (1 - alpha) * x_p
    """

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),  # Single weight per spatial pixel
        )

    def forward(self, recon_s, recon_p, training: bool = True):
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        # alpha is (B, 1, H, W) -> permuted to (B, H, W, 1) and broadcast across C
        alpha = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1))).permute(0, 2, 3, 1)
        fused = alpha * recon_s + (1.0 - alpha) * recon_p
        return fused, alpha


class LearnedSAMPenaltyFusionModule(nn.Module):
    """
    Experiment 3: Learned + SAM Penalty (Current PRISM Baseline).
    Learned single weight per pixel, regularized by a fixed scalar SAM loss.
    """

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, recon_s, recon_p, training: bool = True):
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1))).permute(0, 2, 3, 1)
        fused = alpha * recon_s + (1.0 - alpha) * recon_p
        return fused, alpha


class PhysicsConstrainedTrustRegionFusionModule(nn.Module):
    """
    Experiment 4: Physics-Constrained Trust-Region Fusion (PCTF).
    
    Training:
        Enforces primal-dual constraint E[SAM(x, x_f)] <= epsilon.
        The gate learns candidate spatial reliance alpha_cand per pixel.
    Inference (Ground-Truth Free):
        Evaluates internal physical consistency between streams:
            disagreement = SAM(x_s, x_p)
        If disagreement > epsilon, contracts alpha towards physical spectral stream:
            alpha = alpha_cand * min(1.0, epsilon / (disagreement + 1e-7))
    """

    def __init__(self, channels: int, hidden: int = 64, epsilon: float = 0.08):
        super().__init__()
        self.epsilon = float(epsilon)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, recon_s, recon_p, training: bool = True):
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha_cand = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1))).permute(0, 2, 3, 1)

        if training:
            # During training, the encoder/decoder/gate parameters are guided by the
            # primal-dual Lagrangian objective in loss_terms().
            alpha = alpha_cand
        else:
            # During inference (deployment), ground truth is unavailable.
            # Enforce physical trust region via inter-stream consistency SAM(x_s, x_p):
            stream_disagreement = _pixelwise_sam(recon_s, recon_p)  # (B, H, W, 1)
            trust_contraction = torch.clamp(self.epsilon / (stream_disagreement + 1e-7), max=1.0)
            alpha = alpha_cand * trust_contraction

        fused = alpha * recon_s + (1.0 - alpha) * recon_p
        return fused, alpha


class UncertaintyAwareConstrainedFusionModule(nn.Module):
    """
    Experiment 5: Explicit Uncertainty-Aware Constrained Fusion (U-PCTF).
    
    Tolerance is spatially varying:
        epsilon_i = f(u_i, d_i, s_i)
    where:
        * u_i : spatial latent uncertainty (sigma_s^2)
        * d_i : branch disagreement SAM(x_s, x_p)
        * s_i : spectral signal-to-noise ratio (mean / std)

    Physical direction:
        High discrepancy (d_i) and high uncertainty (u_i) signify potential spatial
        hallucination -> TIGHTEN spectral tolerance (smaller epsilon_i).
        Clean high SNR (s_i) with stream agreement -> RELAX tolerance (allow spatial sharpness).
    """

    def __init__(self, channels: int, hidden: int = 64, eps_min: float = 0.03, eps_max: float = 0.12):
        super().__init__()
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def compute_tolerance(self, recon_s, recon_p, logvar_s=None) -> torch.Tensor:
        """
        Compute explicit spatially varying epsilon_i per pixel without ground truth.
        """
        # 1. Branch disagreement d_i = SAM(recon_s, recon_p) in radians
        d_i = _pixelwise_sam(recon_s, recon_p)  # (B, H, W, 1)
        mean_d = torch.mean(d_i, dim=(1, 2), keepdim=True) + 1e-6
        norm_d = d_i / mean_d

        # 2. Spectral SNR proxy s_i = mean / (std + eps) along spectral bands
        spec_std = torch.std(recon_p, dim=-1, keepdim=True) + 1e-6
        spec_mean = torch.mean(recon_p, dim=-1, keepdim=True)
        s_i = spec_mean / spec_std
        mean_s = torch.mean(s_i, dim=(1, 2), keepdim=True) + 1e-6
        norm_s = s_i / mean_s

        # 3. Latent uncertainty u_i from spatial stream logvar_s
        if logvar_s is not None:
            var_s = torch.exp(0.5 * torch.clamp(logvar_s, -10.0, 10.0))
            u_map = torch.mean(var_s, dim=1, keepdim=True)  # (B, 1, G, G)
            u_map = F.interpolate(u_map, size=(recon_s.shape[1], recon_s.shape[2]), mode="bilinear", align_corners=False)
            u_i = u_map.permute(0, 2, 3, 1)
            mean_u = torch.mean(u_i, dim=(1, 2), keepdim=True) + 1e-6
            norm_u = u_i / mean_u
        else:
            norm_u = torch.ones_like(norm_d)

        # Physically grounded score:
        # High SNR increases trust; High discrepancy & uncertainty penalize trust.
        # tau_i in (0, 1): 1 = high confidence (relax epsilon), 0 = high risk (tighten epsilon)
        tau_i = torch.sigmoid(norm_s - 1.5 * norm_d - 1.0 * norm_u)
        epsilon_i = self.eps_min + (self.eps_max - self.eps_min) * tau_i  # (B, H, W, 1)
        return epsilon_i, d_i

    def forward(self, recon_s, recon_p, logvar_s=None, training: bool = True):
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha_cand = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1))).permute(0, 2, 3, 1)

        epsilon_i, d_i = self.compute_tolerance(recon_s, recon_p, logvar_s)

        if training:
            alpha = alpha_cand
        else:
            # Inference: contract spatial weight if branch disagreement breaches local tolerance
            contraction = torch.clamp(epsilon_i / (d_i + 1e-7), max=1.0)
            alpha = alpha_cand * contraction

        fused = alpha * recon_s + (1.0 - alpha) * recon_p
        return fused, alpha, epsilon_i


# ===========================================================================
# 2. Base Dual-Stream Architecture for Fusion Ablation
# ===========================================================================

class HSI_DualStream_Fusion_Base(HSI_DualStream_PI_VAE):
    """
    Common Dual-Stream base model for the fusion ablation experiments.
    Uses PRISM-NL's Non-Linear spatial stream and standard spectral stream.
    """

    def __init__(self, fusion_module: nn.Module, loss_mode: str = "sam_penalty"):
        super().__init__()
        self.spatial_stream = SpatialEncoderDecoderNL()
        self.spectral_stream = SpectralEncoderDecoder()
        self.fusion = fusion_module
        self.loss_mode = loss_mode

    def forward(self, x, zero_spatial: bool = False):
        spatial_features = self.spatial_stream.encoder(x)
        if zero_spatial:
            spatial_features = torch.zeros_like(spatial_features)
        z_s, mu_s, logvar_s = self.reparameterize(spatial_features)
        recon_s = self.spatial_stream.decoder(z_s)

        spectral_features = self.spectral_stream.encoder(x)
        z_p, mu_p, logvar_p = self.reparameterize(spectral_features)
        recon_p = self.spectral_stream.decoder(z_p)

        if self.loss_mode == "uncertainty":
            fused, alpha, epsilon_i = self.fusion(recon_s, recon_p, logvar_s=logvar_s, training=self.training)
            extra_ctx = {"epsilon_i": epsilon_i}
        else:
            fused, alpha = self.fusion(recon_s, recon_p, training=self.training)
            extra_ctx = {}

        return fused, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p, alpha, extra_ctx

    def loss_terms(self, x, beta: float = 1e-3, lambda_physics: float = 0.3, use_physics: bool = True):
        fused, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p, alpha, extra_ctx = self(x)

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
        elbo_loss = total_mse + beta * total_kld

        # Mode-dependent loss construction
        if self.loss_mode == "unconstrained":
            # Exp 2: Pure ELBO without SAM penalty
            loss = elbo_loss

        elif self.loss_mode == "sam_penalty":
            # Exp 3: Standard PRISM scalar SAM penalty
            loss = elbo_loss + (lambda_physics * sam if use_physics else 0.0)

        elif self.loss_mode == "constrained":
            # Exp 4: Primal-Dual Constrained Optimization
            # min L_ELBO s.t. E[SAM] <= epsilon
            # Lagrangian: J = L_ELBO + lambda * (E[SAM] - epsilon), lambda >= 0
            eps_target = getattr(self.fusion, "epsilon", 0.08)
            lambda_val = F.softplus(self.log_lambda)
            # Primal gradient pushes network to satisfy constraint via lambda.detach() * sam
            # Dual gradient updates lambda via -lambda * (sam.detach() - eps_target)
            primal_sam_penalty = lambda_val.detach() * sam
            dual_ascent_loss = -lambda_val * (sam.detach() - eps_target)
            loss = elbo_loss + primal_sam_penalty + dual_ascent_loss

        elif self.loss_mode == "uncertainty":
            # Exp 5: Uncertainty-Aware Constrained Optimization
            # min L_ELBO s.t. E[ReLU(SAM(x_i, x_f_i) - epsilon_i)] = 0
            epsilon_i = extra_ctx.get("epsilon_i", 0.08)
            pixel_sam = _pixelwise_sam(fused, x)
            violation = torch.relu(pixel_sam - epsilon_i)
            mean_violation = torch.mean(violation)
            lambda_val = F.softplus(self.log_lambda)
            primal_penalty = lambda_val.detach() * torch.mean(violation ** 2)
            dual_ascent_loss = -lambda_val * (mean_violation.detach() - 0.0)
            loss = elbo_loss + primal_penalty + dual_ascent_loss

        elif self.loss_mode == "fixed":
            # Exp 1: Fixed weight blend with standard physics loss
            loss = elbo_loss + (lambda_physics * sam if use_physics else 0.0)
        else:
            loss = elbo_loss

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
        if self.loss_mode == "uncertainty":
            fused, _, _ = self.fusion(recon_s, recon_p, logvar_s=None, training=False)
        else:
            fused, _ = self.fusion(recon_s, recon_p, training=False)
        return fused


# ===========================================================================
# 3. Five Model Registrations
# ===========================================================================

class HSI_DualStream_Fusion_Fixed(HSI_DualStream_Fusion_Base):
    """Experiment 1: Fixed-Weight Fusion (alpha = 0.5 per pixel)."""
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
    """
    Experiment 4: Primal-Dual Constrained Optimization (PCTF).
    min L_ELBO  s.t.  E[SAM] <= epsilon (epsilon = 0.08 rad).
    Dynamically tunes Lagrange multiplier lambda >= 0.
    """
    def __init__(self):
        super().__init__(
            PhysicsConstrainedTrustRegionFusionModule(settings.input_channels, settings.vae_our_fusion_hidden, epsilon=0.08),
            loss_mode="constrained",
        )
        # Learnable Lagrange multiplier (initialized at softplus^-1(0.3) ~ -1.04)
        self.log_lambda = nn.Parameter(torch.tensor(-1.04))


class HSI_DualStream_Fusion_Uncertainty(HSI_DualStream_Fusion_Base):
    """
    Experiment 5: Explicit Uncertainty-Aware Constrained Fusion (U-PCTF).
    Tolerance epsilon_i = f(u_i, d_i, s_i) contracts in high-discrepancy/noisy regions.
    """
    def __init__(self):
        super().__init__(
            UncertaintyAwareConstrainedFusionModule(settings.input_channels, settings.vae_our_fusion_hidden, eps_min=0.03, eps_max=0.12),
            loss_mode="uncertainty",
        )
        # Learnable Lagrange multiplier for the adaptive violation barrier
        self.log_lambda = nn.Parameter(torch.tensor(-1.04))
