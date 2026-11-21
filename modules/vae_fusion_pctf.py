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

2026-11-20 audit fixes (see docs/results_2026-11-18_pctf_grid.md, section 6):
  * the trust-region contraction is applied in training AND eval (it was eval-only);
  * the constraint is on SAM-valid with a per-dataset budget `pctf_epsilon` (was raw SAM, one
    absolute 0.08 rad for every dataset);
  * lambda moves by projected dual ascent on the real violation (was an Adam-driven ramp);
  * U-PCTF: detached tolerance centred on the budget, linear hinge, slack so lambda can fall.

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


def _valid_pixel_mask(x: torch.Tensor, min_energy: float = 1.0e-8) -> torch.Tensor:
    """(B, H, W, 1) float mask of pixels whose TRUE spectrum has energy above `min_energy`.

    A zero-energy pixel has SAM exactly pi/2 whatever the model predicts and a zero
    gradient (CLAUDE.md section 12), so it can neither be improved nor be part of a
    meaningful constraint. About half of CRIMS pixels are like this. The evaluation's
    SAM-valid excludes them with the same threshold; the constraint now does too.
    """
    return (torch.norm(x, p=2, dim=-1, keepdim=True) > min_energy).to(x.dtype)


def _sam_valid_mean(x: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
    """Scalar mean per-pixel SAM (radians) over pixels with non-zero true energy."""
    mask = _valid_pixel_mask(x)
    ang = _pixelwise_sam(fused, x) * mask
    return ang.sum() / mask.sum().clamp(min=1.0)


def _trust_contraction(d: torch.Tensor, eps) -> torch.Tensor:
    """min(1, eps / d). `d` is detached: the contraction is a rule the gate must learn to
    work inside, not a loss path that lets the streams collapse onto each other."""
    return torch.clamp(eps / (d.detach() + 1e-7), max=1.0)


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

    def __init__(self, channels: int, hidden: int = 64, epsilon: float = 0.08, contract: bool = True):
        super().__init__()
        self.epsilon = float(epsilon)
        # FIX (2026-11-20 audit): the contraction used to run in eval only, so the gate
        # was trained without it, early stopping / checkpoint selection / every reported
        # metric used a different function from the one optimised. It now runs in both
        # modes (contract=True). contract=False = plain gate, no trust region.
        self.contract = bool(contract)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, recon_s, recon_p, training: bool = True):
        # `training` is accepted for call-site compatibility and deliberately ignored:
        # train and eval must compute the same fusion.
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha_cand = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1))).permute(0, 2, 3, 1)

        if self.contract:
            # Ground-truth-free trust region via inter-stream consistency SAM(x_s, x_p):
            # the fused spectrum is kept within ~eps radians of the spectral stream.
            stream_disagreement = _pixelwise_sam(recon_s, recon_p)  # (B, H, W, 1)
            alpha = alpha_cand * _trust_contraction(stream_disagreement, self.epsilon)
        else:
            alpha = alpha_cand

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

    def __init__(self, channels: int, hidden: int = 64, eps_min: float = 0.03, eps_max: float = 0.12,
                 contract: bool = True):
        super().__init__()
        # eps_min / eps_max bracket the per-pixel tolerance; the constructor in
        # HSI_DualStream_Fusion_Uncertainty sets them to 0.5x / 1.5x the dataset's
        # pctf_epsilon so that a "typical" pixel gets tolerance == pctf_epsilon.
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.contract = bool(contract)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def compute_tolerance(self, recon_s, recon_p, logvar_s=None) -> torch.Tensor:
        """
        Compute explicit spatially varying epsilon_i per pixel without ground truth.

        FIX (2026-11-20 audit): everything here is computed from DETACHED inputs. The
        tolerance used to be differentiable w.r.t. the stream outputs, so the network
        could shrink its own constraint violation by loosening epsilon_i instead of by
        reconstructing better.
        """
        recon_s = recon_s.detach()
        recon_p = recon_p.detach()
        logvar_s = logvar_s.detach() if logvar_s is not None else None
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
        # The +1.5 offset centres tau at 0.5 for a typical pixel (norm_s = norm_d = norm_u = 1),
        # so a typical pixel's tolerance is the midpoint of [eps_min, eps_max]. Without it
        # tau sat near 0.18 and the realised tolerance was ~0.68x the intended budget.
        tau_i = torch.sigmoid(norm_s - 1.5 * norm_d - 1.0 * norm_u + 1.5)
        epsilon_i = self.eps_min + (self.eps_max - self.eps_min) * tau_i  # (B, H, W, 1)
        return epsilon_i, d_i

    def forward(self, recon_s, recon_p, logvar_s=None, training: bool = True):
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha_cand = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1))).permute(0, 2, 3, 1)

        epsilon_i, d_i = self.compute_tolerance(recon_s, recon_p, logvar_s)

        # Same fusion in train and eval (see PCTF module). `training` is ignored.
        if self.contract:
            alpha = alpha_cand * _trust_contraction(d_i, epsilon_i)
        else:
            alpha = alpha_cand

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
        # Lagrange multiplier for the constrained modes. A NON-persistent buffer, updated by
        # projected dual ascent in _dual_step(); it is not an optimiser parameter. (It used
        # to be a Parameter driven by Adam, whose step is ~lr whatever the violation is, so
        # lambda was a fixed-speed ramp, not a response to how violated the constraint was.)
        self.register_buffer("lambda_dual", torch.tensor(float(settings.pctf_lambda_init)),
                             persistent=False)

    def _dual_step(self, violation: torch.Tensor) -> None:
        """lambda <- clip(lambda + dual_lr * violation, 0, lambda_max); training only.

        The grad-mode check must happen BEFORE entering no_grad (a @no_grad decorator would
        make torch.is_grad_enabled() always False here and silently freeze lambda).
        """
        if not (self.training and torch.is_grad_enabled()):
            return
        with torch.no_grad():
            self.lambda_dual.add_(settings.pctf_dual_lr * float(violation)).clamp_(0.0, settings.pctf_lambda_max)

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
            # FIX (2026-11-20 audit): the constrained quantity is SAM-valid (zero-energy pixels
            # excluded), not raw SAM; epsilon is the per-dataset budget (pctf_epsilon); lambda is
            # moved by projected dual ascent on the actual violation. `sam` (raw) is still what is
            # REPORTED, so monitoring and checkpoint selection stay comparable with the other arms.
            eps_target = float(getattr(self.fusion, "epsilon", settings.pctf_epsilon))
            g = _sam_valid_mean(x, fused)
            lam = self.lambda_dual.detach().clone()
            loss = elbo_loss + lam * (g - eps_target)   # Lagrangian: ELBO + lambda (g - eps)
            self._dual_step(g.detach() - eps_target)

        elif self.loss_mode == "uncertainty":
            # Exp 5: Uncertainty-Aware Constrained Optimization
            # min L_ELBO s.t. E[ReLU(SAM(x_i, x_f_i) - epsilon_i)] = 0
            epsilon_i = extra_ctx["epsilon_i"]            # (B, H, W, 1), already detached
            mask = _valid_pixel_mask(x)
            pixel_sam = _pixelwise_sam(fused, x)
            violation = torch.relu(pixel_sam - epsilon_i) * mask
            mean_violation = violation.sum() / mask.sum().clamp(min=1.0)
            # Slack of 10% of the mean tolerance lets lambda FALL when the constraint is met
            # (with a target of exactly 0 the hinge is never negative, so lambda could only grow).
            slack = 0.1 * epsilon_i.mean().detach()
            lam = self.lambda_dual.detach().clone()
            # Linear hinge (was squared: for violations < 1 rad that is >10x weaker than the
            # lambda*SAM penalty of the learned-gate arm, i.e. almost no physics pressure).
            loss = elbo_loss + lam * (mean_violation - slack)
            self._dual_step(mean_violation.detach() - slack)

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
            PhysicsConstrainedTrustRegionFusionModule(
                settings.input_channels, settings.vae_our_fusion_hidden,
                epsilon=settings.pctf_epsilon, contract=settings.pctf_contract),
            loss_mode="constrained",
        )
        # LEGACY, frozen, unused: kept only so checkpoints from the 2026-11-18 run (which
        # stored this parameter) still load with strict=True. The live multiplier is the
        # `lambda_dual` buffer.
        self.log_lambda = nn.Parameter(torch.tensor(-1.04), requires_grad=False)


class HSI_DualStream_Fusion_Uncertainty(HSI_DualStream_Fusion_Base):
    """
    Experiment 5: Explicit Uncertainty-Aware Constrained Fusion (U-PCTF).
    Tolerance epsilon_i = f(u_i, d_i, s_i) contracts in high-discrepancy/noisy regions.
    """
    def __init__(self):
        e = settings.pctf_epsilon
        super().__init__(
            UncertaintyAwareConstrainedFusionModule(
                settings.input_channels, settings.vae_our_fusion_hidden,
                eps_min=0.5 * e, eps_max=1.5 * e, contract=settings.pctf_contract),
            loss_mode="uncertainty",
        )
        # LEGACY, frozen, unused (checkpoint compatibility only; see Experiment 4).
        self.log_lambda = nn.Parameter(torch.tensor(-1.04), requires_grad=False)
