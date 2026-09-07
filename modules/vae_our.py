"""
modules/vae_our.py
------------------
"vae-our" — the dual-stream, physics-informed VAE (HSI_DualStream_PI_VAE),
a.k.a. PRISM.

Two independent VAE streams each reconstruct the full cube:
  - Spatial stream  : per-pixel 1x1 spectral reduction -> 2D conv pyramid ->
                      8x8 spatial GRID latent (B, d_s, 8, 8).
  - Spectral stream : per-pixel 1D spectral conv -> spatially-resolved latent
                      map (B, d_p, H, W).
A spatially-adaptive gated fusion combines the two reconstructions per pixel
and per band. Each branch decoder ends in a sigmoid (like every baseline
decoder), so the fusion is a plain convex blend with no output head.

ITERATION-1 CHANGES (docs/new_plan.md, 2026-09-04)
==================================================
1. Spatial latent: global 256-vector -> 8x8 grid (see modules/SpatialBranch.py).
2. Fusion: the fixed global `Linear(2C -> C)` could only mix the two streams
   with one spectral recipe applied identically at every pixel. It is replaced
   by AdaptiveGatedFusion: a 2x(3x3 conv) network over the concatenated
   reconstructions emits a per-pixel, per-band gate alpha in [0, 1], and
   `recon_final = alpha * h_s + (1 - alpha) * h_p` (a convex blend of two
   [0, 1] branch reconstructions — no output sigmoid). The gate sees a
   spatial neighbourhood (3x3), so fusion can trust the spatial stream on
   texture and the spectral stream on absorption features, per location.
   `settings.vae_our_adaptive_fusion = False` restores the old Linear fusion
   (the fusion ablation for the paper).
3. Aux-MSE downweight: per-stream reconstruction weights drop from
   0.5/0.25/0.25 to a 0.5 : w : w mix (w = settings.vae_our_aux_mse_weight,
   default 0.1), NORMALISED to sum to 1. The 5:1:1 ratio lets each stream
   specialise instead of being forced to a standalone full reconstruction;
   the normalisation keeps the reconstruction term on the same scale as every
   baseline's single MSE, so `beta` and `lambda_physics` keep meaning the same
   thing across the ablation (the invariant documented in the 2026-08 fix).

Other notes:
  - reparameterize clamps logvar to [-30, 20] before exp (numerical stability),
  - each branch decoder ends in a sigmoid (inputs are max-normalized to [0, 1]
    by the dataloader); the fusion is a convex blend and stays in [0, 1],
  - SAM is intrinsic to this model, so it is always physics-informed
    (``PHYSICS_ONLY`` in modules/registry.py).

I/O convention: channels-last (B, H, W, C) throughout.
"""

import torch
import torch.nn as nn

from modules.SpatialBranch import SpatialEncoderDecoder
from modules.SpectralBranch import SpectralEncoderDecoder
from modules.losses import spectral_angle_mapper_loss, kl_divergence
from utils.config import settings


class AdaptiveGatedFusion(nn.Module):
    """
    Spatially-adaptive gated late fusion.

    alpha = sigmoid( Conv3x3( ReLU( Conv3x3( cat[h_s, h_p] ) ) ) )   # (B,C,H,W)
    fused = alpha * h_s + (1 - alpha) * h_p     # h_s, h_p already in [0, 1]

    alpha is per-pixel AND per-band, computed from a 3x3 neighbourhood of both
    reconstructions, so the fusion can locally arbitrate: spatial stream where
    texture/structure dominates, spectral stream where spectral shape does.
    At init the conv logits are near 0, so alpha starts around 0.5 — an even
    blend, matching the old fusion's operating point. There is no output
    sigmoid: fused is a convex combination of two [0, 1] tensors, so it is in
    [0, 1] by construction. (Squashing it again is the 2026-09-05 bug that
    trapped the reconstruction in [0.5, 0.73].)
    """

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=3, padding=1),
        )

    def _alpha(self, h_s, h_p):
        """Gate map from channels-first stream reconstructions."""
        return torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1)))

    def forward(self, recon_s, recon_p):
        """
        recon_s, recon_p : (B, H, W, C) stream reconstructions (channels-last)
        returns          : (B, H, W, C) fused reconstruction in [0, 1]
        """
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        alpha = self._alpha(h_s, h_p)
        # recon_s / recon_p are already in [0, 1] (each branch decoder ends in a
        # sigmoid), so a convex blend of them is in [0, 1] too — no output head.
        fused = alpha * h_s + (1.0 - alpha) * h_p
        return fused.permute(0, 2, 3, 1)

    @torch.no_grad()
    def gate_map(self, recon_s, recon_p):
        """
        alpha as (B, H, W, C), for visualisation/ablation figures only
        (mean over C gives the per-pixel spatial-vs-spectral reliance map).
        """
        h_s = recon_s.permute(0, 3, 1, 2)
        h_p = recon_p.permute(0, 3, 1, 2)
        return self._alpha(h_s, h_p).permute(0, 2, 3, 1)


class LinearFusion(nn.Module):
    """The pre-Iteration-1 fusion: one global Linear(2C -> C) + sigmoid.

    Kept as the fusion-ablation arm (settings.vae_our_adaptive_fusion = False).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fusion_layer = nn.Linear(channels * 2, channels)

    def forward(self, recon_s, recon_p):
        combined = torch.cat([recon_s, recon_p], dim=-1)      # (B, H, W, 2C)
        return torch.sigmoid(self.fusion_layer(combined))     # (B, H, W, C)


class HSI_DualStream_PI_VAE(nn.Module):
    """Dual-stream gated-late-fusion physics-informed VAE ("vae-our")."""

    def __init__(self):
        super().__init__()

        self.spatial_stream = SpatialEncoderDecoder()
        self.spectral_stream = SpectralEncoderDecoder()

        if settings.vae_our_adaptive_fusion:
            self.fusion = AdaptiveGatedFusion(settings.input_channels,
                                              settings.vae_our_fusion_hidden)
        else:
            self.fusion = LinearFusion(settings.input_channels)

        self.mse_loss_fn = nn.MSELoss()

    def reparameterize(self, z_features):
        """
        Chunk encoder output into mu/logvar (on dim=1) and sample z.

        Spatial:  (B, 2*d_s, G, G)             -> (B, d_s, G, G)
        Spectral: (B, 2*spectral_latent, H, W) -> (B, spectral_latent, H, W)
        """
        mu, logvar = torch.chunk(z_features, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, x):
        # --- Spatial stream ---
        spatial_features = self.spatial_stream.encoder(x)     # (B, 2*d_s, G, G)
        z_s, mu_s, logvar_s = self.reparameterize(spatial_features)
        recon_s = self.spatial_stream.decoder(z_s)            # (B, H, W, C)

        # --- Spectral stream ---
        spectral_features = self.spectral_stream.encoder(x)   # (B, 2*d_p, H, W)
        z_p, mu_p, logvar_p = self.reparameterize(spectral_features)
        recon_p = self.spectral_stream.decoder(z_p)           # (B, H, W, C)

        # --- Adaptive gated late fusion ---
        recon_final = self.fusion(recon_s, recon_p)           # (B, H, W, C)

        return recon_final, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p

    def loss_terms(self, x, beta=1e-3, lambda_physics=0.3, use_physics=True):
        """
        Compute all loss components for one batch.

        vae-our is physics-informed by design; ``use_physics`` is accepted for a
        uniform interface but defaults to True and is expected to stay True (the
        registry marks this model PHYSICS_ONLY).

        Returns:
            dict(loss=, mse=, kld=, sam=) — all scalar tensors.
        """
        recon_final, recon_s, recon_p, mu_s, logvar_s, mu_p, logvar_p = self(x)

        # --- Multi-branch reconstruction MSE, as a weighted MEAN -------------
        # Two invariants, both deliberate:
        #   1. The weights sum to 1, so this model's reconstruction term has the
        #      same magnitude as a baseline's single MSE and a shared
        #      `lambda_physics` / `beta` means the same thing in every cell
        #      (the 2026-08 fix — see git history for the full rationale).
        #   2. The RATIO is 0.5 : w : w with w = vae_our_aux_mse_weight
        #      (default 0.1 -> 5:1:1, from Iteration 1's aux-MSE downweight;
        #      previously 0.5/0.25/0.25 = 2:1:1). The aux terms exist so each
        #      stream stays trained end-to-end, but at 2:1:1 they forced both
        #      streams toward standalone full reconstructions, fighting the
        #      specialisation the fusion is supposed to exploit.
        mse_final = self.mse_loss_fn(recon_final, x)
        mse_spatial = self.mse_loss_fn(recon_s, x)
        mse_spectral = self.mse_loss_fn(recon_p, x)
        w_aux = settings.vae_our_aux_mse_weight
        denom = 0.5 + 2.0 * w_aux
        total_mse = (0.5 * mse_final + w_aux * mse_spatial + w_aux * mse_spectral) / denom

        # Combined KL, mean of the two streams (mean-form primitives, so each is
        # already batch-normalized; the 0.5 makes it comparable to a baseline's
        # single kld rather than twice one).
        total_kld = 0.5 * (kl_divergence(mu_s, logvar_s) + kl_divergence(mu_p, logvar_p))

        # Physics prior (SAM) on the fused reconstruction.
        sam = spectral_angle_mapper_loss(x, recon_final)

        loss = total_mse + beta * total_kld
        if use_physics:
            loss = loss + lambda_physics * sam

        # `mse` is the training objective's reconstruction term (a 3-branch
        # average, specific to this model). `mse_final` is the reconstruction
        # MSE of the fused output — the ONLY one comparable to a baseline's
        # `mse`. Reporting code must use `mse_final` for cross-model comparison.
        return {"loss": loss, "mse": total_mse, "mse_final": mse_final,
                "mse_spatial": mse_spatial, "mse_spectral": mse_spectral,
                "kld": total_kld, "sam": sam, "recon": recon_final}

    @torch.no_grad()
    def reconstruct(self, x):
        """Return the fused reconstruction (B, H, W, C) for inference."""
        recon_final, *_ = self(x)
        return recon_final

    # ------------------------------------------------------------------
    # Downstream-experiment contract (inference/downstream.py)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_latents(self, x):
        """
        Deterministic (mu) latents of both streams, for noise-injection and
        interpolation experiments:

            [ z_spatial  (B, d_s, G, G),
              z_spectral (B, spectral_latent_dim, H, W) ]
        """
        spatial_features = self.spatial_stream.encoder(x)
        mu_s, _ = torch.chunk(spatial_features, 2, dim=1)
        spectral_features = self.spectral_stream.encoder(x)
        mu_p, _ = torch.chunk(spectral_features, 2, dim=1)
        return [mu_s, mu_p]

    @torch.no_grad()
    def decode_latents(self, latents):
        """
        [ z_spatial, z_spectral ] -> fused reconstruction (B, H, W, C).

        Mirrors forward()'s late fusion so a perturbed/interpolated latent pair
        maps back through the exact reconstruction path used in training.
        """
        z_s, z_p = latents
        recon_s = self.spatial_stream.decoder(z_s)
        recon_p = self.spectral_stream.decoder(z_p)
        return self.fusion(recon_s, recon_p)
