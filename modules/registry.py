"""
modules/registry.py
--------------------
Central model registry for the VAE ablation study. train/train.py and
inference/inference.py resolve a CLI ``--model`` name to a class here and never
branch on model type — every model implements the same contract (see
modules/vae_our.py):

    forward(x)                                  # x: (B, H, W, C)
    loss_terms(x, beta, lambda_physics, use_physics) -> dict
    reconstruct(x) -> (B, H, W, C)

``loss_terms`` returns AT LEAST these keys, and every model returns all of them:

    loss       the scalar being optimised
    mse        this model's own reconstruction term (NOT comparable across models
               -- vae-our's is a 3-branch weighted average)
    mse_final  MSE of the final/fused reconstruction. THIS is the cross-model
               comparable fidelity number; for a single-stream model it equals
               `mse`. Reporting and checkpoint selection must use this one.
    kld        KL term (a mean, so latent size does not bias its magnitude)
    sam        spectral angle of the final reconstruction -- comparable as-is
    recon      (B, H, W, C) final reconstruction, so callers can compute PSNR
               and SSIM without a second forward pass

vae-our additionally returns `mse_spatial` / `mse_spectral` for its per-branch
diagnostics (probe P2). Extra keys are permitted; the six above are the contract.

Note for DataParallel wrappers: `recon` is already batched along dim 0 and must
NOT be unsqueezed before the gather, unlike the scalar entries.

Every model additionally implements the downstream-experiment contract used by
inference/downstream.py (latent noise-injection + chemical interpolation):

    encode_latents(x) -> list[Tensor]           # deterministic (mu) latent(s)
    decode_latents(list[Tensor]) -> (B, H, W, C)

The latent list is opaque and model-specific (vae-our returns two tensors, one
per stream; the baselines return one) — downstream.py perturbs/interpolates each
tensor in place and feeds the list straight back to decode_latents.
"""

from modules.vae_our import HSI_DualStream_PI_VAE
from modules.vae_standard import VAE_Standard
from modules.vae_3d import VAE_3D_SpatioSpectral
from modules.vae_1d import VAE_1D_Pixelwise


# CLI name -> model class.
MODELS = {
    "vae-our": HSI_DualStream_PI_VAE,
    "vae-standard": VAE_Standard,
    "vae-3d-spatio-spectral": VAE_3D_SpatioSpectral,
    "vae-1d-pixelwise": VAE_1D_Pixelwise,
}

# Models whose loss is intrinsically physics-informed (SAM baked in). These have
# no separate "standard" variant — train.py rejects --loss standard for them.
PHYSICS_ONLY = {"vae-our"}

MODEL_NAMES = tuple(MODELS.keys())


def build_model(name: str):
    """
    Instantiate a model by its registry name.

    The model reads its dataset-specific dims from the module-global
    ``utils.config.settings`` (configured via ``apply_dataset`` before this call),
    so no shape arguments are needed here.

    Args:
        name : one of MODEL_NAMES.

    Returns:
        An nn.Module implementing the model contract.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_NAMES)}.")
    return MODELS[name]()


# Checkpoint selection criteria. Every cell writes one checkpoint per criterion.
#
# WHY TWO. `monitor = val_loss` used to pick the saved epoch, and val_loss has a
# different FORM in every cell: vae-our carried a 3-branch MSE and a summed KL,
# `physics` cells carry a SAM term, `standard` cells do not. So each cell's
# weights were selected by a different objective, which makes cross-model
# comparison of those weights unsound.
#
# Selecting everything on SAM instead would break the other way: `standard` cells
# train with no SAM term at all, so their SAM is an incidental by-product of an
# MSE trajectory, not a signal (see CLAUDE.md 12 -- those are the cells that
# collapse to sam = pi/2). Writing both costs ~88 MB per cell and lets each
# analysis read the checkpoint selected on the metric that analysis reports,
# uniformly across all cells.
SELECT_CRITERIA = ("sam", "mse")


def checkpoint_name(model: str, loss_type: str,
                    seed: int | None = None,
                    select: str | None = None) -> str:
    """
    Filename (no directory) for a trained checkpoint.

        checkpoint_name("vae-our", "physics")
            -> "vae-our.pt"                            (legacy / physics-only)
        checkpoint_name("vae-standard", "physics")
            -> "vae-standard_physics.pt"               (legacy)
        checkpoint_name("vae-our", "physics", seed=1, select="sam")
            -> "vae-our_seed1_bestsam.pt"

    `seed` is part of the filename because it was not, and `--seed 1` and
    `--seed 2` silently overwrote each other -- which would have made the whole
    seed-robustness exercise measure nothing.

    Omitting `seed`/`select` reproduces the pre-seed-axis names so older
    checkpoints stay resolvable.
    """
    if select is not None and select not in SELECT_CRITERIA:
        raise ValueError(f"select must be one of {SELECT_CRITERIA}, got {select!r}")

    stem = model if model in PHYSICS_ONLY else f"{model}_{loss_type}"
    if seed is not None:
        stem = f"{stem}_seed{seed}"
    if select is not None:
        stem = f"{stem}_best{select}"
    return f"{stem}.pt"


def find_seeds(ckpt_dir, dataset: str, model: str, loss_type: str,
               select: str = "sam") -> list[int]:
    """
    Seeds that actually have a checkpoint on disk for this cell, ascending.

    Used by the inference/probe sweeps to discover the seed axis rather than
    having it hard-coded in two places (it is defined in
    scripts/grid_manifest.sh, which Python cannot read).
    """
    from pathlib import Path
    import re

    stem = model if model in PHYSICS_ONLY else f"{model}_{loss_type}"
    pat = re.compile(rf"^{re.escape(stem)}_seed(\d+)_best{re.escape(select)}\.pt$")
    d = Path(ckpt_dir) / dataset
    if not d.is_dir():
        return []
    return sorted(int(m.group(1)) for m in
                  (pat.match(p.name) for p in d.iterdir()) if m)


def resolve_checkpoint(ckpt_dir, dataset: str, model: str, loss_type: str,
                       seed: int | None = None, select: str = "sam"):
    """
    Path to one checkpoint, tolerating both naming schemes.

    Resolution order:
      1. explicit ``seed`` + ``select``            -> <stem>_seed<N>_best<sel>.pt
      2. no seed, exactly one seed on disk         -> that one
      3. no seed, several on disk                  -> ValueError naming them,
         because silently picking one would make a multi-seed result depend on
         directory order
      4. nothing seeded                            -> the pre-seed-axis name,
         so checkpoints trained before this change stay loadable

    Returns a Path; existence is NOT guaranteed for case 4 (the caller reports
    a missing checkpoint with better context than this function can).
    """
    from pathlib import Path

    base = Path(ckpt_dir) / dataset
    if seed is not None:
        return base / checkpoint_name(model, loss_type, seed=seed, select=select)

    seeds = find_seeds(ckpt_dir, dataset, model, loss_type, select=select)
    if len(seeds) == 1:
        return base / checkpoint_name(model, loss_type, seed=seeds[0], select=select)
    if len(seeds) > 1:
        raise ValueError(
            f"{model}|{dataset}|{loss_type} has {len(seeds)} seeds on disk "
            f"({seeds}). Pass --seed to pick one, or let the sweep iterate them; "
            f"choosing implicitly would make the result depend on file order."
        )
    return base / checkpoint_name(model, loss_type)
