"""
modules/registry.py
--------------------
Central model registry for the VAE ablation study. train/train.py and
inference/inference.py resolve a CLI ``--model`` name to a class here and never
branch on model type — every model implements the same contract (see
modules/vae_our.py):

    forward(x)                                  # x: (B, H, W, C)
    loss_terms(x, beta, lambda_physics, use_physics) -> dict(loss, mse, kld, sam)
    reconstruct(x) -> (B, H, W, C)

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


def checkpoint_name(model: str, loss_type: str) -> str:
    """
    Filename (no directory) for a trained checkpoint, per the naming convention:

        vae-our            -> "vae-our.pt"           (physics-only)
        <model>            -> "<model>_<loss>.pt"    (e.g. vae-standard_physics.pt)
    """
    if model in PHYSICS_ONLY:
        return f"{model}.pt"
    return f"{model}_{loss_type}.pt"
