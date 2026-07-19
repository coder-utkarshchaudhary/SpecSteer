"""
utils/hyperparams.py
--------------------
Per-dataset hyperparameter YAML loader.

Optimization hyperparams (lr, epochs, batch_size, beta, lambda_physics,
weight_decay, early_stopping_patience) vary per dataset. Baseline capacity
knobs (`vae_standard_base_ch`, `vae_3d_base_ch`, `vae_1d_hidden_dims`) also
vary per dataset so each baseline matches vae-our's param count at that
dataset. Everything else on `Settings` (architecture: latent_dim,
reduced_dims, n_2D_conv_blocks, ...) stays constant across datasets.

Usage:
    from utils.hyperparams import load_hyperparams, apply_hyperparams
    from utils.config import apply_dataset

    settings = apply_dataset("IIRS")
    hp = load_hyperparams("IIRS")
    apply_hyperparams(settings, hp)
    # settings.batch_size, settings.vae_standard_base_ch, ... are now overridden
    # hp still holds the optimization keys (epochs, lr, ...) for the caller

Config files live at utils/hyperparam_configs/hyperparam-config-<DATASET>.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Fields mutated in place on the Settings dataclass. Everything else in the YAML
# is treated as an optimization hyperparam and returned in the dict.
_SETTINGS_FIELDS: set[str] = {
    "batch_size",
    "num_workers",
    "vae_standard_base_ch",
    "vae_standard_n_down",
    "vae_standard_latent_ch",
    "vae_3d_base_ch",
    "vae_3d_n_down",
    "vae_3d_latent_ch",
    "vae_1d_hidden_dims",
    "vae_1d_latent_dim",
}

# Optimization keys — returned in the dict for the training loop / notebook to consume.
_OPTIMIZATION_FIELDS: set[str] = {
    "epochs",
    "lr",
    "beta",
    "lambda_physics",
    "seed",
    "weight_decay",
    "early_stopping_patience",
}

_ALL_ALLOWED = _SETTINGS_FIELDS | _OPTIMIZATION_FIELDS


def _config_path(dataset: str, config_dir: str | Path) -> Path:
    key = dataset.upper()
    return Path(config_dir) / f"hyperparam-config-{key}.yaml"


def load_hyperparams(
    dataset: str,
    config_dir: str | Path = "utils/hyperparam_configs",
) -> dict[str, Any]:
    """
    Load hyperparam-config-<DATASET>.yaml. Returns an empty dict if the file
    doesn't exist (caller falls back to CLI/dataclass defaults). Raises
    ValueError on unknown keys so typos fail loudly.
    """
    path = _config_path(dataset, config_dir)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}")
    unknown = set(raw) - _ALL_ALLOWED
    if unknown:
        raise ValueError(
            f"{path}: unknown hyperparam keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_ALL_ALLOWED)}."
        )
    return raw


def apply_hyperparams(settings, hp: dict[str, Any]) -> None:
    """
    Mutate `settings` in place for whitelisted Settings fields present in `hp`.
    Leaves optimization fields alone (caller reads those from `hp` directly).

    Does NOT call `settings.__post_init__()` — none of the whitelisted fields
    feed into any derived (`field(init=False)`) computation.
    """
    for key in _SETTINGS_FIELDS & set(hp):
        value = hp[key]
        if key == "vae_1d_hidden_dims" and isinstance(value, list):
            value = tuple(value)
        setattr(settings, key, value)
