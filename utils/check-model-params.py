"""
utils/check-model-params.py
---------------------------
Fair-ablation parameter audit.

For every (model, dataset) pair, applies the dataset settings + per-dataset
hyperparam YAML overrides, builds the model, runs a dummy forward to
materialize LazyLinear layers, and prints total parameter count. Emits a
4x3 table where every row should read within ~3% of vae-our per column.

Usage (from repo root, with venv active):
    python utils/check-model-params.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.registry import MODELS, MODEL_NAMES, build_model  # noqa: E402
from utils.config import DATASETS, apply_dataset, settings  # noqa: E402
from utils.hyperparams import apply_hyperparams, load_hyperparams  # noqa: E402


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_and_count(model_name: str, dataset: str) -> int:
    apply_dataset(dataset)
    hp = load_hyperparams(dataset)
    apply_hyperparams(settings, hp)

    model = build_model(model_name)
    dummy = torch.randn(2, settings.input_height, settings.input_width, settings.input_channels)
    with torch.no_grad():
        model(dummy)
    return count_params(model)


def main() -> None:
    dataset_names = sorted(DATASETS)
    print(f"{'model':<24} " + "  ".join(f"{d:>14}" for d in dataset_names) + "   vs vae-our")
    print("-" * (24 + 16 * len(dataset_names) + 20))

    reference: dict[str, int] = {}
    for ds in dataset_names:
        if os.path.exists(ds["processed_root"]):
            reference[ds] = build_and_count("vae-our", ds)

    for name in MODEL_NAMES:
        counts = {ds: build_and_count(name, ds) for ds in dataset_names}
        cells = "  ".join(f"{counts[ds]:>14,}" for ds in dataset_names)
        ratios = "  ".join(
            f"{ds}:{100 * (counts[ds] - reference[ds]) / reference[ds]:+.1f}%"
            for ds in dataset_names
        )
        print(f"{name:<24} {cells}   {ratios}")


if __name__ == "__main__":
    main()
