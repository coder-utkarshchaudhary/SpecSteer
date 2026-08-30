"""
scripts/_smoke_fixtures.py
--------------------------
Build synthetic packed shards + checkpoints so scripts/inference_smoke.sh can
drive the real scripts/inference.sh end to end without touching data/packed/ or
model/.

The checkpoints are built through the EXACT training path —
apply_dataset -> apply_hyperparams(load_hyperparams(ds)) -> build_model ->
dummy forward to materialise LazyLinear -> torch.save under
registry.checkpoint_name(...). That is deliberate: a checkpoint built without
apply_hyperparams would load fine into a still-broken inference.py (both sides
wrong identically), and the smoke test would pass for the wrong reason.

Not a general utility — only invoked by scripts/inference_smoke.sh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from modules.registry import (
    MODEL_NAMES,
    PHYSICS_ONLY,
    SELECT_CRITERIA,
    build_model,
    checkpoint_name,
)
from utils.config import DATASETS, apply_dataset, settings
from utils.hyperparams import apply_hyperparams, load_hyperparams

N_ROWS = 16
H = W = 64
SCENES = ("smokeSceneA", "smokeSceneB")   # >=2 so _scene_labels is non-empty


def _write_shard(packed_ds: Path, split: str, C: int, crop_bands, rng) -> None:
    packed_ds.mkdir(parents=True, exist_ok=True)
    arr = rng.random((N_ROWS, H, W, C), dtype=np.float64).astype(np.float16)
    np.save(packed_ds / f"{split}.npy", arr)
    sidecar = {
        "dataset": packed_ds.name,
        "split": split,
        "n": N_ROWS,
        "n_available": N_ROWS,
        "H": H, "W": W, "C": int(C), "raw_C": int(C),
        "crop_bands": crop_bands,
        "dtype": "float16",
        "normalisation": "synthetic smoke fixture",
        "cap": None, "cap_seed": None,
        "processed_root": "SMOKE",
        "patch_max": [1.0] * N_ROWS,
        # scene/<split>/patch_NNNNN.npy — _scene_labels takes parts[0].
        "source_files": [
            f"{SCENES[i % len(SCENES)]}/{split}/patch_{i:05d}.npy"
            for i in range(N_ROWS)
        ],
    }
    (packed_ds / f"{split}.json").write_text(json.dumps(sidecar))


def _cells():
    """(model, loss) — mirrors scripts/grid_manifest.sh GRID_CONFIGS."""
    for m in MODEL_NAMES:
        if m in PHYSICS_ONLY:
            yield m, "physics"
        else:
            yield m, "standard"
            yield m, "physics"


def _save_ckpt(model, path: Path, m: str, ds: str, loss: str, seed: int,
               crit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: (v.half() if v.is_floating_point() else v)
          for k, v in model.state_dict().items()}
    torch.save(
        {
            "epoch": 50,
            "model_state_dict": sd,
            "loss": 0.1234,
            "select": crit,
            "val_sam": 0.1234, "val_mse_final": 0.00123,
            "val_psnr": 30.0, "val_ssim": 0.9,
            "model": m, "dataset": ds, "loss_type": loss, "seed": seed,
            "batch_size": 16, "platform": "smoke",
            "lambda_physics": 0.3, "beta": 1e-3,
        },
        path,
    )


def build_for_dataset(ds: str, seeds: list[int], root: Path) -> None:
    apply_dataset(ds)
    apply_hyperparams(settings, load_hyperparams(ds))
    C = settings.input_channels
    crop = DATASETS[ds.upper()].get("crop_bands")

    rng = np.random.default_rng(20260830)
    for split in ("train", "valid", "test"):
        _write_shard(root / "packed" / ds, split, C, crop, rng)

    dummy = torch.randn(2, H, W, C)
    ckpt_root = root / "model" / ds
    for m, loss in _cells():
        model = build_model(m)
        with torch.no_grad():
            model(dummy)                       # materialise LazyLinear
        model.eval()
        # standard-loss baselines only at the first seed (grid manifest);
        # physics cells at every seed.
        cell_seeds = seeds if loss == "physics" else seeds[:1]
        for seed in cell_seeds:
            for crit in SELECT_CRITERIA:
                fn = checkpoint_name(m, loss, seed=seed, select=crit)
                _save_ckpt(model, ckpt_root / fn, m, ds, loss, seed, crit)
    print(f"  {ds}: shards + {len(list(_cells()))} configs "
          f"x seeds{seeds} x {len(SELECT_CRITERIA)} select written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--datasets", required=True,
                    help="comma-separated (IIRS,AVIRIS,CRIMS)")
    ap.add_argument("--seeds", required=True,
                    help="comma-separated training seeds (e.g. 42 or 42,7,1234)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.root.mkdir(parents=True, exist_ok=True)
    print(f"smoke fixtures -> {args.root}  (datasets={datasets}, seeds={seeds})")
    for ds in datasets:
        build_for_dataset(ds, seeds, args.root)


if __name__ == "__main__":
    main()
