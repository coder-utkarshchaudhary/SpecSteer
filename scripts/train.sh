#!/usr/bin/env bash
# scripts/train.sh
# ----------------
# Launch training for the HSI VAE ablation study (model-agnostic train/train.py).
#
# Run from the repo root:
#   # Single run:
#   bash scripts/train.sh --model vae-our --dataset IIRS --loss physics --epochs 100
#
#   # Full 28-run ablation grid (all models x all datasets x loss regimes):
#   bash scripts/train.sh --all --epochs 100
#
# The grid is:
#   - vae-our             : physics only        -> 1 run/dataset  (4)
#   - vae-standard        : standard + physics  -> 2 runs/dataset (8)
#   - vae-3d-spatio-spectral : standard + physics -> 2 runs/dataset (8)
#   - vae-1d-pixelwise    : standard + physics  -> 2 runs/dataset (8)
#   Total: 4 + 8 + 8 + 8 = 28 trainings across IIRS / M3 / AVIRIS / CRIMS.
#
# Checkpoints are written to ${CKPT_DIR}/<DATASET>/<name>.pt
#   vae-our      -> vae-our.pt
#   others       -> <model>_<loss>.pt
#
# Environment overrides:
#   CKPT_DIR   — checkpoint root directory   (default: model)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-model}"

cd "${REPO_ROOT}"

# --------------------------------------------------------------------------
# --all : run the full ablation grid. Any extra args (e.g. --epochs 100) are
# forwarded to every run.
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--all" ]]; then
    shift
    DATASETS=("IIRS" "M3" "AVIRIS" "CRIMS")
    STANDARD_MODELS=("vae-standard" "vae-3d-spatio-spectral" "vae-1d-pixelwise")

    echo "=============================================="
    echo " HSI VAE Ablation — full grid (28 runs)"
    echo "  ckpt dir : ${REPO_ROOT}/${CKPT_DIR}"
    echo "=============================================="

    for ds in "${DATASETS[@]}"; do
        # vae-our: physics only
        echo ">>> vae-our | ${ds} | physics"
        python train/train.py --model vae-our --dataset "${ds}" --loss physics \
            --ckpt-dir "${CKPT_DIR}" "$@"

        # other models: standard + physics
        for m in "${STANDARD_MODELS[@]}"; do
            for loss in standard physics; do
                echo ">>> ${m} | ${ds} | ${loss}"
                python train/train.py --model "${m}" --dataset "${ds}" --loss "${loss}" \
                    --ckpt-dir "${CKPT_DIR}" "$@"
            done
        done
    done

    echo "Ablation grid complete."
    exit 0
fi

# --------------------------------------------------------------------------
# Single run: forward all args straight through to train/train.py.
# --------------------------------------------------------------------------
python train/train.py --ckpt-dir "${CKPT_DIR}" "$@"
