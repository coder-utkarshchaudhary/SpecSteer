#!/usr/bin/env bash
# scripts/train.sh
# ----------------
# Launch training for the Dual-Stream Physics-Informed VAE.
#
# Run from the repo root:
#   bash scripts/train.sh [extra args forwarded to train/train.py]
#
# Examples:
#   bash scripts/train.sh --epochs 200 --beta 0.005
#   bash scripts/train.sh --no-wandb
#   DATA_ROOT=data/processed CKPT_DIR=runs/exp1 bash scripts/train.sh
#
# One-time W&B setup (run once before first training):
#   wandb login
#
# Environment overrides:
#   DATA_ROOT    — processed data directory  (default: data/processed)
#   CKPT_DIR     — checkpoint directory      (default: checkpoints)
#   WANDB_PROJECT — W&B project name        (default: hsi-pi-vae)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-data/processed}"
CKPT_DIR="${CKPT_DIR:-model/checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-hsi-pi-vae}"

echo "=============================================="
echo " Dual-Stream PI-VAE Training"
echo "  data root   : ${REPO_ROOT}/${DATA_ROOT}"
echo "  ckpt dir    : ${REPO_ROOT}/${CKPT_DIR}"
echo "  wandb proj  : ${WANDB_PROJECT}"
echo "=============================================="
echo

cd "${REPO_ROOT}"

python train/train.py \
    --data-root      "${DATA_ROOT}" \
    --ckpt-dir       "${CKPT_DIR}" \
    --wandb-project  "${WANDB_PROJECT}" \
    "$@"
