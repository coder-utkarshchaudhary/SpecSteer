#!/usr/bin/env bash
# scripts/preprocess.sh
# ---------------------
# Preprocess IIRS / M3 / AVIRIS HSI cubes: normalise (per-cube ~1500 nm
# reference band), Savitzky-Golay smooth, and slice into 64x64xC patches
# saved under data/processed/<DATASET>/. The whole cube is kept (no band
# selection), except M3 which is cropped 85→84 bands so the spectral branch
# round-trips exactly (see utils/config.py DATASETS).
#
# Run from the repo root:
#   bash scripts/preprocess.sh [--dataset {iirs,m3,aviris,all}] [--overwrite]
#
# Extra args are forwarded to utils/dataset/slice.py, e.g.:
#   bash scripts/preprocess.sh --dataset m3 --overwrite
#   bash scripts/preprocess.sh --dataset aviris --limit 1     # smoke test
#
# Override paths (only valid together with a single --dataset):
#   DATA_ROOT="data/original - m3" bash scripts/preprocess.sh --dataset m3
#   OUT_ROOT=data/processed/M3      bash scripts/preprocess.sh --dataset m3
#
# Memory note: peak RAM per cube is roughly 4x the raw cube size (load +
# normalize copy + smooth copy + valid_mask). IIRS ~3.7 GB/cube (~15 GB
# peak); AVIRIS ~4.4 GB/cube (~13-18 GB peak, the largest of the three);
# M3 up to ~3.1 GB/cube (~12 GB peak). Ensure enough free RAM before running
# --dataset all, or process one dataset at a time.

set -euo pipefail

# Repo root is the parent of the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure imports resolve from the repo root (no __init__.py needed)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Configurable overrides (only meaningful together with a single --dataset;
# slice.py enforces this). Leave unset to use utils/config.py DATASETS.
DATA_ROOT="${DATA_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-}"

EXTRA_ARGS=("$@")
if [[ -n "${DATA_ROOT}" ]]; then
    EXTRA_ARGS+=(--data-root "${DATA_ROOT}")
fi
if [[ -n "${OUT_ROOT}" ]]; then
    EXTRA_ARGS+=(--out-root "${OUT_ROOT}")
fi

echo "=============================================="
echo " HSI Preprocessing + Slicing Pipeline"
echo "  IIRS   raw : ${REPO_ROOT}/data/original - IIRS"
echo "  M3     raw : ${REPO_ROOT}/data/original - m3"
echo "  AVIRIS raw : ${REPO_ROOT}/data/original - AVIRIS"
echo "  out root   : ${REPO_ROOT}/data/processed/<DATASET>/ (unless --out-root given)"
echo "=============================================="
echo

cd "${REPO_ROOT}"

python utils/dataset/slice.py "${EXTRA_ARGS[@]}"

echo
echo "Preprocessing complete."
