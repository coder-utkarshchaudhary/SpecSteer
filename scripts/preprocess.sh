#!/usr/bin/env bash
# scripts/preprocess.sh
# ---------------------
# Preprocess all IIRS HSI cubes: select bands, normalise, smooth, and slice
# into (64×64×108) patches saved under data/processed/.
#
# Run from the repo root:
#   bash scripts/preprocess.sh [--overwrite]
#
# All arguments are forwarded to utils/dataset/slice.py.
# Override paths with:
#   DATA_ROOT=path/to/data/original bash scripts/preprocess.sh
#   OUT_ROOT=path/to/data/processed  bash scripts/preprocess.sh

set -euo pipefail

# Repo root is the parent of the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure imports resolve from the repo root (no __init__.py needed)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Configurable defaults (override via env variables)
DATA_ROOT="${DATA_ROOT:-data/original}"
OUT_ROOT="${OUT_ROOT:-data/processed}"

echo "=============================================="
echo " HSI Preprocessing + Slicing Pipeline"
echo "  data root : ${REPO_ROOT}/${DATA_ROOT}"
echo "  out  root : ${REPO_ROOT}/${OUT_ROOT}"
echo "=============================================="
echo

cd "${REPO_ROOT}"

python utils/dataset/slice.py \
    --data-root "${DATA_ROOT}" \
    --out-root  "${OUT_ROOT}" \
    "$@"

echo
echo "Preprocessing complete."
