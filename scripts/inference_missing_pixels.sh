#!/usr/bin/env bash
# scripts/inference_missing_pixels.sh
# -----------------------------------
# Missing-pixel recovery (paper Table 6): 10% of pixels zeroed across all
# bands, checkpoint seed 67, one mask draw. Thin launcher — the work is in
# inference/inference_missing_pixels.py; this does env/CWD setup, the packed-
# shard preflight, a Telegram start ping, and exit-code passthrough (same
# shape as scripts/inference_final.sh).
#
#   bash scripts/inference_missing_pixels.sh --dry-run
#   bash scripts/inference_missing_pixels.sh               # resumable
#   bash scripts/inference_missing_pixels.sh --overwrite
#
# Environment overrides: CKPT_DIR (model), OUT_DIR (results/final/missing_pixels),
# PACKED_ROOT (data/packed).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CKPT_DIR="${CKPT_DIR:-model}"
OUT_DIR="${OUT_DIR:-results/final/missing_pixels}"
PACKED_ROOT="${PACKED_ROOT:-data/packed}"
DATASETS=(IIRS AVIRIS CRIMS)

cd "${REPO_ROOT}"

PY_ARGS=(--ckpt-dir "${CKPT_DIR}" --out-dir "${OUT_DIR}" --packed-root "${PACKED_ROOT}")
DRY_RUN=0
for arg in "$@"; do
    [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
    PY_ARGS+=("${arg}")
done

if [[ ${DRY_RUN} -eq 0 ]]; then
    MISSING_SHARDS=()
    for ds in "${DATASETS[@]}"; do
        for split in valid test; do
            shard="${PACKED_ROOT}/${ds}/${split}.npy"
            [[ -s "${shard}" ]] || MISSING_SHARDS+=("${shard}")
        done
    done
    if [[ ${#MISSING_SHARDS[@]} -gt 0 ]]; then
        echo "ERROR: packed shard(s) missing:"
        for s in "${MISSING_SHARDS[@]}"; do echo "         ${s}"; done
        echo "  Build them with: PYTHONPATH=. python utils/dataset/pack.py --verify"
        exit 4
    fi
    python "${SCRIPT_DIR}/../utils/notify_cli.py" \
        --text "Missing-pixel recovery — launching on $(hostname), ckpt-dir=${CKPT_DIR}, out-dir=${OUT_DIR}" \
        || true
fi

mkdir -p "${OUT_DIR}"
echo "Running: python inference/inference_missing_pixels.py ${PY_ARGS[*]}"
python "${REPO_ROOT}/inference/inference_missing_pixels.py" "${PY_ARGS[@]}"
rc=$?

if [[ ${rc} -ne 0 && ${DRY_RUN} -eq 0 ]]; then
    python "${SCRIPT_DIR}/../utils/notify_cli.py" \
        --text "❌ Missing-pixel recovery EXITED WITH FAILURE (exit_code=${rc}) on $(hostname)." \
        || true
fi

exit "${rc}"
