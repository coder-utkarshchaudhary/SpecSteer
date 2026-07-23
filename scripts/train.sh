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

set -uo pipefail
# NOTE: -e is intentionally omitted. In --all mode, each run is wrapped so one
# failure does not kill the remaining 27 runs.

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

    SKIPPED=()
    FAILED=()

    run_one() {
        # $1=model  $2=dataset  $3=loss  $4=ckpt_name  $5..=forwarded args
        local m="$1" ds="$2" loss="$3" name="$4"
        shift 4
        local ckpt="${CKPT_DIR}/${ds}/${name}.pt"
        if [[ -s "${ckpt}" ]]; then
            echo "[skip] ${m} | ${ds} | ${loss}  (ckpt exists: ${ckpt})"
            SKIPPED+=("${m}|${ds}|${loss}")
            return 0
        fi
        echo ">>> ${m} | ${ds} | ${loss}"
        if python train/train.py --model "${m}" --dataset "${ds}" --loss "${loss}" \
                --ckpt-dir "${CKPT_DIR}" "$@"; then
            return 0
        else
            local rc=$?
            echo "!!! ${m} | ${ds} | ${loss}  FAILED (exit=${rc}) — continuing"
            FAILED+=("${m}|${ds}|${loss}|rc=${rc}")
            return 0
        fi
    }

    for ds in "${DATASETS[@]}"; do
        run_one vae-our "${ds}" physics vae-our "$@"
        for m in "${STANDARD_MODELS[@]}"; do
            for loss in standard physics; do
                run_one "${m}" "${ds}" "${loss}" "${m}_${loss}" "$@"
            done
        done
    done

    echo ""
    echo "=============================================="
    echo " Ablation grid complete."
    echo "  skipped (ckpt exists) : ${#SKIPPED[@]}"
    echo "  failed                : ${#FAILED[@]}"
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "  failed runs:"
        for f in "${FAILED[@]}"; do echo "    - ${f}"; done
    fi
    echo "=============================================="
    exit $(( ${#FAILED[@]} > 0 ? 1 : 0 ))
fi

# --------------------------------------------------------------------------
# Single run: forward all args straight through to train/train.py.
# --------------------------------------------------------------------------
python train/train.py --ckpt-dir "${CKPT_DIR}" "$@"
