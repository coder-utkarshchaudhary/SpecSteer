#!/usr/bin/env bash
# scripts/run_crims_only.sh
# -------------------------
# A helper script to:
#  1. Transfer existing (pre-computed) IIRS & AVIRIS results to Telegram as files.
#  2. Run inference/evaluation specifically for the CRIMS dataset.
#  3. Transfer the final/updated results containing CRIMS to Telegram.
#
# This script is designed to run on the remote machine (GPU compute node)
# where your models and results are stored.
#
# Usage:
#   # To rerun the paper-locking final inference (inference_final.py) for CRIMS:
#   bash scripts/run_crims_only.sh --final
#
#   # To rerun the standard sweep inference (inference.sh) for CRIMS:
#   bash scripts/run_crims_only.sh --standard
#
#   # To run a clean training + eval grid (run_clean_grid.sh) for CRIMS:
#   bash scripts/run_crims_only.sh --grid

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# Try to find a valid python interpreter (default to .venv/bin/python if present)
PY="python"
if [[ -f ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
elif [[ -f ".venv/Scripts/python" ]]; then
    PY=".venv/Scripts/python"
fi

# Parse mode argument
MODE="final"  # default to final
OVERWRITE_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --final)
            MODE="final"
            shift
            ;;
        --standard)
            MODE="standard"
            shift
            ;;
        --grid)
            MODE="grid"
            shift
            ;;
        --overwrite)
            OVERWRITE_ARG="--overwrite"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/run_crims_only.sh [--final | --standard | --grid] [--overwrite]"
            exit 1
            ;;
    esac
done

notify() {
    "${PY}" -u utils/notify_cli.py --text "$1" >/dev/null 2>&1 || true
}

send_file() {
    local filepath="$1"
    local caption="$2"
    if [[ -f "${filepath}" && -s "${filepath}" ]]; then
        echo "Sending ${filepath} via Telegram..."
        "${PY}" -u utils/notify_cli.py --file "${filepath}" --caption "${caption}" >/dev/null 2>&1 || true
    else
        echo "Warning: ${filepath} not found or empty."
    fi
}

echo "=== CRIMS-ONLY INFERENCE RUNNER ==="
echo "Mode selected: ${MODE}"
echo "Repository root: ${REPO_ROOT}"
echo "Using python: ${PY}"
echo "=================================="

# ---------------------------------------------------------------------------
# STEP 1: Transfer existing AVIRIS/IIRS results to Telegram
# ---------------------------------------------------------------------------
echo "Step 1: Identifying and transferring existing results (AVIRIS / IIRS)..."
notify "🚀 CRIMS-only pipeline started on $(hostname). Sending existing AVIRIS/IIRS results first..."

if [[ "${MODE}" == "final" ]]; then
    # Final inference results go to results/final
    FINAL_DIR="${REPO_ROOT}/results/final"
    if [[ -d "${FINAL_DIR}" ]]; then
        echo "Found final results directory: ${FINAL_DIR}"
        send_file "${FINAL_DIR}/reconstruction-quality.csv" "Pre-CRIMS: Final Reconstruction Quality (IIRS/AVIRIS)"
        send_file "${FINAL_DIR}/model-validity-probes.csv" "Pre-CRIMS: Final Validity Probes (IIRS/AVIRIS)"
        send_file "${FINAL_DIR}/noise-recovery.csv" "Pre-CRIMS: Final Noise Recovery (IIRS/AVIRIS)"
        send_file "${FINAL_DIR}/chemical-interpolation.csv" "Pre-CRIMS: Final Chemical Interpolation (IIRS/AVIRIS)"
        send_file "${FINAL_DIR}/missing-pixel-recovery.csv" "Pre-CRIMS: Final Missing Pixel Recovery (IIRS/AVIRIS)"
    else
        echo "No final results directory found at ${FINAL_DIR}."
        notify "⚠️ Pre-run final results directory not found at results/final."
    fi
else
    # Standard or grid results go to results/
    STD_DIR="${REPO_ROOT}/results"
    if [[ -d "${STD_DIR}" ]]; then
        echo "Found standard results directory: ${STD_DIR}"
        send_file "${STD_DIR}/ablation_table.csv" "Pre-CRIMS: Ablation Table (IIRS/AVIRIS)"
        send_file "${STD_DIR}/downstream_table.csv" "Pre-CRIMS: Downstream Table (IIRS/AVIRIS)"
        send_file "${STD_DIR}/probes.csv" "Pre-CRIMS: Falsification Probes (IIRS/AVIRIS)"
        send_file "${STD_DIR}/stats.csv" "Pre-CRIMS: Pairwise Statistics (IIRS/AVIRIS)"
    else
        echo "No standard results directory found at ${STD_DIR}."
        notify "⚠️ Pre-run standard results directory not found at results/."
    fi
fi

# ---------------------------------------------------------------------------
# STEP 2: Run inference on CRIMS (CRISM) dataset
# ---------------------------------------------------------------------------
echo "Step 2: Starting inference for CRIMS dataset..."
notify "🔄 Starting inference for CRIMS dataset (Mode: ${MODE}). This may take a while..."

if [[ "${MODE}" == "final" ]]; then
    echo "Running: ${PY} inference/inference_final.py --datasets CRIMS --overwrite"
    # We use --overwrite to ensure it recomputes CRIMS cells completely
    "${PY}" -u inference/inference_final.py --datasets CRIMS --overwrite
    rc=$?
    
elif [[ "${MODE}" == "standard" ]]; then
    echo "Running: bash scripts/inference.sh --datasets CRIMS"
    bash scripts/inference.sh --datasets CRIMS
    rc=$?
    
elif [[ "${MODE}" == "grid" ]]; then
    # Custom run of clean grid but limited only to CRIMS
    echo "Running clean grid steps specifically for CRIMS..."
    # We can invoke run_clean_grid.sh by temporarily setting the DATASETS env/vars or patching.
    # To be extremely clean and robust, we can run it with a customized temporary copy or inline execution:
    # Creating a temp copy of run_clean_grid.sh with DATASETS=(CRIMS)
    TEMP_GRID_SH="$(mktemp)"
    cp scripts/run_clean_grid.sh "${TEMP_GRID_SH}"
    
    # Replace DATASETS=(IIRS AVIRIS M3 CRIMS) with DATASETS=(CRIMS)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/DATASETS=(IIRS AVIRIS M3 CRIMS)/DATASETS=(CRIMS)/g' "${TEMP_GRID_SH}"
    else
        sed -i 's/DATASETS=(IIRS AVIRIS M3 CRIMS)/DATASETS=(CRIMS)/g' "${TEMP_GRID_SH}"
    fi
    
    echo "Running modified clean grid..."
    bash "${TEMP_GRID_SH}" ${OVERWRITE_ARG}
    rc=$?
    rm -f "${TEMP_GRID_SH}"
fi

if [[ ${rc} -ne 0 ]]; then
    echo "ERROR: CRIMS inference failed with exit code ${rc}!"
    notify "❌ CRIMS inference failed with exit code ${rc} on $(hostname)."
    exit "${rc}"
fi

echo "CRIMS inference completed successfully!"
notify "✅ CRIMS inference completed successfully on $(hostname)."

# ---------------------------------------------------------------------------
# STEP 3: Transfer updated results containing CRIMS to Telegram
# ---------------------------------------------------------------------------
echo "Step 3: Transferring updated results..."
notify "📤 Sending updated results files (including CRIMS) via Telegram..."

if [[ "${MODE}" == "final" ]]; then
    FINAL_DIR="${REPO_ROOT}/results/final"
    send_file "${FINAL_DIR}/reconstruction-quality.csv" "Updated: Final Reconstruction Quality (with CRIMS)"
    send_file "${FINAL_DIR}/model-validity-probes.csv" "Updated: Final Validity Probes (with CRIMS)"
    send_file "${FINAL_DIR}/noise-recovery.csv" "Updated: Final Noise Recovery (with CRIMS)"
    send_file "${FINAL_DIR}/chemical-interpolation.csv" "Updated: Final Chemical Interpolation (with CRIMS)"
    send_file "${FINAL_DIR}/missing-pixel-recovery.csv" "Updated: Final Missing Pixel Recovery (with CRIMS)"
else
    STD_DIR="${REPO_ROOT}/results"
    # For standard and grid modes, regenerate aggregate tables if needed
    echo "Regenerating aggregates..."
    "${PY}" -u inference/verdict.py
    "${PY}" -u inference/aggregate.py
    
    send_file "${STD_DIR}/ablation_table.csv" "Updated: Ablation Table (with CRIMS)"
    send_file "${STD_DIR}/downstream_table.csv" "Updated: Downstream Table (with CRIMS)"
    send_file "${STD_DIR}/probes.csv" "Updated: Falsification Probes (with CRIMS)"
    send_file "${STD_DIR}/stats.csv" "Updated: Pairwise Statistics (with CRIMS)"
fi

notify "🏁 CRIMS-only pipeline execution completed on $(hostname)."
echo "Done!"
