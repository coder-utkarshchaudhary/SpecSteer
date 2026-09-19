#!/usr/bin/env bash
# scripts/run_variant_study.sh
# ----------------------------
# Master wrapper to execute the complete variant and hyperparameter study:
#   Step 1: Run inference/downstream on baseline models.
#   Step 2: Train architecture updates (PRISM variants) with default hyperparams
#           (epochs=60, beta=1e-3) and evaluate.
#   Step 3: Train updates with updated beta=1e-2 (epochs=60, beta=1e-2) and evaluate.
#
# Usage:
#   bash scripts/run_variant_study.sh --all
#   bash scripts/run_variant_study.sh --step 1
#   bash scripts/run_variant_study.sh --step 2
#   bash scripts/run_variant_study.sh --step 3
#
# Overrides:
#   --datasets IIRS,M3   Subset datasets (default: IIRS,M3,AVIRIS,CRIMS)
#   --seeds 42           Subset seeds (default: 42)
#   --select sam         Checkpoint selection metric (default: sam)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATASETS_CSV="IIRS,M3,AVIRIS,CRIMS"
SEEDS_CSV="42"
SELECT="sam"
RUN_STEP1=0
RUN_STEP2=0
RUN_STEP3=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            if [[ "$2" == "1" ]]; then RUN_STEP1=1; fi
            if [[ "$2" == "2" ]]; then RUN_STEP2=1; fi
            if [[ "$2" == "3" ]]; then RUN_STEP3=1; fi
            shift 2
            ;;
        --all)
            RUN_STEP1=1; RUN_STEP2=1; RUN_STEP3=1; shift ;;
        --datasets|--dataset)
            DATASETS_CSV="$2"; shift 2 ;;
        --seeds|--seed)
            SEEDS_CSV="$2"; shift 2 ;;
        --select)
            SELECT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--all] [--step 1|2|3] [--datasets ds1,ds2] [--seeds s1,s2] [--select sam|mse]"
            exit 1
            ;;
    esac
done

# If no steps were explicitly requested, show usage and run all
if (( RUN_STEP1 == 0 && RUN_STEP2 == 0 && RUN_STEP3 == 0 )); then
    echo "No specific step requested. Defaulting to running ALL steps."
    RUN_STEP1=1; RUN_STEP2=1; RUN_STEP3=1
fi

echo "=========================================================="
echo " Master PRISM Variant & Hyperparameter Study"
echo "  Datasets   : ${DATASETS_CSV}"
echo "  Seeds      : ${SEEDS_CSV}"
echo "  Select     : ${SELECT}"
echo "  Steps      : 1=${RUN_STEP1} 2=${RUN_STEP2} 3=${RUN_STEP3}"
echo "=========================================================="

# ------------------------------------------------------------------------------
# STEP 1: Evaluation of baseline models
# ------------------------------------------------------------------------------
if (( RUN_STEP1 )); then
    echo ""
    echo "=========================================================="
    echo " STEP 1: Running evaluation on Baseline Models"
    echo "=========================================================="
    export CKPT_DIR="model"
    export OUT_DIR="results"
    
    # Run evaluation sweep (reconstruction + downstream + probes)
    bash scripts/inference.sh \
        --datasets "${DATASETS_CSV}" \
        --seeds "${SEEDS_CSV}" \
        --select "${SELECT}"
fi

# ------------------------------------------------------------------------------
# STEP 2: Train architecture updates (beta=1e-3, epochs=60) & evaluate
# ------------------------------------------------------------------------------
if (( RUN_STEP2 )); then
    echo ""
    echo "=========================================================="
    echo " STEP 2: Training PRISM variants (epochs=60, beta=1e-3)"
    echo "=========================================================="
    export CKPT_DIR="model_variants_beta_1e-3"
    export OUT_DIR="results_variants_beta_1e-3"
    
    mkdir -p "${CKPT_DIR}" "${OUT_DIR}"
    
    # Train NL, SpecViT, and NL-SpecViT models
    # Same standard beta (1e-3) is loaded automatically from dataset configs
    echo ">>> Launching training for variants (beta=1e-3)..."
    bash scripts/train_variants.sh \
        --datasets "${DATASETS_CSV}" \
        --seeds "${SEEDS_CSV}" \
        --epochs 60 \
        --overwrite
        
    echo ">>> Launching evaluation for variants (beta=1e-3)..."
    bash scripts/inference_variants.sh \
        --datasets "${DATASETS_CSV}" \
        --seeds "${SEEDS_CSV}" \
        --select "${SELECT}"
fi

# ------------------------------------------------------------------------------
# STEP 3: Train architecture updates with beta=1e-2 (epochs=60) & evaluate
# ------------------------------------------------------------------------------
if (( RUN_STEP3 )); then
    echo ""
    echo "=========================================================="
    echo " STEP 3: Training PRISM variants (epochs=60, beta=1e-2)"
    echo "=========================================================="
    export CKPT_DIR="model_variants_beta_1e-2"
    export OUT_DIR="results_variants_beta_1e-2"
    
    mkdir -p "${CKPT_DIR}" "${OUT_DIR}"
    
    # Train NL, SpecViT, and NL-SpecViT models with beta override of 0.01 (1e-2)
    echo ">>> Launching training for variants with beta=1e-2..."
    bash scripts/train_variants.sh \
        --datasets "${DATASETS_CSV}" \
        --seeds "${SEEDS_CSV}" \
        --epochs 60 \
        --beta 0.01 \
        --overwrite
        
    echo ">>> Launching evaluation for variants (beta=1e-2)..."
    bash scripts/inference_variants.sh \
        --datasets "${DATASETS_CSV}" \
        --seeds "${SEEDS_CSV}" \
        --select "${SELECT}"
fi

echo ""
echo "=========================================================="
echo " Master study runs completed successfully!"
echo " Results folders:"
echo "   - Baseline evaluation     : results/"
echo "   - Variant study (beta=1e-3): results_variants_beta_1e-3/"
echo "   - Variant study (beta=1e-2): results_variants_beta_1e-2/"
echo "=========================================================="
