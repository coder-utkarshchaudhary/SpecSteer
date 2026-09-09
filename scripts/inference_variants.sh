#!/usr/bin/env bash
# scripts/inference_variants.sh
# ------------------------------
# Run inference and downstream evaluation on the custom PRISM (vae-our) variants:
#   - vae-our-nl
#   - vae-our-specvit
#   - vae-our-nl-specvit
#
# Usage:
#   bash scripts/inference_variants.sh --datasets IIRS --models vae-our-nl --select sam --seeds 42
#
# Defaults:
#   --models   : vae-our-nl,vae-our-specvit,vae-our-nl-specvit
#   --datasets : IIRS,M3,AVIRIS,CRIMS
#   --seeds    : 42
#   --select   : sam (evaluates the best-SAM checkpoints)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-model}"
OUT_DIR="${OUT_DIR:-results}"
PACKED_ROOT="${PACKED_ROOT:-data/packed}"
INFER_JSON_DIR="${OUT_DIR}/inference"
DOWNSTREAM_DIR="${OUT_DIR}/downstream"

cd "${REPO_ROOT}"

# Default parameters
DATASETS_SUBSET="IIRS,M3,AVIRIS,CRIMS"
MODELS_SUBSET="vae-our-nl,vae-our-specvit,vae-our-nl-specvit"
SEEDS_SUBSET="42"
SELECT="sam"
DO_DOWNSTREAM=1

# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets|--dataset) DATASETS_SUBSET="$2"; shift 2 ;;
        --models|--model)     MODELS_SUBSET="$2"; shift 2 ;;
        --seeds|--seed)       SEEDS_SUBSET="$2"; shift 2 ;;
        --select)             SELECT="$2"; shift 2 ;;
        --skip-downstream)    DO_DOWNSTREAM=0; shift ;;
        *)                    shift ;;
    esac
done

# Convert comma-separated strings to arrays
IFS=',' read -r -a DATASETS <<< "${DATASETS_SUBSET}"
IFS=',' read -r -a MODELS <<< "${MODELS_SUBSET}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_SUBSET}"

echo "=========================================================="
echo " HSI VAE Variant Inference/Evaluation Launcher"
echo "  Models     : ${MODELS_SUBSET}"
echo "  Datasets   : ${DATASETS_SUBSET}"
echo "  Seeds      : ${SEEDS_SUBSET}"
echo "  Select     : ${SELECT}"
echo "=========================================================="

python utils/notify_cli.py --text "PRISM Variants inference/evaluation sweep started
host: $(hostname)
models: ${MODELS_SUBSET}
datasets: ${DATASETS_SUBSET}
seeds: ${SEEDS_SUBSET}
select: best-${SELECT}" >/dev/null 2>&1 || true

mkdir -p "${INFER_JSON_DIR}"

# 1. Run reconstruction/inference evaluation
for ds in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo ">>> Evaluating reconstruction: ${m} | ${ds} | seed ${seed}"
            python inference/inference_variants.py \
                --model "${m}" \
                --dataset "${ds}" \
                --seed "${seed}" \
                --select "${SELECT}" \
                --ckpt-dir "${CKPT_DIR}" \
                --out-dir "${OUT_DIR}" \
                --packed-root "${PACKED_ROOT}"
            echo "----------------------------------------------------------"
        done
    done
done

# 2. Run downstream manifold experiments (latent noise injection + interpolation)
if (( DO_DOWNSTREAM == 1 )); then
    for ds in "${DATASETS[@]}"; do
        for m in "${MODELS[@]}"; do
            echo ">>> Evaluating downstream manifold: ${m} | ${ds}"
            python inference/downstream_variants.py \
                --model "${m}" \
                --dataset "${ds}" \
                --ckpt-dir "${CKPT_DIR}" \
                --save-plots
            echo "----------------------------------------------------------"
        done
    done
fi

echo "=========================================================="
echo " Variants inference/evaluation complete."
echo "=========================================================="

# Run dynamic metrics aggregation and send unified summary to Telegram
echo ">>> Aggregating variant results and notifying Telegram"
python inference/aggregate.py \
    --inference-dir "${INFER_JSON_DIR}" \
    --downstream-dir "${DOWNSTREAM_DIR}" \
    --out-dir "${OUT_DIR}" \
    --telegram

python utils/notify_cli.py --text "PRISM Variants inference/evaluation sweep completed successfully!
host: $(hostname)
models: ${MODELS_SUBSET}
datasets: ${DATASETS_SUBSET}" >/dev/null 2>&1 || true
