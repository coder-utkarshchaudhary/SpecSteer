#!/usr/bin/env bash
# scripts/train_variants.sh
# -------------------------
# Launch training for the custom PRISM (vae-our) variants:
#   - vae-our-nl (Spatially-Adaptive Non-Linear 2D Spectral Projection)
#   - vae-our-specvit (Spectral Vision Transformer)
#   - vae-our-nl-specvit (Hybrid: NL 2D projection + Spectral Transformer)
#
# Usage:
#   bash scripts/train_variants.sh --datasets IIRS --models vae-our-nl --seeds 42
#
# If arguments are omitted, defaults are:
#   --models   : vae-our-nl,vae-our-specvit,vae-our-nl-specvit
#   --datasets : IIRS,M3,AVIRIS,CRIMS
#   --seeds    : 42
#   --loss     : physics (standard training for PI-VAE models)
#   --epochs   : (read from dataset YAML config)
#   --overwrite : 0

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CKPT_DIR="${CKPT_DIR:-model}"

cd "${REPO_ROOT}"

# Default parameters
DATASETS_SUBSET="IIRS,M3,AVIRIS,CRIMS"
MODELS_SUBSET="vae-our-nl,vae-our-specvit,vae-our-nl-specvit"
SEEDS_SUBSET="42"
LOSS="physics"
EPOCHS_OVERRIDE=""
BATCH_SIZE_OVERRIDE=""
OVERWRITE=0
EXTRA_ARGS=()

# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets|--dataset) DATASETS_SUBSET="$2"; shift 2 ;;
        --models|--model)     MODELS_SUBSET="$2"; shift 2 ;;
        --seeds|--seed)       SEEDS_SUBSET="$2"; shift 2 ;;
        --loss)               LOSS="$2"; shift 2 ;;
        --epochs)             EPOCHS_OVERRIDE="$2"; shift 2 ;;
        --batch-size)         BATCH_SIZE_OVERRIDE="$2"; shift 2 ;;
        --overwrite)          OVERWRITE=1; shift ;;
        *)                    EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Convert comma-separated strings to arrays
IFS=',' read -r -a DATASETS <<< "${DATASETS_SUBSET}"
IFS=',' read -r -a MODELS <<< "${MODELS_SUBSET}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_SUBSET}"

echo "=========================================================="
echo " HSI VAE Variant Training Launcher"
echo "  Models     : ${MODELS_SUBSET}"
echo "  Datasets   : ${DATASETS_SUBSET}"
echo "  Seeds      : ${SEEDS_SUBSET}"
echo "  Loss       : ${LOSS}"
echo "  Epochs     : ${EPOCHS_OVERRIDE:-<YAML default>}"
echo "  Batch Size : ${BATCH_SIZE_OVERRIDE:-<Dynamic/YAML defaults>}"
echo "  Overwrite  : ${OVERWRITE}"
echo "  Extra Args : ${EXTRA_ARGS[*]:-<none>}"
echo "=========================================================="

python utils/notify_cli.py --text "PRISM Variants grid training launched
host: $(hostname)
models: ${MODELS_SUBSET}
datasets: ${DATASETS_SUBSET}
seeds: ${SEEDS_SUBSET}
loss: ${LOSS}
epochs: ${EPOCHS_OVERRIDE:-<YAML defaults>}" >/dev/null 2>&1 || true

SKIPPED=()
PASSED=()
FAILED=()

for ds in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            # Formulate the checkpoint stems and names to avoid colliding with standard vae-our
            name_stem="${m}"
            ckpt_sam="${CKPT_DIR}/${ds}/${name_stem}_seed${seed}_bestsam.pt"
            ckpt_mse="${CKPT_DIR}/${ds}/${name_stem}_seed${seed}_bestmse.pt"
            
            # Skip checking if checkpoints already exist (unless overwrite is requested)
            if [[ -s "${ckpt_sam}" && -s "${ckpt_mse}" && "${OVERWRITE}" != "1" ]]; then
                echo "[skip] ${m} | ${ds} | ${LOSS} | seed ${seed} (Checkpoints already exist)"
                SKIPPED+=("${m}|${ds}|${LOSS}|seed${seed}")
                continue
            fi
            
            # Resolve batch size dynamically for 24GB VRAM
            # If not explicitly overridden, downscale Transformer models to avoid attention OOMs
            effective_batch_size=""
            if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
                effective_batch_size="${BATCH_SIZE_OVERRIDE}"
            elif [[ "${m}" == *"specvit"* ]]; then
                # SpecViT has O(C^2) self-attention complexity over H*W=4096 pixels.
                # Safe limits on a 24GB GPU:
                if [[ "${ds}" == "IIRS" || "${ds}" == "M3" ]]; then
                    effective_batch_size="16"  # Safe downscale from 32
                else
                    effective_batch_size="8"   # Safe downscale from 16 for AVIRIS/CRIMS (heavy bands)
                fi
            fi

            if [[ -n "${effective_batch_size}" ]]; then
                echo ">>> Running: ${m} | ${ds} | ${LOSS} | seed ${seed} (Batch Size: ${effective_batch_size})"
            else
                echo ">>> Running: ${m} | ${ds} | ${LOSS} | seed ${seed}"
            fi
            
            cmd=(python train/train_variants.py
                 --model "${m}"
                 --dataset "${ds}"
                 --loss "${LOSS}"
                 --seed "${seed}"
                 --ckpt-dir "${CKPT_DIR}")
            
            if [[ -n "${effective_batch_size}" ]]; then
                cmd+=(--batch-size "${effective_batch_size}")
            fi

            if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
                cmd+=(--epochs "${EPOCHS_OVERRIDE}")
            fi
            
            if (( ${#EXTRA_ARGS[@]} > 0 )); then
                cmd+=("${EXTRA_ARGS[@]}")
            fi
            
            # Execute training
            if "${cmd[@]}"; then
                echo "[OK] Successfully trained ${m} | ${ds} | seed ${seed}"
                PASSED+=("${m}|${ds}|seed${seed}")
            else
                echo "[FAIL] Training failed for ${m} | ${ds} | seed ${seed}"
                FAILED+=("${m}|${ds}|seed${seed}")
            fi
            echo "----------------------------------------------------------"
        done
    done
done

echo ""
echo "=========================================================="
echo " Variants training run complete."
echo "  Successful : ${#PASSED[@]}"
echo "  Skipped    : ${#SKIPPED[@]}"
echo "  Failed     : ${#FAILED[@]}"
echo "=========================================================="

python utils/notify_cli.py --text "PRISM Variants grid training complete!
host: $(hostname)
models: ${MODELS_SUBSET}
datasets: ${DATASETS_SUBSET}
successful: ${#PASSED[@]}
skipped: ${#SKIPPED[@]}
failed: ${#FAILED[@]}" >/dev/null 2>&1 || true

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed runs:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}"
    done
    exit 1
fi
exit 0
