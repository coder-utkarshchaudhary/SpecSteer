#!/usr/bin/env bash
# scripts/run_remote_sweep.sh
# ----------------------------
# Combined master script to coordinate remote sweeps sequentially:
#   Step 1: Run inference on baseline models for existing datasets (IIRS, AVIRIS, CRIMS).
#   Step 2: Train & evaluate only the vae-our-nl variant for all datasets (including M3).
#   Step 3: Train all baseline models for the M3 dataset only.
#   Step 4: Run inference on M3 baseline models.
#
# Ensures proper logging, Telegram notification, and robustness against step failures.
#
# Usage:
#   bash scripts/run_remote_sweep.sh [options]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CKPT_DIR="${CKPT_DIR:-model}"
OUT_DIR="${OUT_DIR:-results}"

# Default parameters
DATASETS_STEP1="IIRS,AVIRIS,CRIMS"
DATASETS_STEP2="IIRS,M3,AVIRIS,CRIMS"
SEEDS_CSV="42"
SELECT="sam"
OVERWRITE=0
SEND_TELEGRAM=1
STEPS_CSV="1,2,3,4"
EXTRA_ARGS=()

# Help message
show_help() {
    echo "Usage: bash scripts/run_remote_sweep.sh [options]"
    echo ""
    echo "Options:"
    echo "  --steps 1,2,3,4         Steps to execute (default: 1,2,3,4)"
    echo "  --seeds 42,7,1234       Seeds to train/evaluate (default: 42)"
    echo "  --select sam|mse        Checkpoint selection metric (default: sam)"
    echo "  --overwrite             Overwrite existing checkpoints/results"
    echo "  --datasets-baseline ds  Datasets for baseline eval in Step 1 (default: IIRS,AVIRIS,CRIMS)"
    echo "  --datasets-variant ds   Datasets for vae-our-nl in Step 2 (default: IIRS,M3,AVIRIS,CRIMS)"
    echo "  --no-telegram           Disable Telegram alerts"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Any other options are forwarded to train.py/train_variants.py as extra arguments."
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS_CSV="$2"; shift 2 ;;
        --seeds|--seed) SEEDS_CSV="$2"; shift 2 ;;
        --select) SELECT="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --datasets-baseline) DATASETS_STEP1="$2"; shift 2 ;;
        --datasets-variant) DATASETS_STEP2="$2"; shift 2 ;;
        --no-telegram) SEND_TELEGRAM=0; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Prepare run directories
LOG_ROOT="${REPO_ROOT}/logs/remote_sweep"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="${LOG_ROOT}/${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

# Symlink or copy to 'latest'
rm -f "${LOG_ROOT}/latest" 2>/dev/null || true
ln -s "${RUN_LOG_DIR}" "${LOG_ROOT}/latest" 2>/dev/null || true

# Global failures collector
SWEEP_FAILURES=()

# Source grid lock to avoid concurrent OOMs
if [[ -f "${SCRIPT_DIR}/grid_lock.sh" ]]; then
    # shellcheck source=scripts/grid_lock.sh
    source "${SCRIPT_DIR}/grid_lock.sh"
fi

# Define execution step runner
run_step() {
    local step_num="$1"
    local step_title="$2"
    local log_file="${RUN_LOG_DIR}/step${step_num}_$(echo "${step_title}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_').log"
    shift 2
    
    echo ""
    echo "=========================================================="
    echo " STEP ${step_num}: ${step_title}"
    echo " Log: ${log_file}"
    echo "=========================================================="
    
    if (( SEND_TELEGRAM )); then
        python utils/notify_cli.py --text "Sweep Step ${step_num} Start: ${step_title} on $(hostname)" >/dev/null 2>&1 || true
    fi
    
    local start_time=$(date +%s)
    set +e
    "$@" 2>&1 | tee "${log_file}"
    local rc=$?
    set -e
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ ${rc} -eq 0 ]]; then
        echo ">>> STEP ${step_num} SUCCESS (duration: ${duration}s)"
        if (( SEND_TELEGRAM )); then
            python utils/notify_cli.py --text "Sweep Step ${step_num} SUCCESS: ${step_title} on $(hostname) (duration: ${duration}s)" >/dev/null 2>&1 || true
        fi
    else
        echo ">>> STEP ${step_num} FAILED (exit code: ${rc}, duration: ${duration}s)"
        if (( SEND_TELEGRAM )); then
            python utils/notify_cli.py --text "Sweep Step ${step_num} FAILED: ${step_title} on $(hostname) (exit: ${rc}, duration: ${duration}s)" >/dev/null 2>&1 || true
        fi
        SWEEP_FAILURES+=("Step ${step_num}: ${step_title} (exit: ${rc})")
    fi
    return ${rc}
}

# Determine which steps are requested
RUN_S1=0; RUN_S2=0; RUN_S3=0; RUN_S4=0
IFS=',' read -r -a STEPS_ARR <<< "${STEPS_CSV}"
for s in "${STEPS_ARR[@]}"; do
    [[ "${s}" == "1" ]] && RUN_S1=1
    [[ "${s}" == "2" ]] && RUN_S2=1
    [[ "${s}" == "3" ]] && RUN_S3=1
    [[ "${s}" == "4" ]] && RUN_S4=1
done

echo "=========================================================="
echo " Master Remote Sweep Coordinator"
echo "  Seeds       : ${SEEDS_CSV}"
echo "  Select      : ${SELECT}"
echo "  Overwrite   : ${OVERWRITE}"
echo "  Telegram    : ${SEND_TELEGRAM}"
echo "  Steps Run   : 1=${RUN_S1} 2=${RUN_S2} 3=${RUN_S3} 4=${RUN_S4}"
echo "  Log Folder  : ${RUN_LOG_DIR}"
echo "=========================================================="

if (( SEND_TELEGRAM )); then
    python utils/notify_cli.py --text "Master Remote Sweep started
host: $(hostname)
seeds: ${SEEDS_CSV}
select: ${SELECT}
steps: 1=${RUN_S1} 2=${RUN_S2} 3=${RUN_S3} 4=${RUN_S4}
logs: ${RUN_LOG_DIR}" >/dev/null 2>&1 || true
fi

# ------------------------------------------------------------------------------
# STEP 1: Baseline Inference on existing datasets
# ------------------------------------------------------------------------------
if (( RUN_S1 )); then
    run_baseline_inference() {
        # Split DATASETS_STEP1 into an array
        IFS=',' read -r -a STEP1_DS_ARR <<< "${DATASETS_STEP1}"
        for ds in "${STEP1_DS_ARR[@]}"; do
            echo ">>> Running baseline inference for dataset ${ds} with old parameter-matched widths..."
            local extra_opts=()
            if [[ "${ds}" == "IIRS" ]]; then
                extra_opts+=(--set "vae_standard_base_ch=86" --set "vae_3d_base_ch=45" --set "vae_1d_hidden_dims=[2748,1374,687]")
            elif [[ "${ds}" == "AVIRIS" ]]; then
                extra_opts+=(--set "vae_standard_base_ch=85" --set "vae_3d_base_ch=46" --set "vae_1d_hidden_dims=[2668,1334,667]")
            elif [[ "${ds}" == "CRIMS" ]]; then
                extra_opts+=(--set "vae_standard_base_ch=85" --set "vae_3d_base_ch=46" --set "vae_1d_hidden_dims=[2656,1328,664]")
            fi
            
            # Run inference.sh for this single dataset
            bash scripts/inference.sh \
                --datasets "${ds}" \
                --seeds "${SEEDS_CSV}" \
                --select "${SELECT}" \
                "${extra_opts[@]}"
        done
    }

    run_step "1" "Inference on Baselines (${DATASETS_STEP1})" run_baseline_inference
fi

# ------------------------------------------------------------------------------
# STEP 2: Train & Evaluate only vae-our-nl for all datasets (including M3)
# ------------------------------------------------------------------------------
if (( RUN_S2 )); then
    # Helper function to run step 2 training + inference sequentially
    run_vae_our_nl() {
        # Check grid lock if script has it
        if command -v acquire_grid_lock >/dev/null 2>&1; then
            acquire_grid_lock "vae-our-nl sweep" || return 9
        fi
        
        local train_flags=()
        if (( OVERWRITE )); then
            train_flags+=(--overwrite)
        fi
        
        echo ">>> Training vae-our-nl..."
        bash scripts/train_variants.sh \
            --models vae-our-nl \
            --datasets "${DATASETS_STEP2}" \
            --seeds "${SEEDS_CSV}" \
            "${train_flags[@]+"${train_flags[@]}"}" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
            
        local train_rc=$?
        if [[ ${train_rc} -ne 0 ]]; then
            echo "WARNING: vae-our-nl training reported errors. Continuing to inference..."
        fi
        
        echo ">>> Running inference on vae-our-nl..."
        bash scripts/inference_variants.sh \
            --models vae-our-nl \
            --datasets "${DATASETS_STEP2}" \
            --seeds "${SEEDS_CSV}" \
            --select "${SELECT}"
    }
    
    run_step "2" "Train & Evaluate vae-our-nl (${DATASETS_STEP2})" run_vae_our_nl
fi

# ------------------------------------------------------------------------------
# STEP 3: Train all Baselines for M3 Dataset only
# ------------------------------------------------------------------------------
if (( RUN_S3 )); then
    # We define all baseline configs for M3
    train_m3_baselines() {
        if command -v acquire_grid_lock >/dev/null 2>&1; then
            acquire_grid_lock "M3 baselines sweep" || return 9
        fi
        
        local M3_CONFIGS=(
            "vae-our|physics|vae-our|1"
            "vae-standard|standard|vae-standard_standard|0"
            "vae-standard|physics|vae-standard_physics|1"
            "vae-1d-pixelwise|standard|vae-1d-pixelwise_standard|0"
            "vae-1d-pixelwise|physics|vae-1d-pixelwise_physics|1"
            "vae-3d-spatio-spectral|physics|vae-3d-spatio-spectral_physics|1"
            "vae-3d-spatio-spectral|standard|vae-3d-spatio-spectral_standard|0"
        )
        
        IFS=',' read -r -a SEEDS_ARR <<< "${SEEDS_CSV}"
        local first_seed="${SEEDS_ARR[0]}"
        
        local passed_cnt=0
        local skipped_cnt=0
        local failed_cnt=0
        
        for cfg in "${M3_CONFIGS[@]}"; do
            local model="${cfg%%|*}"; local rest="${cfg#*|}"
            local loss="${rest%%|*}"; rest="${rest#*|}"
            local name="${rest%%|*}"
            local claim="${rest#*|}"
            
            # Determine which seeds to run for this configuration
            local target_seeds=()
            if [[ "${claim}" == "1" ]]; then
                target_seeds=("${SEEDS_ARR[@]}")
            else
                target_seeds=("${first_seed}")
            fi
            
            for seed in "${target_seeds[@]}"; do
                local ckpt_sam="${CKPT_DIR}/M3/${name}_seed${seed}_bestsam.pt"
                local ckpt_mse="${CKPT_DIR}/M3/${name}_seed${seed}_bestmse.pt"
                
                # Check if checkpoint exists and we should skip
                if [[ -s "${ckpt_sam}" && -s "${ckpt_mse}" && "${OVERWRITE}" != "1" ]]; then
                    echo "[skip] M3 | ${model} | ${loss} | seed ${seed} (Checkpoints already exist)"
                    skipped_cnt=$((skipped_cnt + 1))
                    continue
                fi
                
                echo ">>> Training: M3 | ${model} | ${loss} | seed ${seed}"
                
                # Construct command
                local cmd=(python train/train.py
                     --model "${model}"
                     --dataset "M3"
                     --loss "${loss}"
                     --seed "${seed}"
                     --ckpt-dir "${CKPT_DIR}")
                     
                if (( ${#EXTRA_ARGS[@]} > 0 )); then
                    cmd+=("${EXTRA_ARGS[@]}")
                fi
                
                # Execute training with a retry logic like train.sh
                local attempt=1
                local train_rc=0
                while (( attempt <= 2 )); do
                    echo "    Attempt ${attempt}/2..."
                    set +e
                    "${cmd[@]}"
                    train_rc=$?
                    set -e
                    if [[ ${train_rc} -eq 0 ]]; then
                        break
                    fi
                    echo "    Attempt ${attempt} failed (exit ${train_rc})"
                    (( attempt < 2 )) && { echo "    Retrying after 5s..."; sleep 5; }
                    attempt=$(( attempt + 1 ))
                done
                
                if [[ ${train_rc} -eq 0 ]]; then
                    echo "[OK] Successfully trained ${model} | M3 | ${loss} | seed ${seed}"
                    passed_cnt=$((passed_cnt + 1))
                else
                    echo "[FAIL] Training failed for ${model} | M3 | ${loss} | seed ${seed}"
                    failed_cnt=$((failed_cnt + 1))
                fi
                echo "----------------------------------------------------------"
            done
        done
        
        echo "M3 Baseline training summary:"
        echo "  Passed:  ${passed_cnt}"
        echo "  Skipped: ${skipped_cnt}"
        echo "  Failed:  ${failed_cnt}"
        
        if [[ ${failed_cnt} -gt 0 ]]; then
            return 1
        fi
        return 0
    }
    
    run_step "3" "Train Baselines for M3" train_m3_baselines
fi

# ------------------------------------------------------------------------------
# STEP 4: Run Inference on M3 Baselines
# ------------------------------------------------------------------------------
if (( RUN_S4 )); then
    run_step "4" "Inference on M3 Baselines" \
        bash scripts/inference.sh \
            --datasets "M3" \
            --seeds "${SEEDS_CSV}" \
            --select "${SELECT}"
fi

# ------------------------------------------------------------------------------
# Sweep Completion and Summary
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo " Master remote sweep complete."
echo "=========================================================="
if [[ ${#SWEEP_FAILURES[@]} -gt 0 ]]; then
    echo "The following steps reported warnings/failures:"
    for f in "${SWEEP_FAILURES[@]}"; do
        echo "  - ${f}"
    done
else
    echo "All executed steps completed successfully!"
fi
echo "Logs are available under: ${RUN_LOG_DIR}"
echo "=========================================================="

if (( SEND_TELEGRAM )); then
    if [[ ${#SWEEP_FAILURES[@]} -gt 0 ]]; then
        python utils/notify_cli.py --text "Master Remote Sweep finished with WARNINGS/FAILURES on $(hostname)
Failures:
$(for f in "${SWEEP_FAILURES[@]}"; do echo "  - ${f}"; done)
Logs folder: ${RUN_LOG_DIR}" >/dev/null 2>&1 || true
    else
        python utils/notify_cli.py --text "Master Remote Sweep finished successfully on $(hostname)!
Logs folder: ${RUN_LOG_DIR}" >/dev/null 2>&1 || true
    fi
fi

if [[ ${#SWEEP_FAILURES[@]} -gt 0 ]]; then
    exit 1
fi
exit 0
