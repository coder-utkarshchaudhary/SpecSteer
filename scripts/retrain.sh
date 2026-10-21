#!/usr/bin/env bash
# scripts/retrain.sh
# ------------------
# Physics-Constrained Trust-Region Fusion (PCTF) Retrain Grid
#
# Implements the complete training and evaluation grid for the 5 fusion mechanisms:
#   1. Exp 1/5: Fixed-Weight Fusion (alpha=0.5, parameter-free convex blend)
#   2. Exp 2/5: Learned Unconstrained Fusion (Gate trained without SAM penalty)
#   3. Exp 3/5: Learned + SAM Penalty (Current PRISM regularized fusion baseline)
#   4. Exp 4/5: Constrained SAM Fusion (PCTF: min D s.t. SAM <= epsilon)
#   5. Exp 5/5: Uncertainty-Aware Constrained Fusion (U-PCTF: epsilon = f(disc, snr, unc))
#
# Default Grid:
#   5 models x 4 datasets (IIRS, AVIRIS, M3, CRIMS) x 2 seeds (67, 69) = 40 cells
#   Epochs: 60, Early-Stopping Patience: 3, Selection: best SAM
#
# Per-Cell Pipeline:
#   1. Train (train/train.py) -> [START] / [HB] heartbeats / [OK] to Telegram
#   2. Infer (inference/inference.py) -> whole-cube test split metrics
#   3. Probes (inference/probes.py) -> P1-P4 falsification diagnostics
#   4. Downstream (inference/downstream.py) -> noise recovery & chemical interpolation
#   5. End of Dataset -> verdict.py + aggregate.py (posts CSV tables to Telegram)
#
# Usage:
#   bash scripts/retrain.sh                    # Full 40-cell grid (resumable)
#   bash scripts/retrain.sh --dry-run          # Print the 40-cell schedule and exit
#   bash scripts/retrain.sh --overwrite        # Ignore .done markers and recompute all
#   bash scripts/retrain.sh --no-telegram      # Disable Telegram alerts
#   bash scripts/retrain.sh --select mse       # Select best MSE instead of best SAM
#   bash scripts/retrain.sh --datasets IIRS,AVIRIS  # Run subset of datasets
#   bash scripts/retrain.sh --seeds 67         # Run single seed
#   bash scripts/retrain.sh --include-baselines # Include standard 2D, 1D, and 3D baselines
#
# Resumable: Touches results/.done/<DS>__<stem>_seed<N> only after all 4 steps pass.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../modules/registry.py" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    REPO_ROOT="${SCRIPT_DIR}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "${REPO_ROOT}"

PY="${PY:-python}"
if [[ -f ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
elif [[ -f ".venv/Scripts/python.exe" ]]; then
    PY=".venv/Scripts/python.exe"
fi

# ---------------------------------------------------------------------------
# Default Grid Configuration
# ---------------------------------------------------------------------------

DATASETS=(IIRS AVIRIS M3 CRIMS)
SEEDS=(67 69)
EPOCHS=60
PATIENCE=3
SELECT="sam"
OVERWRITE=0
DRY_RUN=0
SEND_TELEGRAM=1
INCLUDE_BASELINES=0
CKPT_DIR="model"
OUT_DIR="results"

# Primary 5 Fusion Experiments: "<model>|<loss>|<ckpt_stem>|<display_title>"
FUSION_CONFIGS=(
    "vae-fusion-fixed|physics|vae-fusion-fixed|Exp 1/5: Fixed-Weight Fusion (alpha=0.5)"
    "vae-fusion-unconstrained|standard|vae-fusion-unconstrained_standard|Exp 2/5: Learned Unconstrained Fusion (No SAM Penalty)"
    "vae-fusion-sam-penalty|physics|vae-fusion-sam-penalty|Exp 3/5: Learned + SAM Penalty (Baseline PRISM)"
    "vae-fusion-constrained|physics|vae-fusion-constrained|Exp 4/5: Constrained SAM Fusion (PCTF)"
    "vae-fusion-uncertainty|physics|vae-fusion-uncertainty|Exp 5/5: Uncertainty-Aware Constrained Fusion (U-PCTF)"
)

BASELINE_CONFIGS=(
    "vae-standard|physics|vae-standard_physics|Baseline: Standard 2D VAE (Physics)"
    "vae-standard|standard|vae-standard_standard|Baseline: Standard 2D VAE (Standard)"
    "vae-1d-pixelwise|physics|vae-1d-pixelwise_physics|Baseline: 1D Pixelwise VAE (Physics)"
    "vae-1d-pixelwise|standard|vae-1d-pixelwise_standard|Baseline: 1D Pixelwise VAE (Standard)"
    "vae-3d-spatio-spectral|physics|vae-3d-spatio-spectral_physics|Baseline: 3D Spatio-Spectral VAE (Physics)"
    "vae-3d-spatio-spectral|standard|vae-3d-spatio-spectral_standard|Baseline: 3D Spatio-Spectral VAE (Standard)"
)

# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --dry-run              Print planned cell schedule without training
  --overwrite            Ignore existing results/.done markers and retrain all
  --no-telegram          Disable all Telegram notifications
  --select sam|mse       Checkpoint selection metric (default: sam)
  --datasets D1,D2...    Comma-separated datasets (default: IIRS,AVIRIS,M3,CRIMS)
  --seeds S1,S2...       Comma-separated seeds (default: 67,69)
  --epochs N             Number of epochs per cell (default: 60)
  --patience N           Early stopping patience (default: 3)
  --include-baselines    Include standard 2D, 1D, and 3D baseline models in grid
  --ckpt-dir DIR         Checkpoint root directory (default: model)
  --out-dir DIR          Results directory (default: results)
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)           DRY_RUN=1; shift ;;
        --overwrite)         OVERWRITE=1; shift ;;
        --no-telegram)       SEND_TELEGRAM=0; shift ;;
        --select)            SELECT="$2"; shift 2 ;;
        --datasets|--dataset)
            IFS=',' read -r -a DATASETS <<< "$2"; shift 2 ;;
        --seeds|--seed)
            IFS=',' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --epochs)            EPOCHS="$2"; shift 2 ;;
        --patience)          PATIENCE="$2"; shift 2 ;;
        --include-baselines) INCLUDE_BASELINES=1; shift ;;
        --ckpt-dir)          CKPT_DIR="$2"; shift 2 ;;
        --out-dir)           OUT_DIR="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "${SELECT}" != "sam" && "${SELECT}" != "mse" ]]; then
    echo "ERROR: --select must be 'sam' or 'mse', got '${SELECT}'" >&2
    exit 1
fi

CONFIGS=("${FUSION_CONFIGS[@]}")
if (( INCLUDE_BASELINES == 1 )); then
    CONFIGS+=("${BASELINE_CONFIGS[@]}")
fi

TOTAL_PLANNED=$(( ${#DATASETS[@]} * ${#CONFIGS[@]} * ${#SEEDS[@]} ))

# ---------------------------------------------------------------------------
# Dry Run Mode
# ---------------------------------------------------------------------------

if (( DRY_RUN == 1 )); then
    echo "========================================================================"
    echo " Physics-Constrained Trust-Region Fusion (PCTF) Retrain Plan"
    echo " Total planned cells: ${TOTAL_PLANNED}"
    echo " Epochs: ${EPOCHS} | Patience: ${PATIENCE} | Select: ${SELECT}"
    echo " Datasets: ${DATASETS[*]}"
    echo " Seeds: ${SEEDS[*]}"
    echo "========================================================================"
    cell_idx=0
    for ds in "${DATASETS[@]}"; do
        echo ""
        echo "--- Dataset: ${ds} ---"
        for cfg in "${CONFIGS[@]}"; do
            IFS='|' read -r model loss stem title <<< "${cfg}"
            for seed in "${SEEDS[@]}"; do
                cell_idx=$((cell_idx + 1))
                printf "  [%2d/%2d] %-7s | %-26s | %-8s | seed=%s | %s\n" \
                    "${cell_idx}" "${TOTAL_PLANNED}" "${ds}" "${model}" "${loss}" "${seed}" "${title}"
            done
        done
    done
    echo ""
    echo "Dry-run complete. Run without --dry-run to start training."
    exit 0
fi

# ---------------------------------------------------------------------------
# Setup & Logging
# ---------------------------------------------------------------------------

mkdir -p logs "${CKPT_DIR}" "${OUT_DIR}" "${OUT_DIR}/.done" \
    "${OUT_DIR}/inference" "${OUT_DIR}/probes" "${OUT_DIR}/downstream"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
DRIVER_LOG="logs/retrain_pctf_${RUN_TS}.log"
touch "${DRIVER_LOG}"

# Grid lock to prevent duplicate concurrent runs
if [[ -f "${SCRIPT_DIR}/grid_lock.sh" ]]; then
    source "${SCRIPT_DIR}/grid_lock.sh"
    acquire_grid_lock "retrain pctf grid ($(hostname))" || exit 9
elif [[ -f "${REPO_ROOT}/scripts/grid_lock.sh" ]]; then
    source "${REPO_ROOT}/scripts/grid_lock.sh"
    acquire_grid_lock "retrain pctf grid ($(hostname))" || exit 9
fi

log_line() {
    local ts; ts="$(date +%H:%M:%S)"
    echo "${ts} $*"
    echo "${ts} $*" >> "${DRIVER_LOG}"
}

notify() {
    if (( SEND_TELEGRAM == 1 )); then
        "${PY}" -u utils/notify_cli.py --text "$1" >/dev/null 2>&1 || true
    fi
}

VRAM_ABORT=0
VRAM_PATTERN='Refusing to start: only'

run_stage() {
    local label="$1"; shift
    local stage_log; stage_log="$(mktemp)"
    log_line "----- ${label} -----"
    if [[ -t 1 ]]; then
        "$@" 2>&1 | tee /dev/tty \
            | awk '{n=split($0,p,"\r"); if (p[n] ~ /[^[:space:]]/) {print p[n]; fflush()}}' \
            > "${stage_log}"
    else
        "$@" 2>&1 \
            | awk '{n=split($0,p,"\r"); if (p[n] ~ /[^[:space:]]/) {print p[n]; fflush()}}' \
            > "${stage_log}"
    fi
    local rc=${PIPESTATUS[0]}
    cat "${stage_log}" >> "${DRIVER_LOG}"
    if (( rc != 0 )); then
        {
            echo "$(date +%H:%M:%S) [FAIL] ${label} (rc=${rc}) — last 20 lines:"
            tail -n 20 "${stage_log}"
        } | tee -a "${DRIVER_LOG}" >&2
        if grep -qF "${VRAM_PATTERN}" "${stage_log}"; then
            VRAM_ABORT=1
        fi
    fi
    rm -f "${stage_log}"
    return "${rc}"
}

abort_on_vram() {
    local msg="GPU already occupied by another process (VRAM preflight refused to start). "
    msg+="Retrain grid aborting to avoid repeated crashes. Check nvidia-smi, clear conflicting jobs, and rerun."
    echo "" >&2
    echo "############################################################" >&2
    echo "# ABORT: ${msg}" >&2
    echo "############################################################" >&2
    log_line "[ABORT] ${msg}"
    notify "[ABORT] Retrain grid aborted on $(hostname): ${msg}"
    exit 2
}

# ---------------------------------------------------------------------------
# Launch Announcement
# ---------------------------------------------------------------------------

echo "========================================================================"
echo " 🚀 Starting Physics-Constrained Trust-Region Fusion Retrain Grid"
echo "  Total planned cells : ${TOTAL_PLANNED}"
echo "  Datasets            : ${DATASETS[*]}"
echo "  Seeds               : ${SEEDS[*]}"
echo "  Epochs              : ${EPOCHS} (patience: ${PATIENCE}, select: ${SELECT})"
echo "  Driver log          : ${DRIVER_LOG}"
echo "========================================================================"

log_line "Retrain grid launched on $(hostname): ${TOTAL_PLANNED} cells, datasets=${DATASETS[*]}, seeds=${SEEDS[*]}"

notify "🚀 <b>Retrain Grid Launched on $(hostname)</b>
<b>Study:</b> Physics-Constrained Trust-Region Fusion (PCTF)
<b>Total cells:</b> ${TOTAL_PLANNED}
<b>Datasets:</b> ${DATASETS[*]}
<b>Seeds:</b> ${SEEDS[*]}
<b>Epochs:</b> ${EPOCHS} (patience: ${PATIENCE}, select: ${SELECT})
<b>Log:</b> <code>${DRIVER_LOG}</code>"

SKIPPED=()
FAILED=()
DONE_COUNT=0
cell_counter=0

# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------

for ds in "${DATASETS[@]}"; do
    log_line "=================================================================="
    log_line " DATASET: ${ds}"
    log_line "=================================================================="

    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r model loss stem title <<< "${cfg}"

        for seed in "${SEEDS[@]}"; do
            cell_counter=$((cell_counter + 1))
            marker="${OUT_DIR}/.done/${ds}__${stem}_seed${seed}"

            if [[ -f "${marker}" && "${OVERWRITE}" != "1" ]]; then
                SKIPPED+=("${ds}|${model}|${loss}|seed${seed}")
                log_line "[SKIP] [${cell_counter}/${TOTAL_PLANNED}] ${title} | ${ds} | seed=${seed} — already done"
                continue
            fi

            echo ""
            echo "========================================================================"
            echo " [${cell_counter}/${TOTAL_PLANNED}] ${title}"
            echo " Dataset: ${ds} | Model: ${model} | Loss: ${loss} | Seed: ${seed}"
            echo "========================================================================"
            log_line ">>>>> [${cell_counter}/${TOTAL_PLANNED}] ${title} | ${ds} | seed=${seed} <<<<<"

            cell_ok=1

            # 1. Train
            run_stage "train | ${ds} | ${model} | ${loss} | seed${seed}" \
                "${PY}" -u train/train.py \
                    --model "${model}" \
                    --dataset "${ds}" \
                    --loss "${loss}" \
                    --seed "${seed}" \
                    --epochs "${EPOCHS}" \
                    --patience "${PATIENCE}" \
                    --ckpt-dir "${CKPT_DIR}"
            train_rc=$?

            if (( VRAM_ABORT == 1 )); then
                abort_on_vram
            fi

            if (( train_rc != 0 )); then
                FAILED+=("train|${model}|${ds}|${loss}|seed${seed}|rc=${train_rc}")
                log_line "[FAIL] Training failed (rc=${train_rc}) — skipping evaluation for this cell"
                continue
            fi

            # 2. Inference (whole-cube test reconstruction metrics)
            infer_json="${OUT_DIR}/inference/${ds}__${stem}_seed${seed}_${SELECT}.json"
            run_stage "infer | ${ds} | ${model} | ${loss} | seed${seed}" \
                "${PY}" -u inference/inference.py \
                    --model "${model}" \
                    --dataset "${ds}" \
                    --loss "${loss}" \
                    --seed "${seed}" \
                    --select "${SELECT}" \
                    --ckpt-dir "${CKPT_DIR}" \
                    --out-json "${infer_json}"
            infer_rc=$?
            if (( infer_rc != 0 )); then
                FAILED+=("infer|${model}|${ds}|${loss}|seed${seed}|rc=${infer_rc}")
                cell_ok=0
            fi

            # 3. Probes (P1-P4 falsification diagnostics)
            run_stage "probes | ${ds} | ${model} | ${loss} | seed${seed}" \
                "${PY}" -u inference/probes.py \
                    --dataset "${ds}" \
                    --model "${model}" \
                    --loss "${loss}" \
                    --seed "${seed}" \
                    --select "${SELECT}" \
                    --ckpt-dir "${CKPT_DIR}" \
                    --out-dir "${OUT_DIR}/probes"
            probes_rc=$?
            if (( probes_rc != 0 )); then
                FAILED+=("probes|${model}|${ds}|${loss}|seed${seed}|rc=${probes_rc}")
                cell_ok=0
            fi

            # 4. Downstream (noise recovery & chemical interpolation)
            run_stage "downstream | ${ds} | ${model} | ${loss} | seed${seed}" \
                "${PY}" -u inference/downstream.py \
                    --dataset "${ds}" \
                    --models "${model}" \
                    --loss "${loss}" \
                    --seed "${seed}" \
                    --select "${SELECT}" \
                    --ckpt-dir "${CKPT_DIR}" \
                    --out-dir "${OUT_DIR}/downstream"
            downstream_rc=$?
            if (( downstream_rc != 0 )); then
                FAILED+=("downstream|${model}|${ds}|${loss}|seed${seed}|rc=${downstream_rc}")
                cell_ok=0
            fi

            if (( cell_ok == 1 )); then
                mkdir -p "$(dirname "${marker}")"
                touch "${marker}"
                DONE_COUNT=$((DONE_COUNT + 1))
                log_line "[DONE] [${cell_counter}/${TOTAL_PLANNED}] ${ds} | ${model} | seed${seed}"
            else
                log_line "[PARTIAL] [${cell_counter}/${TOTAL_PLANNED}] ${ds} | ${model} | seed${seed} — will retry on next run"
            fi
        done
    done

    # End of dataset: Run verdict and aggregate to produce CSV comparisons & push to Telegram
    log_line "=================================================================="
    log_line " Dataset ${ds} finished — running verdict & aggregate"
    log_line "=================================================================="

    run_stage "verdict | ${ds}" "${PY}" -u inference/verdict.py
    verdict_rc=$?
    if (( verdict_rc != 0 )); then
        FAILED+=("verdict|-|${ds}|-|-|rc=${verdict_rc}")
    fi

    if (( SEND_TELEGRAM == 1 )); then
        run_stage "aggregate | ${ds}" "${PY}" -u inference/aggregate.py \
            --telegram --caption "Cumulative Fusion Ablation Results through ${ds}"
    else
        run_stage "aggregate | ${ds}" "${PY}" -u inference/aggregate.py
    fi
    agg_rc=$?
    if (( agg_rc != 0 )); then
        FAILED+=("aggregate|-|${ds}|-|-|rc=${agg_rc}")
    fi
done

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo " 🏁 Retrain Grid Completed"
echo "  Total planned : ${TOTAL_PLANNED}"
echo "  Completed     : ${DONE_COUNT}"
echo "  Skipped       : ${#SKIPPED[@]}"
echo "  Failed        : ${#FAILED[@]}"
echo "  Log file      : ${DRIVER_LOG}"
echo "========================================================================"

log_line "Retrain grid completed: total=${TOTAL_PLANNED} done=${DONE_COUNT} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}"

if (( ${#FAILED[@]} > 0 )); then
    echo "Failure summary:"
    printf '  - %s\n' "${FAILED[@]}"
    {
        echo "Failures:"
        printf '  - %s\n' "${FAILED[@]}"
    } >> "${DRIVER_LOG}"
fi

notify "🏁 <b>Retrain Grid Finished on $(hostname)</b>
<b>Completed:</b> ${DONE_COUNT}/${TOTAL_PLANNED}
<b>Skipped:</b> ${#SKIPPED[@]}
<b>Failed:</b> ${#FAILED[@]}
<b>Log:</b> <code>${DRIVER_LOG}</code>"

exit $(( ${#FAILED[@]} > 0 ? 1 : 0 ))
