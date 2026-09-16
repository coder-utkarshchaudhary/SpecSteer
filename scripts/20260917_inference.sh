#!/usr/bin/env bash
# scripts/20260917_inference.sh
# ------------------------------
# One-shot follow-up to the 2026-09-15 clean-slate grid + the 2026-09-16 fixes
# (commit 7c9030c: NaN-safe inference/stats.py, crash-hardened
# inference/probes.py, M3 early_stopping_patience 3->10). Four stages, cheapest
# first:
#
#   1. AVIRIS probes-only diagnostic — all 16 (model,loss,seed) cells, against
#      the EXISTING checkpoints (no retraining). Exercises the probes.py crash
#      hardening. If every cell comes back clean, touches the resume markers
#      so stage 4 doesn't waste GPU time retraining AVIRIS too.
#   2. M3 --select mse check — 4 cells (vae-our/vae-our-nl x seed 67/69)
#      against existing checkpoints, no retraining. Informational only.
#   3. Immediate stats refresh — re-runs verdict.py + aggregate.py --telegram
#      against data already on disk, so the CRIMS sam_valid NaN fix lands in
#      Telegram now rather than waiting for stage 4.
#   4. The real retrain — clears M3's resume markers + checkpoints and runs
#      scripts/run_clean_grid.sh, which resumes everything else and redoes
#      M3 under the new patience (and AVIRIS too, if stage 1 didn't clear it).
#
# Usage (foreground, no tmux/nohup — matches scripts/run_clean_grid.sh):
#   bash scripts/20260917_inference.sh                  # all 4 stages
#   bash scripts/20260917_inference.sh --diagnostics-only # stages 1-3 only, skip the retrain
#   bash scripts/20260917_inference.sh --skip-avaris-diag # skip stage 1 (go straight to 2-4)
#
# Safe to re-run: stage 1 is idempotent (just re-reads checkpoints), stage 4
# uses the same resume-by-marker machinery as scripts/run_clean_grid.sh.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

PY="${PY:-python}"
CKPT_DIR="model"
OUT_DIR="results"

DO_RETRAIN=1
DO_AVIRIS_DIAG=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --diagnostics-only)     DO_RETRAIN=0; shift ;;
        --skip-avaris-diag)     DO_AVIRIS_DIAG=0; shift ;;
        -h|--help)
            sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p logs "${OUT_DIR}/.done" "${OUT_DIR}/inference" "${OUT_DIR}/probes"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/20260917_inference_${RUN_TS}.log"
touch "${LOG}"

log() {
    local ts; ts="$(date +%H:%M:%S)"
    echo "${ts} $*"
    echo "${ts} $*" >> "${LOG}"
}

# "<model>|<loss>|<ckpt_stem>" — must match scripts/run_clean_grid.sh's CONFIGS
# exactly (and modules/registry.py's checkpoint_name()).
CONFIGS=(
    "vae-our-nl|physics|vae-our-nl"
    "vae-standard|physics|vae-standard_physics"
    "vae-standard|standard|vae-standard_standard"
    "vae-1d-pixelwise|physics|vae-1d-pixelwise_physics"
    "vae-1d-pixelwise|standard|vae-1d-pixelwise_standard"
    "vae-3d-spatio-spectral|physics|vae-3d-spatio-spectral_physics"
    "vae-3d-spatio-spectral|standard|vae-3d-spatio-spectral_standard"
    "vae-our|physics|vae-our"
)
SEEDS=(67 69)

echo "=============================================="
echo " 2026-09-17 follow-up: AVIRIS diag + M3 select=mse check + stats refresh"
echo "   $( [[ ${DO_RETRAIN} == 1 ]] && echo '+ M3 retrain under patience=10' || echo '(--diagnostics-only: no retrain)' )"
echo "  log: ${LOG}"
echo "=============================================="

# ---------------------------------------------------------------------------
# Stage 1 — AVIRIS probes-only diagnostic (no retraining)
# ---------------------------------------------------------------------------
AVIRIS_ALL_OK=1
if (( DO_AVIRIS_DIAG == 1 )); then
    log "===== Stage 1: AVIRIS probes-only diagnostic (16 cells, existing checkpoints) ====="
    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r model loss stem <<< "${cfg}"
        for seed in "${SEEDS[@]}"; do
            log "  probes | AVIRIS | ${model} | ${loss} | seed${seed}"
            if ! "${PY}" -u inference/probes.py --dataset AVIRIS --model "${model}" --loss "${loss}" \
                    --seed "${seed}" --select sam --ckpt-dir "${CKPT_DIR}" \
                    --out-dir "${OUT_DIR}/probes" 2>&1 | tee -a "${LOG}"; then
                log "  [FAIL] AVIRIS | ${model} | ${loss} | seed${seed}"
                AVIRIS_ALL_OK=0
            fi
        done
    done

    if (( AVIRIS_ALL_OK == 1 )); then
        log "Stage 1: all 16 AVIRIS probe cells OK — writing resume markers so stage 4 won't retrain AVIRIS"
        for cfg in "${CONFIGS[@]}"; do
            IFS='|' read -r model loss stem <<< "${cfg}"
            for seed in "${SEEDS[@]}"; do
                touch "${OUT_DIR}/.done/AVIRIS__${stem}_seed${seed}"
            done
        done
    else
        log "Stage 1: at least one AVIRIS cell still failing — see ${OUT_DIR}/probes/AVIRIS__LOAD_FAILURE.json" \
            "(shared-setup crash) or the per-cell AVIRIS__*.json 'error'/'traceback' fields. No markers written;" \
            "stage 4 will retrain AVIRIS as a side effect of fixing M3."
    fi
else
    log "===== Stage 1 skipped (--skip-avaris-diag) ====="
fi

# ---------------------------------------------------------------------------
# Stage 2 — M3 --select mse check (no retraining, informational)
# ---------------------------------------------------------------------------
log "===== Stage 2: M3 --select mse check (4 cells, existing checkpoints) ====="
for m in vae-our vae-our-nl; do
    for seed in "${SEEDS[@]}"; do
        out_json="${OUT_DIR}/inference/M3__${m}_seed${seed}_mse_check.json"
        log "  inference --select mse | M3 | ${m} | physics | seed${seed}"
        "${PY}" -u inference/inference.py --model "${m}" --dataset M3 --loss physics \
            --seed "${seed}" --select mse --ckpt-dir "${CKPT_DIR}" \
            --out-json "${out_json}" 2>&1 | tee -a "${LOG}" \
            || log "  [FAIL] M3 | ${m} | physics | seed${seed} --select mse (non-fatal, continuing)"
    done
done
log "Stage 2 done — compare results/inference/M3__{vae-our,vae-our-nl}_seed{67,69}_mse_check.json" \
    "against the --select sam numbers already in results/ablation_table.csv"

# ---------------------------------------------------------------------------
# Stage 3 — immediate stats refresh (CRIMS NaN fix), no retraining
# ---------------------------------------------------------------------------
log "===== Stage 3: refresh probes.csv/stats.csv + send corrected CSVs ====="
"${PY}" -u inference/verdict.py 2>&1 | tee -a "${LOG}" \
    || log "  [FAIL] verdict.py (non-fatal, continuing)"
"${PY}" -u inference/aggregate.py --telegram \
    --caption "post-fix refresh: CRIMS sam_valid stats corrected" 2>&1 | tee -a "${LOG}" \
    || log "  [FAIL] aggregate.py (non-fatal, continuing)"

# ---------------------------------------------------------------------------
# Stage 4 — the real retrain: M3 under patience=10 (and AVIRIS, if stage 1
# didn't clear it)
# ---------------------------------------------------------------------------
if (( DO_RETRAIN == 1 )); then
    log "===== Stage 4: clearing M3 markers/checkpoints, re-running the grid ====="
    rm -f "${OUT_DIR}/.done/M3__"*
    rm -rf "${CKPT_DIR}/M3"
    bash "${SCRIPT_DIR}/run_clean_grid.sh"
    rc=$?
    log "Stage 4: run_clean_grid.sh exited ${rc}"
    exit "${rc}"
else
    log "===== Stage 4 skipped (--diagnostics-only) ====="
fi

log "Done. log: ${LOG}"
