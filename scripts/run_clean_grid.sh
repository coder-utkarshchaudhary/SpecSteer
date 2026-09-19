#!/usr/bin/env bash
# scripts/run_clean_grid.sh
# --------------------------
# The 2026-09-14 clean-slate grid: 4 datasets x 8 configs x 2 seeds = 64 cells,
# train -> inference -> probes -> downstream per cell, verdict + aggregate (with
# CSVs sent to Telegram as file attachments) once every 16 cells of a dataset.
#
#   models   : vae-our-nl (physics), vae-standard (physics, standard),
#              vae-1d-pixelwise (physics, standard),
#              vae-3d-spatio-spectral (physics, standard), vae-our (physics)
#   datasets : IIRS, AVIRIS, M3, CRIMS   (in that order)
#   seeds    : 67, 69                    (both, every cell)
#   epochs   : 60      patience: 3
#
# Cell order within a dataset: model -> loss (physics before standard) -> seed
# -- i.e. exactly the CONFIGS table below, seeds innermost. This file is
# self-contained: it does NOT source scripts/grid_manifest.sh, whose dataset
# list (no M3), default seed (69) and claim/other seed asymmetry all conflict
# with this protocol.
#
# Run from the repo root:
#   bash scripts/run_clean_grid.sh              # the whole grid, resumable
#   bash scripts/run_clean_grid.sh --dry-run     # print the 64-cell plan, exit
#   bash scripts/run_clean_grid.sh --overwrite   # ignore resume markers, redo everything
#   bash scripts/run_clean_grid.sh --no-telegram
#   bash scripts/run_clean_grid.sh --select mse  # read/report the best-recon-MSE
#                                                 # checkpoints instead (default: sam)
#
# RESUME. A cell (one model|loss|seed on one dataset) is marked done only after
# training AND all three evaluation steps (inference, probes, downstream) have
# ALL succeeded, by touching results/.done/<DS>__<ckpt_stem>_seed<N>. On a plain
# re-run, any cell with that marker is skipped outright; --overwrite ignores
# markers and reruns everything. This is deliberately coarse: a cell that
# trained fine but failed at, say, probes will retrain from scratch on the next
# run rather than resuming mid-cell — simple beats clever for a script meant to
# survive an unattended interruption in a foreground terminal.
#
# ERRORS. Each of the four per-cell steps is checked independently. A training
# failure skips that cell's three evaluation steps (nothing to evaluate) but
# does NOT stop the dataset — the script logs it, moves to the next cell, and
# reports every failure in the end-of-run summary (exit 1 if any occurred).
# EXCEPTION: if a training failure is a VRAM preflight abort (train/train.py
# refusing to start because the GPU already has >4GB in use by another
# process), the SAME failure would repeat on every remaining cell, so the
# script aborts the whole grid immediately with the compute-app list instead.
#
# LOGGING. The terminal shows the real tqdm progress bars (via `tee` to the
# controlling terminal). This script's own consolidated log,
# logs/run_clean_grid_<ts>.log, gets ONE line per real record — tqdm's
# `\r`-framed progress frames are collapsed to their last frame before being
# written, so the driver log does not get inflated by every tqdm tick. Every
# cell ALSO still gets its own full per-run log via utils/logging_setup.py
# (logs/train_*.log etc.), which already writes one line per epoch and needed
# no change — the inflation this script fixes was specific to a previous
# sweep script (run_remote_sweep.sh) that teed raw stdout verbatim.
#
# PRECONDITIONS, not checked at runtime (see the approved plan):
#   - data/packed/<DS>/{train,valid,test}.npy exists for all four datasets.
#   - model/, logs/, results/ have been emptied by hand before this run (this
#     script recreates the directories it needs but never deletes anything).
#   - The GPU is otherwise idle. scripts/grid_lock.sh refuses a second
#     concurrent grid on this machine, but cannot see another job entirely.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

PY="${PY:-python}"   # override e.g. PY=.venv/bin/python if `python` isn't the venv

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

DATASETS=(IIRS AVIRIS M3 CRIMS)
SEEDS=(67 69)

# "<model>|<loss>|<ckpt_stem>" — ckpt_stem matches modules/registry.py's
# checkpoint_name() exactly (physics-only models carry no _<loss> suffix).
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

EPOCHS=60
PATIENCE=3
CKPT_DIR="model"
OUT_DIR="results"

TOTAL_PLANNED=$(( ${#DATASETS[@]} * ${#CONFIGS[@]} * ${#SEEDS[@]} ))

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

OVERWRITE=0
DRY_RUN=0
SEND_TELEGRAM=1
SELECT="sam"

usage() {
    cat <<EOF
Usage: bash scripts/run_clean_grid.sh [--overwrite] [--dry-run] [--no-telegram] [--select sam|mse]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite)   OVERWRITE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --no-telegram) SEND_TELEGRAM=0; shift ;;
        --select)      SELECT="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "run_clean_grid.sh: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "${SELECT}" != "sam" && "${SELECT}" != "mse" ]]; then
    echo "run_clean_grid.sh: --select must be 'sam' or 'mse', got '${SELECT}'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Dry-run: print the plan and exit, no side effects, no lock.
# ---------------------------------------------------------------------------

if (( DRY_RUN == 1 )); then
    echo "Clean grid plan — ${TOTAL_PLANNED} cells (epochs=${EPOCHS} patience=${PATIENCE} select=${SELECT})"
    echo "----------------------------------------------------------------------"
    n=0
    for ds in "${DATASETS[@]}"; do
        for cfg in "${CONFIGS[@]}"; do
            IFS='|' read -r model loss stem <<< "${cfg}"
            for seed in "${SEEDS[@]}"; do
                n=$((n + 1))
                printf "%3d  %-7s %-24s %-9s seed=%s  (%s)\n" \
                    "${n}" "${ds}" "${model}" "${loss}" "${seed}" "${stem}"
            done
        done
    done
    echo "----------------------------------------------------------------------"
    echo "Total: ${n} cells"
    exit 0
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mkdir -p logs model results results/.done results/inference results/probes results/downstream

RUN_TS="$(date +%Y%m%d-%H%M%S)"
DRIVER_LOG="logs/run_clean_grid_${RUN_TS}.log"
touch "${DRIVER_LOG}"

# One grid per machine — see scripts/grid_lock.sh for why (the v3 double-launch
# OOM incident). Held for the life of THIS process since it is acquired here,
# not inside a subshell/pipeline.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/grid_lock.sh"
acquire_grid_lock "clean grid ($(hostname))" || exit 9

VRAM_ABORT=0
VRAM_PATTERN='Refusing to start: only'

log_line() {
    local ts; ts="$(date +%H:%M:%S)"
    echo "${ts} $*"
    echo "${ts} $*" >> "${DRIVER_LOG}"
}

notify() {
    # Best-effort Telegram text ping; a no-op entirely when --no-telegram.
    if (( SEND_TELEGRAM == 1 )); then
        "${PY}" -u utils/notify_cli.py --text "$1" >/dev/null 2>&1 || true
    fi
}

# run_stage <label> <cmd...>
# Streams the command's combined stdout/stderr to the real terminal (so tqdm
# bars animate exactly as if run directly) while collapsing every `\r`-framed
# line down to its last frame before it reaches the driver log. Returns the
# command's own exit code via PIPESTATUS[0], not the filter's.
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
    local msg="GPU already occupied by another process (VRAM preflight refused to "
    msg+="start). This would repeat identically on every remaining cell, so the "
    msg+="grid is aborting now instead of burning through all ${TOTAL_PLANNED}. "
    msg+="Check 'nvidia-smi' on the lab box, wait for the other job or kill it, "
    msg+="then re-run this script — it resumes from where it stopped."
    echo "" >&2
    echo "############################################################" >&2
    echo "# ABORT: ${msg}" >&2
    echo "############################################################" >&2
    log_line "[ABORT] ${msg}"
    notify "[ABORT] Clean grid aborted on $(hostname): ${msg}
See ${DRIVER_LOG}"
    exit 2
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

echo "=============================================="
echo " Clean grid — ${TOTAL_PLANNED} cells"
echo "  datasets : ${DATASETS[*]}"
echo "  seeds    : ${SEEDS[*]}"
echo "  epochs   : ${EPOCHS}   patience: ${PATIENCE}   select: ${SELECT}"
echo "  overwrite: ${OVERWRITE}"
echo "  log      : ${DRIVER_LOG}"
echo "=============================================="
log_line "Clean grid launched on $(hostname): ${TOTAL_PLANNED} cells, datasets=${DATASETS[*]}, seeds=${SEEDS[*]}"
notify "Clean grid launched on $(hostname)
${TOTAL_PLANNED} cells | datasets: ${DATASETS[*]} | seeds: ${SEEDS[*]}
epochs=${EPOCHS} patience=${PATIENCE} select=${SELECT}
log: ${DRIVER_LOG}"

SKIPPED=()
FAILED=()
DONE_COUNT=0

for ds in "${DATASETS[@]}"; do
    log_line "================ DATASET ${ds} ================"

    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r model loss stem <<< "${cfg}"

        for seed in "${SEEDS[@]}"; do
            marker="${OUT_DIR}/.done/${ds}__${stem}_seed${seed}"

            if [[ -f "${marker}" && "${OVERWRITE}" != "1" ]]; then
                SKIPPED+=("${ds}|${model}|${loss}|seed${seed}")
                log_line "[SKIP] ${ds} | ${model} | ${loss} | seed${seed} — already done (marker present)"
                continue
            fi

            log_line ">>>>> CELL  ${ds} | ${model} | ${loss} | seed${seed} <<<<<"
            cell_ok=1

            # --- 1. train ---------------------------------------------------
            run_stage "train  | ${ds} | ${model} | ${loss} | seed${seed}" \
                "${PY}" -u train/train.py \
                    --model "${model}" --dataset "${ds}" --loss "${loss}" \
                    --seed "${seed}" --epochs "${EPOCHS}" --patience "${PATIENCE}" \
                    --ckpt-dir "${CKPT_DIR}"
            train_rc=$?

            if (( VRAM_ABORT == 1 )); then
                abort_on_vram
            fi

            if (( train_rc != 0 )); then
                FAILED+=("train|${model}|${ds}|${loss}|seed${seed}|rc=${train_rc}")
                log_line "[FAIL] training failed (rc=${train_rc}) — skipping inference/probes/downstream for this cell"
                continue
            fi

            # --- 2. inference (recon metrics) --------------------------------
            infer_json="${OUT_DIR}/inference/${ds}__${stem}_seed${seed}_${SELECT}.json"
            run_stage "infer  | ${ds} | ${model} | ${loss} | seed${seed}" \
                "${PY}" -u inference/inference.py \
                    --model "${model}" --dataset "${ds}" --loss "${loss}" \
                    --seed "${seed}" --select "${SELECT}" --ckpt-dir "${CKPT_DIR}" \
                    --out-json "${infer_json}"
            infer_rc=$?
            if (( infer_rc != 0 )); then
                FAILED+=("infer|${model}|${ds}|${loss}|seed${seed}|rc=${infer_rc}")
                cell_ok=0
            fi

            # --- 3. probes (falsification diagnostics) -----------------------
            run_stage "probes | ${ds} | ${model} | ${loss} | seed${seed}" \
                "${PY}" -u inference/probes.py \
                    --dataset "${ds}" --model "${model}" --loss "${loss}" \
                    --seed "${seed}" --select "${SELECT}" --ckpt-dir "${CKPT_DIR}" \
                    --out-dir "${OUT_DIR}/probes"
            probes_rc=$?
            if (( probes_rc != 0 )); then
                FAILED+=("probes|${model}|${ds}|${loss}|seed${seed}|rc=${probes_rc}")
                cell_ok=0
            fi

            # --- 4. downstream (latent noise/interp) --------------------------
            run_stage "downstream | ${ds} | ${model} | ${loss} | seed${seed}" \
                "${PY}" -u inference/downstream.py \
                    --dataset "${ds}" --models "${model}" --loss "${loss}" \
                    --seed "${seed}" --select "${SELECT}" --ckpt-dir "${CKPT_DIR}" \
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
                log_line "[DONE] ${ds} | ${model} | ${loss} | seed${seed}"
            else
                log_line "[PARTIAL] ${ds} | ${model} | ${loss} | seed${seed} — no marker written, will retry in full on next run"
            fi
        done
    done

    # --- end of dataset: verdict + aggregate (CSVs -> Telegram as files) ----
    log_line "================ ${ds} complete — verdict + aggregate ================"

    run_stage "verdict | ${ds}" "${PY}" -u inference/verdict.py
    verdict_rc=$?
    if (( verdict_rc != 0 )); then
        FAILED+=("verdict|-|${ds}|-|-|rc=${verdict_rc}")
        log_line "[FAIL] verdict.py failed for ${ds} (rc=${verdict_rc}) — continuing to aggregate anyway"
    fi

    if (( SEND_TELEGRAM == 1 )); then
        run_stage "aggregate | ${ds}" "${PY}" -u inference/aggregate.py \
            --telegram --caption "cumulative through ${ds}"
    else
        run_stage "aggregate | ${ds}" "${PY}" -u inference/aggregate.py
    fi
    agg_rc=$?
    if (( agg_rc != 0 )); then
        FAILED+=("aggregate|-|${ds}|-|-|rc=${agg_rc}")
        log_line "[FAIL] aggregate.py failed for ${ds} (rc=${agg_rc})"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "=============================================="
echo " Clean grid finished"
echo "  total cells : ${TOTAL_PLANNED}"
echo "  completed   : ${DONE_COUNT}"
echo "  skipped     : ${#SKIPPED[@]}  (already done on entry)"
echo "  failed      : ${#FAILED[@]}"
echo "  log         : ${DRIVER_LOG}"
echo "=============================================="
log_line "Clean grid finished: total=${TOTAL_PLANNED} completed=${DONE_COUNT} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}"

if (( ${#FAILED[@]} > 0 )); then
    echo "Failures (stage|model|dataset|loss|seed|rc):"
    printf '  %s\n' "${FAILED[@]}"
    {
        echo "Failures:"
        printf '  %s\n' "${FAILED[@]}"
    } >> "${DRIVER_LOG}"
fi

notify "Clean grid finished on $(hostname)
total: ${TOTAL_PLANNED}  completed: ${DONE_COUNT}  skipped: ${#SKIPPED[@]}  failed: ${#FAILED[@]}
log: ${DRIVER_LOG}"

exit $(( ${#FAILED[@]} > 0 ? 1 : 0 ))
