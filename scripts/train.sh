#!/usr/bin/env bash
# scripts/train.sh
# ----------------
# Launch training for the HSI VAE ablation study (model-agnostic train/train.py).
#
# Run from the repo root:
#   # Single run:
#   bash scripts/train.sh --model vae-our --dataset IIRS --loss physics
#
#   # Full 28-run ablation grid (all models x all datasets x loss regimes):
#   bash scripts/train.sh --all
#
#   # Subset — one workstation runs half the grid:
#   bash scripts/train.sh --all --datasets IIRS,M3
#   bash scripts/train.sh --all --datasets AVIRIS,CRIMS
#
#   # Retrain slots that already have a checkpoint:
#   bash scripts/train.sh --all --overwrite
#
# The grid is defined once in scripts/grid_manifest.sh (28 slots).
#
# Environment overrides:
#   CKPT_DIR   — checkpoint root directory   (default: model)
#   EPOCHS     — passed through as --epochs  (default: 100 unless overridden)
#   OVERWRITE  — 1 is equivalent to passing --overwrite
#
# Without --overwrite, a slot whose checkpoint already exists is SKIPPED. That
# skip is printed to stdout only — it never reaches Telegram — which is how a
# previous grid run appeared to lose slots that had in fact just been skipped.
# The summary below now always reports the skip list explicitly.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-model}"
DEFAULT_EPOCHS="${EPOCHS:-100}"

cd "${REPO_ROOT}"

# --------------------------------------------------------------------------
# --all : run the full ablation grid via scripts/grid_manifest.sh
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--all" ]]; then
    shift

    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/grid_manifest.sh"

    # Filter subset (comma-separated).
    DATASETS_SUBSET=""
    EXTRA_ARGS=()
    HAS_EPOCHS=0
    OVERWRITE="${OVERWRITE:-0}"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            # --dataset is accepted as an alias for --datasets: in --all mode the
            # filter takes a comma list, but a single dataset is the common case
            # and typing the singular is the natural thing to do.
            --datasets|--dataset)  DATASETS_SUBSET="$2"; shift 2 ;;
            --epochs)    HAS_EPOCHS=1; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
            --overwrite) OVERWRITE=1; shift ;;
            --allow-concurrent) export ALLOW_CONCURRENT=1; shift ;;
            *)           EXTRA_ARGS+=("$1"); shift ;;
        esac
    done
    if (( HAS_EPOCHS == 0 )); then
        EXTRA_ARGS+=(--epochs "${DEFAULT_EPOCHS}")
    fi

    # Refuse a second concurrent grid on this machine. A no-op when
    # train_fixed.sh already holds the lock (it exports GRID_LOCK_HELD=1).
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/grid_lock.sh"
    acquire_grid_lock "train.sh --all (${DATASETS_SUBSET:-all})" || exit 9

    IFS=',' read -r -a DATASETS_FILTER <<< "${DATASETS_SUBSET}"
    _in_filter() {
        local ds="$1"
        [[ -z "${DATASETS_SUBSET}" ]] && return 0
        for d in "${DATASETS_FILTER[@]}"; do
            [[ "${d}" == "${ds}" ]] && return 0
        done
        return 1
    }

    echo "=============================================="
    echo " HSI VAE Ablation — grid launch"
    echo "  slots     : ${GRID_TOTAL}  (seeds: ${GRID_SEEDS[*]})"
    echo "  datasets  : ${DATASETS_SUBSET:-<all>}"
    echo "  ckpt dir  : ${REPO_ROOT}/${CKPT_DIR}"
    echo "  overwrite : ${OVERWRITE}"
    echo "  extra     : ${EXTRA_ARGS[*]:-<none>}"
    echo "=============================================="

    python utils/notify_cli.py --text "Ablation grid launched
host: $(hostname)
datasets: ${DATASETS_SUBSET:-<all>}
extra: ${EXTRA_ARGS[*]:-<none>}" >/dev/null 2>&1 || true

    SKIPPED=()
    OVERWRITTEN=()
    FAILED=()
    RETRIED=()
    OOMED=()

    for (( slot=1; slot<=GRID_TOTAL; slot++ )); do
        cfg="$(grid_lookup "${slot}")"
        m="${cfg%%|*}"; rest="${cfg#*|}"
        ds="${rest%%|*}"; rest="${rest#*|}"
        loss="${rest%%|*}"; rest="${rest#*|}"
        name="${rest%%|*}"
        seed="${rest#*|}"

        if ! _in_filter "${ds}"; then
            continue
        fi

        # Each cell writes TWO checkpoints (best-SAM and best-recon-MSE); a slot
        # only counts as done when both exist, or a half-finished run would be
        # skipped and silently leave one criterion missing.
        ckpt_sam="${CKPT_DIR}/${ds}/${name}_seed${seed}_bestsam.pt"
        ckpt_mse="${CKPT_DIR}/${ds}/${name}_seed${seed}_bestmse.pt"
        if [[ -s "${ckpt_sam}" && -s "${ckpt_mse}" && "${OVERWRITE}" != "1" ]]; then
            echo "[skip] slot ${slot}  ${m} | ${ds} | ${loss} | seed ${seed}  (ckpts exist)"
            echo "       pass --overwrite (or OVERWRITE=1) to retrain it anyway."
            SKIPPED+=("${m}|${ds}|${loss}|seed${seed}")
            continue
        fi
        if [[ -s "${ckpt_sam}" || -s "${ckpt_mse}" ]]; then
            echo "[overwrite] slot ${slot}  ${m} | ${ds} | ${loss} | seed ${seed}"
            OVERWRITTEN+=("${m}|${ds}|${loss}|seed${seed}")
        fi

        attempt=1
        rc=0
        was_oom=0
        while (( attempt <= 2 )); do
            echo ">>> slot ${slot}/${GRID_TOTAL}  ${m} | ${ds} | ${loss} | seed ${seed}  (attempt ${attempt}/2)"
            slot_log="$(mktemp)"
            if python train/train.py \
                    --model "${m}" --dataset "${ds}" --loss "${loss}" --seed "${seed}" \
                    --ckpt-dir "${CKPT_DIR}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${slot_log}"; then
                if (( attempt > 1 )); then
                    RETRIED+=("${m}|${ds}|${loss}|seed${seed}")
                fi
                rc=0
                rm -f "${slot_log}"
                break
            fi
            rc=${PIPESTATUS[0]}

            # An OOM is NOT retried. Retrying it just burns the same VRAM again
            # and turns one failure into four log entries — which is exactly how
            # the v3 run produced 47 OOM messages from 12 distinct slots. It is a
            # resource verdict, not a flake.
            if grep -qi "OutOfMemoryError\|CUDA out of memory" "${slot_log}"; then
                echo "!!! slot ${slot}  OUT OF MEMORY (exit=${rc}) — not retrying."
                echo "    other processes on the GPU right now:"
                nvidia-smi --query-compute-apps=pid,used_memory,process_name \
                           --format=csv,noheader 2>/dev/null | sed 's/^/      /' \
                    || echo "      (nvidia-smi unavailable)"
                echo "    if another training process is listed, that is the cause."
                OOMED+=("${m}|${ds}|${loss}|seed${seed}")
                was_oom=1
                rm -f "${slot_log}"
                break
            fi

            echo "!!! slot ${slot}  attempt ${attempt} failed (exit=${rc})"
            rm -f "${slot_log}"
            (( attempt < 2 )) && { echo "    retrying after 5s..."; sleep 5; }
            attempt=$(( attempt + 1 ))
        done
        # An OOM is already recorded in OOMED; do not double-count it as a
        # generic failure, or the summary reports one dead cell twice.
        if (( rc != 0 && ! was_oom )); then
            FAILED+=("${m}|${ds}|${loss}|seed${seed}|rc=${rc}")
        fi
    done

    echo ""
    echo "=============================================="
    echo " Ablation grid complete."
    echo "  datasets              : ${DATASETS_SUBSET:-<all>}"
    echo "  skipped (ckpt exists) : ${#SKIPPED[@]}"
    echo "  overwritten           : ${#OVERWRITTEN[@]}"
    echo "  retried (passed on 2) : ${#RETRIED[@]}"
    echo "  out of memory         : ${#OOMED[@]}"
    echo "  failed                : ${#FAILED[@]}"
    if [[ ${#SKIPPED[@]} -gt 0 ]]; then
        echo "  skipped runs (re-run with --overwrite to force):"
        for sk in "${SKIPPED[@]}"; do echo "    - ${sk}"; done
    fi
    if [[ ${#RETRIED[@]} -gt 0 ]]; then
        echo "  retried runs:"
        for r in "${RETRIED[@]}"; do echo "    - ${r}"; done
    fi
    if [[ ${#OOMED[@]} -gt 0 ]]; then
        echo "  OUT OF MEMORY (not retried — check for a concurrent grid):"
        for o in "${OOMED[@]}"; do echo "    - ${o}"; done
    fi
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "  failed runs:"
        for f in "${FAILED[@]}"; do echo "    - ${f}"; done
    fi
    echo "=============================================="

    python utils/notify_cli.py --text "Ablation grid finished
host: $(hostname)
datasets: ${DATASETS_SUBSET:-<all>}
skipped: ${#SKIPPED[@]}  overwritten: ${#OVERWRITTEN[@]}  retried: ${#RETRIED[@]}
oom: ${#OOMED[@]}  failed: ${#FAILED[@]}" >/dev/null 2>&1 || true

    exit $(( ${#FAILED[@]} > 0 ? 1 : 0 ))
fi

# --------------------------------------------------------------------------
# Single run: forward all args straight through to train/train.py.
#
# --overwrite is a grid-level concept (it controls the "skip if a checkpoint
# exists" branch above); a single run always trains and always writes its
# checkpoint, so accept the flag and drop it rather than handing train.py an
# argument it does not define.
# --------------------------------------------------------------------------
SINGLE_ARGS=()
for a in "$@"; do
    [[ "${a}" == "--overwrite" ]] && continue
    SINGLE_ARGS+=("${a}")
done

python train/train.py --ckpt-dir "${CKPT_DIR}" "${SINGLE_ARGS[@]}"
