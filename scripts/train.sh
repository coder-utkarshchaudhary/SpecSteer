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
#   # Subset — one workstation runs half the grid:
#   bash scripts/train.sh --all --datasets IIRS,M3      --epochs 100
#   bash scripts/train.sh --all --datasets AVIRIS,CRIMS --epochs 100
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
# failure does not kill the remaining runs (and each failed run gets one retry).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-model}"

cd "${REPO_ROOT}"

# --------------------------------------------------------------------------
# --all : run the full ablation grid. --datasets IIRS,M3 restricts to a subset.
# Any other extra args (e.g. --epochs 100) are forwarded to every run.
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--all" ]]; then
    shift

    # Default is all four; --datasets replaces it with a comma-separated subset.
    ALL_DATASETS=("IIRS" "M3" "AVIRIS" "CRIMS")
    DATASETS=("${ALL_DATASETS[@]}")

    # Parse the leading options recognised here; anything else falls through
    # to the python train.py forward list.
    EXTRA_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --datasets)
                IFS=',' read -r -a DATASETS <<< "$2"
                shift 2
                ;;
            *)
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done

    # Validate the subset against the known set.
    for ds in "${DATASETS[@]}"; do
        found=0
        for known in "${ALL_DATASETS[@]}"; do
            [[ "${ds}" == "${known}" ]] && { found=1; break; }
        done
        if [[ ${found} -eq 0 ]]; then
            echo "ERROR: unknown dataset '${ds}'. Choose from: ${ALL_DATASETS[*]}"
            exit 2
        fi
    done

    STANDARD_MODELS=("vae-standard" "vae-3d-spatio-spectral" "vae-1d-pixelwise")

    echo "=============================================="
    echo " HSI VAE Ablation — grid launch"
    echo "  datasets : ${DATASETS[*]}"
    echo "  ckpt dir : ${REPO_ROOT}/${CKPT_DIR}"
    echo "  extra    : ${EXTRA_ARGS[*]:-<none>}"
    echo "=============================================="

    SKIPPED=()
    FAILED=()
    RETRIED=()

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

        local attempt=1
        local rc=0
        while (( attempt <= 2 )); do
            echo ">>> ${m} | ${ds} | ${loss}  (attempt ${attempt}/2)"
            if python train/train.py --model "${m}" --dataset "${ds}" --loss "${loss}" \
                    --ckpt-dir "${CKPT_DIR}" "$@"; then
                if (( attempt > 1 )); then
                    RETRIED+=("${m}|${ds}|${loss}")
                fi
                return 0
            fi
            rc=$?
            echo "!!! ${m} | ${ds} | ${loss}  attempt ${attempt} failed (exit=${rc})"
            (( attempt < 2 )) && { echo "    retrying after 5s..."; sleep 5; }
            attempt=$(( attempt + 1 ))
        done
        FAILED+=("${m}|${ds}|${loss}|rc=${rc}")
        return 0
    }

    for ds in "${DATASETS[@]}"; do
        run_one vae-our "${ds}" physics vae-our "${EXTRA_ARGS[@]}"
        for m in "${STANDARD_MODELS[@]}"; do
            for loss in standard physics; do
                run_one "${m}" "${ds}" "${loss}" "${m}_${loss}" "${EXTRA_ARGS[@]}"
            done
        done
    done

    echo ""
    echo "=============================================="
    echo " Ablation grid complete."
    echo "  datasets              : ${DATASETS[*]}"
    echo "  skipped (ckpt exists) : ${#SKIPPED[@]}"
    echo "  retried (passed on 2) : ${#RETRIED[@]}"
    echo "  failed                : ${#FAILED[@]}"
    if [[ ${#RETRIED[@]} -gt 0 ]]; then
        echo "  retried runs:"
        for r in "${RETRIED[@]}"; do echo "    - ${r}"; done
    fi
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
