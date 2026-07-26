#!/usr/bin/env bash
# scripts/inference.sh
# --------------------
# Run the full evaluation sweep after the overnight training grid finishes:
#   1. inference/inference.py on every (model x dataset x loss) checkpoint,
#      dumping metrics JSON under results/inference/.
#   2. inference/downstream.py once per dataset (all four models internally),
#      writing PNGs + JSON under results/downstream/<DATASET>/.
#   3. inference/aggregate.py to roll everything into two CSVs and send one
#      final Telegram summary.
#
# Run from the repo root:
#   bash scripts/inference.sh
#   bash scripts/inference.sh --datasets IIRS,M3          # subset
#   bash scripts/inference.sh --no-telegram               # skip the summary ping
#
# Environment overrides:
#   CKPT_DIR   — checkpoint root directory   (default: model)
#   OUT_DIR    — results root                (default: results)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-model}"
OUT_DIR="${OUT_DIR:-results}"
INFER_JSON_DIR="${OUT_DIR}/inference"
DOWNSTREAM_DIR="${OUT_DIR}/downstream"

cd "${REPO_ROOT}"

ALL_DATASETS=("IIRS" "M3" "AVIRIS" "CRIMS")
DATASETS=("${ALL_DATASETS[@]}")
SEND_TELEGRAM=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            IFS=',' read -r -a DATASETS <<< "$2"
            shift 2
            ;;
        --no-telegram)
            SEND_TELEGRAM=0
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

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

mkdir -p "${INFER_JSON_DIR}" "${DOWNSTREAM_DIR}"

echo "=============================================="
echo " HSI VAE — inference + downstream sweep"
echo "  datasets     : ${DATASETS[*]}"
echo "  ckpt dir     : ${REPO_ROOT}/${CKPT_DIR}"
echo "  out dir      : ${REPO_ROOT}/${OUT_DIR}"
echo "  telegram     : $( ((SEND_TELEGRAM)) && echo yes || echo no )"
echo "=============================================="

SKIPPED=()
FAILED=()

run_inference() {
    # $1=model  $2=dataset  $3=loss  $4=ckpt_name
    local m="$1" ds="$2" loss="$3" name="$4"
    local ckpt="${CKPT_DIR}/${ds}/${name}.pt"
    local out_json="${INFER_JSON_DIR}/${ds}__${name}.json"
    if [[ ! -s "${ckpt}" ]]; then
        echo "[skip] inference  ${m} | ${ds} | ${loss}  (missing ${ckpt})"
        SKIPPED+=("infer|${m}|${ds}|${loss}")
        return 0
    fi
    echo ">>> inference  ${m} | ${ds} | ${loss}"
    if ! python inference/inference.py --model "${m}" --dataset "${ds}" --loss "${loss}" \
            --ckpt-dir "${CKPT_DIR}" --out-json "${out_json}" "${EXTRA_ARGS[@]}"; then
        echo "!!! inference failed: ${m} | ${ds} | ${loss}"
        FAILED+=("infer|${m}|${ds}|${loss}")
    fi
}

# ---- Step 1: 28-cell reconstruction sweep ---------------------------------
for ds in "${DATASETS[@]}"; do
    run_inference vae-our "${ds}" physics vae-our
    for m in "${STANDARD_MODELS[@]}"; do
        for loss in standard physics; do
            run_inference "${m}" "${ds}" "${loss}" "${m}_${loss}"
        done
    done
done

# ---- Step 2: downstream (one call per dataset) ----------------------------
for ds in "${DATASETS[@]}"; do
    echo ">>> downstream ${ds}"
    if ! python inference/downstream.py --dataset "${ds}" --save-plots \
            --ckpt-dir "${CKPT_DIR}" --out-dir "${DOWNSTREAM_DIR}"; then
        echo "!!! downstream failed for ${ds}"
        FAILED+=("downstream|${ds}")
    fi
done

# ---- Step 3: aggregate + single summary ping ------------------------------
AGG_ARGS=(--inference-dir "${INFER_JSON_DIR}" --downstream-dir "${DOWNSTREAM_DIR}" --out-dir "${OUT_DIR}")
if (( SEND_TELEGRAM )); then
    AGG_ARGS+=(--telegram)
fi
python inference/aggregate.py "${AGG_ARGS[@]}"

echo ""
echo "=============================================="
echo " Sweep complete."
echo "  skipped : ${#SKIPPED[@]}"
echo "  failed  : ${#FAILED[@]}"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "  skipped runs:"
    for s in "${SKIPPED[@]}"; do echo "    - ${s}"; done
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  failed runs:"
    for f in "${FAILED[@]}"; do echo "    - ${f}"; done
fi
echo "=============================================="
exit $(( ${#FAILED[@]} > 0 ? 1 : 0 ))
