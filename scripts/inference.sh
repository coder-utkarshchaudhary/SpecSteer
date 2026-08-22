#!/usr/bin/env bash
# scripts/inference.sh
# --------------------
# Run the full evaluation sweep after the overnight training grid finishes:
#   1. inference/inference.py on every (model x dataset x loss) checkpoint,
#      dumping metrics JSON under results/inference/.
#   2. inference/probes.py — the FALSIFICATION SUITE, once per dataset (all
#      models internally). Seven probes against thresholds preregistered in
#      inference/preregistration.yaml. Writes results/probes/.
#   3. inference/downstream.py once per dataset (latent noise + interpolation),
#      writing PNGs + JSON under results/downstream/<DATASET>/.
#   4. inference/verdict.py — probes.csv, stats.csv (paired bootstrap +
#      permutation + Holm) and VERDICT.txt, the readable answer to "why does my
#      model beat or get beaten on each dataset".
#   5. inference/aggregate.py for the reconstruction/downstream CSVs and one
#      final Telegram summary.
#
# Run from the repo root:
#   bash scripts/inference.sh
#   bash scripts/inference.sh --datasets IIRS,M3          # subset
#   bash scripts/inference.sh --probes-only               # just the suite + verdict
#   bash scripts/inference.sh --skip-probes               # old behaviour
#   bash scripts/inference.sh --max-patches 0             # use the whole split
#   bash scripts/inference.sh --no-telegram               # skip the summary ping
#
# Environment overrides:
#   CKPT_DIR     — checkpoint root directory   (default: model)
#   OUT_DIR      — results root                (default: results)
#   MAX_PATCHES  — probe sampling cap          (default: from preregistration)
#   PROBE_BATCH  — probe forward chunk size    (default: from preregistration)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-model}"
OUT_DIR="${OUT_DIR:-results}"
INFER_JSON_DIR="${OUT_DIR}/inference"
DOWNSTREAM_DIR="${OUT_DIR}/downstream"
PROBES_DIR="${OUT_DIR}/probes"

cd "${REPO_ROOT}"

ALL_DATASETS=("IIRS" "M3" "AVIRIS" "CRIMS")
DATASETS=("${ALL_DATASETS[@]}")
SEND_TELEGRAM=1
SELECT="${SELECT:-sam}"
SEEDS_CSV="${SEEDS_CSV:-}"
DO_RECON=1
DO_PROBES=1
DO_DOWNSTREAM=1
EXTRA_ARGS=()
PROBE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets|--dataset)
            IFS=',' read -r -a DATASETS <<< "$2"
            shift 2
            ;;
        --probes-only)
            DO_RECON=0; DO_DOWNSTREAM=0; shift ;;
        --skip-probes)
            DO_PROBES=0; shift ;;
        --skip-downstream)
            DO_DOWNSTREAM=0; shift ;;
        --max-patches)
            PROBE_ARGS+=(--max-patches "$2"); shift 2 ;;
        --probe-batch)
            PROBE_ARGS+=(--probe-batch "$2"); shift 2 ;;
        # Which of each cell's TWO checkpoints to evaluate. Every cell writes
        # best-val-SAM and best-val-recon-MSE; a comparison must read the SAME
        # criterion for every model or it is not a comparison.
        --select)
            SELECT="$2"; shift 2 ;;
        # Which training seed(s), comma-separated. Omit to sweep every seed found.
        --seed|--seeds)
            SEEDS_CSV="$2"; shift 2 ;;
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

mkdir -p "${INFER_JSON_DIR}" "${DOWNSTREAM_DIR}" "${PROBES_DIR}"

[[ -n "${MAX_PATCHES:-}" ]] && PROBE_ARGS+=(--max-patches "${MAX_PATCHES}")
[[ -n "${PROBE_BATCH:-}" ]] && PROBE_ARGS+=(--probe-batch "${PROBE_BATCH}")

# Seeds to evaluate. Default: whatever is actually on disk, so a single-seed
# tree and a three-seed tree both work with no flag. An empty entry means "no
# seed in the filename", i.e. a checkpoint from before the seed axis landed.
if [[ -n "${SEEDS_CSV}" ]]; then
    IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
else
    mapfile -t SEEDS < <(python -c 'import sys; from modules.registry import find_seeds; \
s=set();  [s.update(find_seeds(sys.argv[1], sys.argv[2], m, "physics")) for m in \
["vae-our","vae-standard","vae-3d-spatio-spectral","vae-1d-pixelwise"]]; \
print("\n".join(map(str, sorted(s))))' "${CKPT_DIR}" "${DATASETS[0]}" 2>/dev/null)
    [[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=("")
fi
echo "  select    : best-${SELECT} checkpoints"
echo "  seeds     : ${SEEDS[*]:-<unseeded>}"

PREREG="inference/preregistration.yaml"
if (( DO_PROBES )) && [[ ! -s "${PREREG}" ]]; then
    echo "ERROR: ${PREREG} is missing."
    echo "The probes read every threshold from it and will not run without it —"
    echo "that is the point of preregistering the decision rules."
    exit 3
fi

echo "=============================================="
echo " HSI VAE — inference + downstream sweep"
echo "  datasets     : ${DATASETS[*]}"
echo "  ckpt dir     : ${REPO_ROOT}/${CKPT_DIR}"
echo "  out dir      : ${REPO_ROOT}/${OUT_DIR}"
echo "  telegram     : $( ((SEND_TELEGRAM)) && echo yes || echo no )"
echo "  steps        : recon=${DO_RECON} probes=${DO_PROBES} downstream=${DO_DOWNSTREAM}"
if (( DO_PROBES )); then
    echo "  preregistered: $(grep -m1 '^registered_on:' "${PREREG}" | cut -d'"' -f2)"
fi
echo "=============================================="

SKIPPED=()
FAILED=()

run_inference() {
    # $1=model  $2=dataset  $3=loss  $4=ckpt_name
    local m="$1" ds="$2" loss="$3" name="$4" seed="${5:-}"
    local sfx="" ; local seed_args=()
    if [[ -n "${seed}" ]]; then
        sfx="_seed${seed}"; seed_args=(--seed "${seed}")
    fi
    local ckpt="${CKPT_DIR}/${ds}/${name}${sfx}_best${SELECT}.pt"
    [[ -s "${ckpt}" ]] || ckpt="${CKPT_DIR}/${ds}/${name}.pt"   # pre-seed-axis fallback
    local out_json="${INFER_JSON_DIR}/${ds}__${name}${sfx}_${SELECT}.json"
    if [[ ! -s "${ckpt}" ]]; then
        echo "[skip] inference  ${m} | ${ds} | ${loss}  (missing ${ckpt})"
        SKIPPED+=("infer|${m}|${ds}|${loss}${sfx}")
        return 0
    fi
    echo ">>> inference  ${m} | ${ds} | ${loss}${sfx} [${SELECT}]"
    if ! python inference/inference.py --model "${m}" --dataset "${ds}" --loss "${loss}" \
            --ckpt-dir "${CKPT_DIR}" --out-json "${out_json}" \
            --select "${SELECT}" ${seed_args[@]+"${seed_args[@]}"} "${EXTRA_ARGS[@]}"; then
        echo "!!! inference failed: ${m} | ${ds} | ${loss}${sfx}"
        FAILED+=("infer|${m}|${ds}|${loss}${sfx}")
    fi
}

# ---- Step 1: 28-cell reconstruction sweep ---------------------------------
if (( DO_RECON )); then
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_inference vae-our "${ds}" physics vae-our "${seed}"
    for m in "${STANDARD_MODELS[@]}"; do
        for loss in standard physics; do
            run_inference "${m}" "${ds}" "${loss}" "${m}_${loss}" "${seed}"
        done
    done
  done
done
fi

# ---- Step 2: falsification suite (one call per dataset, all models) -------
# One call per dataset rather than per cell: the trivial-predictor floors and
# the 1000-draw random null are model-independent and get computed once and
# reused across that dataset's seven cells.
if (( DO_PROBES )); then
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    probe_seed_args=(); [[ -n "${seed}" ]] && probe_seed_args=(--seed "${seed}")
    echo ""
    echo ">>> probes ${ds}${seed:+ seed ${seed}} [${SELECT}]"
    if ! python inference/probes.py --dataset "${ds}" --all-models \
            --ckpt-dir "${CKPT_DIR}" --out-dir "${PROBES_DIR}" \
            --select "${SELECT}" ${probe_seed_args[@]+"${probe_seed_args[@]}"} \
            ${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"}; then
        echo "!!! probes failed for ${ds}${seed:+ seed ${seed}}"
        FAILED+=("probes|${ds}|seed${seed}")
    fi
  done
done
fi

# ---- Step 3: downstream (one call per dataset) ----------------------------
if (( DO_DOWNSTREAM )); then
for ds in "${DATASETS[@]}"; do
    echo ">>> downstream ${ds}"
    ds_seed_args=(); [[ -n "${SEEDS[0]}" ]] && ds_seed_args=(--seed "${SEEDS[0]}")
    if ! python inference/downstream.py --dataset "${ds}" --save-plots \
            --ckpt-dir "${CKPT_DIR}" --out-dir "${DOWNSTREAM_DIR}" \
            --select "${SELECT}" ${ds_seed_args[@]+"${ds_seed_args[@]}"}; then
        echo "!!! downstream failed for ${ds}"
        FAILED+=("downstream|${ds}")
    fi
done
fi

# ---- Step 4: verdict — probes.csv, stats.csv, VERDICT.txt -----------------
if (( DO_PROBES )); then
    echo ""
    echo ">>> verdict"
    if ! python inference/verdict.py --probes-dir "${PROBES_DIR}" \
            --out-dir "${OUT_DIR}"; then
        echo "!!! verdict aggregation failed"
        FAILED+=("verdict")
    fi
fi

# ---- Step 5: aggregate + single summary ping ------------------------------
AGG_ARGS=(--inference-dir "${INFER_JSON_DIR}" --downstream-dir "${DOWNSTREAM_DIR}" --out-dir "${OUT_DIR}")
if (( SEND_TELEGRAM )); then
    AGG_ARGS+=(--telegram)
fi
if (( DO_RECON )) || (( DO_DOWNSTREAM )); then
    python inference/aggregate.py "${AGG_ARGS[@]}"
fi

echo ""
echo "=============================================="
echo " Sweep complete."
echo "  skipped : ${#SKIPPED[@]}"
echo "  failed  : ${#FAILED[@]}"
if (( DO_PROBES )); then
    echo ""
    echo "  Results:"
    echo "    ${OUT_DIR}/VERDICT.txt   <- read this one"
    echo "    ${OUT_DIR}/probes.csv    per-cell probe metrics + verdicts"
    echo "    ${OUT_DIR}/stats.csv     pairwise CIs, p-values, effect sizes"
fi
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
