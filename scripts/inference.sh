#!/usr/bin/env bash
# scripts/inference.sh
# --------------------
# Run the full evaluation sweep after the overnight training grid finishes:
#   1. inference/inference.py on every (model x dataset x loss) checkpoint,
#      dumping metrics JSON under results/inference/.
#   2. inference/probes.py — mechanism DIAGNOSTICS (P2 latent rate, P3
#      collapse, P4 spatial reliance + sam_valid), once per dataset (all
#      models internally). No pass/fail adjudication — the falsification
#      layer was demoted 2026-09-04 (docs/new_plan.md). Writes results/probes/.
#   3. inference/downstream.py once per dataset (latent noise + interpolation),
#      writing PNGs + JSON under results/downstream/<DATASET>/.
#   4. inference/verdict.py — probes.csv, stats.csv (paired bootstrap +
#      permutation + Holm) and DIAGNOSTICS.txt, the readable per-dataset
#      summary (rate audit, collapse exclusions, SRI, pairwise SAMv).
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
#   PACKED_ROOT  — packed-shard root           (default: data/packed)
#   MAX_PATCHES  — probe sampling cap          (default: from preregistration)
#   PROBE_BATCH  — probe forward chunk size    (default: from preregistration)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-model}"
OUT_DIR="${OUT_DIR:-results}"
PACKED_ROOT="${PACKED_ROOT:-data/packed}"
INFER_JSON_DIR="${OUT_DIR}/inference"
DOWNSTREAM_DIR="${OUT_DIR}/downstream"
PROBES_DIR="${OUT_DIR}/probes"

cd "${REPO_ROOT}"

# One source of truth for which datasets the grid covers — the grid manifest.
# ALL_DATASETS used to be a second hard-coded list here and it drifted (M3 was
# dropped from the manifest but not from here).
# shellcheck source=scripts/grid_manifest.sh
source "${SCRIPT_DIR}/grid_manifest.sh"
ALL_DATASETS=("${GRID_DATASETS[@]}")
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
        --n-random-draws)
            PROBE_ARGS+=(--n-random-draws "$2"); shift 2 ;;
        --packed-root)
            PACKED_ROOT="$2"; shift 2 ;;
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

# ---- Packed-shard preflight ---------------------------------------------------
# Training ran on data/packed/<DS>/{train,valid,test}.npy and the requirement is
# that evaluation runs on the same test.npy patches. build_dataset() does NOT
# fail on a missing shard — it logs a WARNING and silently falls back to the
# legacy per-patch tree, which is a different (and far slower) code path. Make
# the shard a hard precondition so that can't happen unnoticed.
#   - test.npy  : the split every step evaluates on
#   - train.npy : kept as a precondition for provenance/debugging even though
#                 the P1/P5 probes that consumed it were removed (2026-09-04)
MISSING_SHARDS=()
for ds in "${DATASETS[@]}"; do
    for split in train test; do
        shard="${PACKED_ROOT}/${ds}/${split}.npy"
        [[ -s "${shard}" ]] || MISSING_SHARDS+=("${shard}")
    done
done
if [[ ${#MISSING_SHARDS[@]} -gt 0 ]]; then
    echo "ERROR: packed shard(s) missing — evaluation must run on test.npy patches,"
    echo "       not the legacy per-patch fallback. Missing:"
    for s in "${MISSING_SHARDS[@]}"; do echo "         ${s}"; done
    echo ""
    echo "  Build them with:"
    echo "    PYTHONPATH=. python utils/dataset/pack.py --verify"
    echo "  or point PACKED_ROOT at a staged copy."
    exit 4
fi

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

if (( SEND_TELEGRAM )); then
    python utils/notify_cli.py --text "Inference sweep started
host: $(hostname)
datasets: ${DATASETS[*]}
select: best-${SELECT}
seeds: ${SEEDS[*]:-<unseeded>}
steps: recon=${DO_RECON} probes=${DO_PROBES} downstream=${DO_DOWNSTREAM}" >/dev/null 2>&1 || true
fi

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
    # Pre-seed-axis fallback (<name>.pt) ONLY when no seed was requested. A seeded
    # request that misses must NOT silently load an unseeded checkpoint from an
    # older grid — and it must not pass this guard on the unseeded file only to
    # fail later inside inference.py, which re-resolves from --seed independently.
    if [[ ! -s "${ckpt}" && -z "${seed}" ]]; then
        ckpt="${CKPT_DIR}/${ds}/${name}.pt"
    fi
    local out_json="${INFER_JSON_DIR}/${ds}__${name}${sfx}_${SELECT}.json"
    if [[ ! -s "${ckpt}" ]]; then
        echo "[skip] inference  ${m} | ${ds} | ${loss}${sfx}  (missing ${ckpt})"
        SKIPPED+=("infer|${m}|${ds}|${loss}${sfx}")
        return 0
    fi
    echo ">>> inference  ${m} | ${ds} | ${loss}${sfx} [${SELECT}]"
    # Pass the resolved path explicitly so inference.py evaluates this exact file
    # rather than re-deriving one from --ckpt-dir/--seed/--select (which can and
    # did diverge from the shell's choice).
    if ! python inference/inference.py --model "${m}" --dataset "${ds}" --loss "${loss}" \
            --ckpt "${ckpt}" \
            --ckpt-dir "${CKPT_DIR}" --packed-root "${PACKED_ROOT}/${ds}" --out-json "${out_json}" \
            --select "${SELECT}" ${seed_args[@]+"${seed_args[@]}"} "${EXTRA_ARGS[@]}"; then
        echo "!!! inference failed: ${m} | ${ds} | ${loss}${sfx}"
        FAILED+=("infer|${m}|${ds}|${loss}${sfx}")
    fi
}

# ---- Step 1: reconstruction sweep ---------------------------------------------
# The grid manifest trains the standard-loss baselines at the FIRST seed only
# (the claim is about the physics regime); the physics cells get every seed. So
# the standard-loss rows here run only for GRID_SEEDS[0], and every seed runs physics.
if (( DO_RECON )); then
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_inference vae-our "${ds}" physics vae-our "${seed}"
    for m in "${STANDARD_MODELS[@]}"; do
        run_inference "${m}" "${ds}" physics "${m}_physics" "${seed}"
        if [[ "${seed}" == "${GRID_SEEDS[0]}" ]]; then
            run_inference "${m}" "${ds}" standard "${m}_standard" "${seed}"
        fi
    done
  done
done
fi

# ---- Step 2: mechanism diagnostics (one call per dataset x seed, all models) --
# P2 latent-rate audit, P3 collapse detection, P4 spatial reliance + sam_valid.
# For seeds after the first, only the physics cells exist, so --losses physics.
if (( DO_PROBES )); then
for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    probe_seed_args=(); [[ -n "${seed}" ]] && probe_seed_args=(--seed "${seed}")
    probe_loss_args=()
    [[ -n "${seed}" && "${seed}" != "${GRID_SEEDS[0]}" ]] && probe_loss_args=(--losses physics)
    echo ""
    echo ">>> probes ${ds}${seed:+ seed ${seed}} [${SELECT}]"
    if ! python inference/probes.py --dataset "${ds}" --all-models \
            --ckpt-dir "${CKPT_DIR}" --packed-root "${PACKED_ROOT}/${ds}" \
            --out-dir "${PROBES_DIR}" \
            --select "${SELECT}" ${probe_seed_args[@]+"${probe_seed_args[@]}"} \
            ${probe_loss_args[@]+"${probe_loss_args[@]}"} \
            ${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"}; then
        echo "!!! probes failed for ${ds}${seed:+ seed ${seed}}"
        FAILED+=("probes|${ds}|seed${seed}")
    fi
  done
done
fi

# ---- Step 3: downstream (one call per dataset, first seed) -------------------
if (( DO_DOWNSTREAM )); then
for ds in "${DATASETS[@]}"; do
    echo ">>> downstream ${ds}"
    ds_seed_args=(); [[ -n "${GRID_SEEDS[0]}" ]] && ds_seed_args=(--seed "${GRID_SEEDS[0]}")
    if ! python inference/downstream.py --dataset "${ds}" --save-plots \
            --ckpt-dir "${CKPT_DIR}" --packed-root "${PACKED_ROOT}/${ds}" \
            --out-dir "${DOWNSTREAM_DIR}" \
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
    echo "    ${OUT_DIR}/DIAGNOSTICS.txt   <- read this one"
    echo "    ${OUT_DIR}/probes.csv        per-cell diagnostics (P2/P3/P4 + SAMv)"
    echo "    ${OUT_DIR}/stats.csv         pairwise CIs, p-values, effect sizes"
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

if (( SEND_TELEGRAM )); then
    python utils/notify_cli.py --text "Inference sweep finished
host: $(hostname)
datasets: ${DATASETS[*]}
skipped: ${#SKIPPED[@]}
failed: ${#FAILED[@]}" >/dev/null 2>&1 || true
fi

exit $(( ${#FAILED[@]} > 0 ? 1 : 0 ))
