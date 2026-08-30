#!/usr/bin/env bash
# scripts/inference_smoke.sh
# --------------------------
# Drive the REAL scripts/inference.sh end to end on synthetic fixtures, on CPU,
# in a scratch directory. Run this before the real ~2-3 h sweep to catch a
# staging or plumbing problem in a minute instead of two hours.
#
# It builds fake packed shards and fake checkpoints (through the exact training
# path — see scripts/_smoke_fixtures.py), points inference.sh at them via
# CKPT_DIR / PACKED_ROOT / OUT_DIR, and runs with --max-patches 8 --probe-batch 1
# and CUDA_VISIBLE_DEVICES="" so GPU memory is never a factor. NOTHING is written
# under data/packed/ or model/.
#
#   bash scripts/inference_smoke.sh
#   SMOKE_DATASETS=IIRS bash scripts/inference_smoke.sh     # one dataset, faster
#   SMOKE_SEEDS=42,7,1234 bash scripts/inference_smoke.sh   # full seed axis
#   KEEP=1 bash scripts/inference_smoke.sh                  # keep the scratch dir
#
# Exit 0 = the pipeline ran to completion and produced every expected artifact.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# shellcheck source=scripts/grid_manifest.sh
source "${SCRIPT_DIR}/grid_manifest.sh"

PY="${PY:-.venv/bin/python}"
[[ -x "${PY}" ]] || PY="python"

SMOKE_ROOT="${SMOKE_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/prism_infer_smoke.XXXXXX")}"
SMOKE_DATASETS="${SMOKE_DATASETS:-$(IFS=,; echo "${GRID_DATASETS[*]}")}"
SMOKE_SEEDS="${SMOKE_SEEDS:-42}"
KEEP="${KEEP:-0}"

CKPT_DIR="${SMOKE_ROOT}/model"
PACKED_ROOT="${SMOKE_ROOT}/packed"
OUT_DIR="${SMOKE_ROOT}/results"

cleanup() {
    if [[ "${KEEP}" == "1" ]]; then
        echo "scratch kept: ${SMOKE_ROOT}"
    else
        rm -rf "${SMOKE_ROOT}"
    fi
}
trap cleanup EXIT

echo "=============================================="
echo " inference.sh SMOKE"
echo "  scratch   : ${SMOKE_ROOT}"
echo "  datasets  : ${SMOKE_DATASETS}"
echo "  seeds     : ${SMOKE_SEEDS}"
echo "=============================================="

"${PY}" scripts/_smoke_fixtures.py \
    --root "${SMOKE_ROOT}" --datasets "${SMOKE_DATASETS}" --seeds "${SMOKE_SEEDS}" \
    || { echo "FAIL: fixture build"; exit 1; }

echo ""
echo ">>> running scripts/inference.sh against the fixtures"
CUDA_VISIBLE_DEVICES="" \
CKPT_DIR="${CKPT_DIR}" OUT_DIR="${OUT_DIR}" PACKED_ROOT="${PACKED_ROOT}" \
SELECT=sam SEEDS_CSV="${SMOKE_SEEDS}" \
    bash "${SCRIPT_DIR}/inference.sh" \
        --datasets "${SMOKE_DATASETS}" --no-telegram \
        --max-patches 8 --probe-batch 1 --n-random-draws 16
rc=$?

echo ""
echo ">>> checking expected artifacts"
missing=0
nonempty() { [[ -s "$1" ]] || { echo "  MISSING/empty: $1"; missing=1; }; }
exists()   { [[ -f "$1" ]] || { echo "  MISSING: $1"; missing=1; }; }

nonempty "${OUT_DIR}/ablation_table.csv"
nonempty "${OUT_DIR}/downstream_table.csv"
nonempty "${OUT_DIR}/probes.csv"
# stats.csv is legitimately empty here: every synthetic cell is INVALID (random
# weights cannot beat a mean predictor), so there are 0 pairwise comparisons.
# The real run has valid cells. Require only that the artifact exists.
exists   "${OUT_DIR}/stats.csv"
nonempty "${OUT_DIR}/VERDICT.txt"
first_seed="$(echo "${SMOKE_SEEDS}" | cut -d, -f1)"
IFS=',' read -r -a _dl <<< "${SMOKE_DATASETS}"
for ds in "${_dl[@]}"; do
    nonempty "${OUT_DIR}/downstream/${ds}/downstream_results.json"
    nonempty "${OUT_DIR}/downstream/${ds}/noise_robustness_psnr.png"
    nonempty "${OUT_DIR}/inference/${ds}__vae-our_seed${first_seed}_sam.json"
done

# probes.csv must carry the seed/select columns the verdict step now needs (C3).
hdr="$(head -1 "${OUT_DIR}/probes.csv" 2>/dev/null)"
for col in seed select; do
    echo "${hdr}" | grep -q "${col}" || { echo "  probes.csv has no '${col}' column"; missing=1; }
done
# ablation_table.csv: one row per evaluated cell (7 per dataset x seed for the
# first seed, 4 physics-only for any extra seed).
n_ds=${#_dl[@]}
n_seed=$(echo "${SMOKE_SEEDS}" | tr ',' '\n' | grep -c .)
want=$(( n_ds * (7 + (n_seed - 1) * 4) ))
got=$(($(wc -l < "${OUT_DIR}/ablation_table.csv") - 1))
[[ ${got} -eq ${want} ]] || { echo "  ablation_table.csv: ${got} rows, expected ${want}"; missing=1; }

echo ""
if [[ ${rc} -eq 0 && ${missing} -eq 0 ]]; then
    echo "SMOKE PASSED — inference.sh ran to completion and produced every artifact."
    echo "  ablation_table.csv: ${got} rows"
    exit 0
fi
echo "SMOKE FAILED (inference.sh rc=${rc}, missing artifacts=${missing})"
echo "  scratch left for inspection: ${SMOKE_ROOT}"
KEEP=1
exit 1
