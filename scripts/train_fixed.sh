#!/usr/bin/env bash
# scripts/train_fixed.sh
# ----------------------
# One command, start to finish, for the lab machine:
#
#   1. verify each dataset's on-disk band count against utils/config.py
#   2. PACK  — build the capped fp16 dataset in data/packed/  (the slow part,
#              but it happens exactly once and every later run reuses it)
#   3. ZIP   — data/packed/ -> ./dataset.zip for upload to Kaggle
#   4. SMOKE — 2 epochs across all 28 grid slots, with --overwrite
#   5. FULL  — 30 epochs across all 28 grid slots, with --overwrite
#
# Why this exists alongside scripts/train.sh: train.sh assumes the data is
# already packed and just runs the grid. This wraps it with the one-time
# dataset build and the Kaggle zip, and hard-codes the smoke-then-full sequence
# so nothing has to be remembered. Once data/packed/ exists, either script
# works — train.sh --all --overwrite does steps 4/5 on its own.
#
# Usage (from repo root):
#   bash scripts/train_fixed.sh                      # everything, 2 then 30 epochs
#   bash scripts/train_fixed.sh --datasets IIRS,M3   # restrict to some datasets
#   bash scripts/train_fixed.sh --skip-pack          # data/packed/ already built
#   bash scripts/train_fixed.sh --pack-only          # just build + zip, no training
#   bash scripts/train_fixed.sh --skip-smoke         # go straight to the full grid
#   bash scripts/train_fixed.sh --no-zip             # skip the Kaggle zip
#   bash scripts/train_fixed.sh --dry-run            # print the plan, run nothing
#
# Environment overrides:
#   CKPT_DIR      checkpoint root                    (default: model)
#   SMOKE_EPOCHS  epochs for the smoke pass          (default: 2)
#   FULL_EPOCHS   epochs for the full pass           (default: 30)
#   PATCH_CAP     training patches per dataset       (default: settings.train_patch_cap = 7000)
#   ZIP_NAME      Kaggle archive name at repo root   (default: dataset.zip)
#   WANDB_MODE    online|offline                     (default: offline)
#
# The smoke pass writes to ${CKPT_DIR}_smoke/ so it can never be mistaken for a
# real result, and so the full pass does not see its checkpoints and skip.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CKPT_DIR="${CKPT_DIR:-model}"
SMOKE_CKPT_DIR="${CKPT_DIR}_smoke"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
FULL_EPOCHS="${FULL_EPOCHS:-30}"
ZIP_NAME="${ZIP_NAME:-dataset.zip}"
PACKED_ROOT="data/packed"

ORIGINAL_ARGS="$*"

DATASETS_CSV=""
DO_PACK=1
DO_ZIP=1
DO_SMOKE=1
DO_FULL=1
DRY_RUN=0
PACK_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)   DATASETS_CSV="$2"; shift 2 ;;
        --skip-pack)  DO_PACK=0; shift ;;
        --pack-only)  DO_SMOKE=0; DO_FULL=0; shift ;;
        --skip-smoke) DO_SMOKE=0; shift ;;
        --no-zip)     DO_ZIP=0; shift ;;
        --repack)     PACK_ARGS+=(--overwrite); shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --allow-concurrent) export ALLOW_CONCURRENT=1; shift ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1  (see --help)"; exit 2 ;;
    esac
done

# The venv's interpreter is invoked by absolute path, never via
# `source .venv/bin/activate` — same invariant the HPC scripts rely on (an
# activate script carries a hard-coded VIRTUAL_ENV that silently no-ops when the
# venv has been copied from another machine).
PY="${REPO_ROOT}/.venv/bin/python"
[[ -x "${PY}" ]] || PY="$(command -v python3)"

# Datasets to act on, as an array.
if [[ -n "${DATASETS_CSV}" ]]; then
    IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
else
    DATASETS=(IIRS M3 AVIRIS CRIMS)
fi

# --------------------------------------------------------------------------
# Refuse a second concurrent grid. This is THE fix for the v3 failure: two
# launches four minutes apart fought over one 23.4 GB card and produced 47 OOMs.
# Taken before packing, because step 2 writes data/packed/ and step 3 deletes
# and rewrites the zip — two runs would corrupt both.
# --------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/grid_lock.sh"
GRID_LOCK_CMD="$0 ${ORIGINAL_ARGS}"
if (( ! DRY_RUN )); then
    acquire_grid_lock "train_fixed (${DATASETS[*]})" || exit 9
fi

banner() {
    echo ""
    echo "=============================================================="
    echo " $*"
    echo "=============================================================="
}

run() {
    echo "+ $*"
    (( DRY_RUN )) && return 0
    "$@"
}

# Same as run(), but with CKPT_DIR overridden for that command only. An env
# prefix on a shell function call (`CKPT_DIR=x run ...`) is not reliably scoped
# in bash — it can leak into the caller — so the override is done in a subshell.
run_with_ckpt_dir() {
    local dir="$1"; shift
    echo "+ CKPT_DIR=${dir} $*"
    (( DRY_RUN )) && return 0
    ( export CKPT_DIR="${dir}"; "$@" )
}

notify() {
    "${PY}" utils/notify_cli.py --text "$1" >/dev/null 2>&1 || true
}

STARTED_AT=$(date +%s)

banner "train_fixed — HSI VAE ablation, end to end"
echo "  repo         : ${REPO_ROOT}"
echo "  python       : ${PY}"
echo "  datasets     : ${DATASETS[*]}"
echo "  packed root  : ${PACKED_ROOT}"
echo "  patch cap    : ${PATCH_CAP:-<settings.train_patch_cap>}"
echo "  ckpt dir     : ${CKPT_DIR}  (smoke -> ${SMOKE_CKPT_DIR})"
echo "  epochs       : smoke ${SMOKE_EPOCHS}, full ${FULL_EPOCHS}"
echo "  zip          : $( (( DO_ZIP )) && echo "${ZIP_NAME}" || echo '<skipped>' )"
echo "  steps        : pack=${DO_PACK} zip=${DO_ZIP} smoke=${DO_SMOKE} full=${DO_FULL}"
echo "  wandb        : ${WANDB_MODE}"
(( DRY_RUN )) && echo "  MODE         : DRY RUN — nothing will be executed"

# --------------------------------------------------------------------------
# 1. Band-count verification
# --------------------------------------------------------------------------
# Cheap, and it is the check that would have caught CRIMS being configured as
# 544 bands when its patches are 457 — a mismatch that killed all seven CRIMS
# slots of the previous grid before any notification could be sent.
banner "STEP 1/5 — verify band counts against utils/config.py"
if ! run "${PY}" utils/dataset/inspect_channels.py; then
    echo ""
    echo "!!! Band counts disagree with utils/config.py. Fix that before packing —"
    echo "!!! packing a mismatched cube just moves the failure later."
    exit 3
fi

# Notebook parity: cheap, CPU-only, and the failure it catches (a config value
# edited in three notebooks out of four) is otherwise invisible until Kaggle.
if ! run "${PY}" utils/check_notebook_parity.py; then
    echo ""
    echo "!!! Notebooks disagree with the repo config. They are committed and run"
    echo "!!! on Kaggle, so a drift here produces results that silently used"
    echo "!!! different hyper-parameters. Fix before training."
    exit 4
fi

# --------------------------------------------------------------------------
# 2. Pack
# --------------------------------------------------------------------------
banner "STEP 2/5 — pack to fp16 shards (capped training split)"
if (( DO_PACK )); then
    # ${arr[@]+"${arr[@]}"} rather than "${arr[@]}": under `set -u`, bash
    # before 4.4 errors on expanding an empty array.
    PACK_CMD=("${PY}" utils/dataset/pack.py --verify ${PACK_ARGS[@]+"${PACK_ARGS[@]}"})
    for ds in "${DATASETS[@]}"; do
        PACK_CMD+=(--dataset "${ds}")
    done
    [[ -n "${PATCH_CAP:-}" ]] && PACK_CMD+=(--cap "${PATCH_CAP}")

    notify "Packing dataset on $(hostname): ${DATASETS[*]}"
    if ! run "${PACK_CMD[@]}"; then
        echo "!!! packing failed — aborting before training."
        notify "[FAIL] dataset packing failed on $(hostname)"
        exit 4
    fi
    echo ""
    (( DRY_RUN )) || du -sh "${PACKED_ROOT}"/* 2>/dev/null
else
    echo "  --skip-pack: reusing existing ${PACKED_ROOT}/"
    (( DRY_RUN )) || du -sh "${PACKED_ROOT}"/* 2>/dev/null
fi

# --------------------------------------------------------------------------
# 3. Zip for Kaggle
# --------------------------------------------------------------------------
banner "STEP 3/5 — zip ${PACKED_ROOT}/ -> ${ZIP_NAME} (Kaggle upload)"
if (( DO_ZIP )); then
    if [[ ! -d "${PACKED_ROOT}" ]] && (( ! DRY_RUN )); then
        echo "!!! ${PACKED_ROOT}/ does not exist — nothing to zip."
        exit 5
    fi
    # -0 (store, no deflate): the payload is float16 sensor noise and does not
    # compress meaningfully, so deflate would burn a long time on ~50 GB for a
    # percent or two. Storing keeps the archive streamable and fast to build.
    # -r recurse, -q quiet (the per-file list would be tens of thousands of lines
    # for nothing), and the archive is rebuilt from scratch each time.
    run rm -f "${REPO_ROOT}/${ZIP_NAME}"
    if ! run zip -0 -r -q "${REPO_ROOT}/${ZIP_NAME}" "${PACKED_ROOT}"; then
        echo "!!! zip failed (is 'zip' installed? apt install zip)"
        exit 5
    fi
    (( DRY_RUN )) || ls -lh "${REPO_ROOT}/${ZIP_NAME}"
    echo ""
    echo "  Upload ${ZIP_NAME} to Kaggle as a Dataset, then point the notebooks'"
    echo "  DATA_ROOTS at /kaggle/input/<slug>/data/packed/<DATASET>."
    echo "  (Kaggle caps a single dataset at 100 GB / 50 GB zipped — if the"
    echo "   archive is over, re-run with --datasets to zip them one at a time.)"
else
    echo "  --no-zip: skipped"
fi

# --------------------------------------------------------------------------
# 4. Smoke — 2 epochs across the whole grid
# --------------------------------------------------------------------------
# The point is coverage, not convergence: every one of the 28 (model, dataset,
# loss) slots must actually start, train, and write a checkpoint. --overwrite is
# essential here — without it, any leftover checkpoint silently consumes its
# slot and the smoke pass reports success for a run that never happened.
banner "STEP 4/5 — SMOKE: ${SMOKE_EPOCHS} epochs x 28 slots (--overwrite)"
SMOKE_RC=0
if (( DO_SMOKE )); then
    SMOKE_ARGS=(--all --overwrite --epochs "${SMOKE_EPOCHS}")
    [[ -n "${DATASETS_CSV}" ]] && SMOKE_ARGS+=(--datasets "${DATASETS_CSV}")
    notify "Smoke grid starting on $(hostname): ${SMOKE_EPOCHS} epochs x 28 slots"
    run_with_ckpt_dir "${SMOKE_CKPT_DIR}" bash scripts/train.sh "${SMOKE_ARGS[@]}"
    SMOKE_RC=$?
    if (( SMOKE_RC != 0 )); then
        echo ""
        echo "!!! smoke pass reported failures (exit ${SMOKE_RC})."
        echo "!!! Scroll up for the 'failed runs:' list. Not launching the full"
        echo "!!! grid — ${FULL_EPOCHS} epochs x 28 slots is too expensive to"
        echo "!!! start on top of a known-broken slot."
        notify "[FAIL] smoke grid failed on $(hostname) (exit ${SMOKE_RC}) — full grid NOT launched"
        exit 6
    fi
else
    echo "  --skip-smoke: skipped"
fi

# --------------------------------------------------------------------------
# 5. Full grid
# --------------------------------------------------------------------------
banner "STEP 5/5 — FULL: ${FULL_EPOCHS} epochs x 28 slots (--overwrite)"
FULL_RC=0
if (( DO_FULL )); then
    FULL_ARGS=(--all --overwrite --epochs "${FULL_EPOCHS}")
    [[ -n "${DATASETS_CSV}" ]] && FULL_ARGS+=(--datasets "${DATASETS_CSV}")
    notify "Full grid starting on $(hostname): ${FULL_EPOCHS} epochs x 28 slots"
    run bash scripts/train.sh "${FULL_ARGS[@]}"
    FULL_RC=$?
else
    echo "  full grid skipped"
fi

# --------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - STARTED_AT ))
banner "train_fixed complete — $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
echo "  packed data : ${PACKED_ROOT}/"
(( DO_ZIP )) && echo "  kaggle zip  : ${REPO_ROOT}/${ZIP_NAME}"
(( DO_SMOKE )) && echo "  smoke ckpts : ${SMOKE_CKPT_DIR}/  (rc=${SMOKE_RC})"
(( DO_FULL )) && echo "  full ckpts  : ${CKPT_DIR}/  (rc=${FULL_RC})"
echo ""
echo "  Next: bash scripts/inference.sh   # reconstruction metrics on the test split"
if [[ "${WANDB_MODE}" == "offline" ]]; then
    echo "        wandb sync wandb/offline-run-*   # push the offline runs"
fi

notify "train_fixed finished on $(hostname) in $(( ELAPSED / 3600 ))h$(( (ELAPSED % 3600) / 60 ))m (smoke rc=${SMOKE_RC}, full rc=${FULL_RC})"
exit "${FULL_RC}"
