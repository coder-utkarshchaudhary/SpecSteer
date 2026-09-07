#!/usr/bin/env bash
# scripts/run_entire_grid.sh
# --------------------------
# The full post-fix ablation grid + evaluation sweep.
#
# Launch ONLY after `bash scripts/run.sh` (the post-fix smoke) passes the
# PASS conditions in docs/arch_fix_and_rerun.md §3.
#
#   bash scripts/run_entire_grid.sh
#
# What it does:
#   1. Trains the grid at TWO seeds {69, 67}, BOTH the physics and standard
#      loss arms, all 3 datasets. train.sh gives every config the seeds in
#      GRID_SEEDS, so one single-seed pass runs both arms at that seed; two
#      passes give both arms at both seeds.
#        3 datasets x 7 configs x 2 seeds = 42 cells.
#   2. Runs the evaluation sweep once per seed (GRID_SEEDS=<seed> makes the
#      standard-arm gate in inference.sh fire for that seed), for both
#      checkpoint selections (--select sam and --select mse).
#
# Fresh dirs by default so the broken model_iclr/ run from scripts/run.sh is
# neither reused nor destroyed:
#   checkpoints -> model_fix/<DS>/
#   results     -> results_fix/
#
# Resumable: a re-run skips any cell whose two checkpoints already exist.
# Fix the cause of any crash and re-run the SAME command.
#
# Environment overrides:
#   CKPT_DIR   checkpoint root      (default: model_fix)
#   OUT_DIR    results root         (default: results_fix)
#   SEEDS      space-separated list (default: "69 67")

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CKPT_DIR="${CKPT_DIR:-model_fix}"
OUT_DIR="${OUT_DIR:-results_fix}"
SEEDS="${SEEDS:-69 67}"

mkdir -p logs "${CKPT_DIR}" "${OUT_DIR}"

echo "=============================================================="
echo " PRISM — full grid + eval sweep"
echo "   seeds      : ${SEEDS}   (both physics and standard arms)"
echo "   ckpt dir   : ${REPO_ROOT}/${CKPT_DIR}"
echo "   out dir    : ${REPO_ROOT}/${OUT_DIR}"
echo "   started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

python utils/notify_cli.py --text "PRISM full grid launched
host: $(hostname)
seeds: ${SEEDS} (physics + standard)
ckpt: ${CKPT_DIR}" >/dev/null 2>&1 || true

# ---- fairness invariant (cheap, catches config drift) -----------------------
python utils/match_latent_rate.py --exact --check || {
    echo "!!! latent-rate check FAILED — the grid is not fair. Aborting."
    exit 2
}

# ---- 1. training grid: one pass per seed -----------------------------------
# GRID_SEEDS=<seed> => train.sh runs EVERY config (claim + standard) at <seed>.
for seed in ${SEEDS}; do
    echo ""
    echo ">>> TRAIN GRID — seed ${seed}  ($(date '+%H:%M:%S'))"
    GRID_SEEDS="${seed}" CKPT_DIR="${CKPT_DIR}" \
        bash scripts/train.sh --all
    rc=$?
    echo ">>> train grid seed ${seed} exit ${rc}"
done

# ---- 2. evaluation sweep: one pass per seed, both selections --------------
# GRID_SEEDS=<seed> makes inference.sh's standard-arm gate (== GRID_SEEDS[0])
# fire for this seed, so standard cells are evaluated at every seed too.
for seed in ${SEEDS}; do
    echo ""
    echo ">>> EVAL — seed ${seed}, select sam  ($(date '+%H:%M:%S'))"
    GRID_SEEDS="${seed}" CKPT_DIR="${CKPT_DIR}" OUT_DIR="${OUT_DIR}" \
        bash scripts/inference.sh --seeds "${seed}" --select sam --no-telegram

    echo ""
    echo ">>> EVAL — seed ${seed}, select mse (recon only)  ($(date '+%H:%M:%S'))"
    GRID_SEEDS="${seed}" CKPT_DIR="${CKPT_DIR}" OUT_DIR="${OUT_DIR}" \
        bash scripts/inference.sh --seeds "${seed}" --select mse \
            --skip-probes --skip-downstream --no-telegram
done

echo ""
echo "=============================================================="
echo " Full grid + eval sweep complete.  $(date '+%Y-%m-%d %H:%M:%S')"
echo "   read: ${OUT_DIR}/DIAGNOSTICS.txt"
echo "         ${OUT_DIR}/ablation_table.csv"
echo "         ${OUT_DIR}/stats.csv"
echo "=============================================================="

python utils/notify_cli.py --text "PRISM full grid + eval FINISHED
host: $(hostname)
seeds: ${SEEDS}
results: ${OUT_DIR}/" >/dev/null 2>&1 || true
