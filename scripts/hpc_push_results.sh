#!/usr/bin/env bash
# scripts/hpc_push_results.sh <array_idx> <rc> <ckpt_root> <dataset> <ckpt_name>
# --------------------------------------------------------------------------
# Runs ON THE COMPUTE NODE, called from scripts/hpc_pbs_job.pbs after a
# training run's exit code has been captured. Best-effort: pushes this
# slot's checkpoint + logs + wandb offline run back to the LOGIN node
# (compute -> login is the direction that always works over PBS scheduling
# networks — never the reverse). Never affects the caller's exit status.
#
# Two-tier delivery:
#   1. A `logs/pending_push/<idx>` marker is written UNCONDITIONALLY and
#      FIRST — before any transfer is attempted — so the login-node
#      collector (scripts/hpc_collector.sh) is the single source of truth
#      for "this slot still needs pulling", regardless of whether the push
#      below succeeds, partially succeeds, or is skipped entirely.
#   2. If PUSH_RESULTS_FROM_JOB=1 (only set when preflight probe 6 confirmed
#      the compute node can reach the login node), attempt the push. On
#      success the marker is removed immediately — the collector never has
#      to touch this slot. On failure the marker is left for the collector.
#
# Env expected (passed in via `qsub -v` on hpc_pbs_job.pbs, NOT read from a
# config file — the compute node has no hpc_config.env):
#   HPC_COMPUTE_REPO_ROOT   cwd of the running job (falls back to $PWD)
#   HPC_LOGIN_REPO_ROOT     where to push to on the login node
#   HPC_USER / HPC_HOST     login node ssh target
#   PUSH_RESULTS_FROM_JOB   "1" to attempt the push; default "0" (marker-only)

set -uo pipefail

ARRAY_IDX="${1:?usage: hpc_push_results.sh <array_idx> <rc> <ckpt_root> <dataset> <ckpt_name>}"
RC="${2:-?}"
CKPT_ROOT="${3:-model}"
DATASET="${4:-}"
CKPT_NAME="${5:-}"

COMPUTE_ROOT="${HPC_COMPUTE_REPO_ROOT:-${PBS_O_WORKDIR:-$(pwd)}}"
cd "${COMPUTE_ROOT}" || exit 0

mkdir -p logs/pending_push
: > "logs/pending_push/${ARRAY_IDX}"   # unconditional, first — see header

_notify() {
    local text="$1"
    if [[ -x "${COMPUTE_ROOT}/.venv/bin/python" ]]; then
        PYTHONPATH="${COMPUTE_ROOT}" "${COMPUTE_ROOT}/.venv/bin/python" \
            "${COMPUTE_ROOT}/utils/notify_cli.py" --text "${text}" >/dev/null 2>&1 || true
    fi
}

PUSH_RESULTS_FROM_JOB="${PUSH_RESULTS_FROM_JOB:-0}"
if [[ "${PUSH_RESULTS_FROM_JOB}" != "1" ]]; then
    echo "hpc_push_results: PUSH_RESULTS_FROM_JOB!=1 — leaving marker for the login-node collector."
    exit 0
fi

if [[ -z "${HPC_USER:-}" || -z "${HPC_HOST:-}" || -z "${HPC_LOGIN_REPO_ROOT:-}" ]]; then
    echo "hpc_push_results: HPC_USER/HPC_HOST/HPC_LOGIN_REPO_ROOT not set — cannot push; leaving marker."
    exit 0
fi

RSYNC_OPTS=(-a --info=progress2 --partial --compress --timeout=20)
LOGIN_DEST="${HPC_USER}@${HPC_HOST}:${HPC_LOGIN_REPO_ROOT}"

t0=$(date +%s)
ok=1

if [[ -n "${DATASET}" && -n "${CKPT_NAME}" && -s "${CKPT_ROOT}/${DATASET}/${CKPT_NAME}.pt" ]]; then
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" "mkdir -p '${HPC_LOGIN_REPO_ROOT}/${CKPT_ROOT}/${DATASET}'" 2>/dev/null || ok=0
    rsync "${RSYNC_OPTS[@]}" "${CKPT_ROOT}/${DATASET}/${CKPT_NAME}.pt" \
        "${LOGIN_DEST}/${CKPT_ROOT}/${DATASET}/" || ok=0
fi

# Logs: whole directory, minus the local-only bookkeeping subpaths. Cheap —
# rsync only ships deltas, and a failed run's log is exactly what you want
# to have on the lab side to debug without another ssh hop.
rsync "${RSYNC_OPTS[@]}" \
    --exclude='pending_push/' --exclude='notify_queue.jsonl*' --exclude='grid_done/' \
    logs/ "${LOGIN_DEST}/logs/" || ok=0

if ls wandb/offline-run-* >/dev/null 2>&1; then
    rsync "${RSYNC_OPTS[@]}" wandb/ "${LOGIN_DEST}/wandb/" || ok=0
fi

dt=$(( $(date +%s) - t0 ))

if (( ok )); then
    rm -f "logs/pending_push/${ARRAY_IDX}"
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
        "mkdir -p '${HPC_LOGIN_REPO_ROOT}/logs/grid_done' && : > '${HPC_LOGIN_REPO_ROOT}/logs/grid_done/${ARRAY_IDX}'" 2>/dev/null || true
    echo "hpc_push_results: slot ${ARRAY_IDX} -> login OK (${dt}s)"
    _notify "[XFER] slot ${ARRAY_IDX} (rc=${RC}) -> login OK (${dt}s)"
else
    echo "hpc_push_results: slot ${ARRAY_IDX} -> login FAILED; marker left for collector."
    _notify "[XFER] slot ${ARRAY_IDX} (rc=${RC}) -> login FAILED (no route or transient error); queued for the login-node collector."
fi

exit 0
