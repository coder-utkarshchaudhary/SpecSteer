#!/usr/bin/env bash
# scripts/hpc_pull_results.sh
# ---------------------------
# Rsync checkpoints, wandb offline runs, and logs from the HPC LOGIN node
# back to the lab machine. Safe to run multiple times — rsync only copies
# what has changed.
#
# NOTE: this always pulls from the LOGIN node (HPC_LOGIN_REPO_ROOT), never
# the compute node directly — the lab machine cannot reach the compute node's
# filesystem (see docs/hpc_wiki.md's two-node topology section). Results get
# to the login node either via scripts/hpc_push_results.sh (compute node
# pushes at the end of each job, if PUSH_RESULTS_FROM_JOB=1) or via
# scripts/hpc_collector.sh (a login-node tmux session that pulls from
# compute on a timer) — this script is the final lab-facing leg regardless
# of which of those got the data to the login node.
#
# Reads scripts/hpc_config.env for HPC_USER / HPC_HOST / HPC_LOGIN_REPO_ROOT.
#
# Usage:
#   bash scripts/hpc_pull_results.sh                   # pull everything
#   bash scripts/hpc_pull_results.sh --only ckpt       # model/ + model_smoke/ + wandb/
#   bash scripts/hpc_pull_results.sh --only wandb      # only wandb/
#   bash scripts/hpc_pull_results.sh --only logs       # only logs/
#
# After a full pull, run  `wandb sync wandb/offline-run-*`  from the lab
# machine to push the offline runs to the wandb server.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_FILE="${SCRIPT_DIR}/hpc_config.env"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERR: ${CONFIG_FILE} not found. Run scripts/hpc_launch.sh setup first." >&2
    exit 1
fi
# shellcheck source=/dev/null
set -a; source "${CONFIG_FILE}"; set +a

# shellcheck source=hpc_common.sh
source "${SCRIPT_DIR}/hpc_common.sh"
resolve_hpc_roots

ONLY="all"
if [[ "${1:-}" == "--only" ]]; then
    ONLY="${2:-all}"
fi

RSYNC_OPTS=(-a --human-readable --info=progress2 --partial --compress)

pull() {
    local remote_sub="$1" local_sub="$2"
    echo ""
    echo ">>> pulling ${remote_sub}"
    mkdir -p "${REPO_ROOT}/${local_sub}"
    rsync "${RSYNC_OPTS[@]}" \
        "${HPC_USER}@${HPC_HOST}:${HPC_LOGIN_REPO_ROOT}/${remote_sub}/" \
        "${REPO_ROOT}/${local_sub}/" || echo "!!! ${remote_sub} pull failed (may not exist yet)"
}

case "${ONLY}" in
    all)
        pull model model
        pull model_smoke model_smoke
        pull checkpoints checkpoints
        pull wandb wandb
        pull logs logs_hpc
        ;;
    ckpt|model|checkpoints)
        # Includes wandb/ too — small, and it's what lets per-slot pulls
        # bring offline wandb runs home incrementally instead of only in
        # the final end-of-grid sweep.
        pull model model
        pull model_smoke model_smoke
        pull checkpoints checkpoints
        pull wandb wandb
        ;;
    wandb)
        pull wandb wandb
        ;;
    logs)
        pull logs logs_hpc
        ;;
    *)
        echo "ERR: unknown --only value: ${ONLY}" >&2
        exit 2
        ;;
esac

echo ""
echo "=========================================="
echo " Pull complete."
if [[ "${ONLY}" == "all" || "${ONLY}" == "wandb" || "${ONLY}" == "ckpt" || "${ONLY}" == "model" || "${ONLY}" == "checkpoints" ]]; then
    echo ""
    echo " Sync offline wandb runs to the server:"
    echo "   cd ${REPO_ROOT}"
    echo "   wandb sync wandb/offline-run-*"
fi
echo "=========================================="
