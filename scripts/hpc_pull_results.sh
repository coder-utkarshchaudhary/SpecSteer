#!/usr/bin/env bash
# scripts/hpc_pull_results.sh
# ---------------------------
# Rsync checkpoints, wandb offline runs, and logs from the HPC back to the lab
# machine. Safe to run multiple times — rsync only copies what has changed.
#
# Reads scripts/hpc_config.env for HPC_USER / HPC_HOST / HPC_PROJECT_DIR.
#
# Usage:
#   bash scripts/hpc_pull_results.sh                   # pull everything
#   bash scripts/hpc_pull_results.sh --only ckpt       # only model/
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

HPC_PROJECT_DIR="${HPC_PROJECT_DIR:-${HPC_HOME}/prism}"

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
        "${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/${remote_sub}/" \
        "${REPO_ROOT}/${local_sub}/" || echo "!!! ${remote_sub} pull failed (may not exist yet)"
}

case "${ONLY}" in
    all)
        pull model model
        pull checkpoints checkpoints
        pull wandb wandb
        pull logs logs_hpc
        ;;
    ckpt|model|checkpoints)
        pull model model
        pull checkpoints checkpoints
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
if [[ "${ONLY}" == "all" || "${ONLY}" == "wandb" ]]; then
    echo ""
    echo " Sync offline wandb runs to the server:"
    echo "   cd ${REPO_ROOT}"
    echo "   wandb sync wandb/offline-run-*"
fi
echo "=========================================="
