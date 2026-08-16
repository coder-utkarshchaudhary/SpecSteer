#!/usr/bin/env bash
# scripts/hpc_collector.sh
# -------------------------
# Runs INSIDE A TMUX SESSION ON THE LOGIN NODE (started by scripts/hpc_launch.sh
# over login_ssh, session name "prism_collector"). Polls the COMPUTE node for
# `logs/pending_push/<idx>` markers and PULLS each one back — pull, not push,
# because login -> compute is the direction that's already proven to work
# (the reverse may not be, see scripts/hpc_push_results.sh's header).
#
# This is the fallback path for result return, and the single source of
# truth: scripts/hpc_push_results.sh writes the pending_push marker
# unconditionally before it even attempts its own optional push, so a slot
# is only ever considered "returned" once THIS script has pulled it and
# removed the marker — regardless of whether the job's own push succeeded,
# partially succeeded, or never ran (PUSH_RESULTS_FROM_JOB=0).
#
# No dependency on the project's Python venv — pure bash/rsync/ssh, so it
# runs fine on the login node even before/without bootstrap.
#
# Usage (as started by hpc_launch.sh):
#   bash scripts/hpc_collector.sh \
#       --inner-host hpc --compute-root /path/on/compute \
#       --login-root /path/on/login --interval 120

set -uo pipefail

INNER_HOST="hpc"
COMPUTE_ROOT=""
LOGIN_ROOT=""
INTERVAL=120

while [[ $# -gt 0 ]]; do
    case "$1" in
        --inner-host)   INNER_HOST="$2"; shift 2 ;;
        --compute-root) COMPUTE_ROOT="$2"; shift 2 ;;
        --login-root)   LOGIN_ROOT="$2"; shift 2 ;;
        --interval)     INTERVAL="$2"; shift 2 ;;
        *) echo "hpc_collector: unknown flag $1" >&2; exit 1 ;;
    esac
done
: "${COMPUTE_ROOT:?--compute-root is required}"
: "${LOGIN_ROOT:?--login-root is required}"

cd "${LOGIN_ROOT}" || { echo "hpc_collector: cannot cd to ${LOGIN_ROOT}" >&2; exit 1; }
mkdir -p logs/grid_done logs/collector_pulled

RSYNC_OPTS=(-a --info=progress2 --partial --compress --timeout=30)

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }

log "collector started"
log "  inner host   : ${INNER_HOST}"
log "  compute root : ${COMPUTE_ROOT}"
log "  login root   : ${LOGIN_ROOT}"
log "  interval     : ${INTERVAL}s"

pull_slot() {
    local idx="$1" ok=1
    log "pulling slot ${idx} from compute node"
    rsync "${RSYNC_OPTS[@]}" "${INNER_HOST}:${COMPUTE_ROOT}/model/" "model/" 2>&1 | tail -3 || ok=0
    rsync "${RSYNC_OPTS[@]}" "${INNER_HOST}:${COMPUTE_ROOT}/model_smoke/" "model_smoke/" 2>&1 | tail -3 || true
    rsync "${RSYNC_OPTS[@]}" \
        --exclude='pending_push/' --exclude='notify_queue.jsonl*' --exclude='grid_done/' \
        "${INNER_HOST}:${COMPUTE_ROOT}/logs/" "logs/" 2>&1 | tail -3 || ok=0
    rsync "${RSYNC_OPTS[@]}" "${INNER_HOST}:${COMPUTE_ROOT}/wandb/" "wandb/" 2>&1 | tail -3 || true

    if (( ok )); then
        : > "logs/grid_done/${idx}"
        : > "logs/collector_pulled/${idx}"
        ssh -o BatchMode=yes "${INNER_HOST}" "rm -f '${COMPUTE_ROOT}/logs/pending_push/${idx}'" 2>/dev/null || true
        log "slot ${idx} pulled OK"
    else
        log "slot ${idx} pull FAILED — will retry next cycle (marker left on compute node)"
    fi
}

while true; do
    pending="$(ssh -o BatchMode=yes "${INNER_HOST}" "ls '${COMPUTE_ROOT}/logs/pending_push/' 2>/dev/null" || true)"
    if [[ -n "${pending}" ]]; then
        while read -r idx; do
            [[ -z "${idx}" ]] && continue
            [[ -f "logs/collector_pulled/${idx}" ]] && continue
            pull_slot "${idx}"
        done <<< "${pending}"
    fi
    sleep "${INTERVAL}"
done
