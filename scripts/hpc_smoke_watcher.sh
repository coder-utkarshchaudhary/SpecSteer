#!/usr/bin/env bash
# scripts/hpc_smoke_watcher.sh
# ----------------------------
# Lab-side background watcher launched by scripts/hpc_launch.sh right after
# the smoke job is submitted (two-hop: qstat/qsub run via compute_ssh, from
# scripts/hpc_common.sh, since only the compute node has PBS). Responsibilities:
#
#   1. Poll qstat for the smoke PBS job until it reaches a terminal state.
#   2. If the smoke run succeeded:
#        - Send Telegram: "smoke OK, sleeping N min then submitting full grid"
#        - Sleep SMOKE_DELAY_SECS
#        - Submit the full array (HPC_ARRAY_RANGE, model/ ckpts, EPOCHS)
#        - Write the full job id to logs/hpc_jobid_full
#        - Send Telegram: "full grid launched, JOBID=..."
#   3. If the smoke run failed:
#        - Fetch the failing log's tail directly via compute_ssh (no rsync —
#          the lab machine cannot reach the compute node's filesystem)
#        - Send Telegram with the log tail and the exception
#        - Do NOT submit the full array. Exit.
#
# Idempotent: writes state to logs/smoke_watcher.state so it survives being
# killed and restarted. Reads scripts/hpc_config.env for all HPC coordinates.
#
# Not intended to be invoked by hand — the launcher wires it up. But it works
# standalone:
#   HPC_SMOKE_JOBID=1234.padum bash scripts/hpc_smoke_watcher.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_FILE="${SCRIPT_DIR}/hpc_config.env"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "smoke_watcher: config not found at ${CONFIG_FILE}" >&2
    exit 1
fi
# shellcheck source=/dev/null
set -a; source "${CONFIG_FILE}"; set +a

# shellcheck source=hpc_common.sh
source "${SCRIPT_DIR}/hpc_common.sh"
resolve_hpc_roots

SMOKE_DELAY_SECS="${SMOKE_DELAY_SECS:-600}"      # 10 min default
SMOKE_POLL_INTERVAL="${SMOKE_POLL_INTERVAL:-30}" # every 30 s
FULL_ARRAY_RANGE="${FULL_ARRAY_RANGE:-1-60}"
EPOCHS="${EPOCHS:-100}"
PUSH_RESULTS_FROM_JOB="${PUSH_RESULTS_FROM_JOB:-0}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
WATCHER_LOG="${LOG_DIR}/smoke_watcher.log"
STATE_FILE="${LOG_DIR}/smoke_watcher.state"

SMOKE_JOBID="${HPC_SMOKE_JOBID:-}"
if [[ -z "${SMOKE_JOBID}" && -s "${LOG_DIR}/hpc_jobid_smoke" ]]; then
    SMOKE_JOBID="$(cat "${LOG_DIR}/hpc_jobid_smoke")"
fi
if [[ -z "${SMOKE_JOBID}" ]]; then
    echo "smoke_watcher: no smoke job id (HPC_SMOKE_JOBID / logs/hpc_jobid_smoke both empty)" >&2
    exit 2
fi

log() {
    printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${WATCHER_LOG}"
}

_notify() {
    # Best-effort Telegram send via the lab-side notify_cli.py.
    local text="$1"
    (
        set -a; source "${CONFIG_FILE}"; set +a
        PYTHONPATH="${REPO_ROOT}" python3 "${REPO_ROOT}/utils/notify_cli.py" --text "${text}"
    ) >/dev/null 2>&1 || true
}

log "watcher started"
log "  smoke jobid    : ${SMOKE_JOBID}"
log "  poll interval  : ${SMOKE_POLL_INTERVAL}s"
log "  delay after ok : ${SMOKE_DELAY_SECS}s"
log "  full range     : ${FULL_ARRAY_RANGE}"

echo "SMOKE_JOBID=${SMOKE_JOBID}" > "${STATE_FILE}"
echo "STATE=polling" >> "${STATE_FILE}"

# ---------------------------------------------------------------------------
# Poll qstat until every array element is terminal (two-hop, via compute_ssh).
# ---------------------------------------------------------------------------
_qstat_master_state() {
    compute_ssh "qstat -f -x '${SMOKE_JOBID}' 2>/dev/null | awk '/^ *job_state = / {print \$3; exit}'" \
        || echo "?"
}

_qstat_master_exit() {
    compute_ssh "qstat -f -x '${SMOKE_JOBID}' 2>/dev/null | awk '/^ *Exit_status = / {print \$3; exit}'" \
        || echo ""
}

state=""
exit_code=""
while true; do
    state="$(_qstat_master_state)"
    state="$(echo "${state}" | tr -d '[:space:]')"
    log "qstat ${SMOKE_JOBID}: state=${state:-<empty>}"

    if [[ "${state}" == "F" || "${state}" == "X" || -z "${state}" || "${state}" == "?" ]]; then
        exit_code="$(_qstat_master_exit)"
        exit_code="$(echo "${exit_code}" | tr -d '[:space:]')"
        log "smoke terminal. exit_code=${exit_code:-<unknown>}"
        break
    fi
    sleep "${SMOKE_POLL_INTERVAL}"
done

# ---------------------------------------------------------------------------
# Fetch the smoke log's tail directly via compute_ssh — the lab machine
# cannot rsync from the compute node's filesystem (only the login node can
# reach it), so read the file remotely instead of pulling it locally.
# ---------------------------------------------------------------------------
log "fetching smoke log tail (two-hop)"
smoke_tail="$(compute_ssh "ls -t '${HPC_COMPUTE_REPO_ROOT}/logs'/pbs_*_1.out 2>/dev/null | head -1 | xargs -r tail -n 60" 2>/dev/null)"
smoke_log_name="$(compute_ssh "ls -t '${HPC_COMPUTE_REPO_ROOT}/logs'/pbs_*_1.out 2>/dev/null | head -1" 2>/dev/null | tr -d '[:space:]')"

# ---------------------------------------------------------------------------
# Decide: success (exit_code == 0) or failure (anything else / unknown).
# ---------------------------------------------------------------------------
if [[ "${exit_code}" == "0" ]]; then
    echo "STATE=smoke_ok" >> "${STATE_FILE}"
    log "SMOKE OK — sleeping ${SMOKE_DELAY_SECS}s before full submit"
    _notify "$(printf '[OK] smoke run passed (job %s)\nSleeping %ss then submitting full grid (%s).\nWatcher log: %s' \
        "${SMOKE_JOBID}" "${SMOKE_DELAY_SECS}" "${FULL_ARRAY_RANGE}" "${WATCHER_LOG}")"
    sleep "${SMOKE_DELAY_SECS}"

    # Submit the full array.
    log "submitting full array (range=${FULL_ARRAY_RANGE}, epochs=${EPOCHS})"
    qsub_extra=""
    if [[ -n "${HPC_PROJECT_CODE:-}" ]]; then
        qsub_extra="-P ${HPC_PROJECT_CODE}"
    fi
    full_cmd="cd '${HPC_COMPUTE_REPO_ROOT}' && qsub \
        -q '${HPC_QUEUE}' \
        -l '${HPC_SELECT}' \
        -l walltime='${HPC_WALLTIME}' \
        -J '${FULL_ARRAY_RANGE}' \
        ${qsub_extra} \
        -v HPC_PROJECT_DIR='${HPC_COMPUTE_REPO_ROOT}',HPC_COMPUTE_REPO_ROOT='${HPC_COMPUTE_REPO_ROOT}',HPC_LOGIN_REPO_ROOT='${HPC_LOGIN_REPO_ROOT}',HPC_USER='${HPC_USER}',HPC_HOST='${HPC_HOST}',PUSH_RESULTS_FROM_JOB='${PUSH_RESULTS_FROM_JOB}',OVERWRITE='${OVERWRITE:-0}',EPOCHS='${EPOCHS}',SMOKE_MODE=0,WANDB_PROJECT='${WANDB_PROJECT:-hsi-pi-vae}',WANDB_ENTITY='${WANDB_ENTITY:-}',EXTRA_TRAIN_ARGS='${EXTRA_TRAIN_ARGS:-}' \
        scripts/hpc_pbs_job.pbs"
    full_jobid="$(compute_ssh "${full_cmd}" 2>&1 | tail -1 | tr -d '[:space:]')"
    if [[ -z "${full_jobid}" ]]; then
        log "!! full array submit produced empty jobid"
        _notify "[FAIL] full array submit failed after smoke passed. Watcher log: ${WATCHER_LOG}"
        echo "STATE=full_submit_failed" >> "${STATE_FILE}"
        exit 5
    fi
    echo "${full_jobid}" > "${LOG_DIR}/hpc_jobid_full"
    echo "${full_jobid}" > "${LOG_DIR}/hpc_jobid"   # canonical pointer for --stop / --status
    echo "STATE=full_submitted" >> "${STATE_FILE}"
    echo "FULL_JOBID=${full_jobid}" >> "${STATE_FILE}"
    log "full array submitted: ${full_jobid}"

    # Launch the per-run pull-back watcher: it pulls each slot's checkpoint
    # the moment that slot finishes, instead of waiting for the whole grid.
    log "starting grid watcher (per-slot pull-back)"
    export HPC_FULL_JOBID="${full_jobid}"
    export FULL_ARRAY_RANGE
    nohup bash "${SCRIPT_DIR}/hpc_grid_watcher.sh" \
        >> "${LOG_DIR}/grid_watcher.log" 2>&1 &
    grid_watcher_pid=$!
    echo "${grid_watcher_pid}" > "${LOG_DIR}/grid_watcher.pid"
    log "grid watcher pid=${grid_watcher_pid} (logs: ${LOG_DIR}/grid_watcher.log)"

    _notify "$(printf '[LAUNCHED] Full ablation grid submitted.\njobid: %s\nrange: %s\nepochs: %s\nCheckpoints will be pulled back to the lab machine as each run finishes.' \
        "${full_jobid}" "${FULL_ARRAY_RANGE}" "${EPOCHS}")"
    exit 0
else
    echo "STATE=smoke_failed" >> "${STATE_FILE}"
    log "SMOKE FAILED — not launching full grid"
    tail_txt="${smoke_tail:-(no log found)}"
    _notify "$(printf '[FAIL] smoke run failed (job %s, exit=%s). Full grid NOT submitted.\nTail of %s:\n%s' \
        "${SMOKE_JOBID}" "${exit_code:-?}" "${smoke_log_name:-?}" "${tail_txt}")"
    exit 3
fi
