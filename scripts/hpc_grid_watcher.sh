#!/usr/bin/env bash
# scripts/hpc_grid_watcher.sh
# ----------------------------
# Lab-side background watcher launched by scripts/hpc_smoke_watcher.sh right
# after it submits the full 28-run array. Responsibilities:
#
#   1. Each poll cycle, first check the LOGIN node for
#      logs/grid_done/<idx> markers (written either by scripts/hpc_push_results.sh
#      right after a successful compute->login push, or by the login-node
#      collector tmux session — see scripts/hpc_collector.sh). Any slot with
#      a marker is pulled immediately, without waiting on qstat.
#   2. Otherwise, poll `qstat -f -x -t <full_jobid>` (two-hop, via
#      scripts/hpc_common.sh's compute_ssh) for the per-array-element state of
#      every slot in the grid, and pull any slot that just went terminal.
#   3. Pulling a slot means:
#        - bash scripts/hpc_pull_results.sh --only ckpt   (model/ + checkpoints/)
#        - bash scripts/hpc_pull_results.sh --only logs   (pbs_*.out + train logs)
#      — both of which pull from the LOGIN node, never the compute node
#      directly (the lab machine cannot reach the compute node's filesystem).
#      Then send a Telegram ping with the slot's (model, dataset, loss, exit).
#   4. Once every slot in the range is accounted for, do one final
#      `--only all` sweep (wandb/ included) and exit.
#
# Why per-slot instead of one pull at the very end: the full grid runs for
# many hours across many array elements: if the HPC job, the SSH tunnel, or
# the lab machine dies partway through, a single end-of-grid pull would lose
# every checkpoint already produced by runs that finished earlier. Pulling
# right after each slot's own terminal state makes that already-persisted:
# a later crash only threatens *in-flight* runs, never completed ones.
#
# Idempotent: successfully-pulled slots are marked with a marker file under
# logs/grid_watcher/pulled/<idx> and never re-pulled; a rsync failure leaves
# the slot unmarked so the next poll retries it (rsync itself is incremental,
# so retrying a slot that partially landed is cheap and correct).
#
# Not intended to be invoked by hand — hpc_smoke_watcher.sh wires it up. But
# it works standalone (e.g. to re-attach after the lab machine rebooted):
#   HPC_FULL_JOBID=1234[].padum FULL_ARRAY_RANGE=1-28 bash scripts/hpc_grid_watcher.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_FILE="${SCRIPT_DIR}/hpc_config.env"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "grid_watcher: config not found at ${CONFIG_FILE}" >&2
    exit 1
fi
# shellcheck source=/dev/null
set -a; source "${CONFIG_FILE}"; set +a

# shellcheck source=hpc_common.sh
source "${SCRIPT_DIR}/hpc_common.sh"
resolve_hpc_roots

GRID_POLL_INTERVAL="${GRID_POLL_INTERVAL:-600}"    # a 24h array job doesn't need 30s polling
GRID_EMPTY_STREAK_MAX=3                            # consecutive failed/empty qstat queries before giving up on a slot

LOG_DIR="${REPO_ROOT}/logs"
STATE_DIR="${LOG_DIR}/grid_watcher"
PULLED_DIR="${STATE_DIR}/pulled"
SEEN_DIR="${STATE_DIR}/seen"
mkdir -p "${LOG_DIR}" "${PULLED_DIR}" "${SEEN_DIR}"
WATCHER_LOG="${LOG_DIR}/grid_watcher.log"
STATE_FILE="${LOG_DIR}/grid_watcher.state"

FULL_JOBID="${HPC_FULL_JOBID:-}"
if [[ -z "${FULL_JOBID}" && -s "${LOG_DIR}/hpc_jobid_full" ]]; then
    FULL_JOBID="$(cat "${LOG_DIR}/hpc_jobid_full")"
fi
if [[ -z "${FULL_JOBID}" ]]; then
    echo "grid_watcher: no full job id (HPC_FULL_JOBID / logs/hpc_jobid_full both empty)" >&2
    exit 2
fi

# Assumes a plain "N-M" range (what hpc_launch.sh/hpc_config.env.example ever
# produce). A PBS step range ("1-28:2") or comma list would break both the
# count below and the slot loop further down — not handled.
FULL_ARRAY_RANGE="${FULL_ARRAY_RANGE:-${HPC_ARRAY_RANGE:-1-28}}"
IFS='-' read -r RANGE_START RANGE_END <<< "${FULL_ARRAY_RANGE}"
RANGE_START="${RANGE_START:-1}"
RANGE_END="${RANGE_END:-${RANGE_START}}"
SLOT_TOTAL=$(( RANGE_END - RANGE_START + 1 ))

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/grid_manifest.sh"  # defines its own GRID_TOTAL (=28) — unrelated to SLOT_TOTAL above, don't conflate

log() {
    printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${WATCHER_LOG}"
}

_notify() {
    local text="$1"
    (
        set -a; source "${CONFIG_FILE}"; set +a
        PYTHONPATH="${REPO_ROOT}" python3 "${REPO_ROOT}/utils/notify_cli.py" --text "${text}"
    ) >/dev/null 2>&1 || true
}

log "watcher started"
log "  full jobid     : ${FULL_JOBID}"
log "  array range    : ${FULL_ARRAY_RANGE}  (${SLOT_TOTAL} slots)"
log "  poll interval  : ${GRID_POLL_INTERVAL}s"

echo "FULL_JOBID=${FULL_JOBID}" > "${STATE_FILE}"
echo "RANGE=${FULL_ARRAY_RANGE}" >> "${STATE_FILE}"
echo "STATE=polling" >> "${STATE_FILE}"

# ---------------------------------------------------------------------------
# One qstat query -> lines "<idx> STATE <s>" / "<idx> EXIT <code>", one pair
# per array element currently known to PBS. Two-hop via compute_ssh (only
# the compute node has qstat).
# ---------------------------------------------------------------------------
_qstat_slots() {
    compute_ssh "qstat -f -x -t '${FULL_JOBID}' 2>/dev/null" | awk '
        /^Job Id: / {
            line = $0
            s = index(line, "["); e = index(line, "]")
            idx = (s > 0 && e > s) ? substr(line, s + 1, e - s - 1) : ""
            next
        }
        /^ *job_state = / { if (idx != "") print idx, "STATE", $3 }
        /^ *Exit_status = / { if (idx != "") print idx, "EXIT", $3 }
    '
}

# List logs/grid_done/<idx> markers already on the LOGIN node — the fast
# path (hpc_push_results.sh or the collector already moved this slot's
# results, no need to wait on qstat).
_login_grid_done() {
    login_ssh "ls '${HPC_LOGIN_REPO_ROOT}/logs/grid_done/' 2>/dev/null" 2>/dev/null
}

# pull_slot <idx> <exit_code> — pull one slot's ckpt + logs back, notify, mark done.
pull_slot() {
    local idx="$1" exitc="$2"
    local cfg model dataset loss ckpt_name tag
    cfg="$(grid_lookup "${idx}" 2>/dev/null || true)"
    model="${cfg%%|*}"; local rest="${cfg#*|}"
    dataset="${rest%%|*}"; rest="${rest#*|}"
    loss="${rest%%|*}"
    ckpt_name="${rest#*|}"

    if [[ "${exitc}" == "0" ]]; then tag="OK"; else tag="FAIL"; fi
    log "slot ${idx} -> ${model}/${dataset}/${loss} [exit=${exitc:-?}] — pulling from login node"

    # hpc_pull_results.sh swallows rsync errors internally and always exits 0
    # (it just echoes "!!! ... pull failed" and continues), so its exit code
    # can't tell us whether the checkpoint actually landed. Check the artifact
    # itself instead. On a crashed run (exitc != 0, or exitc unknown "?" from
    # the grid_done fast path) there may legitimately be no checkpoint to
    # wait for, so mark it done regardless — the final catch-all sweep at the
    # end still re-syncs anything that shows up late.
    bash "${SCRIPT_DIR}/hpc_pull_results.sh" --only ckpt >> "${WATCHER_LOG}" 2>&1
    bash "${SCRIPT_DIR}/hpc_pull_results.sh" --only logs >> "${WATCHER_LOG}" 2>&1

    local ckpt_path="${REPO_ROOT}/model/${dataset}/${ckpt_name}.pt"
    if [[ "${exitc}" != "0" || -s "${ckpt_path}" ]]; then
        touch "${PULLED_DIR}/${idx}"
        log "slot ${idx} pulled OK (${PULLED_DIR##*/}: $(ls "${PULLED_DIR}" | wc -l | tr -d ' ')/${SLOT_TOTAL})"
        _notify "$(printf '[%s] slot %s/%s done: %s | %s | %s\nckpt: model/%s/%s.pt\nprogress: %s/%s slots pulled' \
            "${tag}" "${idx}" "${RANGE_END}" "${model}" "${dataset}" "${loss}" \
            "${dataset}" "${ckpt_name}" "$(ls "${PULLED_DIR}" | wc -l | tr -d ' ')" "${SLOT_TOTAL}")"
    else
        log "slot ${idx} exited 0 but ${ckpt_path} not found locally yet — will retry next poll"
    fi
}

# ---------------------------------------------------------------------------
# Main poll loop.
# ---------------------------------------------------------------------------
empty_streak=0
while true; do
    n_pulled=$(ls "${PULLED_DIR}" 2>/dev/null | wc -l | tr -d ' ')
    if (( n_pulled >= SLOT_TOTAL )); then
        log "all ${SLOT_TOTAL} slots pulled — done"
        break
    fi

    # Fast path: anything the login node already flagged done, pull now.
    done_now="$(_login_grid_done)"
    if [[ -n "${done_now}" ]]; then
        while read -r idx; do
            [[ -z "${idx}" ]] && continue
            [[ -f "${PULLED_DIR}/${idx}" ]] && continue
            pull_slot "${idx}" "?"
        done <<< "${done_now}"
    fi

    n_pulled=$(ls "${PULLED_DIR}" 2>/dev/null | wc -l | tr -d ' ')
    if (( n_pulled >= SLOT_TOTAL )); then
        log "all ${SLOT_TOTAL} slots pulled — done"
        break
    fi

    raw="$(_qstat_slots)"
    if [[ -z "${raw}" ]]; then
        empty_streak=$(( empty_streak + 1 ))
        log "qstat query empty (streak=${empty_streak}/${GRID_EMPTY_STREAK_MAX})"
        if (( empty_streak >= GRID_EMPTY_STREAK_MAX )); then
            log "job no longer queryable — treating all un-pulled slots in range as terminal"
            for (( idx=RANGE_START; idx<=RANGE_END; idx++ )); do
                [[ -f "${PULLED_DIR}/${idx}" ]] && continue
                pull_slot "${idx}" "?"
            done
            break
        fi
        sleep "${GRID_POLL_INTERVAL}"
        continue
    fi
    empty_streak=0

    STATE_ARR=()
    EXIT_ARR=()
    while read -r idx kind val; do
        [[ -z "${idx}" ]] && continue
        if [[ "${kind}" == "STATE" ]]; then
            STATE_ARR[idx]="${val}"
            case "${val}" in
                F|X) : ;;  # terminal — handled below
                *) touch "${SEEN_DIR}/${idx}" ;;  # queued/running at least once
            esac
        elif [[ "${kind}" == "EXIT" ]]; then
            EXIT_ARR[idx]="${val}"
        fi
    done <<< "${raw}"

    for (( idx=RANGE_START; idx<=RANGE_END; idx++ )); do
        [[ -f "${PULLED_DIR}/${idx}" ]] && continue
        st="${STATE_ARR[idx]:-}"
        if [[ "${st}" == "F" || "${st}" == "X" ]]; then
            pull_slot "${idx}" "${EXIT_ARR[idx]:-}"
        elif [[ -z "${st}" && -f "${SEEN_DIR}/${idx}" ]]; then
            # Was running/queued before, now absent from qstat -> purged after finishing.
            pull_slot "${idx}" "?"
        fi
    done

    sleep "${GRID_POLL_INTERVAL}"
done

# ---------------------------------------------------------------------------
# Final catch-all sweep: wandb/ + any stragglers in logs/, once the whole
# array is accounted for.
# ---------------------------------------------------------------------------
log "final full pull (wandb + logs + model + model_smoke)"
bash "${SCRIPT_DIR}/hpc_pull_results.sh" >> "${WATCHER_LOG}" 2>&1 || log "!! final pull failed"

echo "STATE=grid_complete" >> "${STATE_FILE}"
_notify "$(printf '[GRID DONE] all %s slots of the ablation grid are terminal and pulled back.\nRun: wandb sync wandb/offline-run-* on the lab machine.' "${SLOT_TOTAL}")"
log "grid watcher finished"
