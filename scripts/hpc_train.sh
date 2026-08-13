#!/usr/bin/env bash
# scripts/hpc_train.sh
# ---------------------
# Lab-side "resume/launch" entry point for use AFTER data already lives on
# the HPC (i.e. after a prior scripts/hpc_launch.sh rsync, and after someone
# has fixed the dataset folder casing on the HPC side via `mv m3 M3` /
# `mv crims CRIMS`). Run this INSTEAD of re-running hpc_launch.sh from
# scratch — it skips the wheel build / rsync / bootstrap steps entirely and
# assumes those already succeeded.
#
# What this does, in order:
#
#   1. Refuses to run if a smoke/grid job or watcher already appears to be
#      in flight (logs/*.pid still alive, or logs/hpc_jobid_* still
#      non-terminal in qstat) — never risks a duplicate submission.
#   2. SSHes into the HPC (using HPC_HOME / HPC_PROJECT_DIR from
#      hpc_config.env — whatever path that points at, /home or /scratch, is
#      not this script's concern) and inspects data/processed/ with
#      `tree -L 3` (installed on both machines if missing, best-effort) so a
#      human can visually confirm the four dataset folders are correctly
#      cased: IIRS, M3, AVIRIS, CRIMS. The actual pass/fail gate is a
#      `find`-based *.npy count per folder (tree's box-drawing output isn't
#      reliable to grep) — same style/threshold as hpc_launch.sh's local
#      data check.
#   3. If the data check FAILS: prints + Telegrams a message saying the data
#      isn't correctly in place yet and to run `bash scripts/hpc_launch.sh`,
#      then exits without touching PBS.
#   4. If the data check PASSES: brings up relay + tunnel + forwarder (if
#      not already up) and submits the smoke job exactly like
#      hpc_launch.sh's steps 5-9, so hpc_smoke_watcher.sh carries it through
#      to the full 28-run grid automatically.
#
# Modes:
#   bash scripts/hpc_train.sh                 # check + launch
#   bash scripts/hpc_train.sh --dry-run       # checks only, no qsub/relay/tunnel
#
# Driven by the same scripts/hpc_config.env as hpc_launch.sh.

set -uo pipefail
# NOTE: -e intentionally omitted, same reasoning as hpc_launch.sh — keep
# going through diagnostics and give a clear message instead of a bare exit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
    C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_B=$'\033[34m'; C_N=$'\033[0m'
else
    C_R=""; C_G=""; C_Y=""; C_B=""; C_N=""
fi

log_step() { echo ""; echo "${C_B}>>> $*${C_N}"; }
log_ok()   { echo "${C_G}    OK: $*${C_N}"; }
log_warn() { echo "${C_Y}    WARN: $*${C_N}"; }
log_err()  { echo "${C_R}    ERR: $*${C_N}"; }

fatal() { log_err "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
MODE="run"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) MODE="dry-run"; shift ;;
        -h|--help)
            sed -n '2,35p' "$0"
            exit 0
            ;;
        *) fatal "unknown flag: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Load config (same file + same required fields as hpc_launch.sh)
# ---------------------------------------------------------------------------
CONFIG_FILE="${SCRIPT_DIR}/hpc_config.env"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    fatal "config not found: ${CONFIG_FILE}
Copy ${CONFIG_FILE}.example -> ${CONFIG_FILE} and fill the FILL_ME blanks
(same config hpc_launch.sh uses). See docs/hpc_wiki.md."
fi
# shellcheck source=/dev/null
set -a; source "${CONFIG_FILE}"; set +a

_check_filled() {
    local name value
    name="$1"; value="${!name:-}"
    if [[ -z "${value}" || "${value}" == "FILL_ME" ]]; then
        fatal "config: '${name}' is empty or still FILL_ME in ${CONFIG_FILE}.
See docs/hpc_wiki.md to learn how to obtain it."
    fi
}
_check_filled HPC_USER
_check_filled HPC_HOST
_check_filled HPC_HOME
_check_filled HPC_QUEUE
_check_filled TELEGRAM_BOT_TOKEN
_check_filled TELEGRAM_CHAT_ID
_check_filled WANDB_API_KEY

export HPC_PROJECT_DIR="${HPC_PROJECT_DIR:-${HPC_HOME}/prism}"
export LAB_TUNNEL_PORT="${LAB_TUNNEL_PORT:-8765}"
export LAB_REPO_ROOT="${LAB_REPO_ROOT:-${REPO_ROOT}}"
export EPOCHS="${EPOCHS:-100}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
TRAIN_LOG="${LOG_DIR}/hpc_train_${TS}.log"

RELAY_PID_FILE="${LOG_DIR}/relay.pid"
TUNNEL_PID_FILE="${LOG_DIR}/tunnel.pid"
JOBID_FILE="${LOG_DIR}/hpc_jobid"
FORWARDER_SESSION="specsteer_forwarder"

DATA_DIR_REMOTE="${HPC_PROJECT_DIR}/data/processed"
REQUIRED_DATASETS=(IIRS M3 AVIRIS CRIMS)
MIN_TOTAL_PATCHES=1000   # same threshold hpc_launch.sh uses locally

exec > >(tee -a "${TRAIN_LOG}") 2>&1

echo "=========================================="
echo " HPC TRAIN  (mode=${MODE})"
echo "  ts               : ${TS}"
echo "  hpc target       : ${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}"
echo "  remote data root : ${DATA_DIR_REMOTE}"
echo "  train log        : ${TRAIN_LOG}"
echo "=========================================="

# hpc_ssh — run a remote command in a LOGIN shell so PBS binaries are on
# PATH. See hpc_launch.sh for why a plain non-interactive ssh isn't enough.
hpc_ssh() {
    # Even `bash -l -s` may miss the PBS PATH when the scheduler binaries are
    # added by an interactive-only rc guard (the actual "qsub: command not
    # found"). Prepend a best-effort fix: read PBS_EXEC from /etc/pbs.conf
    # (present on every PBS Pro node) and fall back to the common install dirs.
    # No-op for non-PBS commands (tmux/tail) since qsub is already found.
    local pbs_fix='if ! command -v qsub >/dev/null 2>&1; then [ -r /etc/pbs.conf ] && . /etc/pbs.conf; for d in "${PBS_EXEC:-/opt/pbs}/bin" /opt/pbs/bin /opt/pbs/default/bin; do [ -x "$d/qsub" ] && export PATH="$d:$PATH" && break; done; fi;'
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" 'bash -l -s' <<< "${pbs_fix} $1"
}

_notify() {
    local text="$1"
    (
        set -a; source "${CONFIG_FILE}"; set +a
        PYTHONPATH="${REPO_ROOT}" python3 "${REPO_ROOT}/utils/notify_cli.py" --text "${text}"
    ) >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# 1. Refuse to double-submit: bail if a watcher or non-terminal job exists.
# ---------------------------------------------------------------------------
log_step "1/5  Checking for an already in-flight run"

_qstat_state_of() {
    local jobid="$1"
    [[ -z "${jobid}" ]] && return 1
    hpc_ssh "qstat -f -x '${jobid}' 2>/dev/null | awk '/^ *job_state = / {print \$3; exit}'"
}

active=0
reasons=()

for pf in "${LOG_DIR}/smoke_watcher.pid" "${LOG_DIR}/grid_watcher.pid"; do
    if [[ -s "${pf}" ]] && kill -0 "$(cat "${pf}")" 2>/dev/null; then
        active=1
        reasons+=("$(basename "${pf}") is alive (pid=$(cat "${pf}")) — a watcher is already managing a run.")
    fi
done

for jf in "${LOG_DIR}/hpc_jobid_full" "${LOG_DIR}/hpc_jobid_smoke" "${JOBID_FILE}"; do
    if [[ -s "${jf}" ]]; then
        jobid="$(cat "${jf}")"
        state="$(_qstat_state_of "${jobid}" | tr -d '[:space:]')"
        if [[ -n "${state}" && "${state}" != "F" && "${state}" != "X" ]]; then
            active=1
            reasons+=("$(basename "${jf}")=${jobid} is still state=${state} in PBS (not terminal).")
        fi
    fi
done

if (( active )); then
    log_err "A run already appears to be in flight — refusing to submit a duplicate."
    for r in "${reasons[@]}"; do echo "      - ${r}"; done
    echo ""
    echo "    Check full status with : bash scripts/hpc_launch.sh --status"
    echo "    If this is stale, clear it first with : bash scripts/hpc_launch.sh --stop"
    exit 4
fi
log_ok "no in-flight smoke/grid job or watcher detected — safe to proceed"

# ---------------------------------------------------------------------------
# 2. Data check: tree -L 3 for a human to eyeball, *.npy counts for the gate.
# ---------------------------------------------------------------------------
log_step "2/5  Checking HPC data/processed layout"

_ensure_remote_tree() {
    if ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" 'command -v tree >/dev/null 2>&1'; then
        return 0
    fi
    log_warn "'tree' not found on the HPC — attempting a best-effort install"
    if ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
        'command -v apt-get >/dev/null 2>&1 && sudo -n apt-get install -y tree >/dev/null 2>&1'; then
        log_ok "installed tree on the HPC"
        return 0
    fi
    log_warn "could not install tree on the HPC (no passwordless sudo — expected on a shared cluster). Falling back to a 'find'-based listing."
    return 1
}

_ensure_local_tree() {
    if command -v tree >/dev/null 2>&1; then
        return 0
    fi
    log_warn "'tree' not found locally — attempting a best-effort install"
    if command -v apt-get >/dev/null 2>&1 && sudo -n apt-get install -y tree >/dev/null 2>&1; then
        log_ok "installed tree locally"
        return 0
    fi
    log_warn "could not install tree locally (no passwordless sudo). Falling back to a 'find'-based listing."
    return 1
}

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run: would ensure tree is installed on both machines and inspect ${DATA_DIR_REMOTE})"
else
    _ensure_local_tree || true
    _ensure_remote_tree || true

    log_step "  remote tree -L 3 ${DATA_DIR_REMOTE}"
    if ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" 'command -v tree >/dev/null 2>&1'; then
        ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" "tree -L 3 '${DATA_DIR_REMOTE}'" || \
            log_warn "tree failed to run remotely (path may not exist yet)"
    else
        ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" "find '${DATA_DIR_REMOTE}' -maxdepth 3 2>/dev/null | sort" || \
            log_warn "remote listing failed (path may not exist yet)"
    fi
fi

# The actual gate: per-dataset *.npy count via find, exact-case paths, same
# style/threshold hpc_launch.sh already uses for the local check. tree above
# is for a human to visually catch a wrong-case folder; its box-drawing
# output isn't reliable to parse programmatically.
data_ok=1
declare -A ds_counts
total_patches=0

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run: would count *.npy under each of ${REQUIRED_DATASETS[*]})"
else
    log_step "  counting *.npy per dataset (exact case)"
    for ds in "${REQUIRED_DATASETS[@]}"; do
        n="$(ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
            "find '${DATA_DIR_REMOTE}/${ds}' -name '*.npy' 2>/dev/null | wc -l" | tr -d '[:space:]')"
        n="${n:-0}"
        ds_counts[${ds}]="${n}"
        total_patches=$(( total_patches + n ))
        if (( n == 0 )); then
            log_err "${ds}: 0 patches found at ${DATA_DIR_REMOTE}/${ds} (wrong case, missing, or empty)"
            data_ok=0
        else
            log_ok "${ds}: ${n} patches"
        fi
    done

    if (( total_patches < MIN_TOTAL_PATCHES )); then
        log_err "total patches (${total_patches}) below sanity threshold (${MIN_TOTAL_PATCHES})"
        data_ok=0
    fi
fi

if [[ "${MODE}" != "dry-run" && "${data_ok}" -eq 0 ]]; then
    msg="[DATA NOT READY] hpc_train.sh found data/processed on the HPC is missing or mis-cased (expected IIRS, M3, AVIRIS, CRIMS with patches under each). Please run: bash scripts/hpc_launch.sh to (re-)sync data, or fix folder casing on the HPC, then re-run hpc_train.sh."
    log_err "Data check failed. NOT launching PBS jobs."
    log_err "Ask whoever renamed the folders (crims -> CRIMS, m3 -> M3) to double check, or run: bash scripts/hpc_launch.sh"
    _notify "${msg}"
    exit 3
fi

if [[ "${MODE}" == "dry-run" ]]; then
    log_ok "dry-run: data check skipped (would gate here)"
else
    log_ok "data check passed — ${total_patches} total patches across ${REQUIRED_DATASETS[*]}"
fi

# ---------------------------------------------------------------------------
# 3. Relay + reverse tunnel + HPC forwarder (idempotent — reuse hpc_launch.sh's
#    exact steps 5-7; skip entirely if already alive).
# ---------------------------------------------------------------------------
log_step "3/5  Relay + reverse tunnel + HPC forwarder"

if [[ -s "${RELAY_PID_FILE}" ]] && kill -0 "$(cat "${RELAY_PID_FILE}")" 2>/dev/null; then
    log_ok "relay already running (pid=$(cat "${RELAY_PID_FILE}"))"
else
    if [[ "${MODE}" == "dry-run" ]]; then
        echo "    (dry-run: would launch notify_relay.py)"
    else
        export TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID
        nohup python3 "${REPO_ROOT}/utils/notify_relay.py" \
            --port "${LAB_TUNNEL_PORT}" --bind 127.0.0.1 \
            >> "${LOG_DIR}/relay.log" 2>&1 &
        relay_pid=$!
        echo "${relay_pid}" > "${RELAY_PID_FILE}"
        sleep 1
        if kill -0 "${relay_pid}" 2>/dev/null; then
            log_ok "relay pid=${relay_pid} (logs: ${LOG_DIR}/relay.log)"
        else
            fatal "relay failed to start. Check ${LOG_DIR}/relay.log"
        fi
    fi
fi

if [[ -s "${TUNNEL_PID_FILE}" ]] && kill -0 "$(cat "${TUNNEL_PID_FILE}")" 2>/dev/null; then
    log_ok "tunnel already running (pid=$(cat "${TUNNEL_PID_FILE}"))"
else
    if [[ "${MODE}" == "dry-run" ]]; then
        echo "    (dry-run: would open autossh reverse tunnel)"
    else
        COMMON_SSH_OPTS=(
            -N -T
            -o BatchMode=yes
            -o ServerAliveInterval=30
            -o ServerAliveCountMax=3
            -o ExitOnForwardFailure=yes
            -R "${LAB_TUNNEL_PORT}:localhost:${LAB_TUNNEL_PORT}"
        )
        AUTOSSH_BIN="$(command -v autossh || command -v ssh)"
        if [[ "$(basename "${AUTOSSH_BIN}")" == "autossh" ]]; then
            AUTOSSH_GATETIME=0 nohup "${AUTOSSH_BIN}" -M 0 \
                "${COMMON_SSH_OPTS[@]}" "${HPC_USER}@${HPC_HOST}" \
                >> "${LOG_DIR}/tunnel.log" 2>&1 &
        else
            nohup "${AUTOSSH_BIN}" \
                "${COMMON_SSH_OPTS[@]}" "${HPC_USER}@${HPC_HOST}" \
                >> "${LOG_DIR}/tunnel.log" 2>&1 &
        fi
        tunnel_pid=$!
        echo "${tunnel_pid}" > "${TUNNEL_PID_FILE}"
        sleep 3
        if ! kill -0 "${tunnel_pid}" 2>/dev/null; then
            log_err "tunnel process exited immediately. Last lines of tunnel.log:"
            tail -n 15 "${LOG_DIR}/tunnel.log" 2>/dev/null | sed 's/^/      /'
            fatal "reverse tunnel failed to start. See hpc_launch.sh's troubleshooting notes."
        fi

        probe=$(ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
            "curl -s -m 8 http://localhost:${LAB_TUNNEL_PORT}/healthz 2>/dev/null || \
             (command -v python3 >/dev/null && python3 -c \"import urllib.request,sys; sys.stdout.write(urllib.request.urlopen('http://localhost:${LAB_TUNNEL_PORT}/healthz', timeout=8).read().decode())\" 2>/dev/null) || echo PROBE_FAILED" \
            2>/dev/null | tr -d '[:space:]')
        if [[ "${probe}" == "ok" ]]; then
            log_ok "tunnel verified end-to-end (pid=${tunnel_pid})"
        else
            log_warn "tunnel is up (pid=${tunnel_pid}) but HPC could not reach the relay through it (probe='${probe:-empty}'). Telegram may be delayed."
        fi
    fi
fi

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run: would start/verify HPC forwarder tmux session)"
else
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" bash <<EOF
set -uo pipefail
cd '${HPC_PROJECT_DIR}'
if tmux has-session -t '${FORWARDER_SESSION}' 2>/dev/null; then
    echo '    forwarder tmux session already exists — leaving as-is.'
    exit 0
fi
tmux new-session -d -s '${FORWARDER_SESSION}' \\
    "source .venv/bin/activate && python scripts/notify_forwarder.py \\
        --queue '${HPC_PROJECT_DIR}/logs/notify_queue.jsonl' \\
        --url 'http://localhost:${LAB_TUNNEL_PORT}/notify' \\
        --interval 3 2>&1 | tee -a logs/forwarder.log"
echo "    tmux session '${FORWARDER_SESSION}' started."
EOF
    if [[ $? -ne 0 ]]; then
        log_warn "forwarder tmux setup failed. Heartbeats will still land in logs/notify_queue.jsonl on the HPC."
    else
        log_ok "forwarder running on HPC"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Submit smoke run (slot 1) and launch the smoke -> full-grid watcher.
# ---------------------------------------------------------------------------
log_step "4/5  Submit smoke run (slot 1, ${SMOKE_EPOCHS:-5} epochs)"

SMOKE_EPOCHS="${SMOKE_EPOCHS:-5}"
SMOKE_DELAY_SECS="${SMOKE_DELAY_SECS:-600}"
FULL_ARRAY_RANGE="${FULL_ARRAY_RANGE:-${HPC_ARRAY_RANGE:-1-28}}"
WATCHER_PID_FILE="${LOG_DIR}/smoke_watcher.pid"

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run: would submit smoke job for slot 1)"
    echo ""
    echo "    grid preview:"
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/scripts/grid_manifest.sh"
    grid_print
else
    qsub_extra=()
    if [[ -n "${HPC_PROJECT_CODE:-}" ]]; then
        qsub_extra+=(-P "${HPC_PROJECT_CODE}")
    fi

    smoke_qsub_cmd="cd '${HPC_PROJECT_DIR}' && qsub \
        -q '${HPC_QUEUE}' \
        -l '${HPC_SELECT}' \
        -l walltime='01:00:00' \
        -J '1-1' \
        ${qsub_extra[*]:-} \
        -v HPC_PROJECT_DIR='${HPC_PROJECT_DIR}',EPOCHS='${SMOKE_EPOCHS}',SMOKE_MODE=1,SMOKE_EPOCHS='${SMOKE_EPOCHS}',WANDB_PROJECT='${WANDB_PROJECT:-hsi-pi-vae}',WANDB_ENTITY='${WANDB_ENTITY:-}',EXTRA_TRAIN_ARGS='${EXTRA_TRAIN_ARGS:-}' \
        scripts/hpc_pbs_job.pbs"

    SMOKE_JOBID=$(hpc_ssh "${smoke_qsub_cmd}") || fatal "qsub smoke failed"
    SMOKE_JOBID=$(echo "${SMOKE_JOBID}" | tail -1 | xargs)
    echo "${SMOKE_JOBID}" > "${LOG_DIR}/hpc_jobid_smoke"
    echo "${SMOKE_JOBID}" > "${JOBID_FILE}"
    log_ok "smoke job submitted: ${SMOKE_JOBID}"
    log_step "  if smoke passes, watcher will wait ${SMOKE_DELAY_SECS}s then submit full grid (${FULL_ARRAY_RANGE})"

    export HPC_SMOKE_JOBID="${SMOKE_JOBID}"
    export SMOKE_DELAY_SECS FULL_ARRAY_RANGE EPOCHS EXTRA_TRAIN_ARGS
    nohup bash "${REPO_ROOT}/scripts/hpc_smoke_watcher.sh" \
        >> "${LOG_DIR}/smoke_watcher.log" 2>&1 &
    watcher_pid=$!
    echo "${watcher_pid}" > "${WATCHER_PID_FILE}"
    log_ok "smoke watcher pid=${watcher_pid} (logs: ${LOG_DIR}/smoke_watcher.log)"

    _notify "$(printf '[RESUMED] hpc_train.sh confirmed data on the HPC and submitted the smoke job.\njobid: %s\nIf it passes, the full grid (%s) auto-submits after %ss.' \
        "${SMOKE_JOBID}" "${FULL_ARRAY_RANGE}" "${SMOKE_DELAY_SECS}")"
fi

# ---------------------------------------------------------------------------
# 5. Monitoring cheatsheet
# ---------------------------------------------------------------------------
log_step "5/5  Monitoring cheatsheet"

cat <<EOF

    Watch this run's log:
      tail -f ${TRAIN_LOG}

    Watch smoke watcher progress:
      tail -f ${LOG_DIR}/smoke_watcher.log

    Watch grid pull-back watcher (after smoke passes):
      tail -f ${LOG_DIR}/grid_watcher.log

    Status / stop (shared with hpc_launch.sh):
      bash scripts/hpc_launch.sh --status
      bash scripts/hpc_launch.sh --stop

EOF
log_ok "hpc_train.sh complete."
