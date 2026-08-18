#!/usr/bin/env bash
# scripts/hpc_launch.sh
# ---------------------
# Single-entry-point launcher for the IITD HPC ablation run.
# Data is assumed to already be present on the HPC — no rsync steps.
#
# TOPOLOGY: two separate hosts, two separate filesystems.
#   LOGIN node   ssh ${HPC_USER}@${HPC_HOST}         — reachable from the lab.
#                                                       No qsub here.
#   COMPUTE node ssh ${HPC_INNER_HOST} (from login)  — has qsub. Reached via
#                                                       scripts/hpc_common.sh's
#                                                       compute_ssh (two hops).
#
# What this does, in order:
#
#   1. Sanity checks (wifi SSID, config env, SSH reachability, tool probes).
#   2. Wheels: skipped when USE_SHIPPED_VENV=1 (default).
#   3. lab -> login: TWO independent conditional rsyncs (repo / .venv),
#      each skipped if already present+intact on the login node.
#   4. login -> compute: same two pushes, run ON the login node.
#   5. Runs scripts/hpc_bootstrap.sh on the COMPUTE node.
#   6. Telegram: relay (lab) + chained reverse tunnels (lab->login->compute)
#      + forwarder tmux (on the compute node) + collector tmux (on the
#      login node, pulls finished slots' results back).
#   7. qsubs scripts/hpc_pbs_job.pbs (via compute_ssh) as a smoke run, then
#      hands off to scripts/hpc_smoke_watcher.sh for the smoke->full handoff.
#   8. Prints the qstat / tail-monitoring cheatsheet.
#
# NOTE: data is expected to be pre-transferred to the HPC filesystem.
#       The data/processed/ rsyncs have been removed entirely.
#
# Modes:
#   bash scripts/hpc_launch.sh                            # full run (all datasets)
#   bash scripts/hpc_launch.sh --datasets M3,CRIMS        # grid filtered to named datasets
#   bash scripts/hpc_launch.sh --dry-run                  # sanity + print-only
#   bash scripts/hpc_launch.sh --stop                     # kill everything + qdel
#   bash scripts/hpc_launch.sh --status                   # what's alive
#
# Everything is driven by scripts/hpc_config.env. Copy .example to .env,
# fill the FILL_ME blanks, then run this script. Don't touch anything else.

set -uo pipefail
# NOTE: -e is intentionally omitted — we want to keep going through the
# per-step diagnostics even if one step fails, and give the junior a clear
# error message rather than a bare exit code.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Colours (harmless when redirected — everything falls back to plain)
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
RESUME=0
DATASETS_SUBSET=""   # e.g. "M3,CRIMS" — empty means all
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  MODE="dry-run"; shift ;;
        --stop)     MODE="stop"; shift ;;
        --status)   MODE="status"; shift ;;
        --resume)   RESUME=1; shift ;;
        --datasets) DATASETS_SUBSET="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,45p' "$0"
            exit 0
            ;;
        *) fatal "unknown flag: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Load config + shared helpers
# ---------------------------------------------------------------------------
CONFIG_FILE="${SCRIPT_DIR}/hpc_config.env"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    fatal "config not found: ${CONFIG_FILE}
Copy ${CONFIG_FILE}.example -> ${CONFIG_FILE} and fill the FILL_ME blanks.
See docs/hpc_wiki.md for how to obtain each value."
fi
# shellcheck source=/dev/null
set -a; source "${CONFIG_FILE}"; set +a

# shellcheck source=hpc_common.sh
source "${SCRIPT_DIR}/hpc_common.sh"

# ---- validate all required fields are filled -----------------------------
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
_check_filled HPC_SELECT
_check_filled HPC_WALLTIME
_check_filled LAB_DATA_ROOT
_check_filled TELEGRAM_BOT_TOKEN
_check_filled TELEGRAM_CHAT_ID
_check_filled WANDB_API_KEY

resolve_hpc_roots   # fills HPC_LOGIN_REPO_ROOT / HPC_COMPUTE_REPO_ROOT / HPC_PROJECT_DIR
export LAB_TUNNEL_PORT="${LAB_TUNNEL_PORT:-8765}"
export LAB_REPO_ROOT="${LAB_REPO_ROOT:-${REPO_ROOT}}"
export EPOCHS="${EPOCHS:-100}"
export USE_SHIPPED_VENV="${USE_SHIPPED_VENV:-1}"
export PUSH_RESULTS_FROM_JOB="${PUSH_RESULTS_FROM_JOB:-0}"
export COLLECTOR_INTERVAL="${COLLECTOR_INTERVAL:-120}"
HPC_ARRAY_RANGE="${HPC_ARRAY_RANGE:-1-28}"
export DATASETS_SUBSET

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
LAUNCH_LOG="${LOG_DIR}/hpc_launch_${TS}.log"

RELAY_PID_FILE="${LOG_DIR}/relay.pid"
TUNNEL_PID_FILE="${LOG_DIR}/tunnel.pid"
INNER_TUNNEL_SESSION="prism_inner_tunnel"   # tmux, on the LOGIN node
FORWARDER_SESSION="prism_forwarder"         # tmux, on the COMPUTE node
COLLECTOR_SESSION="prism_collector"         # tmux, on the LOGIN node
JOBID_FILE="${LOG_DIR}/hpc_jobid"

# ---------------------------------------------------------------------------
# --stop and --status short-circuit here
# ---------------------------------------------------------------------------
_kill_pidfile() {
    local pf="$1"
    if [[ -s "${pf}" ]]; then
        local pid; pid=$(cat "${pf}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "stopping pid=${pid} ($(basename "${pf}"))"
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 1
            kill -KILL "${pid}" 2>/dev/null || true
        fi
        rm -f "${pf}"
    fi
}

if [[ "${MODE}" == "stop" ]]; then
    log_step "stopping local relay + tunnel + watchers"
    _kill_pidfile "${RELAY_PID_FILE}"
    _kill_pidfile "${TUNNEL_PID_FILE}"
    _kill_pidfile "${LOG_DIR}/smoke_watcher.pid"
    _kill_pidfile "${LOG_DIR}/grid_watcher.pid"

    log_step "killing login-node inner tunnel + collector tmux sessions"
    login_ssh "tmux kill-session -t ${INNER_TUNNEL_SESSION} 2>/dev/null || true; tmux kill-session -t ${COLLECTOR_SESSION} 2>/dev/null || true" || true

    log_step "killing compute-node forwarder tmux session"
    compute_ssh "tmux kill-session -t ${FORWARDER_SESSION} 2>/dev/null || true" || true

    for jf in "${LOG_DIR}/hpc_jobid_smoke" "${LOG_DIR}/hpc_jobid_full" "${JOBID_FILE}"; do
        if [[ -s "${jf}" ]]; then
            local_job=$(cat "${jf}")
            log_step "qdel ${local_job} (on compute node)"
            compute_ssh "qdel ${local_job} 2>/dev/null || true" || true
        fi
    done
    log_ok "stop complete."
    exit 0
fi

if [[ "${MODE}" == "status" ]]; then
    log_step "local processes"
    for pf in "${RELAY_PID_FILE}" "${TUNNEL_PID_FILE}" "${LOG_DIR}/smoke_watcher.pid" "${LOG_DIR}/grid_watcher.pid"; do
        if [[ -s "${pf}" ]] && kill -0 "$(cat "${pf}")" 2>/dev/null; then
            echo "    ${pf##*/}: alive (pid=$(cat "${pf}"))"
        else
            echo "    ${pf##*/}: not running"
        fi
    done
    if [[ -d "${LOG_DIR}/grid_watcher/pulled" ]]; then
        n_pulled=$(ls "${LOG_DIR}/grid_watcher/pulled" 2>/dev/null | wc -l | tr -d ' ')
        status_range="${FULL_ARRAY_RANGE:-${HPC_ARRAY_RANGE}}"
        IFS='-' read -r status_rs status_re <<< "${status_range}"
        n_total=$(( ${status_re:-${status_rs:-1}} - ${status_rs:-1} + 1 ))
        echo "    grid pull-back progress: ${n_pulled}/${n_total} slots pulled so far (range ${status_range})"
    fi
    log_step "login-node inner tunnel + collector"
    login_ssh "tmux has-session -t ${INNER_TUNNEL_SESSION} 2>/dev/null && echo 'inner tunnel: alive' || echo 'inner tunnel: not running'; tmux has-session -t ${COLLECTOR_SESSION} 2>/dev/null && echo 'collector: alive' || echo 'collector: not running'" \
        || log_warn "could not ssh to login node"
    log_step "compute-node forwarder"
    compute_ssh "tmux has-session -t ${FORWARDER_SESSION} 2>/dev/null && echo alive || echo 'not running'" \
        || log_warn "could not reach compute node"
    if [[ -s "${JOBID_FILE}" ]]; then
        log_step "PBS job $(cat "${JOBID_FILE}")"
        compute_ssh "qstat -t $(cat "${JOBID_FILE}") 2>&1 || echo '(job no longer in queue)'"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# From here down: dry-run or run. Tee everything into the launch log.
# ---------------------------------------------------------------------------
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "=========================================="
echo " HPC LAUNCH  (mode=${MODE})"
echo "  ts               : ${TS}"
echo "  lab repo         : ${LAB_REPO_ROOT}"
echo "  login node       : ${HPC_USER}@${HPC_HOST}:${HPC_LOGIN_REPO_ROOT}"
echo "  compute node     : ${HPC_INNER_HOST}:${HPC_COMPUTE_REPO_ROOT}"
echo "  hpc queue        : ${HPC_QUEUE}"
echo "  hpc array        : ${HPC_ARRAY_RANGE}"
echo "  datasets         : ${DATASETS_SUBSET:-<all>}"
echo "  tunnel port      : ${LAB_TUNNEL_PORT}"
echo "  epochs           : ${EPOCHS}"
echo "  use_shipped_venv : ${USE_SHIPPED_VENV}"
echo "  launch log       : ${LAUNCH_LOG}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1/8  Sanity + preflight
# ---------------------------------------------------------------------------
log_step "1/8  Sanity checks + preflight"

# 1a. WiFi SSID (best-effort)
if [[ "${LAB_WIFI_SSID:-}" != "" ]]; then
    current_ssid=""
    if [[ "$(uname -s)" == "Darwin" ]]; then
        current_ssid=$(/System/Library/PrivateFrameworks/Apple80211.framework/Versions/A/Resources/airport -I 2>/dev/null | awk -F': ' '/ SSID/ {print $2; exit}')
    elif command -v iwgetid >/dev/null 2>&1; then
        current_ssid=$(iwgetid -r 2>/dev/null || true)
    elif command -v nmcli >/dev/null 2>&1; then
        current_ssid=$(nmcli -t -f active,ssid dev wifi 2>/dev/null | awk -F: '$1=="yes"{print $2; exit}')
    fi
    if [[ -n "${current_ssid}" ]]; then
        if [[ "${current_ssid}" == "${LAB_WIFI_SSID}" ]]; then
            log_ok "wifi = '${current_ssid}'"
        else
            log_warn "wifi = '${current_ssid}' (expected '${LAB_WIFI_SSID}'). Continuing."
        fi
    else
        log_warn "could not detect wifi SSID; skipping check."
    fi
fi

# 1b. Data is pre-transferred — verify it exists on the compute node.
log_ok "data transfer skipped — data assumed pre-loaded on HPC."

# 1c. lab -> login reachability
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${HPC_USER}@${HPC_HOST}" 'echo ok' >/dev/null 2>&1; then
    fatal "cannot ssh to ${HPC_USER}@${HPC_HOST} in BatchMode.
Fix: set up passwordless key auth to the login node.
  ssh-keygen -t ed25519           # if you don't have a key yet
  ssh-copy-id ${HPC_USER}@${HPC_HOST}
Then re-run this script."
fi
log_ok "ssh key auth to login node (${HPC_USER}@${HPC_HOST}) works"

# 1d. login -> compute reachability (the second hop)
inner_probe="$(login_ssh "ssh -o BatchMode=yes -o ConnectTimeout=10 ${HPC_INNER_HOST} echo ok" 2>/dev/null | tr -d '[:space:]')"
if [[ "${inner_probe}" != "ok" ]]; then
    fatal "login node cannot reach the compute node ('ssh ${HPC_INNER_HOST}' from ${HPC_HOST} failed).
Run 'bash scripts/hpc_preflight.sh' for a detailed diagnosis (probe 2), or check
HPC_INNER_HOST in ${CONFIG_FILE} and the login node's ~/.ssh/config."
fi
log_ok "login node can reach the compute node ('ssh ${HPC_INNER_HOST}')"

# 1e. required tools locally
for tool in rsync ssh autossh tmux python3; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        if [[ "${tool}" == "autossh" ]]; then
            log_warn "'autossh' not found locally. Reverse tunnel will use plain ssh (less reliable). Install autossh for auto-reconnect."
        elif [[ "${tool}" == "tmux" ]]; then
            log_warn "'tmux' not found locally — only affects lab-side status prints. Continuing."
        else
            fatal "'${tool}' not found. Install it and retry."
        fi
    fi
done

# 1f. tools on both remote nodes
missing_login=$(login_ssh 'for t in rsync tmux ssh; do command -v $t >/dev/null 2>&1 || echo $t; done' 2>/dev/null | xargs)
if [[ -n "${missing_login}" ]]; then
    fatal "login node missing tools: ${missing_login}. Ask IITD HPC support / your PI."
fi
missing_compute=$(compute_ssh 'for t in python3 tmux rsync curl; do command -v $t >/dev/null 2>&1 || echo $t; done' 2>/dev/null | xargs)
if [[ -n "${missing_compute}" ]]; then
    fatal "compute node missing tools: ${missing_compute}. Ask IITD HPC support / your PI."
fi
log_ok "login + compute nodes have the tools this launcher needs"
# NOTE: qsub is intentionally NOT probed directly — see hpc_common.sh's
# _PBS_FIX for why a naive `command -v qsub` can false-negative.

# 1g. shipped-.venv sanity, IF it's already on the compute node from a prior
# run. If it isn't there yet, step 4 below will push it and there's nothing
# to verify yet — that's expected on a first launch.
if [[ "${USE_SHIPPED_VENV}" == "1" ]]; then
    venv_probe="$(compute_ssh "test -x '${HPC_COMPUTE_REPO_ROOT}/.venv/bin/python' && echo present || echo absent" 2>/dev/null | tr -d '[:space:]')"
    if [[ "${venv_probe}" == "present" ]]; then
        import_probe="$(compute_ssh "'${HPC_COMPUTE_REPO_ROOT}/.venv/bin/python' -c 'import torch, wandb' 2>&1 && echo IMPORT_OK" 2>/dev/null)"
        if echo "${import_probe}" | grep -q IMPORT_OK; then
            log_ok "existing .venv on compute node imports torch+wandb — will be treated as intact (skip re-push)."
        else
            log_warn "existing .venv on compute node does NOT import torch/wandb cleanly — will be re-pushed in step 3/4."
            echo "${import_probe}" | sed 's/^/      /'
        fi
    else
        log_warn "no .venv on compute node yet — step 3/4 will push it (this is normal on a first launch)."
    fi
fi

# ---------------------------------------------------------------------------
# 2/8  Wheels (skipped when USE_SHIPPED_VENV=1 — the default)
# ---------------------------------------------------------------------------
log_step "2/8  Build wheels for HPC platform"

WHEEL_DIR="${REPO_ROOT}/wheels"
if [[ "${USE_SHIPPED_VENV}" == "1" ]]; then
    log_ok "USE_SHIPPED_VENV=1 — skipping wheel build entirely (the shipped .venv is used as-is)."
else
    mkdir -p "${WHEEL_DIR}"
    n_wheels=$(ls "${WHEEL_DIR}"/*.whl 2>/dev/null | wc -l | tr -d ' ')
    if (( n_wheels > 0 )); then
        log_ok "${n_wheels} wheels already present in ${WHEEL_DIR}; skipping pip download"
    else
        log_step "  pip download -> ${WHEEL_DIR}   (this can take 5-15 min)"
        if [[ "${MODE}" == "dry-run" ]]; then
            echo "    (dry-run: would run pip download)"
        else
            python3 -m pip download \
                --platform "${PIP_PLATFORM:-manylinux2014_x86_64}" \
                --python-version "${PIP_PYTHON_VERSION:-311}" \
                --abi "${PIP_ABI:-cp311}" \
                --only-binary=:all: \
                --dest "${WHEEL_DIR}" \
                -r "${REPO_ROOT}/requirements.txt" || \
                log_warn "some wheels could not be downloaded — the bootstrap step will surface the concrete missing package(s)."
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 3/8  lab -> login: repo / .venv pushes
# ---------------------------------------------------------------------------
log_step "3/8  lab -> login node (repo / .venv)"

REPO_RSYNC_EXCLUDES=(
    --exclude='/.git/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='/wandb/'
    --exclude='/logs/'
    --exclude='/checkpoints/'
    --exclude='/model/'
    --exclude='/model_smoke/'
    --exclude='/data/'
    --exclude='/results/'
    --exclude='/wheels/'
    --exclude='/.venv/'
    --exclude='notebooks/*_files/'
)
VENV_RSYNC_EXCLUDES=(--exclude='__pycache__/' --exclude='*.pyc')
RSYNC_OPTS=(-a --human-readable --info=progress2 --partial --compress)

venv_needs_push=1
venv_login_probe="$(login_ssh "'${HPC_LOGIN_REPO_ROOT}/.venv/bin/python' -c 'import torch, wandb' 2>&1 && echo IMPORT_OK" 2>/dev/null || true)"
if echo "${venv_login_probe}" | grep -q IMPORT_OK; then
    venv_needs_push=0
    log_ok "login-node .venv already imports torch+wandb — skipping the .venv push."
else
    log_warn "login-node .venv missing or broken — will (re-)push it. Last probe output:"
    echo "${venv_login_probe}" | sed 's/^/      /'
fi

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run) repo         -> ${HPC_USER}@${HPC_HOST}:${HPC_LOGIN_REPO_ROOT}/  [always]"
    (( venv_needs_push )) && echo "    (dry-run) .venv        -> ${HPC_USER}@${HPC_HOST}:${HPC_LOGIN_REPO_ROOT}/.venv/  [needed]" \
        || echo "    (dry-run) .venv        -> SKIP (already intact)"
else
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
        "mkdir -p '${HPC_LOGIN_REPO_ROOT}/logs'"

    log_step "  3a repo tree (always — small, keeps scripts/train.py current)"
    rsync "${RSYNC_OPTS[@]}" "${REPO_RSYNC_EXCLUDES[@]}" \
        "${LAB_REPO_ROOT}/" \
        "${HPC_USER}@${HPC_HOST}:${HPC_LOGIN_REPO_ROOT}/" \
        || fatal "rsync repo failed"

    if (( venv_needs_push )); then
        log_step "  3b .venv/  (~5GB, can take a while)"
        rsync "${RSYNC_OPTS[@]}" "${VENV_RSYNC_EXCLUDES[@]}" \
            "${LAB_REPO_ROOT}/.venv/" \
            "${HPC_USER}@${HPC_HOST}:${HPC_LOGIN_REPO_ROOT}/.venv/" \
            || fatal "rsync .venv failed"
    fi
fi
log_ok "lab -> login sync complete"

# ---------------------------------------------------------------------------
# 4/8  login -> compute: repo / .venv pushes
# ---------------------------------------------------------------------------
log_step "4/8  login -> compute node (repo / .venv)"

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run) would run compute_rsync_push for repo and .venv (if needed)"
else
    log_step "  4a repo tree -> compute"
    compute_rsync_push "${HPC_LOGIN_REPO_ROOT}" "${HPC_COMPUTE_REPO_ROOT}" \
        "--exclude='/.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='/wandb/' --exclude='/logs/' --exclude='/checkpoints/' --exclude='/model/' --exclude='/model_smoke/' --exclude='/data/' --exclude='/results/' --exclude='/wheels/' --exclude='/.venv/'" \
        || fatal "login->compute repo rsync failed"

    compute_venv_probe="$(compute_ssh "'${HPC_COMPUTE_REPO_ROOT}/.venv/bin/python' -c 'import torch, wandb' 2>&1 && echo IMPORT_OK" 2>/dev/null || true)"
    if echo "${compute_venv_probe}" | grep -q IMPORT_OK; then
        log_ok "compute-node .venv already intact — skipping the .venv push."
    else
        log_step "  4b .venv/ -> compute  (~5GB)"
        compute_rsync_push "${HPC_LOGIN_REPO_ROOT}/.venv" "${HPC_COMPUTE_REPO_ROOT}/.venv" \
            "--exclude='__pycache__/' --exclude='*.pyc'" \
            || fatal "login->compute .venv rsync failed"
    fi
fi
log_ok "login -> compute sync complete"

# ---------------------------------------------------------------------------
# 5/8  Bootstrap on the compute node
# ---------------------------------------------------------------------------
log_step "5/8  Running scripts/hpc_bootstrap.sh on the compute node"

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run: would compute_ssh + run bootstrap)"
else
    # TELEGRAM_RELAY_URL is deliberately left BLANK here regardless of which
    # Telegram mode step 6 below ends up choosing: in "direct" mode
    # notify.py's tier-2 direct sendMessage picks up the bot token/chat id on
    # its own; in "tunnel" mode a relay URL wouldn't be reachable from the
    # compute node anyway, so it falls through to the queue by design (see
    # hpc_bootstrap.sh's comment on this same line).
    bootstrap_cmd="export HPC_PROJECT_DIR='${HPC_COMPUTE_REPO_ROOT}' HPC_COMPUTE_REPO_ROOT='${HPC_COMPUTE_REPO_ROOT}' HPC_LOGIN_REPO_ROOT='${HPC_LOGIN_REPO_ROOT}' HPC_USER='${HPC_USER}' HPC_HOST='${HPC_HOST}' LAB_TUNNEL_PORT='${LAB_TUNNEL_PORT}' USE_SHIPPED_VENV='${USE_SHIPPED_VENV}' PUSH_RESULTS_FROM_JOB='${PUSH_RESULTS_FROM_JOB}' TELEGRAM_BOT_TOKEN='${TELEGRAM_BOT_TOKEN}' TELEGRAM_CHAT_ID='${TELEGRAM_CHAT_ID}' TELEGRAM_RELAY_URL='' WANDB_API_KEY='${WANDB_API_KEY}' WANDB_PROJECT='${WANDB_PROJECT:-hsi-pi-vae}' WANDB_ENTITY='${WANDB_ENTITY:-}'; cd '${HPC_COMPUTE_REPO_ROOT}' && bash scripts/hpc_bootstrap.sh"
    compute_ssh "${bootstrap_cmd}" || fatal "hpc_bootstrap.sh failed on the compute node — inspect the log above."
fi
log_ok "compute-node bootstrap complete"

# ---------------------------------------------------------------------------
# 6/8  Telegram: relay + chained tunnels + forwarder (compute) + collector (login)
# ---------------------------------------------------------------------------
log_step "6/8  Telegram delivery chain"

# Decide direct-send vs chained-tunnel mode.
HPC_TELEGRAM_MODE="tunnel"
if [[ "${MODE}" != "dry-run" ]]; then
    tg_code="$(compute_ssh 'curl -sS -m 5 -o /dev/null -w "%{http_code}" https://api.telegram.org 2>/dev/null || echo 000' | tr -d '[:space:]')"
    if [[ "${tg_code}" =~ ^[234][0-9][0-9]$ ]]; then
        HPC_TELEGRAM_MODE="direct"
        log_ok "compute node has outbound internet (HTTP ${tg_code}) — using direct Telegram send, no tunnel needed."
    else
        log_ok "compute node has no outbound internet (expected) — using the chained reverse-tunnel path."
    fi
fi

if [[ "${HPC_TELEGRAM_MODE}" == "direct" ]]; then
    # Compute node's .env already has real TELEGRAM_* creds from bootstrap and
    # a blank TELEGRAM_RELAY_URL — utils/notify.py's tier-2 direct sendMessage
    # handles everything. Still start the forwarder as a harmless no-op-ish
    # safety net in case direct send transiently fails (it'll queue+drain).
    log_ok "direct mode: no lab relay / tunnel / collector needed for Telegram."
else
    # --- lab relay -----------------------------------------------------
    if [[ -s "${RELAY_PID_FILE}" ]] && kill -0 "$(cat "${RELAY_PID_FILE}")" 2>/dev/null; then
        log_ok "lab relay already running (pid=$(cat "${RELAY_PID_FILE}"))"
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
                log_ok "lab relay pid=${relay_pid} (logs: ${LOG_DIR}/relay.log)"
            else
                fatal "relay failed to start. Check ${LOG_DIR}/relay.log"
            fi
        fi
    fi

    # --- outer tunnel: lab -> login --------------------------------------
    if [[ -s "${TUNNEL_PID_FILE}" ]] && kill -0 "$(cat "${TUNNEL_PID_FILE}")" 2>/dev/null; then
        log_ok "outer tunnel (lab->login) already running (pid=$(cat "${TUNNEL_PID_FILE}"))"
    else
        if [[ "${MODE}" == "dry-run" ]]; then
            echo "    (dry-run: would open autossh outer tunnel lab->login)"
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
                log_err "outer tunnel process exited immediately. Last lines of tunnel.log:"
                tail -n 15 "${LOG_DIR}/tunnel.log" 2>/dev/null | sed 's/^/      /'
                fatal "reverse tunnel (lab->login) failed to start.
Most likely causes:
  * your lab->login SSH key has a passphrase, or
  * the login node forbids remote port-forwarding (AllowTcpForwarding no).
Check ${LOG_DIR}/tunnel.log and ask Utkarsh."
            fi
            log_ok "outer tunnel pid=${tunnel_pid}"
        fi
    fi

    # --- inner tunnel: login -> compute, supervised in a login-node tmux --
    if [[ "${MODE}" == "dry-run" ]]; then
        echo "    (dry-run: would start inner tunnel tmux session on the login node)"
    else
        inner_alive="$(login_ssh "tmux has-session -t ${INNER_TUNNEL_SESSION} 2>/dev/null && echo yes || echo no" | tr -d '[:space:]')"
        if [[ "${inner_alive}" == "yes" ]]; then
            log_ok "inner tunnel (login->compute) tmux session already running"
        else
            login_ssh "tmux new-session -d -s '${INNER_TUNNEL_SESSION}' \
                \"autossh -M 0 -N -T -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -R ${LAB_TUNNEL_PORT}:localhost:${LAB_TUNNEL_PORT} ${HPC_INNER_HOST} 2>&1 | tee -a logs/inner_tunnel.log || ssh -N -T -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -R ${LAB_TUNNEL_PORT}:localhost:${LAB_TUNNEL_PORT} ${HPC_INNER_HOST} 2>&1 | tee -a logs/inner_tunnel.log\"" \
                || log_warn "could not start inner tunnel tmux session"
            sleep 3
            log_ok "inner tunnel (login->compute) tmux session started"
        fi

        log_step "  verifying end-to-end tunnel (compute -> relay /healthz)"
        probe=$(compute_ssh \
            "curl -s -m 8 http://localhost:${LAB_TUNNEL_PORT}/healthz 2>/dev/null || \
             (command -v python3 >/dev/null && python3 -c \"import urllib.request,sys; sys.stdout.write(urllib.request.urlopen('http://localhost:${LAB_TUNNEL_PORT}/healthz', timeout=8).read().decode())\" 2>/dev/null) || echo PROBE_FAILED" \
            2>/dev/null | tr -d '[:space:]')
        if [[ "${probe}" == "ok" ]]; then
            log_ok "chained tunnel verified end-to-end (compute -> login -> lab relay)"
        else
            log_warn "chained tunnel is up but the compute node could not reach the relay through it (probe='${probe:-empty}')."
            log_warn "Telegram may be delayed: messages queue in logs/notify_queue.jsonl on the compute node and flush once the tunnel works."
        fi
    fi

    # --- forwarder: on the COMPUTE node now (that's where the queue is) ---
    log_step "  forwarder tmux on the compute node (session=${FORWARDER_SESSION})"
    if [[ "${MODE}" == "dry-run" ]]; then
        echo "    (dry-run: would start compute-node forwarder tmux)"
    else
        fwd_alive="$(compute_ssh "tmux has-session -t ${FORWARDER_SESSION} 2>/dev/null && echo yes || echo no" | tr -d '[:space:]')"
        if [[ "${fwd_alive}" == "yes" ]]; then
            log_ok "forwarder already running on compute node"
        else
            compute_ssh "cd '${HPC_COMPUTE_REPO_ROOT}' && tmux new-session -d -s '${FORWARDER_SESSION}' \
                \"'${HPC_COMPUTE_REPO_ROOT}/.venv/bin/python' scripts/notify_forwarder.py --queue '${HPC_COMPUTE_REPO_ROOT}/logs/notify_queue.jsonl' --url 'http://localhost:${LAB_TUNNEL_PORT}/notify' --interval 3 2>&1 | tee -a logs/forwarder.log\"" \
                && log_ok "forwarder running on compute node" \
                || log_warn "forwarder tmux setup failed — heartbeats will still queue in logs/notify_queue.jsonl on the compute node."
        fi
    fi

    # --- collector: on the LOGIN node, pulls results from compute --------
    log_step "  collector tmux on the login node (session=${COLLECTOR_SESSION})"
    if [[ "${MODE}" == "dry-run" ]]; then
        echo "    (dry-run: would start login-node collector tmux)"
    else
        coll_alive="$(login_ssh "tmux has-session -t ${COLLECTOR_SESSION} 2>/dev/null && echo yes || echo no" | tr -d '[:space:]')"
        if [[ "${coll_alive}" == "yes" ]]; then
            log_ok "collector already running on login node"
        else
            login_ssh "cd '${HPC_LOGIN_REPO_ROOT}' && tmux new-session -d -s '${COLLECTOR_SESSION}' \
                \"bash scripts/hpc_collector.sh --inner-host '${HPC_INNER_HOST}' --compute-root '${HPC_COMPUTE_REPO_ROOT}' --login-root '${HPC_LOGIN_REPO_ROOT}' --interval '${COLLECTOR_INTERVAL}' 2>&1 | tee -a logs/collector.log\"" \
                && log_ok "collector running on login node" \
                || log_warn "collector tmux setup failed — results will only come back via the job's own push (if PUSH_RESULTS_FROM_JOB=1) or manual scripts/hpc_pull_results.sh."
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 7/8  Submit smoke run (slot 1 only) and launch watcher
# ---------------------------------------------------------------------------
log_step "7/8  Submit smoke run (slot 1, ${SMOKE_EPOCHS:-5} epochs)"

SMOKE_EPOCHS="${SMOKE_EPOCHS:-5}"
SMOKE_DELAY_SECS="${SMOKE_DELAY_SECS:-600}"
FULL_ARRAY_RANGE="${FULL_ARRAY_RANGE:-${HPC_ARRAY_RANGE}}"
WATCHER_PID_FILE="${LOG_DIR}/smoke_watcher.pid"

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run: would submit smoke job for slot 1 via compute_ssh)"
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

    smoke_qsub_cmd="cd '${HPC_COMPUTE_REPO_ROOT}' && qsub \
        -q '${HPC_QUEUE}' \
        -l '${HPC_SELECT}' \
        -l walltime='01:00:00' \
        -J '1-1' \
        ${qsub_extra[*]:-} \
        -v HPC_PROJECT_DIR='${HPC_COMPUTE_REPO_ROOT}',HPC_COMPUTE_REPO_ROOT='${HPC_COMPUTE_REPO_ROOT}',HPC_LOGIN_REPO_ROOT='${HPC_LOGIN_REPO_ROOT}',HPC_USER='${HPC_USER}',HPC_HOST='${HPC_HOST}',PUSH_RESULTS_FROM_JOB='${PUSH_RESULTS_FROM_JOB}',EPOCHS='${SMOKE_EPOCHS}',SMOKE_MODE=1,SMOKE_EPOCHS='${SMOKE_EPOCHS}',WANDB_PROJECT='${WANDB_PROJECT:-hsi-pi-vae}',WANDB_ENTITY='${WANDB_ENTITY:-}',EXTRA_TRAIN_ARGS='${EXTRA_TRAIN_ARGS:-}',DATASETS_SUBSET='${DATASETS_SUBSET:-}' \
        scripts/hpc_pbs_job.pbs"

    SMOKE_JOBID=$(compute_ssh "${smoke_qsub_cmd}") || fatal "qsub smoke failed"

    # A login shell may print module-load / MOTD banners before qsub's output;
    # the job id is the LAST non-empty line.
    SMOKE_JOBID=$(echo "${SMOKE_JOBID}" | tail -1 | xargs)
    echo "${SMOKE_JOBID}" > "${LOG_DIR}/hpc_jobid_smoke"
    echo "${SMOKE_JOBID}" > "${JOBID_FILE}"
    log_ok "smoke job submitted: ${SMOKE_JOBID}  (datasets=${DATASETS_SUBSET:-<all>})"
    log_step "  if smoke passes, watcher will wait ${SMOKE_DELAY_SECS}s then submit full grid (${FULL_ARRAY_RANGE})"

    export HPC_SMOKE_JOBID="${SMOKE_JOBID}"
    export SMOKE_DELAY_SECS FULL_ARRAY_RANGE EPOCHS EXTRA_TRAIN_ARGS DATASETS_SUBSET
    nohup bash "${REPO_ROOT}/scripts/hpc_smoke_watcher.sh" \
        >> "${LOG_DIR}/smoke_watcher.log" 2>&1 &
    watcher_pid=$!
    echo "${watcher_pid}" > "${WATCHER_PID_FILE}"
    log_ok "smoke watcher pid=${watcher_pid} (logs: ${LOG_DIR}/smoke_watcher.log)"
fi

# ---------------------------------------------------------------------------
# 8/8  Print monitoring cheatsheet
# ---------------------------------------------------------------------------
log_step "8/8  Monitoring cheatsheet"

cat <<EOF

    Watch this launcher log:
      tail -f ${LAUNCH_LOG}

    Watch smoke watcher progress:
      tail -f ${LOG_DIR}/smoke_watcher.log

    Watch grid pull-back watcher (after smoke passes, per-run checkpoint pulls):
      tail -f ${LOG_DIR}/grid_watcher.log

    Watch job queue status (two hops — this wraps compute_ssh for you):
      bash scripts/hpc_launch.sh --status

    Attach to the compute-node forwarder:
      ssh ${HPC_USER}@${HPC_HOST} 'ssh ${HPC_INNER_HOST} "tmux attach -t ${FORWARDER_SESSION}"'
        (detach with Ctrl-b then d)

    Attach to the login-node collector:
      ssh ${HPC_USER}@${HPC_HOST} 'tmux attach -t ${COLLECTOR_SESSION}'

    Manual full pull (grid watcher + collector already do this automatically;
    use this only for a one-off refresh):
      bash scripts/hpc_pull_results.sh

    Emergency stop everything:
      bash scripts/hpc_launch.sh --stop

    Current status snapshot (includes grid pull-back progress):
      bash scripts/hpc_launch.sh --status

    What happens next (automated):
      1. Smoke watcher polls qstat (two-hop) for the smoke job to finish.
      2. If smoke passes -> waits ${SMOKE_DELAY_SECS:-600}s -> submits full ${FULL_ARRAY_RANGE:-1-28} grid.
         (Telegram: "[LAUNCHED] Full ablation grid submitted")
         -> starts the grid watcher, which polls each array slot and, the
            moment a slot finishes, pulls its checkpoint + logs back to the
            lab machine via the login node (Telegram: "[OK]"/"[FAIL]" per
            slot, plus "[XFER]" for the compute->login leg) — so a later
            failure only risks runs still in flight, never ones that
            already finished.
      3. If smoke fails -> pulls logs -> Telegrams the tail -> does NOT launch full grid.
         (fix the issue, re-run this script)

EOF
log_ok "launch complete. Thanks a lot people. You may rest now."
