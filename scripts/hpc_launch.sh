#!/usr/bin/env bash
# scripts/hpc_launch.sh
# ---------------------
# Single-entry-point launcher for the IITD HPC ablation run.
#
# What this does, in order:
#
#   1. Sanity checks (wifi SSID, config env, local processed data, ssh key).
#   2. Downloads pip wheels for the HPC's platform (once, cached).
#   3. Rsyncs repo + wheels + data/processed/ to the HPC.
#   4. Runs scripts/hpc_bootstrap.sh on the HPC login node (offline pip install).
#   5. Starts utils/notify_relay.py on the lab machine (backgrounded).
#   6. Brings up an autossh reverse tunnel: HPC :$PORT -> lab :$PORT.
#   7. Starts scripts/notify_forwarder.py on the HPC login node inside tmux.
#   8. qsubs scripts/hpc_pbs_job.pbs as an array over $HPC_ARRAY_RANGE.
#   9. Prints the qstat / tail-monitoring commands.
#
# Modes:
#   bash scripts/hpc_launch.sh                 # full run
#   bash scripts/hpc_launch.sh --dry-run       # sanity + print-only (no qsub, no rsync)
#   bash scripts/hpc_launch.sh --stop          # kill tunnel + forwarder + relay + qdel array
#   bash scripts/hpc_launch.sh --status        # what's alive
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
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) MODE="dry-run"; shift ;;
        --stop)    MODE="stop"; shift ;;
        --status)  MODE="status"; shift ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *) fatal "unknown flag: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_FILE="${SCRIPT_DIR}/hpc_config.env"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    fatal "config not found: ${CONFIG_FILE}
Copy ${CONFIG_FILE}.example -> ${CONFIG_FILE} and fill the FILL_ME blanks.
See docs/hpc_wiki.md for how to obtain each value."
fi
# shellcheck source=/dev/null
set -a; source "${CONFIG_FILE}"; set +a

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
_check_filled HPC_SCRATCH
_check_filled HPC_QUEUE
_check_filled TELEGRAM_BOT_TOKEN
_check_filled TELEGRAM_CHAT_ID
_check_filled WANDB_API_KEY
export HPC_PROJECT_DIR="${HPC_PROJECT_DIR:-${HPC_HOME}/prism}"
export LAB_TUNNEL_PORT="${LAB_TUNNEL_PORT:-8765}"
export LAB_REPO_ROOT="${LAB_REPO_ROOT:-${REPO_ROOT}}"
export EPOCHS="${EPOCHS:-100}"
HPC_ARRAY_RANGE="${HPC_ARRAY_RANGE:-1-28}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
LAUNCH_LOG="${LOG_DIR}/hpc_launch_${TS}.log"

RELAY_PID_FILE="${LOG_DIR}/relay.pid"
TUNNEL_PID_FILE="${LOG_DIR}/tunnel.pid"
JOBID_FILE="${LOG_DIR}/hpc_jobid"
FORWARDER_SESSION="specsteer_forwarder"

# ---------------------------------------------------------------------------
# hpc_ssh — run a remote command in a LOGIN shell.
#
# IITD's PBS Pro only puts the scheduler binaries (qsub/qstat/qdel) on PATH via
# /etc/profile.d, which a plain non-interactive `ssh host 'cmd'` never sources —
# that's the "qsub: command not found" (/opt/pbs/.../bin/qsub missing) the
# junior hit. Piping the command over stdin to `bash -l -s` sources the login
# profile AND sidesteps nested-quoting issues in the command string. Use this
# for anything that calls qsub/qstat/qdel; plain ssh is fine for tmux/tail/etc.
# ---------------------------------------------------------------------------
hpc_ssh() {
    # Even `bash -l -s` may miss the PBS PATH when the scheduler binaries are
    # added by an interactive-only rc guard (the actual "qsub: command not
    # found" the junior hit). Prepend a best-effort fix: read PBS_EXEC from
    # /etc/pbs.conf (present on every PBS Pro node) and fall back to the common
    # install dirs. No-op for non-PBS commands (tmux/tail) since qsub is found.
    local pbs_fix='if ! command -v qsub >/dev/null 2>&1; then [ -r /etc/pbs.conf ] && . /etc/pbs.conf; for d in "${PBS_EXEC:-/opt/pbs}/bin" /opt/pbs/bin /opt/pbs/default/bin; do [ -x "$d/qsub" ] && export PATH="$d:$PATH" && break; done; fi;'
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" 'bash -l -s' <<< "${pbs_fix} $1"
}

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

    log_step "killing HPC forwarder tmux session"
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
        "tmux kill-session -t ${FORWARDER_SESSION} 2>/dev/null || true" || true

    # Kill both smoke and full jobs if present.
    for jf in "${LOG_DIR}/hpc_jobid_smoke" "${LOG_DIR}/hpc_jobid_full" "${JOBID_FILE}"; do
        if [[ -s "${jf}" ]]; then
            local_job=$(cat "${jf}")
            log_step "qdel ${local_job}"
            hpc_ssh "qdel ${local_job} 2>/dev/null || true" || true
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
    log_step "HPC forwarder"
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
        "tmux has-session -t ${FORWARDER_SESSION} 2>/dev/null && echo alive || echo not running" \
        || log_warn "could not ssh"
    if [[ -s "${JOBID_FILE}" ]]; then
        log_step "PBS job $(cat "${JOBID_FILE}")"
        hpc_ssh "qstat -t $(cat "${JOBID_FILE}") 2>&1 || echo '(job no longer in queue)'"
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
echo "  lab data root    : ${LAB_DATA_ROOT}"
echo "  hpc target       : ${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}"
echo "  hpc queue        : ${HPC_QUEUE}"
echo "  hpc array        : ${HPC_ARRAY_RANGE}"
echo "  tunnel port      : ${LAB_TUNNEL_PORT}"
echo "  epochs           : ${EPOCHS}"
echo "  launch log       : ${LAUNCH_LOG}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
log_step "1/9  Sanity checks"

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
            log_warn "wifi = '${current_ssid}' (expected '${LAB_WIFI_SSID}'). Continuing — but if uploads stall, switch networks."
        fi
    else
        log_warn "could not detect wifi SSID; skipping check."
    fi
fi

# 1b. Local processed data present
if [[ ! -d "${LAB_DATA_ROOT}" ]]; then
    fatal "processed data not found at:
         ${LAB_DATA_ROOT}
Is the external drive mounted? Adjust LAB_DATA_ROOT in ${CONFIG_FILE}."
fi
n_iirs=$(find "${LAB_DATA_ROOT}/IIRS"   -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
n_m3=$(find   "${LAB_DATA_ROOT}/M3"     -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
n_av=$(find   "${LAB_DATA_ROOT}/AVIRIS" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
n_cr=$(find   "${LAB_DATA_ROOT}/CRIMS"  -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
log_ok "patches: IIRS=${n_iirs}  M3=${n_m3}  AVIRIS=${n_av}  CRIMS=${n_cr}"
if (( n_iirs + n_m3 + n_av + n_cr < 1000 )); then
    fatal "very few patches under ${LAB_DATA_ROOT} — did preprocessing run?"
fi

# 1c. ssh reachability (BatchMode: fails fast if key auth isn't set up)
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${HPC_USER}@${HPC_HOST}" 'echo ok' >/dev/null 2>&1; then
    fatal "cannot ssh to ${HPC_USER}@${HPC_HOST} in BatchMode.
Fix: set up passwordless key auth to the HPC.
  ssh-keygen -t ed25519           # if you don't have a key yet
  ssh-copy-id ${HPC_USER}@${HPC_HOST}
Then re-run this script."
fi
log_ok "ssh key auth to ${HPC_USER}@${HPC_HOST} works"

# 1d. required tools locally
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

# 1e. hpc has the tools we need
# NOTE: qsub is intentionally NOT checked here. On IITD's PBS Pro the scheduler
# binaries are not on the login-node PATH for non-interactive ssh sessions; the
# queue is assigned automatically when a node becomes available. Probing for
# qsub gives a false negative and blocks launch.
missing_hpc=$(ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
    'for t in python3 tmux; do command -v $t >/dev/null 2>&1 || echo $t; done' 2>/dev/null | xargs)
if [[ -n "${missing_hpc}" ]]; then
    fatal "HPC missing tools: ${missing_hpc}. Ask IITD HPC support / your PI."
fi
log_ok "hpc has python3, tmux"

# ---------------------------------------------------------------------------
# 2. Build wheels
# ---------------------------------------------------------------------------
log_step "2/9  Build wheels for HPC platform"

WHEEL_DIR="${REPO_ROOT}/wheels"
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

# ---------------------------------------------------------------------------
# 3. Rsync repo + wheels + data
# ---------------------------------------------------------------------------
log_step "3/9  Rsync to HPC (repo + wheels + processed data)"

RSYNC_OPTS=(-a --human-readable --info=progress2 --partial --compress)

RSYNC_EXCLUDES=(
    --exclude='.git/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='wandb/'
    --exclude='logs/'
    --exclude='checkpoints/'
    --exclude='model/'
    --exclude='data/'
    --exclude='results/'
    --exclude='notebooks/*_files/'
)

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run: would rsync)"
    echo "    repo -> ${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/"
    echo "    wheels -> ${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/wheels/"
    echo "    data  -> ${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/data/processed/"
else
    # Ensure remote parents exist.
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
        "mkdir -p '${HPC_PROJECT_DIR}/data/processed' '${HPC_PROJECT_DIR}/wheels' '${HPC_PROJECT_DIR}/logs'"

    log_step "  3a repo tree"
    rsync "${RSYNC_OPTS[@]}" "${RSYNC_EXCLUDES[@]}" \
        "${LAB_REPO_ROOT}/" \
        "${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/" \
        || fatal "rsync repo failed"

    log_step "  3b wheels/"
    rsync "${RSYNC_OPTS[@]}" \
        "${WHEEL_DIR}/" \
        "${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/wheels/" \
        || fatal "rsync wheels failed"

    log_step "  3c data/processed/  (this is the long one — expect 1-3 h over lab wifi)"
    rsync "${RSYNC_OPTS[@]}" \
        "${LAB_DATA_ROOT}/" \
        "${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/data/processed/" \
        || fatal "rsync data failed"
fi
log_ok "rsync complete"

# ---------------------------------------------------------------------------
# 4. Bootstrap on HPC (offline pip install)
# ---------------------------------------------------------------------------
log_step "4/9  Running scripts/hpc_bootstrap.sh on HPC"

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run: would ssh + run bootstrap)"
else
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" bash <<EOF
set -euo pipefail
export HPC_PROJECT_DIR='${HPC_PROJECT_DIR}'
export LAB_TUNNEL_PORT='${LAB_TUNNEL_PORT}'
export TELEGRAM_BOT_TOKEN='${TELEGRAM_BOT_TOKEN}'
export TELEGRAM_CHAT_ID='${TELEGRAM_CHAT_ID}'
export WANDB_API_KEY='${WANDB_API_KEY}'
export WANDB_PROJECT='${WANDB_PROJECT:-hsi-pi-vae}'
export WANDB_ENTITY='${WANDB_ENTITY:-}'
cd '${HPC_PROJECT_DIR}'
bash scripts/hpc_bootstrap.sh
EOF
    if [[ $? -ne 0 ]]; then
        fatal "hpc_bootstrap.sh failed on the HPC — inspect the log above."
    fi
fi
log_ok "hpc bootstrap complete"

# ---------------------------------------------------------------------------
# 5. Start local relay
# ---------------------------------------------------------------------------
log_step "5/9  Start lab-side notify_relay.py (background)"

if [[ -s "${RELAY_PID_FILE}" ]] && kill -0 "$(cat "${RELAY_PID_FILE}")" 2>/dev/null; then
    log_ok "already running (pid=$(cat "${RELAY_PID_FILE}"))"
else
    if [[ "${MODE}" == "dry-run" ]]; then
        echo "    (dry-run: would launch notify_relay.py)"
    else
        # Export creds so the relay's TelegramNotifier picks them up.
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

# ---------------------------------------------------------------------------
# 6. Reverse SSH tunnel
# ---------------------------------------------------------------------------
log_step "6/9  Reverse SSH tunnel (HPC:${LAB_TUNNEL_PORT} -> lab:${LAB_TUNNEL_PORT})"

if [[ -s "${TUNNEL_PID_FILE}" ]] && kill -0 "$(cat "${TUNNEL_PID_FILE}")" 2>/dev/null; then
    log_ok "already running (pid=$(cat "${TUNNEL_PID_FILE}"))"
else
    if [[ "${MODE}" == "dry-run" ]]; then
        echo "    (dry-run: would open autossh tunnel)"
    else
        # The tunnel authenticates lab -> HPC using the SAME key that Step 1c
        # already proved works in BatchMode. We repeat BatchMode=yes here so
        # that if the key ever needs a passphrase, the tunnel dies immediately
        # (which we detect below) instead of silently blocking on a hidden
        # password prompt inside nohup.
        #
        # ExitOnForwardFailure=yes makes ssh exit non-zero if the HPC refuses
        # the remote (-R) forward (e.g. sshd 'AllowTcpForwarding no'), so a
        # forwarding-disabled login node is caught rather than looking "up".
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
            fatal "reverse tunnel failed to start.
Most likely causes:
  * your lab->HPC SSH key has a passphrase (make a passphraseless key, or
    load it into an agent: eval \$(ssh-agent); ssh-add ~/.ssh/id_ed25519), or
  * the HPC login node forbids remote port-forwarding (AllowTcpForwarding no).
Check ${LOG_DIR}/tunnel.log and ask Utkarsh."
        fi

        # End-to-end probe: from the HPC login node, curl the relay THROUGH the
        # reverse tunnel. This is the only check that proves the -R forward is
        # actually live (a running ssh process alone does not guarantee it).
        log_step "  verifying tunnel end-to-end (HPC -> relay /healthz)"
        probe=$(ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
            "curl -s -m 8 http://localhost:${LAB_TUNNEL_PORT}/healthz 2>/dev/null || \
             (command -v python3 >/dev/null && python3 -c \"import urllib.request,sys; sys.stdout.write(urllib.request.urlopen('http://localhost:${LAB_TUNNEL_PORT}/healthz', timeout=8).read().decode())\" 2>/dev/null) || echo PROBE_FAILED" \
            2>/dev/null | tr -d '[:space:]')
        if [[ "${probe}" == "ok" ]]; then
            log_ok "tunnel verified end-to-end (pid=${tunnel_pid})"
        else
            log_warn "tunnel process is up (pid=${tunnel_pid}) but the HPC could not reach the relay through it (probe='${probe:-empty}')."
            log_warn "Telegram may be delayed: messages still accumulate in logs/notify_queue.jsonl on the HPC and flush once the tunnel works."
            log_warn "If this persists, the login node likely blocks remote forwarding — tell Utkarsh; he may need a different relay path."
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 7. Start HPC-side forwarder in tmux
# ---------------------------------------------------------------------------
log_step "7/9  Start HPC forwarder in tmux (session=${FORWARDER_SESSION})"

if [[ "${MODE}" == "dry-run" ]]; then
    echo "    (dry-run: would start tmux forwarder)"
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
        log_warn "forwarder tmux setup failed. Heartbeats will still land in logs/notify_queue.jsonl on the HPC — pull them back and drain manually."
    else
        log_ok "forwarder running on HPC (attach: ssh ${HPC_USER}@${HPC_HOST} 'tmux attach -t ${FORWARDER_SESSION}')"
    fi
fi

# ---------------------------------------------------------------------------
# 8. Submit smoke run (slot 1 only, few epochs) and launch watcher
# ---------------------------------------------------------------------------
log_step "8/9  Submit smoke run (slot 1, ${SMOKE_EPOCHS:-5} epochs)"

SMOKE_EPOCHS="${SMOKE_EPOCHS:-5}"
SMOKE_DELAY_SECS="${SMOKE_DELAY_SECS:-600}"
FULL_ARRAY_RANGE="${FULL_ARRAY_RANGE:-${HPC_ARRAY_RANGE}}"
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

    SMOKE_JOBID=$(hpc_ssh "${smoke_qsub_cmd}") || \
        fatal "qsub smoke failed"

    # A login shell may print module-load / MOTD banners before qsub's output;
    # the job id is the LAST non-empty line. (Mirrors the full-array capture in
    # hpc_smoke_watcher.sh.)
    SMOKE_JOBID=$(echo "${SMOKE_JOBID}" | tail -1 | xargs)
    echo "${SMOKE_JOBID}" > "${LOG_DIR}/hpc_jobid_smoke"
    echo "${SMOKE_JOBID}" > "${JOBID_FILE}"
    log_ok "smoke job submitted: ${SMOKE_JOBID}"
    log_step "  if smoke passes, watcher will wait ${SMOKE_DELAY_SECS}s then submit full grid (${FULL_ARRAY_RANGE})"

    # Launch the watcher in the background (nohup'd so the junior can close the terminal).
    export HPC_SMOKE_JOBID="${SMOKE_JOBID}"
    export SMOKE_DELAY_SECS FULL_ARRAY_RANGE EPOCHS EXTRA_TRAIN_ARGS
    nohup bash "${REPO_ROOT}/scripts/hpc_smoke_watcher.sh" \
        >> "${LOG_DIR}/smoke_watcher.log" 2>&1 &
    watcher_pid=$!
    echo "${watcher_pid}" > "${WATCHER_PID_FILE}"
    log_ok "smoke watcher pid=${watcher_pid} (logs: ${LOG_DIR}/smoke_watcher.log)"
fi

# ---------------------------------------------------------------------------
# 9. Print monitoring cheatsheet
# ---------------------------------------------------------------------------
log_step "9/9  Monitoring cheatsheet"

cat <<EOF

    Watch this launcher log:
      tail -f ${LAUNCH_LOG}

    Watch smoke watcher progress:
      tail -f ${LOG_DIR}/smoke_watcher.log

    Watch grid pull-back watcher (after smoke passes, per-run checkpoint pulls):
      tail -f ${LOG_DIR}/grid_watcher.log

    Watch job queue status (qstat needs a login shell for PATH -> use ssh -t):
      ssh -t ${HPC_USER}@${HPC_HOST} "qstat -t $(cat "${JOBID_FILE}" 2>/dev/null)"

    Attach to the HPC forwarder:
      ssh ${HPC_USER}@${HPC_HOST} 'tmux attach -t ${FORWARDER_SESSION}'
        (detach with Ctrl-b then d)

    Tail a specific run's log (once it starts on a compute node):
      ssh ${HPC_USER}@${HPC_HOST} 'tail -f ${HPC_PROJECT_DIR}/logs/train_*.log'

    Manual full pull (grid watcher already does this per-slot automatically;
    use this only for a one-off refresh, e.g. to grab wandb/ mid-run):
      bash scripts/hpc_pull_results.sh

    Emergency stop everything:
      bash scripts/hpc_launch.sh --stop

    Current status snapshot (includes grid pull-back progress):
      bash scripts/hpc_launch.sh --status

    What happens next (automated):
      1. Smoke watcher polls qstat for the smoke job to finish.
      2. If smoke passes → waits ${SMOKE_DELAY_SECS:-600}s → submits full ${FULL_ARRAY_RANGE:-1-28} grid.
         (Telegram: "[LAUNCHED] Full ablation grid submitted")
         → starts the grid watcher, which polls each array slot and, the moment
           a slot finishes, immediately pulls its checkpoint + logs back to the
           lab machine (Telegram: "[OK]"/"[FAIL] slot N/M done: ...") — so a
           later failure (tunnel drop, HPC outage, lab machine reboot) only
           risks runs still in flight, never runs that already finished.
      3. If smoke fails  → pulls logs → Telegrams the tail → does NOT launch full grid.
         (fix the issue, re-run this script)

EOF
log_ok "launch complete. Thanks a lot people. You may rest now."
