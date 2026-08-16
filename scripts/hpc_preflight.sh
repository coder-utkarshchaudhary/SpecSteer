#!/usr/bin/env bash
# scripts/hpc_preflight.sh
# ------------------------
# READ-ONLY. Run this BEFORE scripts/hpc_launch.sh, and any time the HPC
# topology changes (new login node, venv re-shipped, etc). It answers
# questions no amount of reading the scripts can: whether the two-hop SSH
# path actually works, whether the shipped .venv can run on the compute
# node, and which Telegram/result-return mode is available.
#
# Nothing here mutates state on any machine (aside from a throwaway rsync
# dry-run comparison in probe 8, which touches nothing outside /tmp).
#
# Usage:
#   bash scripts/hpc_preflight.sh
#
# Exit code is nonzero iff a BLOCKING probe (1-4) failed — those are the ones
# scripts/hpc_launch.sh also runs inline and refuses to proceed past.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -t 1 ]]; then
    C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_B=$'\033[34m'; C_N=$'\033[0m'
else
    C_R=""; C_G=""; C_Y=""; C_B=""; C_N=""
fi
step() { echo ""; echo "${C_B}>>> $*${C_N}"; }
ok()   { echo "${C_G}    OK: $*${C_N}"; }
warn() { echo "${C_Y}    WARN: $*${C_N}"; }
err()  { echo "${C_R}    FAIL: $*${C_N}"; }

CONFIG_FILE="${SCRIPT_DIR}/hpc_config.env"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    err "config not found: ${CONFIG_FILE}. Copy .example -> hpc_config.env and fill it first."
    exit 1
fi
# shellcheck source=/dev/null
set -a; source "${CONFIG_FILE}"; set +a

# shellcheck source=hpc_common.sh
source "${SCRIPT_DIR}/hpc_common.sh"
resolve_hpc_roots

BLOCKED=0

echo "=========================================="
echo " HPC PREFLIGHT"
echo "  login node   : ${HPC_USER}@${HPC_HOST}"
echo "  inner (hpc)  : ${HPC_INNER_HOST}"
echo "  login root   : ${HPC_LOGIN_REPO_ROOT}"
echo "  compute root : ${HPC_COMPUTE_REPO_ROOT}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. lab -> login
# ---------------------------------------------------------------------------
step "1. lab -> login node (ssh ${HPC_USER}@${HPC_HOST})"
if ssh -o BatchMode=yes -o ConnectTimeout=10 "${HPC_USER}@${HPC_HOST}" 'echo ok' 2>/dev/null | grep -q ok; then
    ok "reachable in BatchMode"
else
    err "cannot reach the login node in BatchMode. Set up passwordless key auth first (ssh-copy-id)."
    BLOCKED=1
fi

# ---------------------------------------------------------------------------
# 2. login -> compute
# ---------------------------------------------------------------------------
step "2. login -> compute node (ssh ${HPC_INNER_HOST} from the login node)"
inner_probe="$(login_ssh "ssh -o BatchMode=yes -o ConnectTimeout=10 ${HPC_INNER_HOST} echo ok" 2>/dev/null | tr -d '[:space:]')"
if [[ "${inner_probe}" == "ok" ]]; then
    ok "login node can reach '${HPC_INNER_HOST}' in BatchMode"
else
    err "login node could NOT reach '${HPC_INNER_HOST}'. Check HPC_INNER_HOST in hpc_config.env and that the login node's ~/.ssh/config or DNS resolves it."
    echo "    diagnostic — ~/.ssh/config on the login node (Host ${HPC_INNER_HOST}):"
    login_ssh "grep -A5 '^Host ${HPC_INNER_HOST}' ~/.ssh/config 2>/dev/null || echo '(no matching Host block — ${HPC_INNER_HOST} may be a plain resolvable hostname instead, or not configured at all)'" | sed 's/^/      /'
    BLOCKED=1
fi

# ---------------------------------------------------------------------------
# 3. tool availability on both nodes
# ---------------------------------------------------------------------------
step "3. required tools on both remote nodes"
if (( BLOCKED == 0 )); then
    missing_compute="$(compute_ssh 'for t in qsub rsync tmux curl python3; do command -v "$t" >/dev/null 2>&1 || echo "$t"; done' 2>/dev/null | xargs)"
    if [[ -z "${missing_compute}" ]]; then
        ok "compute node has qsub, rsync, tmux, curl, python3"
    else
        err "compute node missing: ${missing_compute}"
        BLOCKED=1
    fi

    missing_login="$(login_ssh 'for t in rsync tmux ssh; do command -v "$t" >/dev/null 2>&1 || echo "$t"; done' 2>/dev/null | xargs)"
    if [[ -z "${missing_login}" ]]; then
        ok "login node has rsync, tmux, ssh"
    else
        err "login node missing: ${missing_login}"
        BLOCKED=1
    fi
else
    warn "skipped — hop 1/2 not established"
fi

# ---------------------------------------------------------------------------
# 4. shipped .venv — the gate for the whole "reuse it as-is" premise
# ---------------------------------------------------------------------------
step "4. shipped .venv on the compute node"
if (( BLOCKED == 0 )); then
    pyvenv_home="$(compute_ssh "grep '^home' '${HPC_COMPUTE_REPO_ROOT}/.venv/pyvenv.cfg' 2>/dev/null | sed 's/^home *= *//'" | tr -d '[:space:]')"
    if [[ -z "${pyvenv_home}" ]]; then
        warn ".venv/pyvenv.cfg not found (or unreadable) at ${HPC_COMPUTE_REPO_ROOT}/.venv on the compute node yet. This is expected before the first rsync — hpc_launch.sh will push it."
        warn "cannot verify interpreter compatibility until the venv is on the compute node — treating as non-blocking for now. Re-run preflight after hpc_launch.sh's push to confirm it actually runs."
    else
        base_python="${pyvenv_home}/python3.10"
        base_python_ok="$(compute_ssh "[ -x '${base_python}' ] && echo yes || echo no" 2>/dev/null | tr -d '[:space:]')"
        if [[ "${base_python_ok}" == "yes" ]]; then
            ok "base interpreter ${base_python} exists on the compute node"
            probe_py="$(compute_ssh "'${HPC_COMPUTE_REPO_ROOT}/.venv/bin/python' -c \"import sys,torch,wandb; print(sys.version.split()[0]); print(torch.__version__, torch.cuda.is_available())\" 2>&1")"
            if echo "${probe_py}" | tail -1 | grep -qE 'True|False'; then
                ok "shipped .venv imports torch + wandb on the compute node:"
                echo "${probe_py}" | sed 's/^/      /'
            else
                err "shipped .venv failed to import torch/wandb on the compute node. Output:"
                echo "${probe_py}" | sed 's/^/      /'
                err "THE 'REUSE THE .venv AS-IS' PREMISE IS DEAD. Do not proceed to a 28-run grid on top of this."
                echo "    Most likely cause: rsync's RSYNC_EXCLUDES were unanchored and silently dropped"
                echo "    .venv/lib/python3.10/site-packages/{wandb,logs,model,data,...}/ subfolders."
                echo "    Fix: re-push .venv with the anchored excludes (scripts/hpc_launch.sh does this now),"
                echo "    then re-run this preflight."
                BLOCKED=1
            fi
        else
            err "base interpreter ${base_python} does NOT exist on the compute node."
            err "THE 'REUSE THE .venv AS-IS' PREMISE IS DEAD — this cluster doesn't have the Python version the venv was built against."
            echo "    Options: ask IITD HPC support for a python3.10 module, or rebuild the venv"
            echo "    against whatever python3 IS on the compute node, or fall back to the"
            echo "    offline-wheels bootstrap path (set USE_SHIPPED_VENV=0 in hpc_config.env)."
            BLOCKED=1
        fi
    fi
else
    warn "skipped — hop 1/2 not established"
fi

# ---------------------------------------------------------------------------
# 5. compute node outbound internet (decides the Telegram mode)
# ---------------------------------------------------------------------------
step "5. compute node outbound internet (Telegram direct-send eligibility)"
if (( BLOCKED == 0 )); then
    code="$(compute_ssh 'curl -sS -m 5 -o /dev/null -w "%{http_code}" https://api.telegram.org 2>/dev/null || echo 000' | tr -d '[:space:]')"
    if [[ "${code}" =~ ^[234][0-9][0-9]$ ]]; then
        ok "compute node reached api.telegram.org (HTTP ${code}) — direct-send mode is available (no tunnel needed)."
        echo "    HPC_TELEGRAM_MODE=direct is viable; the chained-tunnel path still works too."
    else
        warn "compute node could NOT reach api.telegram.org (probe='${code}') — expected on IITD compute nodes."
        echo "    Will use the chained reverse-tunnel path (lab -> login -> compute)."
    fi
else
    warn "skipped — hop 1/2 not established"
fi

# ---------------------------------------------------------------------------
# 6. compute -> login (decides push-vs-pull for results)
# ---------------------------------------------------------------------------
step "6. compute -> login node (decides whether jobs can push results themselves)"
if (( BLOCKED == 0 )); then
    back_probe="$(compute_ssh "ssh -o BatchMode=yes -o ConnectTimeout=5 ${HPC_USER}@${HPC_HOST} echo ok 2>/dev/null || echo FAIL" | tr -d '[:space:]')"
    if [[ "${back_probe}" == "ok" ]]; then
        ok "compute node can reach the login node — safe to set PUSH_RESULTS_FROM_JOB=1"
    else
        warn "compute node cannot reach the login node (probe='${back_probe:-empty}') — expected if it has no route/DNS to it."
        echo "    Keep PUSH_RESULTS_FROM_JOB=0 (default). The login-node collector (pull-based) will be the only path — that's fine, just slower to notice."
    fi
else
    warn "skipped — hop 1/2 not established"
fi

# ---------------------------------------------------------------------------
# 7. disk free
# ---------------------------------------------------------------------------
step "7. disk free at both repo roots"
if (( BLOCKED == 0 )); then
    login_ssh "df -h '${HPC_LOGIN_REPO_ROOT}' 2>/dev/null || df -h \$(dirname '${HPC_LOGIN_REPO_ROOT}') 2>/dev/null" | sed 's/^/    [login]   /'
    compute_ssh "df -h '${HPC_COMPUTE_REPO_ROOT}' 2>/dev/null || df -h \$(dirname '${HPC_COMPUTE_REPO_ROOT}') 2>/dev/null" | sed 's/^/    [compute] /'
    echo "    (the .venv alone is ~5.1 GB; data/processed is ~15k patches — make sure there's headroom)"
else
    warn "skipped — hop 1/2 not established"
fi

echo ""
echo "=========================================="
if (( BLOCKED )); then
    err "PREFLIGHT FAILED — one or more blocking probes (1-4) failed. Fix them before running scripts/hpc_launch.sh."
    exit 1
else
    ok "preflight passed. Safe to run: bash scripts/hpc_launch.sh"
    exit 0
fi
