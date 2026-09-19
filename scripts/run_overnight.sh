#!/usr/bin/env bash
# scripts/run_overnight.sh
# ------------------------
# Fire-and-forget wrapper around scripts/train.sh for the two-workstation
# split. Detaches with nohup, tees the wall-clock output to a hostname-stamped
# log, and stashes the PID so the run can be inspected or killed later.
#
# Usage (from repo root):
#   bash scripts/run_overnight.sh --datasets IIRS,M3      --epochs 100
#   bash scripts/run_overnight.sh --datasets AVIRIS,CRIMS --epochs 100
#
#   bash scripts/run_overnight.sh --all --epochs 100        # everything on one box
#   bash scripts/run_overnight.sh --status                  # is a run alive?
#   bash scripts/run_overnight.sh --stop                    # kill the most recent
#
# Behavior:
#   - Verifies the shared drive is mounted before launching.
#   - Activates .venv/ if present; otherwise assumes system python.
#   - Exports PYTHONPATH + WANDB_MODE=offline.
#   - Launches nohup ...train.sh --all --datasets $DS $FORWARD > $LOG 2>&1 &
#   - Writes logs/overnight.pid so subsequent invocations can find the child.
#
# Safe to close the ssh window after the "PID:" line is printed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PID_FILE="logs/overnight.pid"

# ---- subcommands: --status / --stop ---------------------------------------
if [[ "${1:-}" == "--status" ]]; then
    if [[ ! -s "${PID_FILE}" ]]; then
        echo "no PID file at ${PID_FILE} — no run recorded."
        exit 0
    fi
    pid=$(cat "${PID_FILE}")
    if kill -0 "${pid}" 2>/dev/null; then
        echo "running: pid=${pid}"
        ps -o pid,etime,cmd -p "${pid}"
    else
        echo "not running: pid=${pid} (stale ${PID_FILE})"
    fi
    exit 0
fi

if [[ "${1:-}" == "--stop" ]]; then
    if [[ ! -s "${PID_FILE}" ]]; then
        echo "no PID file at ${PID_FILE} — nothing to stop."
        exit 0
    fi
    pid=$(cat "${PID_FILE}")
    if kill -0 "${pid}" 2>/dev/null; then
        echo "stopping pid=${pid} (and its children)..."
        # negative PID kills the whole process group (nohup + child python).
        kill -TERM -"${pid}" 2>/dev/null || kill -TERM "${pid}"
        sleep 2
        if kill -0 "${pid}" 2>/dev/null; then
            echo "  still alive, sending SIGKILL"
            kill -KILL -"${pid}" 2>/dev/null || kill -KILL "${pid}"
        fi
        echo "stopped."
    else
        echo "pid=${pid} already gone."
    fi
    rm -f "${PID_FILE}"
    exit 0
fi

# ---- argument parsing (launch path) ---------------------------------------
DATASETS_CSV=""
USE_ALL=0
FORWARD=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS_CSV="$2"; shift 2 ;;
        --all)      USE_ALL=1; shift ;;
        -h|--help)  sed -n '2,25p' "$0"; exit 0 ;;
        *)          FORWARD+=("$1"); shift ;;
    esac
done

if [[ -z "${DATASETS_CSV}" && ${USE_ALL} -eq 0 ]]; then
    echo "ERROR: pass either --datasets IIRS,M3 (comma list) or --all."
    exit 2
fi

# ---- sanity: shared drive mounted -----------------------------------------
# Reuse the hard-coded default from utils/config.py so the check matches the
# path the training code will actually try to open.
DATA_ROOT="${PRISM_DATA_ROOT:-/media/yashdeep/New Volume 21/UTKARSH_CHAUDHARY_prism/data/processed}"
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: processed data not found at:"
    echo "         ${DATA_ROOT}"
    echo "Is the external drive mounted? Export PRISM_DATA_ROOT if it's"
    echo "mounted at a different path, e.g."
    echo "  export PRISM_DATA_ROOT=/mnt/prism/data/processed"
    exit 3
fi

# ---- sanity: refuse to start if one is already running --------------------
if [[ -s "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "ERROR: an overnight run is already in progress:"
    ps -o pid,etime,cmd -p "$(cat "${PID_FILE}")"
    echo "Use --status or --stop first."
    exit 4
fi

# ---- environment ----------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p logs model

if [[ -d .venv ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
    echo ">>> venv: $(which python)"
fi

# ---- launch nohup'd train.sh ---------------------------------------------
TS="$(date +%F_%H%M)"
HOST="$(hostname)"
LOG="logs/overnight_${HOST}_${TS}.log"

TRAIN_ARGS=(--all)
if [[ -n "${DATASETS_CSV}" ]]; then
    TRAIN_ARGS+=(--datasets "${DATASETS_CSV}")
fi
TRAIN_ARGS+=("${FORWARD[@]}")

echo "=============================================="
echo " Overnight training launch"
echo "  host      : ${HOST}"
echo "  data root : ${DATA_ROOT}"
echo "  datasets  : ${DATASETS_CSV:-<all>}"
echo "  extra     : ${FORWARD[*]:-<none>}"
echo "  log       : ${LOG}"
echo "=============================================="

# setsid puts the child in its own process group so --stop can hit the
# whole tree with kill -TERM -<pgid>.
setsid nohup bash scripts/train.sh "${TRAIN_ARGS[@]}" > "${LOG}" 2>&1 &
child_pid=$!

echo "${child_pid}" > "${PID_FILE}"
echo ""
echo "PID: ${child_pid}   (saved to ${PID_FILE})"
echo "Log: ${LOG}"
echo ""
echo "Watch it:   tail -f ${LOG}"
echo "Status:     bash scripts/run_overnight.sh --status"
echo "Kill it:    bash scripts/run_overnight.sh --stop"
echo ""
echo "Safe to close this ssh session now."
