#!/usr/bin/env bash
# scripts/grid_lock.sh
# --------------------
# A single advisory lock that makes a second concurrent grid launch impossible.
#
# WHY THIS EXISTS
# ---------------
# The v3 grid was launched twice, four minutes apart (logs/retrain_*_1021.log and
# *_1025.log). Nothing refused the second launch, so two 28-slot grids ran on one
# 23.4 GB card and produced 47 OOMs — every one of which names a second multi-GiB
# process. It also made Telegram look haunted: the first run printed ALL DONE
# while the second kept sending heartbeats.
#
# Single-run peaks were 14.4-14.6 GB against a 20 GB budget, so the batch sizes
# were never the problem. Concurrency was.
#
# USAGE
#   source "${SCRIPT_DIR}/grid_lock.sh"
#   acquire_grid_lock "full grid" || exit 9
#
# The lock is released automatically when the holding process exits (flock is
# tied to the file descriptor, so a kill -9 releases it too — no stale locks).
#
# ENV
#   ALLOW_CONCURRENT=1   skip the lock entirely. For the deliberate case: two
#                        different datasets on one card with room for both.
#   GRID_LOCK_HELD=1     set automatically for child processes, so train_fixed.sh
#                        calling train.sh does not deadlock against itself.
#   GRID_LOCK_FILE       override the lock path (default: logs/.grid.lock)

GRID_LOCK_FILE="${GRID_LOCK_FILE:-${REPO_ROOT:-.}/logs/.grid.lock}"
GRID_LOCK_INFO="${GRID_LOCK_FILE}.info"

acquire_grid_lock() {
    local label="${1:-grid}"

    if [[ "${GRID_LOCK_HELD:-0}" == "1" ]]; then
        # A parent (train_fixed.sh) already holds it on our behalf.
        return 0
    fi
    if [[ "${ALLOW_CONCURRENT:-0}" == "1" ]]; then
        echo "[lock] ALLOW_CONCURRENT=1 — not taking ${GRID_LOCK_FILE}."
        echo "[lock] You are responsible for the VRAM budget of every concurrent run."
        export GRID_LOCK_HELD=1
        return 0
    fi
    if ! command -v flock >/dev/null 2>&1; then
        echo "[lock] WARNING: flock not found; cannot guard against a concurrent grid." >&2
        export GRID_LOCK_HELD=1
        return 0
    fi

    mkdir -p "$(dirname "${GRID_LOCK_FILE}")"
    # A dedicated fd, held for the lifetime of this shell. Assigning to a
    # variable (bash 4.1+) avoids colliding with a hard-coded descriptor number.
    exec {GRID_LOCK_FD}>>"${GRID_LOCK_FILE}"

    if ! flock -n "${GRID_LOCK_FD}"; then
        echo ""
        echo "=============================================================="
        echo " REFUSED: another grid is already running on this machine."
        echo "=============================================================="
        if [[ -r "${GRID_LOCK_INFO}" ]]; then
            sed 's/^/  /' "${GRID_LOCK_INFO}"
        else
            echo "  (no holder info recorded; check: ps aux | grep train)"
        fi
        echo ""
        echo "  Two grids on one GPU is what produced 47 OOMs in the v3 run."
        echo "  Wait for it to finish, or:"
        echo "    ALLOW_CONCURRENT=1 <your command>     # only if VRAM genuinely fits"
        echo "=============================================================="
        return 1
    fi

    # Holder info is written to a SEPARATE file: the lock file itself is held
    # open by flock, and truncating it under the lock is needless risk.
    {
        echo "label   : ${label}"
        echo "pid     : $$"
        echo "started : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "host    : $(hostname)"
        echo "cwd     : $(pwd)"
        echo "cmd     : ${GRID_LOCK_CMD:-${0##*/} $*}"
    } > "${GRID_LOCK_INFO}" 2>/dev/null || true

    echo "[lock] acquired ${GRID_LOCK_FILE} (pid $$, ${label})"
    # Children (train.sh under train_fixed.sh) must not try to re-acquire.
    export GRID_LOCK_HELD=1
    return 0
}
