# scripts/hpc_common.sh
# ----------------------
# Shared two-hop SSH/rsync helpers for the HPC scripts.
#
# Topology: IITD Padum has a LOGIN node (reachable directly from the lab as
# ${HPC_USER}@${HPC_HOST}) and a separate COMPUTE node with its own
# filesystem, reachable only from the login node as `ssh ${HPC_INNER_HOST}`
# (default alias "hpc" — set HPC_INNER_HOST in hpc_config.env if yours
# differs). qsub/qstat/qdel only exist on the compute node.
#
# Source this file AFTER `set -a; source hpc_config.env; set +a` so
# HPC_USER / HPC_HOST / HPC_INNER_HOST are already in the environment.
#
#   source "${SCRIPT_DIR}/hpc_common.sh"
#   login_ssh   "<cmd run on the login node>"
#   compute_ssh "<cmd run on the compute node, via the login node>"
#
# Not meant to be executed directly.

: "${HPC_INNER_HOST:=hpc}"

# ---------------------------------------------------------------------------
# resolve_hpc_roots — fills in HPC_LOGIN_REPO_ROOT / HPC_COMPUTE_REPO_ROOT
# with back-compat fallbacks so old configs (only HPC_PROJECT_DIR set) keep
# working: HPC_PROJECT_DIR -> HPC_LOGIN_REPO_ROOT -> HPC_COMPUTE_REPO_ROOT.
# Call once, right after sourcing hpc_config.env, before any *_ssh call.
# ---------------------------------------------------------------------------
resolve_hpc_roots() {
    export HPC_PROJECT_DIR="${HPC_PROJECT_DIR:-${HPC_HOME:-}/prism}"
    export HPC_LOGIN_REPO_ROOT="${HPC_LOGIN_REPO_ROOT:-${HPC_PROJECT_DIR}}"
    export HPC_COMPUTE_REPO_ROOT="${HPC_COMPUTE_REPO_ROOT:-${HPC_LOGIN_REPO_ROOT}}"
    # Keep HPC_PROJECT_DIR itself pointed at the compute root from here on —
    # it's still what hpc_pbs_job.pbs reads via qsub -v for back-compat.
    export HPC_PROJECT_DIR="${HPC_COMPUTE_REPO_ROOT}"
}

# ---------------------------------------------------------------------------
# _PBS_FIX — best-effort PATH repair for PBS Pro binaries.
#
# Even a login shell (`bash -l`) can miss qsub/qstat/qdel when they're added
# by an interactive-only rc guard. Read PBS_EXEC from /etc/pbs.conf (present
# on every PBS Pro node) and fall back to the common install dirs. No-op for
# non-PBS commands (tmux/rsync/find/...) since qsub is already found.
# ---------------------------------------------------------------------------
_PBS_FIX='if ! command -v qsub >/dev/null 2>&1; then [ -r /etc/pbs.conf ] && . /etc/pbs.conf; for d in "${PBS_EXEC:-/opt/pbs}/bin" /opt/pbs/bin /opt/pbs/default/bin; do [ -x "$d/qsub" ] && export PATH="$d:$PATH" && break; done; fi;'

# ---------------------------------------------------------------------------
# login_ssh "<cmd>" — run a command on the login node in a login shell (so
# PATH is sourced from /etc/profile.d, tools like tmux/rsync are found, etc).
# ---------------------------------------------------------------------------
login_ssh() {
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" 'bash -l -s' <<< "$1"
}

# ---------------------------------------------------------------------------
# compute_ssh "<cmd>" — run a command on the COMPUTE node, two hops out
# (lab -> login -> compute). The payload is base64-encoded before crossing
# the first hop so nested quoting in <cmd> (which itself may contain single
# quotes, e.g. awk scripts) never has to survive two levels of shell
# re-parsing. base64's alphabet (A-Za-z0-9+/=) is safe inside the single
# quotes wrapping the inner ssh call.
# ---------------------------------------------------------------------------
compute_ssh() {
    local payload
    payload="$(printf '%s' "${_PBS_FIX} $1" | base64 | tr -d '\n')"
    ssh -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" \
        "ssh -o BatchMode=yes ${HPC_INNER_HOST} 'echo ${payload} | base64 -d | bash -l -s'"
}

# ---------------------------------------------------------------------------
# compute_rsync_push <local_login_path> <remote_compute_path> [extra rsync args]
#
# Pushes a directory from the LOGIN node's filesystem to the COMPUTE node's
# filesystem. Must run ON the login node (the lab cannot reach the compute
# node directly), so this shells out via login_ssh and runs rsync there,
# targeting `${HPC_INNER_HOST}:` as its rsync destination.
# ---------------------------------------------------------------------------
compute_rsync_push() {
    local local_login_path="$1" remote_compute_path="$2"
    shift 2
    local extra_args="$*"
    login_ssh "mkdir -p '${remote_compute_path}' && rsync -a --human-readable --info=progress2 --partial --compress ${extra_args} '${local_login_path}/' '${HPC_INNER_HOST}:${remote_compute_path}/'"
}

# ---------------------------------------------------------------------------
# count_npy_remote <login|compute> <data_root> <dataset_subdir>
# Echoes the *.npy count under <data_root>/<dataset_subdir> on the given node.
# Exact-case match — dataset casing bugs (m3 vs M3) are the whole point.
# ---------------------------------------------------------------------------
count_npy_remote() {
    local node="$1" data_root="$2" dataset="$3"
    local cmd="find '${data_root}/${dataset}' -name '*.npy' 2>/dev/null | wc -l"
    local n
    case "${node}" in
        login)   n="$(login_ssh "${cmd}" 2>/dev/null | tr -d '[:space:]')" ;;
        compute) n="$(compute_ssh "${cmd}" 2>/dev/null | tr -d '[:space:]')" ;;
        *) echo "count_npy_remote: unknown node '${node}'" >&2; echo 0; return 1 ;;
    esac
    echo "${n:-0}"
}
