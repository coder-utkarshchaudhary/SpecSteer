#!/usr/bin/env bash
# scripts/install_on_hpc.sh
# -------------------------
# Runs on the IITD Padum login node, once, after the bundle has been scp'd.
# Turns a freshly-extracted repo into a runnable workstation:
#   1. Sanity-checks the environment (login node, expected files present)
#   2. Fills the <REPLACE_ME> placeholders in the PBS files with $PROJECT_CODE
#   3. Creates .venv and installs from local wheels (no PyPI access needed)
#   4. Sets up $SCRATCH/prism-data and symlinks data/original + data/processed
#   5. Moves any shipped-in processed data over to $SCRATCH
#   6. Prints the next commands (interactive smoke → qsub)
#
# Usage (on HPC, from repo root):
#   PROJECT_CODE=cc IITD_EMAIL=you@iitd.ac.in \
#     WHEELS_DIR=$HOME/prism-wheels \
#     bash scripts/install_on_hpc.sh
#
# The variables can also be passed as flags; env vars win if both are set.

set -euo pipefail

# ---- flag parsing ---------------------------------------------------------
PROJECT_CODE="${PROJECT_CODE:-}"
IITD_EMAIL="${IITD_EMAIL:-}"
WHEELS_DIR="${WHEELS_DIR:-${HOME}/prism-wheels}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-code) PROJECT_CODE="$2"; shift 2 ;;
        --email)        IITD_EMAIL="$2"; shift 2 ;;
        --wheels-dir)   WHEELS_DIR="$2"; shift 2 ;;
        -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
        *)              echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=============================================="
echo " PRISM HPC install"
echo "  repo root  : ${REPO_ROOT}"
echo "  wheels dir : ${WHEELS_DIR}"
echo "  \$HOME      : ${HOME}"
echo "  \$SCRATCH   : ${SCRATCH:-<unset>}"
echo "=============================================="

# ---- refuse to run on a compute node --------------------------------------
# Compute jobs set PBS_JOBID; login sessions do not. Login-node-only work
# below (git ops, big pip install, symlink setup) shouldn't burn compute time.
if [[ -n "${PBS_JOBID:-}" ]]; then
    echo "FATAL: PBS_JOBID is set (${PBS_JOBID}); you're inside a compute job."
    echo "       Run this script on a login node instead."
    exit 3
fi

# ---- required inputs -----------------------------------------------------
: "${SCRATCH:?SCRATCH is unset. Are you on Padum? Login again or contact hpchelp.}"

if [[ -z "${PROJECT_CODE}" ]]; then
    # Guess from $HOME: /home/<proj>/<user>
    PROJECT_CODE="$(echo "${HOME}" | cut -d/ -f3)"
    echo ">>> inferred PROJECT_CODE=${PROJECT_CODE} from \$HOME. Override with --project-code."
fi
if [[ -z "${IITD_EMAIL}" ]]; then
    IITD_EMAIL="$(whoami)@iitd.ac.in"
    echo ">>> defaulted IITD_EMAIL=${IITD_EMAIL}. Override with --email."
fi

# ---- 1. patch PBS placeholders -------------------------------------------
echo ""
echo ">>> [1/4] filling PBS placeholders (project=${PROJECT_CODE}, email=${IITD_EMAIL})"
for pbs in scripts/hpc_train.pbs; do
    [[ -f "${pbs}" ]] || continue
    sed -i.bak \
        -e "s|<REPLACE_ME>|${PROJECT_CODE}|g" \
        -e "s|REPLACE_ME@iitd\.ac\.in|${IITD_EMAIL}|g" \
        "${pbs}"
    rm -f "${pbs}.bak"
    echo "  patched ${pbs}"
done

# ---- 2. create venv + install from local wheels --------------------------
echo ""
echo ">>> [2/4] creating .venv and installing from ${WHEELS_DIR}"
if [[ ! -d "${WHEELS_DIR}" ]]; then
    echo "FATAL: wheels dir ${WHEELS_DIR} not found."
    echo "       Did rsync of wheels/ finish? See README.txt in the bundle."
    exit 4
fi

if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# Upgrade pip from a shipped wheel if one is present; otherwise skip (offline).
python -m pip install --no-index --find-links "${WHEELS_DIR}" --upgrade pip || true

REQ_FILE="requirements-hpc.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
    echo "FATAL: ${REQ_FILE} missing. Was the code tarball extracted here?"
    exit 5
fi
python -m pip install --no-index --find-links "${WHEELS_DIR}" -r "${REQ_FILE}"

echo "  pip install done. python: $(which python)"

# ---- 3. wire data paths through $SCRATCH ----------------------------------
echo ""
echo ">>> [3/4] wiring \$SCRATCH/prism-data → data/{original,processed}"
mkdir -p "${SCRATCH}/prism-data/original" "${SCRATCH}/prism-data/processed" data

# If the user scp'd processed data straight into repo's data/processed/ (not
# recommended, but common), move it under $SCRATCH before symlinking.
if [[ -d data/processed ]] && [[ ! -L data/processed ]]; then
    echo "  moving pre-existing data/processed/ → \$SCRATCH"
    # Merge into SCRATCH rather than overwrite; each dataset directory is
    # its own subtree so name collisions are rare.
    rsync -a --remove-source-files data/processed/ "${SCRATCH}/prism-data/processed/"
    rm -rf data/processed
fi

ln -sfn "${SCRATCH}/prism-data/original"  data/original
ln -sfn "${SCRATCH}/prism-data/processed" data/processed
echo "  data/original  -> $(readlink data/original)"
echo "  data/processed -> $(readlink data/processed)"

# Quick census
for ds in IIRS M3 AVIRIS crims; do
    d="data/processed/${ds}/train"
    if [[ -d "${d}" ]]; then
        n=$(find "${d}" -name 'patch_*.npy' | wc -l | tr -d ' ')
        echo "    ${ds}: ${n} train patches"
    else
        echo "    ${ds}: no train patches at ${d}"
    fi
done

# ---- 4. next steps --------------------------------------------------------
echo ""
echo ">>> [4/4] install complete."
cat <<EOF

Next steps
----------
  # Login-node sanity check
  source .venv/bin/activate
  export PYTHONPATH=\$PWD:\$PYTHONPATH
  python3 utils/check-model-params.py

  # (Optional) 1-hour GPU smoke test
  qsub -P ${PROJECT_CODE} -I -l select=1:ncpus=4:ngpus=1 -l walltime=1:00:00
  # ...on the compute node:
  cd \$HOME/$(basename "${REPO_ROOT}")
  source .venv/bin/activate
  export PYTHONPATH=\$PWD:\$PYTHONPATH
  export WANDB_MODE=offline
  python train/train.py --model vae-our --dataset IIRS --loss physics --epochs 1

  # Full 28-run grid
  qsub scripts/hpc_train.pbs
  qstat -u \$USER
  tail -f logs/hpc/train28.out
EOF
