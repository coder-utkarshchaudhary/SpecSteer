#!/usr/bin/env bash
# scripts/build_hpc_bundle.sh
# ---------------------------
# Build a self-contained HPC transfer bundle on the lab Mac. After this
# runs, two scp/rsync commands drop the whole project (code, Linux wheels,
# processed data) onto the IITD Padum cluster; the cluster then behaves like
# an offline workstation.
#
# Why this exists: Padum compute nodes have no outbound internet. Everything
# that touches PyPI, Google Drive, or wandb.ai has to happen here first.
#
# Usage (from repo root):
#   bash scripts/build_hpc_bundle.sh                       # gdown + wheels + tar
#   bash scripts/build_hpc_bundle.sh --skip-gdown          # data all local
#   bash scripts/build_hpc_bundle.sh --drive-url <URL>     # override drive folder
#   bash scripts/build_hpc_bundle.sh --out-dir /tmp/bundle
#
# Output layout (default: build/hpc_bundle/):
#   build/hpc_bundle/
#     code.tar.gz            # source tree + install_on_hpc.sh + requirements-hpc.txt
#     wheels/                # ~2 GB of linux_x86_64 cp310 wheels
#     data/processed/        # per-dataset patch tree (staged, ~35 GB)
#     install.sh             # convenience alias -> unpacks and calls install_on_hpc.sh
#     README.txt             # exact scp/rsync commands to run next

set -euo pipefail

# ---- argument parsing -----------------------------------------------------
DRIVE_URL_DEFAULT="https://drive.google.com/drive/folders/1QjwlQRSCgLFKT4f3SHYTOyFSAKXiIAlZ"
DRIVE_URL="${DRIVE_URL:-${DRIVE_URL_DEFAULT}}"
OUT_DIR="build/hpc_bundle"
SKIP_GDOWN=0
PY_VERSION="3.10"
PY_PLATFORM="manylinux2014_x86_64"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --drive-url)      DRIVE_URL="$2"; shift 2 ;;
        --out-dir)        OUT_DIR="$2"; shift 2 ;;
        --skip-gdown)     SKIP_GDOWN=1; shift ;;
        --python-version) PY_VERSION="$2"; shift 2 ;;
        --platform)       PY_PLATFORM="$2"; shift 2 ;;
        --torch-index)    TORCH_INDEX="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0"; exit 0 ;;
        *)
            echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BUNDLE_DIR="${REPO_ROOT}/${OUT_DIR}"
WHEELS_DIR="${BUNDLE_DIR}/wheels"
DATA_STAGE="${BUNDLE_DIR}/data/processed"
LOCAL_DATA="${REPO_ROOT}/data/processed"

echo "=============================================="
echo " PRISM HPC bundle builder"
echo "  repo root : ${REPO_ROOT}"
echo "  out dir   : ${BUNDLE_DIR}"
echo "  py target : cp${PY_VERSION//./}, ${PY_PLATFORM}"
echo "  torch idx : ${TORCH_INDEX}"
echo "  drive url : ${DRIVE_URL}"
echo "  skip gdown: ${SKIP_GDOWN}"
echo "=============================================="

mkdir -p "${BUNDLE_DIR}" "${WHEELS_DIR}" "${DATA_STAGE}"

# ---- sanity: local tools --------------------------------------------------
command -v python3 >/dev/null || { echo "FATAL: python3 missing"; exit 3; }
command -v tar     >/dev/null || { echo "FATAL: tar missing";     exit 3; }

# ---- stage 1: fetch IIRS/AVIRIS processed data via gdown ------------------
# The lab hard drive covers M3 + CRIMS; IIRS + AVIRIS come from Drive.
if [[ ${SKIP_GDOWN} -eq 0 ]]; then
    echo ""
    echo ">>> [1/4] gdown IIRS + AVIRIS processed data → ${LOCAL_DATA}"
    python3 -m pip install --quiet --upgrade gdown
    mkdir -p "${LOCAL_DATA}"

    # gdown --folder streams the whole Drive folder into the current dir.
    # We assume the Drive layout has IIRS/ and AVIRIS/ subfolders that match
    # utils/config.py::DATASETS[<KEY>]["processed_root"].
    for ds in IIRS AVIRIS; do
        if [[ -d "${LOCAL_DATA}/${ds}/train" ]]; then
            echo "[skip] ${ds}: already present at ${LOCAL_DATA}/${ds}"
            continue
        fi
        echo ">>> pulling ${ds}"
        # Users can override DRIVE_URL to a per-dataset folder if the parent
        # link is too broad. Fallback: whole folder pull.
        python3 -m gdown --folder --continue --remaining-ok \
                -O "${LOCAL_DATA}" "${DRIVE_URL}" || {
            echo "WARN: gdown returned non-zero. See wiki-hpc.md §4 fallback."
        }
        # Google Drive folders may deliver files inside an arbitrary top-level
        # dir — leave verification to stage 3 rather than second-guessing.
    done
else
    echo ">>> [1/4] --skip-gdown set; assuming all four datasets already at ${LOCAL_DATA}"
fi

# ---- stage 2: pip download wheels for linux_x86_64 / cp310 ----------------
echo ""
echo ">>> [2/4] downloading linux wheels → ${WHEELS_DIR}"
REQ_FILE="${REPO_ROOT}/requirements-hpc.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
    echo "FATAL: ${REQ_FILE} missing"; exit 4
fi

# Two-pass wheel fetch: torch stack from pytorch index, everything else from PyPI.
# --only-binary=:all: forces wheel-only (no sdists that would need compilation on HPC).
python3 -m pip download \
    --dest "${WHEELS_DIR}" \
    --python-version "${PY_VERSION}" \
    --platform "${PY_PLATFORM}" \
    --only-binary=:all: \
    --extra-index-url "${TORCH_INDEX}" \
    -r "${REQ_FILE}"

wheel_count=$(find "${WHEELS_DIR}" -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')
wheel_bytes=$(du -sh "${WHEELS_DIR}" | awk '{print $1}')
echo ">>> ${wheel_count} wheels, ${wheel_bytes} total"

# ---- stage 3: stage processed data into the bundle ------------------------
echo ""
echo ">>> [3/4] staging processed data → ${DATA_STAGE}"
if [[ ! -d "${LOCAL_DATA}" ]]; then
    echo "WARN: ${LOCAL_DATA} missing; the bundle will ship without data."
    echo "      Drop M3/CRIMS in from the external drive and re-run, or scp"
    echo "      data separately per README.txt."
else
    # Hardlink where possible to avoid duplicating ~35 GB on the Mac.
    # cp -al is macOS-safe; falls back to a plain cp if the source isn't on
    # the same FS as the bundle out dir.
    for ds in IIRS M3 AVIRIS CRIMS; do
        # config.py uses lowercase "crims" for CRIMS's processed_root — honour it.
        src_name="${ds}"
        [[ "${ds}" == "CRIMS" ]] && src_name="crims"
        src="${LOCAL_DATA}/${src_name}"
        dst="${DATA_STAGE}/${src_name}"
        if [[ ! -d "${src}" ]]; then
            echo "  [miss] ${src_name}: no local copy at ${src}"
            continue
        fi
        n=$(find "${src}" -name 'patch_*.npy' | wc -l | tr -d ' ')
        sz=$(du -sh "${src}" | awk '{print $1}')
        echo "  [ok]   ${src_name}: ${n} patches, ${sz}"
        rm -rf "${dst}"
        # Try hardlink clone first (fast, no extra disk). If that fails
        # (cross-device), fall back to a normal copy.
        if ! cp -al "${src}" "${dst}" 2>/dev/null; then
            cp -R "${src}" "${dst}"
        fi
    done
fi

# ---- stage 4: bundle the code + install helper ----------------------------
echo ""
echo ">>> [4/4] tarring code → ${BUNDLE_DIR}/code.tar.gz"

# Copy install_on_hpc.sh + requirements-hpc.txt into the bundle root so the
# HPC-side unpack step is one command.
cp "${REPO_ROOT}/scripts/install_on_hpc.sh" "${BUNDLE_DIR}/install.sh"
chmod +x "${BUNDLE_DIR}/install.sh"

EXCLUDE_FILE="${REPO_ROOT}/scripts/hpc_bundle.exclude"

# tar the repo root itself (as "prism/") so unpacking creates a clean dir.
tar \
    --exclude-from="${EXCLUDE_FILE}" \
    -C "$(dirname "${REPO_ROOT}")" \
    -czf "${BUNDLE_DIR}/code.tar.gz" \
    "$(basename "${REPO_ROOT}")"

code_bytes=$(du -sh "${BUNDLE_DIR}/code.tar.gz" | awk '{print $1}')
data_bytes=$(du -sh "${DATA_STAGE}" 2>/dev/null | awk '{print $1}' || echo "0")

cat > "${BUNDLE_DIR}/README.txt" <<EOF
PRISM HPC bundle
================
built : $(date -Iseconds)
code  : code.tar.gz (${code_bytes})
whls  : wheels/     (${wheel_bytes}, ${wheel_count} files)
data  : data/processed/ (${data_bytes})

STEP 1 — ship code + wheels to \$HOME on the login node
   scp code.tar.gz install.sh <user>@hpc.iitd.ac.in:~/
   rsync -avP wheels/ <user>@hpc.iitd.ac.in:~/prism-wheels/

STEP 2 — ship processed data to \$SCRATCH (huge, resumable)
   rsync -avP data/processed/ <user>@hpc.iitd.ac.in:/scratch/<proj>/<user>/prism-data/processed/

STEP 3 — on the HPC login node
   cd ~
   tar -xzf code.tar.gz             # creates ./prism/
   mv install.sh prism/scripts/     # (or just run ./install.sh — it'll cd in)
   cd prism
   PROJECT_CODE=<your-code> bash scripts/install_on_hpc.sh

STEP 4 — smoke test + submit (see docs/wiki-hpc.md §5 onwards)
   qsub scripts/hpc_train.pbs
EOF

echo ""
echo "=============================================="
echo " Bundle ready at: ${BUNDLE_DIR}"
echo "   code.tar.gz : ${code_bytes}"
echo "   wheels/     : ${wheel_bytes} (${wheel_count} files)"
echo "   data/       : ${data_bytes}"
echo ""
echo " Next steps: cat ${BUNDLE_DIR}/README.txt"
echo "=============================================="
