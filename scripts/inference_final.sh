#!/usr/bin/env bash
# scripts/inference_final.sh
# ---------------------------
# The FROZEN, paper-locking evaluation run. Training is frozen; this is the
# single command that produces the exact numbers the paper cites. Thin
# launcher only — all the actual work is inference/inference_final.py; this
# script does env/CWD setup, a hard packed-shard preflight (mirroring
# scripts/inference.sh's — build_dataset() falls back silently to the slow
# legacy per-patch tree on a missing shard, and a frozen run must never do
# that unnoticed), one Telegram start ping, and exit-code passthrough.
#
# Datasets/models/losses/seeds are NOT configurable via env here — they are
# fixed in inference/inference_final.py itself (IIRS/AVIRIS/CRIMS x
# vae-our-nl/vae-standard/vae-1d-pixelwise/vae-3d-spatio-spectral, checkpoint
# seeds 67+69 averaged, RNG seeds 67/69/1234 for noise+masking) — this is a
# locked grid, not a sweep to be resized from the command line.
#
# Run from the repo root:
#   bash scripts/inference_final.sh --dry-run             # print the plan, load nothing
#   bash scripts/inference_final.sh                        # the real run
#   bash scripts/inference_final.sh --no-telegram
#   bash scripts/inference_final.sh --ckpt-dir model --out-dir results/final
#
# Environment overrides:
#   CKPT_DIR     — checkpoint root directory   (default: model)
#   OUT_DIR      — results root for the five CSVs (default: results/final —
#                  DELIBERATELY separate from results/, the exploratory
#                  sweep's output dir; this run never reads or writes there)
#   PACKED_ROOT  — packed-shard root           (default: data/packed)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_DIR:-model}"
OUT_DIR="${OUT_DIR:-results/final}"
PACKED_ROOT="${PACKED_ROOT:-data/packed}"
DATASETS=(IIRS AVIRIS CRIMS)

cd "${REPO_ROOT}"

PY_ARGS=(--ckpt-dir "${CKPT_DIR}" --out-dir "${OUT_DIR}" --packed-root "${PACKED_ROOT}")
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            PY_ARGS+=(--dry-run)
            shift
            ;;
        *)
            PY_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Packed-shard preflight ----------------------------------------------
# The frozen run evaluates the COMPLETE valid+test split — both must exist
# and be non-empty, or build_dataset() falls back to the slow legacy
# per-patch tree without saying so. Skipped for --dry-run: the Python side's
# own --dry-run already audits each dataset gracefully (warns, doesn't
# crash, on a missing shard), which is more informative pre-staging than a
# hard exit here — this hard preflight exists to protect the REAL run.
if [[ ${DRY_RUN} -eq 0 ]]; then
    MISSING_SHARDS=()
    for ds in "${DATASETS[@]}"; do
        for split in valid test; do
            shard="${PACKED_ROOT}/${ds}/${split}.npy"
            [[ -s "${shard}" ]] || MISSING_SHARDS+=("${shard}")
        done
    done
    if [[ ${#MISSING_SHARDS[@]} -gt 0 ]]; then
        echo "ERROR: packed shard(s) missing — the frozen run evaluates valid+test"
        echo "       in full, not the legacy per-patch fallback. Missing:"
        for s in "${MISSING_SHARDS[@]}"; do echo "         ${s}"; done
        echo ""
        echo "  Build them with:"
        echo "    PYTHONPATH=. python utils/dataset/pack.py --verify"
        echo "  or point PACKED_ROOT at a staged copy."
        exit 4
    fi
fi

mkdir -p "${OUT_DIR}"

if [[ ${DRY_RUN} -eq 0 ]]; then
    python "${SCRIPT_DIR}/../utils/notify_cli.py" \
        --text "Inference final — launching on $(hostname), ckpt-dir=${CKPT_DIR}, out-dir=${OUT_DIR}" \
        || true
fi

echo "Running: python inference/inference_final.py ${PY_ARGS[*]}"
python "${REPO_ROOT}/inference/inference_final.py" "${PY_ARGS[@]}"
rc=$?

exit "${rc}"
