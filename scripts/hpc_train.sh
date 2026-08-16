#!/usr/bin/env bash
# scripts/hpc_train.sh
# ---------------------
# Thin "resume" wrapper. Historically this duplicated most of
# scripts/hpc_launch.sh (data check -> relay/tunnel/forwarder -> smoke qsub)
# for the case where data was already staged on the HPC. That behaviour is
# now native to hpc_launch.sh itself (each rsync step independently checks
# what's already present/intact — on the login node AND on the compute
# node — and skips what doesn't need re-pushing), so this script just
# delegates with the equivalent flag.
#
# Modes:
#   bash scripts/hpc_train.sh                 # check + launch (resume)
#   bash scripts/hpc_train.sh --dry-run       # checks only, no qsub/relay/tunnel

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/hpc_launch.sh" --resume "$@"
