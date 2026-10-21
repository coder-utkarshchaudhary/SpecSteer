#!/usr/bin/env bash
# retrain.sh
# ----------
# Convenience root wrapper for scripts/retrain.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/scripts/retrain.sh" "$@"
