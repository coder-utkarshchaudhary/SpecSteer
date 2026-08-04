#!/usr/bin/env bash

set -euo pipefail

# ==========================
# Configuration
# ==========================
USER="yashdeep"
HOST="192.168.1.36"

SOURCE="data/processed/"
DEST="/media/yashdeep/New Volume 21/UTKARSH_CHAUDHARY_prism/prism/data/processed"

# ==========================
# Configuration v2 -> Vultr
# ==========================
# USER="root"
# HOST="38.128.232.57"
# PORT="46508"

# SOURCE="data/processed/"
# DEST="/workspace/prism/data/processed"

# ==========================
# Connectivity check
# ==========================
if ! ping -c 1 -W 2 "$HOST" >/dev/null 2>&1; then
    echo "Error: $HOST is unreachable."
    echo "Make sure you are on the same local network."
    exit 1
fi

echo "Connected to $HOST"
echo

# ==========================
# Sync
# ==========================
rsync \
    -a \
    --human-readable \
    --info=progress2 \
    --partial \
    --compress \
    -e "ssh -i ~/.ssh/id_rsa -p $PORT" \
    "$SOURCE" \
    "$USER@$HOST:$DEST"

echo
echo "Transfer complete."