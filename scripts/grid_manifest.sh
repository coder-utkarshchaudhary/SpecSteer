#!/usr/bin/env bash
# scripts/grid_manifest.sh
# -------------------------
# Single source of truth for the 28-run ablation grid.
#
# Sourced by scripts/train.sh (--all mode) and scripts/hpc_pbs_job.pbs so the
# grid is defined once. Each entry maps a 1-indexed slot to a
# "<model>|<dataset>|<loss>|<ckpt_name>" tuple, matching the naming rule in
# modules/registry.py:checkpoint_name().
#
# Layout of the 28 slots (7 configs per dataset, 4 datasets):
#   slots 1..7   IIRS   (vae-our + 3 baselines x standard/physics)
#   slots 8..14  M3
#   slots 15..21 AVIRIS
#   slots 22..28 CRIMS

# Datasets in the exact PBS array order.
GRID_DATASETS=("IIRS" "M3" "AVIRIS" "CRIMS")

# For each dataset, we emit 7 configs in this fixed order:
#   1. vae-our                    physics    -> vae-our.pt
#   2. vae-standard               standard   -> vae-standard_standard.pt
#   3. vae-standard               physics    -> vae-standard_physics.pt
#   4. vae-3d-spatio-spectral     standard   -> vae-3d-spatio-spectral_standard.pt
#   5. vae-3d-spatio-spectral     physics    -> vae-3d-spatio-spectral_physics.pt
#   6. vae-1d-pixelwise           standard   -> vae-1d-pixelwise_standard.pt
#   7. vae-1d-pixelwise           physics    -> vae-1d-pixelwise_physics.pt

GRID_CONFIGS=(
    "vae-our|physics|vae-our"
    "vae-standard|standard|vae-standard_standard"
    "vae-standard|physics|vae-standard_physics"
    "vae-3d-spatio-spectral|standard|vae-3d-spatio-spectral_standard"
    "vae-3d-spatio-spectral|physics|vae-3d-spatio-spectral_physics"
    "vae-1d-pixelwise|standard|vae-1d-pixelwise_standard"
    "vae-1d-pixelwise|physics|vae-1d-pixelwise_physics"
)

GRID_CONFIGS_PER_DATASET=${#GRID_CONFIGS[@]}   # 7
GRID_TOTAL=$(( ${#GRID_DATASETS[@]} * GRID_CONFIGS_PER_DATASET ))  # 28

# grid_lookup <1-indexed slot> -> echoes "<model>|<dataset>|<loss>|<ckpt_name>"
grid_lookup() {
    local idx=$1
    if (( idx < 1 || idx > GRID_TOTAL )); then
        echo "grid_lookup: slot ${idx} out of range 1..${GRID_TOTAL}" >&2
        return 1
    fi
    local zero=$(( idx - 1 ))
    local ds_i=$(( zero / GRID_CONFIGS_PER_DATASET ))
    local cf_i=$(( zero % GRID_CONFIGS_PER_DATASET ))
    local ds="${GRID_DATASETS[$ds_i]}"
    local cfg="${GRID_CONFIGS[$cf_i]}"
    # cfg = "<model>|<loss>|<ckpt_name>"
    local model="${cfg%%|*}"
    local rest="${cfg#*|}"
    local loss="${rest%%|*}"
    local name="${rest#*|}"
    echo "${model}|${ds}|${loss}|${name}"
}

# grid_print — dump the full grid (used by --dry-run).
grid_print() {
    local i
    for (( i=1; i<=GRID_TOTAL; i++ )); do
        printf "  %2d  %s\n" "${i}" "$(grid_lookup "${i}")"
    done
}
