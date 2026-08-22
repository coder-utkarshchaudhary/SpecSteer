#!/usr/bin/env bash
# scripts/grid_manifest.sh
# -------------------------
# Single source of truth for the ablation grid, now with a SEED axis.
#
# Sourced by scripts/train.sh (--all mode) and scripts/hpc_pbs_job.pbs so the
# grid is defined once. Each slot maps a 1-indexed number to a
# "<model>|<dataset>|<loss>|<ckpt_name>|<seed>" tuple.
#
# WHY SEEDS
# ---------
# The v3 grid was accidentally launched twice, which handed us a free
# replication: the same cell at the SAME seed 42, run twice, differed by
# 0.0005-0.0036 rad SAM. Several of the differences the ablation is meant to
# report are that size. Without repeats, a 0.0003 rad "loss" on M3 is
# indistinguishable from nondeterminism — and that is exactly what it turned out
# to be.
#
# SEED BUDGET
# -----------
# Seeds are NOT applied uniformly, because most of the extra GPU time would go to
# cells no claim depends on. The claim is "vae-our beats each baseline in the
# physics regime", so:
#
#   CLAIM cells  (vae-our + the 3 baselines at --loss physics)  -> GRID_SEEDS
#   OTHER cells  (the 3 baselines at --loss standard)           -> first seed only
#
# 4 datasets x 4 claim configs x 3 seeds = 48, plus 4 x 3 other configs x 1 = 12.
# 60 runs.

# Datasets in the exact PBS array order.
GRID_DATASETS=("IIRS" "M3" "AVIRIS" "CRIMS")

# Seeds. Override with e.g. GRID_SEEDS="42 7" to split work across machines.
# shellcheck disable=SC2206
GRID_SEEDS=(${GRID_SEEDS:-42 7 1234})

# For each dataset, 7 configs in this fixed order. Field 4 marks whether the cell
# carries the claim (and therefore gets every seed).
#   "<model>|<loss>|<ckpt_stem>|<claim:1|0>"
GRID_CONFIGS=(
    "vae-our|physics|vae-our|1"
    "vae-standard|physics|vae-standard_physics|1"
    "vae-3d-spatio-spectral|physics|vae-3d-spatio-spectral_physics|1"
    "vae-1d-pixelwise|physics|vae-1d-pixelwise_physics|1"
    "vae-standard|standard|vae-standard_standard|0"
    "vae-3d-spatio-spectral|standard|vae-3d-spatio-spectral_standard|0"
    "vae-1d-pixelwise|standard|vae-1d-pixelwise_standard|0"
)

GRID_CONFIGS_PER_DATASET=${#GRID_CONFIGS[@]}   # 7

# Build the flat slot table once, at source time. Doing it here rather than with
# arithmetic in grid_lookup keeps the seed asymmetry (claim cells get every seed,
# the rest get one) in a single place instead of spread across index maths.
GRID_SLOTS=()
_build_grid_slots() {
    local ds cfg model loss name claim seed
    for ds in "${GRID_DATASETS[@]}"; do
        for cfg in "${GRID_CONFIGS[@]}"; do
            model="${cfg%%|*}";  local rest="${cfg#*|}"
            loss="${rest%%|*}";  rest="${rest#*|}"
            name="${rest%%|*}"
            claim="${rest#*|}"
            if [[ "${claim}" == "1" ]]; then
                for seed in "${GRID_SEEDS[@]}"; do
                    GRID_SLOTS+=("${model}|${ds}|${loss}|${name}|${seed}")
                done
            else
                GRID_SLOTS+=("${model}|${ds}|${loss}|${name}|${GRID_SEEDS[0]}")
            fi
        done
    done
}
_build_grid_slots

GRID_TOTAL=${#GRID_SLOTS[@]}

# grid_lookup <1-indexed slot> -> "<model>|<dataset>|<loss>|<ckpt_stem>|<seed>"
#
# NOTE the trailing <seed> field: callers that split on '|' and expected four
# fields must be updated. scripts/train.sh and scripts/hpc_pbs_job.pbs are.
grid_lookup() {
    local idx=$1
    if (( idx < 1 || idx > GRID_TOTAL )); then
        echo "grid_lookup: slot ${idx} out of range 1..${GRID_TOTAL}" >&2
        return 1
    fi
    echo "${GRID_SLOTS[$(( idx - 1 ))]}"
}

# grid_print — dump the full grid (used by --dry-run).
grid_print() {
    local i
    for (( i=1; i<=GRID_TOTAL; i++ )); do
        printf "  %3d  %s\n" "${i}" "$(grid_lookup "${i}")"
    done
}
