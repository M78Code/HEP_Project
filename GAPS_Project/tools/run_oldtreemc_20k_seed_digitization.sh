#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
INPUT_DIR=${INPUT_DIR:-/mnt/aohba/oldtreemc_source_disjoint_1m_skims}
CRANE=${CRANE:-"$HOME/simpledet_treerec_fix/build/analysis/CraneBaseProcessing"}
JOBS=${JOBS:-4}
EVENTS_PER_FILE=${EVENTS_PER_FILE:-1000}
EXPECTED_EVENTS=${EXPECTED_EVENTS:-20000}
SEED_A=${SEED_A:-20260907}
SEED_B=${SEED_B:-20260908}

cd "$PROJECT"

run_variant() {
    local name=$1
    local seed_base=$2
    local output_dir="/mnt/aohba/oldtreemc_seed_audit_20k_${name}_digitized"
    local log_dir="$HOME/oldtreemc_seed_audit_20k_${name}_logs"
    local command=(
        python tools/export/digitize_treemc_skims.py
        --input-dir "$INPUT_DIR"
        --output-dir "$output_dir"
        --log-dir "$log_dir"
        --crane "$CRANE"
        --jobs "$JOBS"
        --events-per-file "$EVENTS_PER_FILE"
        --expected-events "$EXPECTED_EVENTS"
    )
    if [[ -n "$seed_base" ]]; then
        command+=(--seed-base "$seed_base")
    fi
    echo "[VARIANT START] $name seed_base=${seed_base:-legacy-default}"
    "${command[@]}"
    echo "[VARIANT DONE] $name output=$output_dir"
}

run_variant legacy ""
run_variant seed_a "$SEED_A"
run_variant seed_b "$SEED_B"

echo "OLD TREEMC 20K DIGITIZATION SEED SETS: COMPLETE"
