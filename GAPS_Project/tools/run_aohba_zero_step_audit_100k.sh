#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
OUTDIR=${OUTDIR:-"$HOME/aohba_zero_step_audit_100k"}
LOGDIR=${LOGDIR:-"$HOME/aohba_zero_step_audit_100k_logs"}
MAX_EVENTS=${MAX_EVENTS:-100000}

cd "$PROJECT"
source "$HOME/setup_ynakagami_root.sh"
mkdir -p build/tools "$OUTDIR" "$LOGDIR"

/usr/bin/g++ -std=c++17 -O2 $(root-config --cflags) \
    -I/home/ynakagami/simpledet/SimpleDet/common/include \
    -I/home/ynakagami/simpledet/build/install/gaps-v1.7.0/include/gaps \
    tools/export/export_treemc_topiso_like_csv.cc \
    -L/home/ynakagami/simpledet/build/common -lGAPSCommon \
    $(root-config --libs) -o build/tools/export_treemc_topiso_like_csv

audit_particle() {
    local group=$1 selection=$2 particle=$3 label=$4 input=$5 geometry=$6
    local output="$OUTDIR/$group/${particle}.json"
    mkdir -p "$OUTDIR/$group"
    echo "[START] $group/$particle selection=$selection"
    env LD_PRELOAD=/home/ynakagami/GEANT/install/lib/libG4geometry.so:/home/ynakagami/GEANT/install/lib/libG4event.so:/home/ynakagami/simpledet/build/common/libGAPSCommon.so \
        build/tools/export_treemc_topiso_like_csv \
        --input "$input" --geometry-file "$geometry" --max-events "$MAX_EVENTS" \
        --target-label "$label" --selection "$selection" \
        --zero-step-audit-output "$output" \
        >"$LOGDIR/${group}_${particle}.log" 2>&1
    echo "[DONE] $group/$particle: $output"
}

audit_group() {
    local group=$1 selection=$2
    audit_particle "$group" "$selection" antiP 0 \
        '/mnt/aohba/GAPS_Sim_2tof/antiP/*.root' \
        /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root &
    local pid_antip=$!
    audit_particle "$group" "$selection" antiD 1 \
        '/mnt/aohba/GAPS_Sim_2tof/antiD/*.root' \
        /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root &
    local pid_antid=$!
    wait "$pid_antip"
    wait "$pid_antid"
}

# Profile the broad paper-like base first, then the complete strict subset.
audit_group summary_atrest_topology legacy-atrest-toptrigger
audit_group strict_track_topology legacy-atrest-strict-toptrigger

echo
echo '===== zero-step audit summary ====='
python - "$OUTDIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(root.glob('*/*.json')):
    data = json.loads(path.read_text())
    n = data['selected_events']
    print(f"\n{path.parent.name}/{path.stem}: selected={n:,}")
    for key in (
        'events_with_zero_step',
        'events_with_zero_step_in_tracker',
        'events_with_zero_step_same_index_kinetic_zero',
        'events_with_zero_step_same_index_tracker_kinetic_zero',
    ):
        value = data[key]
        print(f"  {key}: {value:,} ({value / n:.2%})")
    print('  zero-step locations:',
          f"tracker={data['zero_steps_in_tracker']:,},",
          f"umbrella={data['zero_steps_in_umbrella']:,},",
          f"cube={data['zero_steps_in_cube']:,},",
          f"elsewhere={data['zero_steps_elsewhere']:,}")
    print('  zero-step values:',
          f"mean edep={data['mean_edep_at_zero_step']:.6g},",
          f"mean KE={data['mean_kinetic_energy_at_zero_step']:.6g}")
PY

echo "AOHBA ZERO-STEP AUDIT: COMPLETE"
