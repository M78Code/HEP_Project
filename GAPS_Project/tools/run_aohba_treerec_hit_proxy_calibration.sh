#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-}
case "$PHASE" in
    export|audit) ;;
    *) echo "usage: $0 [export|audit]" >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
CANDIDATES=${CANDIDATES:-/mnt/aohba/aohba_treerec_hit_proxy_calibration_candidates_300k}
RESULT=${RESULT:-"$PROJECT/results/aohba_treerec_hit_proxy_calibration_50k"}
CANDIDATES_PER_CLASS=${CANDIDATES_PER_CLASS:-300000}
AUDIT_EVENTS_PER_CLASS=${AUDIT_EVENTS_PER_CLASS:-50000}
LOGDIR=${LOGDIR:-"$HOME/aohba_treerec_hit_proxy_calibration_logs"}

cd "$PROJECT"

activate_naka() {
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate naka
}

compile_exporter() {
    source "$HOME/setup_ynakagami_root.sh"
    mkdir -p build/tools "$CANDIDATES" "$LOGDIR"
    /usr/bin/g++ -std=c++17 -O2 $(root-config --cflags) \
        -I/home/ynakagami/simpledet/SimpleDet/common/include \
        -I/home/ynakagami/simpledet/build/install/gaps-v1.7.0/include/gaps \
        tools/export/export_treemc_topiso_like_csv.cc \
        -L/home/ynakagami/simpledet/build/common -lGAPSCommon \
        $(root-config --libs) -o build/tools/export_treemc_topiso_like_csv
}

export_particle() {
    local particle=$1 label=$2 input=$3 geometry=$4
    local output="$CANDIDATES/$particle"
    [[ -f "$output/_SUCCESS" ]] && { echo "[SKIP] $particle"; return; }
    [[ ! -e "$output" ]] || {
        echo "ERROR: incomplete candidate directory: $output" >&2
        return 1
    }
    echo "[START] $particle: legacy-atrest + truth Umbrella-to-Cube"
    env LD_PRELOAD=/home/ynakagami/GEANT/install/lib/libG4geometry.so:/home/ynakagami/GEANT/install/lib/libG4event.so:/home/ynakagami/simpledet/build/common/libGAPSCommon.so \
        build/tools/export_treemc_topiso_like_csv \
        --input "$input" --geometry-file "$geometry" --output-npy-dir "$output" \
        --max-events "$CANDIDATES_PER_CLASS" --target-label "$label" \
        --selection legacy-atrest-toptrigger --provenance-only \
        --truth-selection-flags \
        >"$LOGDIR/export_${particle}.log" 2>&1
    echo "[DONE] $particle"
}

run_export() {
    compile_exporter
    export_particle antiP 0 '/mnt/aohba/GAPS_Sim_2tof/antiP/*.root' \
        /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root &
    local pid_antip=$!
    export_particle antiD 1 '/mnt/aohba/GAPS_Sim_2tof/antiD/*.root' \
        /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root &
    local pid_antid=$!
    wait "$pid_antip"
    wait "$pid_antid"
}

run_audit() {
    for particle in antiP antiD; do
        test -f "$CANDIDATES/$particle/_SUCCESS" || {
            echo "ERROR: missing completed candidate pool: $CANDIDATES/$particle" >&2
            exit 1
        }
        test -f "$CANDIDATES/$particle/truth_strict_stop.npy" || {
            echo "ERROR: missing truth audit flags: rerun export with updated exporter" >&2
            exit 1
        }
    done
    activate_naka
    mkdir -p "$RESULT"
    python -u src/data_parse/calibrate_treerec_hit_stop_proxy.py \
        --provenance-dir "$CANDIDATES" \
        --events-per-class "$AUDIT_EVENTS_PER_CLASS" \
        --output "$RESULT/calibration.json"
}

echo "phase: $PHASE"
echo "TreeRec hitseries stopping-proxy calibration; no GPU is used."
case "$PHASE" in
    export) run_export ;;
    audit) run_audit ;;
esac
echo "AOHBA TREEREC HITSERIES PROXY CALIBRATION $PHASE: COMPLETE"
