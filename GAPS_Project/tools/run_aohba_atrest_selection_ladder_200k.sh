#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-}
case "$PHASE" in
    export-match|cache-train|compare) ;;
    *) echo "usage: $0 [export-match|cache-train|compare]" >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
CANDIDATES=/mnt/aohba/aohba_legacy_atrest_vs_strict_candidates_300k
MATCHED=/mnt/aohba/aohba_atrest_selection_ladder_beta_matched_100k_per_class
LOGDIR="$HOME/aohba_atrest_selection_ladder_200k_logs"
RESULT_ROOT="$PROJECT/results/aohba_atrest_selection_ladder_200k"
CACHE_PREFIX=aohba_atrest_selection_ladder
SEED=20260825
EVENTS_PER_CLASS=100000
CANDIDATES_PER_CLASS=300000
STRICT=strict_track_topology
SUMMARY=summary_atrest_topology
LEGACY=legacy_atrest

cd "$PROJECT"

activate_naka() {
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate naka
}

run_export_match() {
    source "$HOME/setup_ynakagami_root.sh"
    mkdir -p build/tools "$CANDIDATES" "$LOGDIR"
    for group in "$STRICT" "$LEGACY"; do
        for particle in antiP antiD; do
            [[ -f "$CANDIDATES/$group/$particle/_SUCCESS" ]] || {
                echo "ERROR: missing existing candidate pool: $CANDIDATES/$group/$particle" >&2
                exit 1
            }
        done
    done
    /usr/bin/g++ -std=c++17 -O2 $(root-config --cflags) \
        -I/home/ynakagami/simpledet/SimpleDet/common/include \
        -I/home/ynakagami/simpledet/build/install/gaps-v1.7.0/include/gaps \
        tools/export/export_treemc_topiso_like_csv.cc \
        -L/home/ynakagami/simpledet/build/common -lGAPSCommon \
        $(root-config --libs) -o build/tools/export_treemc_topiso_like_csv

    export_one() {
        local particle=$1 label=$2 input=$3 geometry=$4
        local output="$CANDIDATES/$SUMMARY/$particle"
        [[ -f "$output/_SUCCESS" ]] && { echo "[SKIP] $SUMMARY/$particle"; return; }
        [[ ! -e "$output" ]] || { echo "ERROR: incomplete $output" >&2; return 1; }
        mkdir -p "$CANDIDATES/$SUMMARY"
        echo "[START] $SUMMARY/$particle selection=legacy-atrest-toptrigger"
        env LD_PRELOAD=/home/ynakagami/GEANT/install/lib/libG4geometry.so:/home/ynakagami/GEANT/install/lib/libG4event.so:/home/ynakagami/simpledet/build/common/libGAPSCommon.so \
            build/tools/export_treemc_topiso_like_csv \
            --input "$input" --geometry-file "$geometry" --output-npy-dir "$output" \
            --max-events "$CANDIDATES_PER_CLASS" --target-label "$label" \
            --selection legacy-atrest-toptrigger --provenance-only \
            >"$LOGDIR/export_${SUMMARY}_${particle}.log" 2>&1
        echo "[DONE] $SUMMARY/$particle"
    }
    export_one antiP 0 '/mnt/aohba/GAPS_Sim_2tof/antiP/*.root' /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root &
    local pid_antip=$!
    export_one antiD 1 '/mnt/aohba/GAPS_Sim_2tof/antiD/*.root' /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root &
    local pid_antid=$!
    wait "$pid_antip"
    wait "$pid_antid"

    activate_naka
    [[ ! -e "$MATCHED" ]] || { echo "ERROR: output already exists: $MATCHED" >&2; exit 1; }
    python -u src/data_parse/match_treemc_beta_bins.py \
        --output-dir "$MATCHED" --events-per-class "$EVENTS_PER_CLASS" --seed "$SEED" \
        --group "$STRICT=$CANDIDATES/$STRICT" \
        --group "$SUMMARY=$CANDIDATES/$SUMMARY" \
        --group "$LEGACY=$CANDIDATES/$LEGACY"
}

run_cache_train() {
    activate_naka
    [[ -f "$MATCHED/_SUCCESS" ]] || { echo "ERROR: missing $MATCHED/_SUCCESS" >&2; exit 1; }
    PROJECT="$PROJECT" MATCHED="$MATCHED" RESULT_ROOT="$RESULT_ROOT" \
        CACHE_PREFIX="$CACHE_PREFIX" GPU=0 SEED="$SEED" \
        GROUP_A="$STRICT" GROUP_B="$SUMMARY" \
        TRAIN_GROUPS="$STRICT $SUMMARY $LEGACY" \
        bash tools/run_aohba_stopping_ablation_200k.sh cache-train
}

latest_run() {
    local group=$1 tag="${CACHE_PREFIX}_${1}_200k_global_log_seed${SEED}"
    find "$RESULT_ROOT/$group" -mindepth 1 -maxdepth 1 -type d -name "*_${tag}" \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
}

run_compare() {
    activate_naka
    local strict_run summary_run legacy_run
    strict_run=$(latest_run "$STRICT")
    summary_run=$(latest_run "$SUMMARY")
    legacy_run=$(latest_run "$LEGACY")
    for run in "$strict_run" "$summary_run" "$legacy_run"; do
        [[ -f "$run/evaluation_test/labels.npy" ]] || { echo "ERROR: incomplete evaluation: $run" >&2; exit 1; }
    done
    python src/scripts/visual/compare_binary_eval.py \
        --item "Strict track stop + Umbrella-to-Cube" "$strict_run/evaluation_test" \
        --item "Summary at-rest + Umbrella-to-Cube" "$summary_run/evaluation_test" \
        --item "Legacy tracker-at-rest summary" "$legacy_run/evaluation_test" \
        --out-dir "$RESULT_ROOT/comparison" --x-min 0.90 --y-max 20000 \
        --mark-efficiencies 0.90 0.95 0.98 0.99
    echo "figure : $RESULT_ROOT/comparison/rejection_compare.png"
    echo "metrics: $RESULT_ROOT/comparison/metrics_compare.json"
}

echo "phase: $PHASE"
echo "selection ladder: $LEGACY -> $SUMMARY -> $STRICT"
echo "events/class/group: $EVENTS_PER_CLASS; GPU: 0; serial: yes"
case "$PHASE" in
    export-match) run_export_match ;;
    cache-train) run_cache_train ;;
    compare) run_compare ;;
esac
echo "AOHBA AT-REST SELECTION LADDER 200K $PHASE: COMPLETE"
