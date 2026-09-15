#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-}
case "$PHASE" in
    export-match|cache-train|compare) ;;
    *) echo "usage: $0 [export-match|cache-train|compare]" >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
CANDIDATES=/mnt/aohba/aohba_strict_stop_evidence_candidates_300k
MATCHED=/mnt/aohba/aohba_strict_stop_evidence_beta_matched_100k_per_class
LOGDIR="$HOME/aohba_strict_stop_evidence_ladder_200k_logs"
RESULT_ROOT="$PROJECT/results/aohba_strict_stop_evidence_ladder_200k"
CACHE_PREFIX=aohba_strict_stop_evidence_ladder
SEED=20260825
EVENTS_PER_CLASS=100000
CANDIDATES_PER_CLASS=300000
SUMMARY=summary_atrest_topology
KINETIC=kinetic_zero_in_tracker_topology
ZERO_STEP=zero_step_topology
STRICT=strict_track_topology

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
    local group=$1 selection=$2 particle=$3 label=$4 input=$5 geometry=$6
    local output="$CANDIDATES/$group/$particle"
    [[ -f "$output/_SUCCESS" ]] && { echo "[SKIP] $group/$particle"; return; }
    [[ ! -e "$output" ]] || {
        echo "ERROR: incomplete candidate directory: $output" >&2
        return 1
    }
    mkdir -p "$CANDIDATES/$group"
    echo "[START] $group/$particle selection=$selection"
    env LD_PRELOAD=/home/ynakagami/GEANT/install/lib/libG4geometry.so:/home/ynakagami/GEANT/install/lib/libG4event.so:/home/ynakagami/simpledet/build/common/libGAPSCommon.so \
        build/tools/export_treemc_topiso_like_csv \
        --input "$input" --geometry-file "$geometry" --output-npy-dir "$output" \
        --max-events "$CANDIDATES_PER_CLASS" --target-label "$label" \
        --selection "$selection" --provenance-only \
        >"$LOGDIR/export_${group}_${particle}.log" 2>&1
    echo "[DONE] $group/$particle"
}

export_group() {
    local group=$1 selection=$2
    export_particle "$group" "$selection" antiP 0 \
        '/mnt/aohba/GAPS_Sim_2tof/antiP/*.root' \
        /mnt/aohba/GAPS_Sim_2tof/antiP/antiP_2tof_FTFP_BERT_1781424263.root &
    local pid_antip=$!
    export_particle "$group" "$selection" antiD 1 \
        '/mnt/aohba/GAPS_Sim_2tof/antiD/*.root' \
        /mnt/aohba/GAPS_Sim_2tof/antiD/antiD_2tof_FTFP_BERT_1781253355.root &
    local pid_antid=$!
    wait "$pid_antip"
    wait "$pid_antid"
}

run_export_match() {
    compile_exporter
    # All four pools are made with the same summary-at-rest and topology base.
    export_group "$SUMMARY" legacy-atrest-toptrigger
    export_group "$KINETIC" legacy-atrest-kinetic-zero-toptrigger
    export_group "$ZERO_STEP" legacy-atrest-zero-step-toptrigger
    export_group "$STRICT" legacy-atrest-strict-toptrigger

    activate_naka
    [[ ! -e "$MATCHED" ]] || {
        echo "ERROR: output already exists: $MATCHED" >&2
        exit 1
    }
    python -u src/data_parse/match_treemc_beta_bins.py \
        --output-dir "$MATCHED" --events-per-class "$EVENTS_PER_CLASS" --seed "$SEED" \
        --group "$SUMMARY=$CANDIDATES/$SUMMARY" \
        --group "$KINETIC=$CANDIDATES/$KINETIC" \
        --group "$ZERO_STEP=$CANDIDATES/$ZERO_STEP" \
        --group "$STRICT=$CANDIDATES/$STRICT"
}

run_cache_train() {
    activate_naka
    [[ -f "$MATCHED/_SUCCESS" ]] || { echo "ERROR: missing $MATCHED/_SUCCESS" >&2; exit 1; }
    PROJECT="$PROJECT" MATCHED="$MATCHED" RESULT_ROOT="$RESULT_ROOT" \
        CACHE_PREFIX="$CACHE_PREFIX" GPU=0 SEED="$SEED" \
        GROUP_A="$SUMMARY" GROUP_B="$KINETIC" \
        SELECTION_A=legacy-atrest-toptrigger \
        SELECTION_B=legacy-atrest-kinetic-zero-toptrigger \
        LABEL_A='Summary at-rest + Umbrella-to-Cube' \
        LABEL_B='Summary at-rest + K=0 in tracker + Umbrella-to-Cube' \
        TRAIN_GROUPS="$SUMMARY $KINETIC $ZERO_STEP $STRICT" \
        bash tools/run_aohba_stopping_ablation_200k.sh cache-train
}

latest_run() {
    local group=$1 tag="${CACHE_PREFIX}_${1}_200k_global_log_seed${SEED}"
    find "$RESULT_ROOT/$group" -mindepth 1 -maxdepth 1 -type d -name "*_${tag}" \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
}

run_compare() {
    activate_naka
    local summary_run kinetic_run zero_step_run strict_run
    summary_run=$(latest_run "$SUMMARY")
    kinetic_run=$(latest_run "$KINETIC")
    zero_step_run=$(latest_run "$ZERO_STEP")
    strict_run=$(latest_run "$STRICT")
    for run in "$summary_run" "$kinetic_run" "$zero_step_run" "$strict_run"; do
        [[ -f "$run/evaluation_test/labels.npy" ]] || { echo "ERROR: incomplete evaluation: $run" >&2; exit 1; }
    done
    python src/scripts/visual/compare_binary_eval.py \
        --item 'Summary at-rest + Umbrella-to-Cube' "$summary_run/evaluation_test" \
        --item 'Summary at-rest + K=0 in tracker' "$kinetic_run/evaluation_test" \
        --item 'Summary at-rest + zero step' "$zero_step_run/evaluation_test" \
        --item 'Summary at-rest + K=0 + zero step' "$strict_run/evaluation_test" \
        --out-dir "$RESULT_ROOT/comparison" --x-min 0.90 --y-max 20000 \
        --mark-efficiencies 0.90 0.95 0.98 0.99
    echo "figure : $RESULT_ROOT/comparison/rejection_compare.png"
    echo "metrics: $RESULT_ROOT/comparison/metrics_compare.json"
}

echo "phase: $PHASE"
echo "strict-stop evidence ladder: summary -> K=0 -> zero step -> both"
echo "events/class/group: $EVENTS_PER_CLASS; GPU: 0; serial: yes"
case "$PHASE" in
    export-match) run_export_match ;;
    cache-train) run_cache_train ;;
    compare) run_compare ;;
esac
echo "AOHBA STRICT-STOP EVIDENCE LADDER 200K $PHASE: COMPLETE"
