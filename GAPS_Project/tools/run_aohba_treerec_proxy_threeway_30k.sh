#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-}
case "$PHASE" in
    prepare-match|cache-train|compare) ;;
    *)
        echo "usage: $0 [prepare-match|cache-train|compare]" >&2
        exit 2
        ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
SOURCE_CANDIDATES=${SOURCE_CANDIDATES:-/mnt/aohba/aohba_treerec_hit_proxy_calibration_candidates_300k}
DERIVED=${DERIVED:-/mnt/aohba/aohba_treerec_proxy_threeway_candidates_300k}
MATCHED=${MATCHED:-/mnt/aohba/aohba_treerec_proxy_threeway_beta_matched_30k}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_treerec_proxy_threeway_30k"}
EVENTS_PER_CLASS=${EVENTS_PER_CLASS:-30000}
# Keep small pilots at the same 8:1:1 split when EVENTS_PER_CLASS is overridden.
TRAIN_PER_CLASS=${TRAIN_PER_CLASS:-$((EVENTS_PER_CLASS * 8 / 10))}
VAL_PER_CLASS=${VAL_PER_CLASS:-$((EVENTS_PER_CLASS / 10))}
TEST_PER_CLASS=${TEST_PER_CLASS:-$((EVENTS_PER_CLASS - TRAIN_PER_CLASS - VAL_PER_CLASS))}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
CACHE_PREFIX=${CACHE_PREFIX:-aohba_treerec_proxy_threeway}
GROUPS=(summary_atrest_topology treerec_logistic_proxy truth_strict_topology)

cd "$PROJECT"

activate_naka() {
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate naka
}

latest_run_dir() {
    local root=$1 tag=$2
    find "$root" -mindepth 1 -maxdepth 1 -type d -name "*_${tag}" -print 2>/dev/null | sort | tail -1
}

check_split_sizes() {
    local total=$((TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS))
    [[ "$total" -eq "$EVENTS_PER_CLASS" ]] || {
        echo "ERROR: train + val + test must equal EVENTS_PER_CLASS" >&2
        exit 2
    }
}

prepare_match() {
    activate_naka
    for particle in antiP antiD; do
        [[ -f "$SOURCE_CANDIDATES/$particle/_SUCCESS" ]] || {
            echo "ERROR: source candidate pool is incomplete: $SOURCE_CANDIDATES/$particle" >&2
            exit 1
        }
    done
    if [[ ! -f "$DERIVED/_SUCCESS" ]]; then
        [[ ! -e "$DERIVED" ]] || {
            echo "ERROR: incomplete derived provenance exists: $DERIVED" >&2
            exit 1
        }
        python -u src/data_parse/build_treerec_proxy_threeway_provenance.py \
            --source-dir "$SOURCE_CANDIDATES" \
            --output-dir "$DERIVED" \
            --seed "$SEED"
    else
        echo "[SKIP] derived three-way provenance: $DERIVED"
    fi
    if [[ -f "$MATCHED/_SUCCESS" ]]; then
        echo "[SKIP] beta-matched provenance: $MATCHED"
        return
    fi
    [[ ! -e "$MATCHED" ]] || {
        echo "ERROR: incomplete matched provenance exists: $MATCHED" >&2
        exit 1
    }
    python -u src/data_parse/match_treemc_beta_bins.py \
        --output-dir "$MATCHED" \
        --events-per-class "$EVENTS_PER_CLASS" \
        --seed "$SEED" \
        --group "summary_atrest_topology=$DERIVED/summary_atrest_topology" \
        --group "treerec_logistic_proxy=$DERIVED/treerec_logistic_proxy" \
        --group "truth_strict_topology=$DERIVED/truth_strict_topology"
}

run_group() {
    local group=$1
    local provenance="$MATCHED/$group"
    local cache="/mnt/aohba/${CACHE_PREFIX}_${group}_${EVENTS_PER_CLASS}_global_log"
    local tag="${CACHE_PREFIX}_${group}_${EVENTS_PER_CLASS}_global_log_seed${SEED}"
    local result_dir="$RESULT_ROOT/$group"
    local run_dir checkpoint model

    if [[ -f "$cache/_SUCCESS" && -f "$cache/cache_manifest.json" ]]; then
        echo "[SKIP] $group cache: $cache"
    elif [[ -e "$cache" ]]; then
        echo "ERROR: incomplete cache exists: $cache" >&2
        exit 1
    else
        echo "[CACHE START] $group"
        python -u src/data_parse/build_aohba_fixedgrid_selected_treerec_cache.py \
            --fixedgrid-raw-dir "$provenance" \
            --output-dir "$cache" \
            --chunk-size 10000 \
            --k 8 \
            --train-events-per-class "$TRAIN_PER_CLASS" \
            --val-events-per-class "$VAL_PER_CLASS" \
            --test-events-per-class "$TEST_PER_CLASS"
        echo "[CACHE DONE] $group"
    fi

    mkdir -p "$result_dir"
    run_dir=$(latest_run_dir "$result_dir" "$tag" || true)
    if [[ -n "$run_dir" && -f "$run_dir/evaluation_test/metrics.json" ]]; then
        echo "[SKIP] $group training/evaluation: $run_dir"
        return
    fi
    checkpoint=""
    if [[ -n "$run_dir" ]]; then
        checkpoint=$(find "$run_dir" -maxdepth 1 -type f -name '*_last_checkpoint.pth' -print | sort | tail -1)
    fi
    local train_args=(
        --split-cache-dir "$cache"
        --epochs 80
        --model gravnet
        --batch-size 512
        --num-workers 2
        --non-blocking-transfer
        --seed "$SEED"
        --dataset-tag "$tag"
        --result-dir "$result_dir"
    )
    if [[ -n "$checkpoint" ]]; then
        echo "[RESUME] $checkpoint"
        train_args+=(--resume-checkpoint "$checkpoint")
    fi
    echo "[TRAIN START] $group on physical GPU $GPU (serial)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/train_aohba.py "${train_args[@]}"
    run_dir=$(latest_run_dir "$result_dir" "$tag")
    model=$(find "$run_dir" -maxdepth 1 -type f -name '*_best.pth' -print | sort | tail -1)
    [[ -n "$model" ]] || { echo "ERROR: best checkpoint missing: $group" >&2; exit 1; }
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/evaluate_aohba_split_cache.py \
            --cache-dir "$cache" --model-path "$model" \
            --output-dir "$run_dir/evaluation_test" --model gravnet \
            --batch-size 512 --seed "$SEED"
    echo "[TRAIN DONE] $group"
    echo "evaluation: $run_dir/evaluation_test"
}

cache_train() {
    activate_naka
    check_split_sizes
    [[ -f "$MATCHED/_SUCCESS" ]] || {
        echo "ERROR: matched provenance is incomplete: $MATCHED" >&2
        exit 1
    }
    local group
    for group in "${GROUPS[@]}"; do
        run_group "$group"
    done
}

compare() {
    activate_naka
    local group tag run_dir
    local args=()
    for group in "${GROUPS[@]}"; do
        tag="${CACHE_PREFIX}_${group}_${EVENTS_PER_CLASS}_global_log_seed${SEED}"
        run_dir=$(latest_run_dir "$RESULT_ROOT/$group" "$tag")
        [[ -n "$run_dir" && -f "$run_dir/evaluation_test/labels.npy" && -f "$run_dir/evaluation_test/scores.npy" ]] || {
            echo "ERROR: evaluation missing for $group" >&2
            exit 1
        }
        case "$group" in
            summary_atrest_topology) args+=(--item "Summary at-rest + Umbrella-to-Cube" "$run_dir/evaluation_test") ;;
            treerec_logistic_proxy) args+=(--item "TreeRec hitseries logistic proxy" "$run_dir/evaluation_test") ;;
            truth_strict_topology) args+=(--item "Truth strict stop (K=0 + zero step)" "$run_dir/evaluation_test") ;;
        esac
    done
    local out="$RESULT_ROOT/comparison"
    python src/scripts/visual/compare_binary_eval.py "${args[@]}" \
        --out-dir "$out" --x-min 0.80 --y-max 3500 \
        --mark-efficiencies 0.90 0.95 0.98 0.99
    echo "figure : $out/rejection_compare.png"
    echo "metrics: $out/metrics_compare.json"
}

echo "phase                 : $PHASE"
echo "source candidates     : $SOURCE_CANDIDATES"
echo "matched events/class  : $EVENTS_PER_CLASS ($TRAIN_PER_CLASS/$VAL_PER_CLASS/$TEST_PER_CLASS)"
echo "physical GPU          : $GPU (serial only)"
echo "seed                  : $SEED"

case "$PHASE" in
    prepare-match) prepare_match ;;
    cache-train) cache_train ;;
    compare) compare ;;
esac
echo "AOHBA TREEREC PROXY THREE-WAY 30K $PHASE: COMPLETE"
