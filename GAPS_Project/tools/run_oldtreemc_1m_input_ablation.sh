#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
CACHE=${CACHE:-/mnt/aohba/oldtreemc_source_disjoint_1m_global_log}
SEED=${SEED:-20260825}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-"$PROJECT/results/oldtreemc_1m_input_ablation_${STAMP}"}
BASELINE_EVAL=${BASELINE_EVAL:-"$PROJECT/results/20260906-000617_GravNet_6b_h128_oldtreemc_source_disjoint_1m_global_log_baseline_seed20260825/evaluation_test"}

cd "$PROJECT"
mkdir -p "$RUN_ROOT"

run_one() {
    local mode=$1
    local gpu=$2
    local parent="$RUN_ROOT/$mode"
    local train_log="$HOME/train_oldtreemc_1m_ablation_${mode}_${STAMP}.log"
    local eval_log="$HOME/evaluate_oldtreemc_1m_ablation_${mode}_${STAMP}.log"

    mkdir -p "$parent"
    echo "[TRAIN START] mode=$mode gpu=$gpu"
    set +e
    env CUDA_VISIBLE_DEVICES="$gpu" python src/scripts/train_aohba.py \
        --split-cache-dir "$CACHE" \
        --epochs 80 \
        --model gravnet \
        --input-ablation "$mode" \
        --batch-size 512 \
        --num-workers 2 \
        --non-blocking-transfer \
        --seed "$SEED" \
        --dataset-tag "oldtreemc_source_disjoint_1m_${mode}_seed${SEED}" \
        --result-dir "$parent" \
        2>&1 | tee "$train_log"
    local train_status=${PIPESTATUS[0]}
    set -e
    if (( train_status != 0 )); then
        echo "[TRAIN FAILED] mode=$mode status=$train_status"
        return "$train_status"
    fi

    local model
    model=$(find "$parent" -type f -name '*_best.pth' -print | sort | tail -1)
    if [[ -z "$model" ]]; then
        echo "[TRAIN FAILED] mode=$mode best checkpoint missing"
        return 1
    fi
    local run_dir
    run_dir=$(dirname "$model")
    local eval_dir="$run_dir/evaluation_test"

    echo "[EVAL START] mode=$mode gpu=$gpu model=$model"
    set +e
    env CUDA_VISIBLE_DEVICES="$gpu" python src/scripts/evaluate_aohba_split_cache.py \
        --cache-dir "$CACHE" \
        --model-path "$model" \
        --output-dir "$eval_dir" \
        --model gravnet \
        --input-ablation "$mode" \
        --batch-size 512 \
        --seed "$SEED" \
        2>&1 | tee "$eval_log"
    local eval_status=${PIPESTATUS[0]}
    set -e
    if (( eval_status != 0 )); then
        echo "[EVAL FAILED] mode=$mode status=$eval_status"
        return "$eval_status"
    fi
    printf '%s\n' "$eval_dir" > "$RUN_ROOT/$mode.eval_dir"
    echo "[DONE] mode=$mode evaluation=$eval_dir"
}

export PROJECT CACHE SEED GPU0 GPU1 STAMP RUN_ROOT BASELINE_EVAL
export -f run_one

run_pair() {
    local first=$1
    local second=$2
    run_one "$first" "$GPU0" &
    local first_pid=$!
    run_one "$second" "$GPU1" &
    local second_pid=$!
    local failed=0
    wait "$first_pid" || failed=1
    wait "$second_pid" || failed=1
    return "$failed"
}

run_pair node_only event_only
run_pair no_energy no_time

baseline="$BASELINE_EVAL"
node_only=$(cat "$RUN_ROOT/node_only.eval_dir")
event_only=$(cat "$RUN_ROOT/event_only.eval_dir")
no_energy=$(cat "$RUN_ROOT/no_energy.eval_dir")
no_time=$(cat "$RUN_ROOT/no_time.eval_dir")

python src/scripts/visual/compare_binary_eval.py \
    --item "Full: node + 45D event" "$baseline" \
    --item "Node only" "$node_only" \
    --item "45D event only" "$event_only" \
    --item "No energy information" "$no_energy" \
    --item "No timing information" "$no_time" \
    --out-dir "$RUN_ROOT/comparison" \
    --x-min 0.5 \
    --y-max 100000 \
    --zero-fpr-mode cap \
    --mark-efficiencies 0.90 0.95 0.98 0.99

echo "ABLATION COMPLETE"
echo "results: $RUN_ROOT"
echo "figure : $RUN_ROOT/comparison/rejection_compare.png"
echo "metrics: $RUN_ROOT/comparison/metrics_compare.json"
