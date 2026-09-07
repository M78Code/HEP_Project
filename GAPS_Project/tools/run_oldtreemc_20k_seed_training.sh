#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
TRAIN_SEED=${TRAIN_SEED:-20260825}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-"$PROJECT/results/oldtreemc_20k_digitization_seed_ablation_${STAMP}"}

cd "$PROJECT"
mkdir -p "$RUN_ROOT"

build_cache() {
    local name=$1
    local digitized="/mnt/aohba/oldtreemc_seed_audit_20k_${name}_digitized"
    local cache="/mnt/aohba/oldtreemc_seed_audit_20k_${name}_global_log"
    echo "[CACHE START] $name"
    python src/data_parse/build_oldtreemc_source_disjoint_treerec_cache.py \
        --pilot-reco-dir "$digitized" \
        --test-reco-dir "$digitized" \
        --output-dir "$cache" \
        --expected-train 16000 \
        --expected-val 2000 \
        --expected-test 2000 \
        --test-antip-source 1627528714 \
        --test-antid-source 1627550286
    echo "[CACHE DONE] $name cache=$cache"
}

for name in legacy seed_a seed_b; do
    build_cache "$name"
done

run_model() {
    local name=$1
    local gpu=$2
    local cache="/mnt/aohba/oldtreemc_seed_audit_20k_${name}_global_log"
    local parent="$RUN_ROOT/$name"
    mkdir -p "$parent"

    echo "[TRAIN START] $name gpu=$gpu"
    env CUDA_VISIBLE_DEVICES="$gpu" python src/scripts/train_aohba.py \
        --split-cache-dir "$cache" \
        --epochs 80 \
        --model gravnet \
        --batch-size 512 \
        --num-workers 2 \
        --non-blocking-transfer \
        --seed "$TRAIN_SEED" \
        --dataset-tag "oldtreemc_seed_audit_20k_${name}_seed${TRAIN_SEED}" \
        --result-dir "$parent" \
        2>&1 | tee "$HOME/train_oldtreemc_seed_audit_20k_${name}.log"

    local model
    model=$(find "$parent" -type f -name '*_best.pth' -print | sort | tail -1)
    if [[ -z "$model" ]]; then
        echo "best checkpoint missing for $name" >&2
        return 1
    fi
    local eval_dir="$(dirname "$model")/evaluation_test"

    echo "[EVAL START] $name gpu=$gpu"
    env CUDA_VISIBLE_DEVICES="$gpu" python \
        src/scripts/evaluate_aohba_split_cache.py \
        --cache-dir "$cache" \
        --model-path "$model" \
        --output-dir "$eval_dir" \
        --model gravnet \
        --batch-size 512 \
        --seed "$TRAIN_SEED" \
        2>&1 | tee "$HOME/evaluate_oldtreemc_seed_audit_20k_${name}.log"
    printf '%s\n' "$eval_dir" >"$RUN_ROOT/$name.eval_dir"
    echo "[MODEL DONE] $name evaluation=$eval_dir"
}

run_model legacy "$GPU0" &
legacy_pid=$!
run_model seed_a "$GPU1" &
seed_a_pid=$!
failed=0
wait "$legacy_pid" || failed=1
wait "$seed_a_pid" || failed=1
if (( failed )); then
    echo "legacy or seed_a failed" >&2
    exit 1
fi
run_model seed_b "$GPU0"

legacy=$(cat "$RUN_ROOT/legacy.eval_dir")
seed_a=$(cat "$RUN_ROOT/seed_a.eval_dir")
seed_b=$(cat "$RUN_ROOT/seed_b.eval_dir")

python src/scripts/visual/compare_binary_eval.py \
    --item "Repeated default RNG stream" "$legacy" \
    --item "Independent streams seed-base A" "$seed_a" \
    --item "Independent streams seed-base B" "$seed_b" \
    --out-dir "$RUN_ROOT/comparison" \
    --x-min 0.5 \
    --y-max 100000 \
    --zero-fpr-mode cap \
    --mark-efficiencies 0.90 0.95 0.98 0.99

echo "DIGITIZATION SEED ABLATION COMPLETE"
echo "results: $RUN_ROOT"
echo "figure : $RUN_ROOT/comparison/rejection_compare.png"
echo "metrics: $RUN_ROOT/comparison/metrics_compare.json"
