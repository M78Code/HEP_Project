#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-}
case "$PHASE" in
    cache-train) ;;
    *) echo "usage: $0 cache-train" >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
PROVENANCE=${PROVENANCE:-/mnt/aohba/aohba_treerec_hit_proxy_calibration_candidates_300k}
CACHE=${CACHE:-/mnt/aohba/aohba_treerec_truth_stop_selector_source_holdout_global_log}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_treerec_truth_stop_selector_source_holdout"}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
TRAIN_PER_CELL=${TRAIN_PER_CELL:-10000}
VAL_PER_CELL=${VAL_PER_CELL:-2000}
TEST_PER_CELL=${TEST_PER_CELL:-3000}
TAG="aohba_treerec_truth_stop_selector_source_holdout_seed${SEED}"

cd "$PROJECT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka

latest_run_dir() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -name "*_${2}" -print 2>/dev/null | sort | tail -1
}

if [[ -f "$CACHE/_SUCCESS" && -f "$CACHE/cache_manifest.json" ]]; then
    echo "[SKIP] cache complete: $CACHE"
elif [[ -e "$CACHE" ]]; then
    echo "ERROR: incomplete cache exists: $CACHE" >&2
    exit 1
else
    echo "[CACHE START] source-file-held-out TreeRec truth-stop selector"
    python -u src/data_parse/build_aohba_treerec_truth_stop_selector_cache.py \
        --provenance-dir "$PROVENANCE" --output-dir "$CACHE" --seed "$SEED" \
        --k 8 --chunk-size 10000 \
        --train-events-per-cell "$TRAIN_PER_CELL" \
        --val-events-per-cell "$VAL_PER_CELL" \
        --test-events-per-cell "$TEST_PER_CELL"
    echo "[CACHE DONE]"
fi

mkdir -p "$RESULT_ROOT"
RUN_DIR=$(latest_run_dir "$RESULT_ROOT" "$TAG" || true)
if [[ -n "$RUN_DIR" && -f "$RUN_DIR/evaluation_test/metrics.json" ]]; then
    echo "[SKIP] training/evaluation complete: $RUN_DIR"
    exit 0
fi
CHECKPOINT=""
if [[ -n "$RUN_DIR" ]]; then
    CHECKPOINT=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_last_checkpoint.pth' -print | sort | tail -1)
fi
ARGS=(
    --split-cache-dir "$CACHE" --epochs 80 --model gravnet --batch-size 512
    --num-workers 2 --non-blocking-transfer --seed "$SEED"
    --dataset-tag "$TAG" --result-dir "$RESULT_ROOT"
)
if [[ -n "$CHECKPOINT" ]]; then
    echo "[RESUME] $CHECKPOINT"
    ARGS+=(--resume-checkpoint "$CHECKPOINT")
fi
echo "[TRAIN START] truth-stop selector on physical GPU $GPU (serial only)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python -u src/scripts/train_aohba.py "${ARGS[@]}"

RUN_DIR=$(latest_run_dir "$RESULT_ROOT" "$TAG")
MODEL=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_best.pth' -print | sort | tail -1)
[[ -n "$MODEL" ]] || { echo "ERROR: best checkpoint missing" >&2; exit 1; }
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    python -u src/scripts/evaluate_aohba_split_cache.py \
        --cache-dir "$CACHE" --model-path "$MODEL" \
        --output-dir "$RUN_DIR/evaluation_test" --model gravnet \
        --batch-size 512 --seed "$SEED"

echo "evaluation: $RUN_DIR/evaluation_test"
echo "AOHBA TREEREC TRUTH-STOP SELECTOR: COMPLETE"
