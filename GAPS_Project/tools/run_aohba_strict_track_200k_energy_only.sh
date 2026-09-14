#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
CACHE=${CACHE:-/mnt/aohba/archive_20260913_after_submission_mc1/aohba_stopping_ablation_strict_track_200k_global_log}
BASELINE_ROOT=${BASELINE_ROOT:-"$PROJECT/results/aohba_stopping_summary_only_ablation_200k/strict_track"}
NO_ENERGY_ROOT=${NO_ENERGY_ROOT:-"$PROJECT/results/aohba_strict_track_200k_input_ablation/no_energy"}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_strict_track_200k_input_ablation/energy_only"}
COMPARE_DIR=${COMPARE_DIR:-"$PROJECT/results/aohba_strict_track_200k_input_ablation/comparison_energy"}
SEED=${SEED:-20260825}
GPU=${GPU:-0}
TAG="aohba_strict_track_200k_energy_only_seed${SEED}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
cd "$PROJECT"

latest_eval()
{
    local root=$1
    find "$root" -type f -path '*/evaluation_test/labels.npy' \
        -printf '%T@ %h\n' 2>/dev/null |
        sort -nr |
        head -1 |
        cut -d' ' -f2-
}

latest_run()
{
    find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name "*_${TAG}" -printf '%T@ %p\n' 2>/dev/null |
        sort -nr |
        head -1 |
        cut -d' ' -f2-
}

test -f "$CACHE/_SUCCESS" || {
    echo "ERROR: strict-track 200K cache is missing or incomplete: $CACHE" >&2
    exit 1
}

FULL_EVAL=$(latest_eval "$BASELINE_ROOT")
NO_ENERGY_EVAL=$(latest_eval "$NO_ENERGY_ROOT")
for directory in "$FULL_EVAL" "$NO_ENERGY_EVAL"; do
    test -n "$directory" &&
    test -f "$directory/labels.npy" &&
    test -f "$directory/scores.npy" || {
        echo "ERROR: comparison evaluation is missing: $directory" >&2
        exit 1
    }
done

mkdir -p "$RESULT_ROOT" "$COMPARE_DIR"
RUN_DIR=$(latest_run || true)

if test -n "$RUN_DIR" &&
   test -f "$RUN_DIR/evaluation_test/metrics.json"; then
    echo "[SKIP] energy-only training and evaluation already complete: $RUN_DIR"
else
    CHECKPOINT=""
    if test -n "$RUN_DIR"; then
        CHECKPOINT=$(
            find "$RUN_DIR" -maxdepth 1 -type f \
                -name '*_last_checkpoint.pth' -print |
                sort |
                tail -1
        )
    fi

    TRAIN_ARGS=(
        --split-cache-dir "$CACHE"
        --epochs 80
        --model gravnet
        --input-ablation energy_only
        --batch-size 512
        --num-workers 2
        --non-blocking-transfer
        --seed "$SEED"
        --dataset-tag "$TAG"
        --result-dir "$RESULT_ROOT"
    )
    if test -n "$CHECKPOINT"; then
        echo "[RESUME] $CHECKPOINT"
        TRAIN_ARGS+=(--resume-checkpoint "$CHECKPOINT")
    fi

    echo "[TRAIN START] hit-energy-only input on physical GPU $GPU"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/train_aohba.py "${TRAIN_ARGS[@]}"

    RUN_DIR=$(latest_run)
    MODEL=$(
        find "$RUN_DIR" -maxdepth 1 -type f -name '*_best.pth' -print |
        sort |
        tail -1
    )
    test -n "$MODEL" || {
        echo "ERROR: best checkpoint was not found" >&2
        exit 1
    }

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/evaluate_aohba_split_cache.py \
        --cache-dir "$CACHE" \
        --model-path "$MODEL" \
        --output-dir "$RUN_DIR/evaluation_test" \
        --model gravnet \
        --input-ablation energy_only \
        --batch-size 512 \
        --seed "$SEED"
fi

ENERGY_ONLY_EVAL="$RUN_DIR/evaluation_test"
python -u src/scripts/visual/compare_binary_eval.py \
    --item "Full TreeRec input" "$FULL_EVAL" \
    --item "Without energy information" "$NO_ENERGY_EVAL" \
    --item "Hit energy only" "$ENERGY_ONLY_EVAL" \
    --out-dir "$COMPARE_DIR" \
    --x-min 0.20 \
    --y-max 1000000 \
    --zero-fpr-mode cap \
    --mark-efficiencies 0.90 0.95 0.98 0.99

echo "AOHBA STRICT TRACK-STOP 200K ENERGY-ONLY ABLATION: COMPLETE"
echo "result : $RUN_DIR"
echo "figure : $COMPARE_DIR/rejection_compare.png"
echo "metrics: $COMPARE_DIR/metrics_compare.json"
