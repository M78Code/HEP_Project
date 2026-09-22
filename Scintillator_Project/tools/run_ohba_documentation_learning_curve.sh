#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-$HOME/HEP_Project/Scintillator_Project}"
OUTPUT="${OUTPUT:-$PROJECT/results/ohba_hybrid_report_20260922/learning_curve_run}"
SPLIT_SEED="${SPLIT_SEED:-42}"

cd "$PROJECT"
echo "[START] documentation-only learning-curve run"
echo "output     : $OUTPUT"
echo "split seed : $SPLIT_SEED"
echo "GPU        : ${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 src/scripts/repeat_ohba_all_usable_splits.py \
  --output-root "$OUTPUT" \
  --split-seeds "$SPLIT_SEED"

HISTORY="$OUTPUT/split_${SPLIT_SEED}_model_20260825/history.json"
python3 src/scripts/plot_ohba_learning_history.py "$HISTORY"
cp "$OUTPUT/split_${SPLIT_SEED}_model_20260825/learning_curve.png" "$(dirname "$OUTPUT")/learning_curve.png"
cp "$OUTPUT/split_${SPLIT_SEED}_model_20260825/test_residual_comparison.png" "$(dirname "$OUTPUT")/same_position_residual_comparison.png"
echo "OHBA DOCUMENTATION LEARNING CURVE: COMPLETE"
