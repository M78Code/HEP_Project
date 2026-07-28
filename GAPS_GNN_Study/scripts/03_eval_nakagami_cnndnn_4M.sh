#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained Nakagami CNN+DNN checkpoint.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_PARENT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

GPU="${GPU:-0}"
DATA_DIR="${DATA_DIR:-dataset/nakagami_atrest_voxel_gnn_4M}"
MODEL_PATH="${MODEL_PATH:-results/20260720-183528_CNNDNNFig72_nakagami_fig72_cnndnn_4M_rerun/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/20260720-183528_CNNDNNFig72_nakagami_fig72_cnndnn_4M_rerun/evaluation_test}"
BATCH_SIZE="${BATCH_SIZE:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FILE="${LOG_FILE:-$HOME/evaluate_nakagami_fig72_cnndnn_4M.log}"
LIMIT_ARGS=()

if [[ -n "${MAX_EVENTS:-}" ]]; then
  LIMIT_ARGS+=(--max-events "$MAX_EVENTS")
fi

CUDA_VISIBLE_DEVICES="$GPU" python -u src/eval/evaluate_nakagami_cnndnn.py \
  --data-dir "$DATA_DIR" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --split test \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --amp \
  "${LIMIT_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
