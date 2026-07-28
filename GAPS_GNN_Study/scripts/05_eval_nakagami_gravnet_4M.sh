#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained Nakagami sparse-voxel GravNet checkpoint.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_PARENT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"

GPU="${GPU:-0}"
DATA_DIR="${DATA_DIR:-dataset/nakagami_atrest_voxel_gnn_4M}"
DATASET_TAG="${DATASET_TAG:-nakagami_atrest_sparse_voxel_gravnet_4M_tof11}"
if [[ -z "${MODEL_PATH:-}" ]]; then
  LATEST_DIR="$(find results -maxdepth 1 -type d -name "*_${DATASET_TAG}" | sort | tail -n 1)"
  if [[ -z "$LATEST_DIR" ]]; then
    echo "No trained result found for DATASET_TAG=${DATASET_TAG}. Run scripts/04_train_nakagami_gravnet_4M.sh first or set MODEL_PATH." >&2
    exit 1
  fi
  MODEL_PATH="${LATEST_DIR}/best.pt"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$MODEL_PATH")/evaluation_test}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FILE="${LOG_FILE:-$HOME/evaluate_nakagami_atrest_sparse_voxel_gravnet_4M_tof11.log}"

CUDA_VISIBLE_DEVICES="$GPU" python -u src/eval/evaluate_nakagami_gravnet.py \
  --data-dir "$DATA_DIR" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --split test \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  2>&1 | tee "$LOG_FILE"
