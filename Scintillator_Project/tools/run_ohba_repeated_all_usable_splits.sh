#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/results/ohba_all_usable_repeated_splits}"

export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scintillator_matplotlib_${USER:-user}}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_ROOT"

echo "[START] repeated all-usable event-split benchmark"
echo "project      : $PROJECT_DIR"
echo "output       : $OUTPUT_ROOT"
echo "split seeds  : ${SPLIT_SEEDS:-42,31415,271828}"
echo "model seed   : ${MODEL_SEED:-20260825}"
echo "physical GPU : $GPU_ID (serial trials)"

"$PYTHON_BIN" -m Scintillator_Project.src.scripts.repeat_ohba_all_usable_splits \
    --output-root "$OUTPUT_ROOT" \
    --split-seeds "${SPLIT_SEEDS:-42,31415,271828}" \
    --model-seed "${MODEL_SEED:-20260825}" \
    --epochs "${EPOCHS:-300}" \
    --patience "${PATIENCE:-40}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --learning-rate "${LEARNING_RATE:-3e-4}" \
    --num-workers "${NUM_WORKERS:-2}"

echo "OHBA REPEATED ALL-USABLE SPLIT BENCHMARK: COMPLETE"
