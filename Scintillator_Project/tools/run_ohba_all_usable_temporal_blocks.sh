#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/results/ohba_all_usable_temporal_blocks}"

export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scintillator_matplotlib_${USER:-user}}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_ROOT"

echo "[START] all-usable chronological event-block benchmark"
echo "project      : $PROJECT_DIR"
echo "output       : $OUTPUT_ROOT"
echo "model seeds  : ${MODEL_SEEDS:-20260825,20260826,20260827}"
echo "physical GPU : $GPU_ID (serial trials)"

"$PYTHON_BIN" -m Scintillator_Project.src.scripts.evaluate_ohba_all_usable_temporal_blocks \
    --output-root "$OUTPUT_ROOT" \
    --model-seeds "${MODEL_SEEDS:-20260825,20260826,20260827}" \
    --epochs "${EPOCHS:-300}" \
    --patience "${PATIENCE:-40}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --learning-rate "${LEARNING_RATE:-3e-4}" \
    --num-workers "${NUM_WORKERS:-2}"

echo "OHBA CHRONOLOGICAL EVENT-BLOCK BENCHMARK: COMPLETE"
