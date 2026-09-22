#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/results/ohba_quality_selected_hybrid}"
QUALITY_RESULTS="${QUALITY_RESULTS:-$PROJECT_DIR/results/ohba_quality_selection/quality_selection_results.json}"
SEEDS="${SEEDS:-20260825}"
GPU_ID="${GPU_ID:-0}"

export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scintillator_matplotlib_${USER:-user}}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_ROOT"

echo "[START] quality-selected hybrid waveform benchmark"
echo "project         : $PROJECT_DIR"
echo "quality results : $QUALITY_RESULTS"
echo "output root     : $OUTPUT_ROOT"
echo "seeds           : $SEEDS (serial)"
echo "physical GPU    : $GPU_ID"

for SEED in $SEEDS; do
    RUN_DIR="$OUTPUT_ROOT/seed_${SEED}"
    echo "[RUN] seed=$SEED"
    "$PYTHON_BIN" -m Scintillator_Project.src.scripts.train_ohba_quality_selected_hybrid \
        --seed "$SEED" \
        --epochs "${EPOCHS:-300}" \
        --patience "${PATIENCE:-40}" \
        --batch-size "${BATCH_SIZE:-128}" \
        --learning-rate "${LEARNING_RATE:-3e-4}" \
        --num-workers "${NUM_WORKERS:-2}" \
        --quality-results "$QUALITY_RESULTS" \
        --output-dir "$RUN_DIR"
done

echo "OHBA QUALITY-SELECTED HYBRID BENCHMARK: COMPLETE"
