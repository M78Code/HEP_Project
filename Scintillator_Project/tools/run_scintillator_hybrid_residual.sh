#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(dirname "$PROJECT_DIR")"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results/scintillator_hybrid_residual}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_DIR/dataset/cache}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-25}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEEDS="${SEEDS:-20260825 20260826 20260827}"

export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scintillator_matplotlib_${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" "$RESULT_ROOT"

echo "[START] scintillator physics-guided hybrid residual study"
echo "project : $PROJECT_DIR"
echo "results : $RESULT_ROOT"
echo "cache   : $CACHE_DIR"
echo "events  : 16,514 train / 3,533 val / 3,555 test"
echo "seeds   : $SEEDS (serial)"

for seed in $SEEDS; do
    output_dir="$RESULT_ROOT/seed_$seed"
    mkdir -p "$output_dir"
    echo "[TRAIN START] seed=$seed"
    "$PYTHON_BIN" -m Scintillator_Project.src.scripts.train_hybrid_residual \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --batch-size "$BATCH_SIZE" \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$output_dir" \
        2>&1 | tee "$output_dir/train.log"
    echo "[TRAIN DONE] seed=$seed"
done

"$PYTHON_BIN" -m Scintillator_Project.src.scripts.summarize_hybrid_residual \
    "$RESULT_ROOT" 2>&1 | tee "$RESULT_ROOT/summary.log"

echo "SCINTILLATOR HYBRID RESIDUAL STUDY: COMPLETE"
