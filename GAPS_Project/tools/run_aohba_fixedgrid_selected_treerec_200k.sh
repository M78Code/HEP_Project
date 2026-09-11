#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-all}
if [[ "$PHASE" != "cache" && "$PHASE" != "train" && "$PHASE" != "all" ]]; then
    echo "usage: $0 [cache|train|all]" >&2
    exit 2
fi

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
RAW=${RAW:-/mnt/aohba/aohba_treemc_fixedgrid_direct_200k_raw}
FIXEDGRID=${FIXEDGRID:-/mnt/aohba/aohba_treemc_fixedgrid_direct_200k}
CACHE=${CACHE:-/mnt/aohba/aohba_fixedgrid_selected_treerec_200k_global_log}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_fixedgrid_selected_treerec_200k"}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
TAG="aohba_fixedgrid_selected_treerec_200k_global_log_seed${SEED}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
cd "$PROJECT"

run_cache()
{
    if [[ -f "$CACHE/_SUCCESS" && -f "$CACHE/cache_manifest.json" ]]; then
        echo "[SKIP] paired cache already complete: $CACHE"
        return
    fi
    if [[ -e "$CACHE" ]]; then
        echo "ERROR: incomplete cache exists: $CACHE" >&2
        exit 1
    fi
    python -u src/data_parse/build_aohba_fixedgrid_selected_treerec_cache.py \
        --fixedgrid-raw-dir "$RAW" \
        --fixedgrid-dataset-dir "$FIXEDGRID" \
        --output-dir "$CACHE" \
        --chunk-size 10000 \
        --k 8
}

latest_run_dir()
{
    find "$RESULT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name "*_${TAG}" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-
}

run_train()
{
    [[ -f "$CACHE/_SUCCESS" ]] || {
        echo "ERROR: paired cache is incomplete: $CACHE" >&2
        exit 1
    }
    mkdir -p "$RESULT_ROOT"

    local run_dir model checkpoint
    run_dir=$(latest_run_dir || true)
    if [[ -n "$run_dir" && -f "$run_dir/evaluation_test/metrics.json" ]]; then
        echo "[SKIP] training and evaluation already complete: $run_dir"
        return
    fi

    checkpoint=""
    if [[ -n "$run_dir" ]]; then
        checkpoint=$(find "$run_dir" -maxdepth 1 -type f \
            -name '*_last_checkpoint.pth' -print | sort | tail -1)
    fi

    train_args=(
        --split-cache-dir "$CACHE"
        --epochs 80
        --model gravnet
        --batch-size 512
        --num-workers 2
        --non-blocking-transfer
        --seed "$SEED"
        --dataset-tag "$TAG"
        --result-dir "$RESULT_ROOT"
    )
    if [[ -n "$checkpoint" ]]; then
        echo "[RESUME] $checkpoint"
        train_args+=(--resume-checkpoint "$checkpoint")
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/train_aohba.py "${train_args[@]}"

    run_dir=$(latest_run_dir)
    model=$(find "$run_dir" -maxdepth 1 -type f \
        -name '*_best.pth' -print | sort | tail -1)
    [[ -n "$model" ]] || {
        echo "ERROR: best checkpoint not found" >&2
        exit 1
    }

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/evaluate_aohba_split_cache.py \
        --cache-dir "$CACHE" \
        --model-path "$model" \
        --output-dir "$run_dir/evaluation_test" \
        --model gravnet \
        --batch-size 512 \
        --seed "$SEED"

    echo "model: $model"
    echo "evaluation: $run_dir/evaluation_test"
}

echo "phase      : $PHASE"
echo "raw export : $RAW"
echo "fixed-grid : $FIXEDGRID"
echo "cache      : $CACHE"
echo "physical GPU: $GPU"
echo "seed       : $SEED"

case "$PHASE" in
    cache) run_cache ;;
    train) run_train ;;
    all) run_cache; run_train ;;
esac

echo "AOHBA FIXED-GRID-SELECTED TREEREC 200K $PHASE: COMPLETE"
