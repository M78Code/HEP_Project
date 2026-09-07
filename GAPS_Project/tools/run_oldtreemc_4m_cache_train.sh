#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-all}
if [[ "$PHASE" != "cache" && "$PHASE" != "train" && \
      "$PHASE" != "all" ]]; then
    echo "usage: $0 [cache|train|all]" >&2
    exit 2
fi

PROJECT=${PROJECT:-$HOME/HEP_Project/GAPS_Project}
DIGITIZED=${DIGITIZED:-/mnt/aohba/oldtreemc_source_disjoint_4m_digitized}
CACHE=${CACHE:-/mnt/aohba/oldtreemc_source_disjoint_4m_global_log}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT/results}
DATASET_TAG=${DATASET_TAG:-oldtreemc_source_disjoint_4m_global_log_seed${SEED}}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
cd "$PROJECT"

run_cache()
{
    if [[ -f "$CACHE/cache_manifest.json" ]]; then
        echo "[SKIP] existing cache manifest: $CACHE/cache_manifest.json"
        return
    fi
    if [[ -e "$CACHE" ]]; then
        echo "ERROR: cache directory exists without cache_manifest.json: $CACHE" >&2
        exit 1
    fi
    [[ -f "$DIGITIZED/digitization_manifest.json" ]] || {
        echo "ERROR: missing 4M digitization manifest: $DIGITIZED" >&2
        exit 1
    }

    python src/data_parse/build_oldtreemc_source_disjoint_treerec_cache.py \
        --pilot-reco-dir "$DIGITIZED" \
        --test-reco-dir "$DIGITIZED" \
        --output-dir "$CACHE" \
        --expected-train 3200000 \
        --expected-val 400000 \
        --expected-test 400000 \
        --test-antip-source 1627528714 \
        --test-antid-source 1627550286
}

run_train()
{
    [[ -f "$CACHE/cache_manifest.json" ]] || {
        echo "ERROR: missing 4M cache manifest: $CACHE" >&2
        exit 1
    }

    local train_log="$HOME/train_oldtreemc_source_disjoint_4m_global_log_seed${SEED}.log"
    env CUDA_VISIBLE_DEVICES="$GPU" python src/scripts/train_aohba.py \
        --split-cache-dir "$CACHE" \
        --epochs 80 \
        --model gravnet \
        --batch-size 512 \
        --num-workers 2 \
        --non-blocking-transfer \
        --seed "$SEED" \
        --dataset-tag "$DATASET_TAG" \
        --result-dir "$RESULT_ROOT" \
        2>&1 | tee "$train_log"

    local model
    model=$(find "$RESULT_ROOT" -maxdepth 2 -type f \
        -name "*${DATASET_TAG}*_best.pth" -print | sort | tail -1)
    if [[ -z "$model" ]]; then
        echo "ERROR: best checkpoint not found for $DATASET_TAG" >&2
        exit 1
    fi
    local eval_dir="$(dirname "$model")/evaluation_test"
    env CUDA_VISIBLE_DEVICES="$GPU" python \
        src/scripts/evaluate_aohba_split_cache.py \
        --cache-dir "$CACHE" \
        --model-path "$model" \
        --output-dir "$eval_dir" \
        --model gravnet \
        --batch-size 512 \
        --seed "$SEED" \
        2>&1 | tee "$HOME/evaluate_oldtreemc_source_disjoint_4m_global_log_seed${SEED}.log"

    echo "model     : $model"
    echo "evaluation: $eval_dir"
}

echo "phase      : $PHASE"
echo "digitized  : $DIGITIZED"
echo "cache      : $CACHE"
echo "GPU        : $GPU"
echo "train seed : $SEED"
df -h /mnt/aohba

case "$PHASE" in
    cache)
        run_cache
        ;;
    train)
        run_train
        ;;
    all)
        run_cache
        run_train
        ;;
esac

echo "OLD TREEMC SOURCE-DISJOINT 4M $PHASE: COMPLETE"
