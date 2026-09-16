#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-}
case "$PHASE" in
    train-score) ;;
    *) echo "usage: $0 train-score" >&2; exit 2 ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
PROVENANCE=${PROVENANCE:-/mnt/aohba/aohba_treerec_hit_proxy_calibration_candidates_300k}
# The candidate provenance contains four independent source ROOT files.
# Keep entire files together; four folds are the maximum valid OOF partition.
CACHE_ROOT=${CACHE_ROOT:-/mnt/aohba/aohba_treerec_truth_stop_oof_4fold}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_treerec_truth_stop_oof_4fold"}
SCORE_DIR=${SCORE_DIR:-"$RESULT_ROOT/oof_scores"}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
FOLDS=${FOLDS:-4}
TRAIN_PER_CELL=${TRAIN_PER_CELL:-10000}
VAL_PER_CELL=${VAL_PER_CELL:-2000}
TEST_PER_CELL=${TEST_PER_CELL:-3000}

cd "$PROJECT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka

latest_run_dir() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -name "*_${2}" -print 2>/dev/null | sort | tail -1
}

for ((fold=0; fold<FOLDS; fold++)); do
    CACHE="$CACHE_ROOT/fold_${fold}_global_log"
    FOLD_ROOT="$RESULT_ROOT/fold_${fold}"
    TAG="aohba_treerec_truth_stop_oof_fold${fold}_seed${SEED}"
    SCORE="$SCORE_DIR/fold_${fold}.npz"
    if [[ -f "$CACHE/_SUCCESS" && -f "$CACHE/cache_manifest.json" ]]; then
        echo "[SKIP] fold $fold cache: $CACHE"
    elif [[ -e "$CACHE" ]]; then
        echo "ERROR: incomplete fold $fold cache: $CACHE" >&2
        exit 1
    else
        echo "[CACHE START] fold $fold/$((FOLDS - 1))"
        python -u src/data_parse/build_aohba_treerec_truth_stop_oof_fold_cache.py \
            --provenance-dir "$PROVENANCE" --output-dir "$CACHE" \
            --folds "$FOLDS" --fold-index "$fold" --seed "$SEED" --k 8 \
            --train-events-per-cell "$TRAIN_PER_CELL" \
            --val-events-per-cell "$VAL_PER_CELL" \
            --test-events-per-cell "$TEST_PER_CELL"
    fi
    mkdir -p "$FOLD_ROOT"
    RUN_DIR=$(latest_run_dir "$FOLD_ROOT" "$TAG" || true)
    if [[ -z "$RUN_DIR" || ! -f "$RUN_DIR/evaluation_test/metrics.json" ]]; then
        CHECKPOINT=""
        if [[ -n "$RUN_DIR" ]]; then
            CHECKPOINT=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_last_checkpoint.pth' -print | sort | tail -1)
        fi
        ARGS=(--split-cache-dir "$CACHE" --epochs 80 --model gravnet --batch-size 512
              --num-workers 2 --non-blocking-transfer --seed "$SEED"
              --dataset-tag "$TAG" --result-dir "$FOLD_ROOT")
        [[ -z "$CHECKPOINT" ]] || ARGS+=(--resume-checkpoint "$CHECKPOINT")
        echo "[TRAIN START] fold $fold on physical GPU $GPU (serial)"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
            python -u src/scripts/train_aohba.py "${ARGS[@]}"
        RUN_DIR=$(latest_run_dir "$FOLD_ROOT" "$TAG")
        MODEL=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_best.pth' -print | sort | tail -1)
        [[ -n "$MODEL" ]] || { echo "ERROR: fold $fold best checkpoint missing" >&2; exit 1; }
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
            python -u src/scripts/evaluate_aohba_split_cache.py \
                --cache-dir "$CACHE" --model-path "$MODEL" \
                --output-dir "$RUN_DIR/evaluation_test" --model gravnet --batch-size 512 --seed "$SEED"
    else
        echo "[SKIP] fold $fold training/evaluation: $RUN_DIR"
    fi
    if [[ ! -f "$SCORE" ]]; then
        MODEL=$(find "$RUN_DIR" -maxdepth 1 -type f -name '*_best.pth' -print | sort | tail -1)
        [[ -n "$MODEL" ]] || { echo "ERROR: fold $fold model missing for OOF scoring" >&2; exit 1; }
        echo "[SCORE START] fold $fold unseen source ROOT events on physical GPU $GPU"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
            python -u src/data_parse/score_aohba_treerec_truth_stop_oof_fold.py \
                --provenance-dir "$PROVENANCE" --folds "$FOLDS" --fold-index "$fold" \
                --model-path "$MODEL" --normalizer "$CACHE/node_feature_normalizer.json" \
                --output "$SCORE" --seed "$SEED"
    else
        echo "[SKIP] fold $fold OOF score: $SCORE"
    fi
done

python -u src/data_parse/summarize_aohba_treerec_truth_stop_oof.py \
    --provenance-dir "$PROVENANCE" --score-dir "$SCORE_DIR" --folds "$FOLDS" \
    --output "$RESULT_ROOT/oof_summary.json"
echo "AOHBA TREEREC TRUTH-STOP OOF ${FOLDS}-FOLD: COMPLETE"
