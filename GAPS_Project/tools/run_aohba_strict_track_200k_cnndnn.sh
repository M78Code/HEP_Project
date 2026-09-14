#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail

PHASE=${1:-all}
case "$PHASE" in
    dataset|train|compare|all) ;;
    *)
        echo "usage: $0 [dataset|train|compare|all]" >&2
        exit 2
        ;;
esac

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
GRAPH_CACHE=${GRAPH_CACHE:-/mnt/aohba/archive_20260913_after_submission_mc1/aohba_stopping_ablation_strict_track_200k_global_log}
DATASET=${DATASET:-/mnt/aohba/aohba_2tof_strict_track_stop_top_trigger_200k_treerec_cnndnn_10x12x12}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT/results/aohba_strict_track_200k_cnndnn"}
GRAVNET_ROOT=${GRAVNET_ROOT:-"$PROJECT/results/aohba_stopping_summary_only_ablation_200k/strict_track"}
COMPARE_DIR=${COMPARE_DIR:-"$RESULT_ROOT/comparison_gravnet_vs_cnndnn_high_efficiency"}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
SPLIT_SUFFIX=cnndnn_10x12x12
TAG="aohba_strict_track_stop_treerec_200k_cnndnn_seed${SEED}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka
cd "$PROJECT"

latest_run()
{
    local root=$1
    local pattern=$2
    find "$root" -mindepth 1 -maxdepth 1 -type d \
        -name "$pattern" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-
}

run_dataset()
{
    test -f "$GRAPH_CACHE/_SUCCESS" || {
        echo "ERROR: strict-track 200K graph cache is missing: $GRAPH_CACHE" >&2
        exit 1
    }
    if test -f "$DATASET/_SUCCESS" -a -f "$DATASET/dataset_manifest.json"; then
        echo "[SKIP] CNN+DNN dataset already complete: $DATASET"
        return
    fi
    test ! -e "$DATASET" || {
        echo "ERROR: incomplete output already exists: $DATASET" >&2
        exit 1
    }

    python -u src/data_parse/export_graph_cache_cnndnn_fixedgrid.py \
        --cache-dir "$GRAPH_CACHE" \
        --output-dir "$DATASET" \
        --split-suffix "$SPLIT_SUFFIX" \
        --grid-x 12 \
        --grid-y 12
}

run_train()
{
    test -f "$DATASET/_SUCCESS" -a -f "$DATASET/dataset_manifest.json" || {
        echo "ERROR: CNN+DNN dataset is incomplete: $DATASET" >&2
        exit 1
    }
    mkdir -p "$RESULT_ROOT"

    local run_dir resume_args=()
    run_dir=$(latest_run "$RESULT_ROOT" "*_CNNDNNFig72_${TAG}" || true)
    if test -n "$run_dir" -a -f "$run_dir/evaluation_test/metrics.json"; then
        echo "[SKIP] CNN+DNN training and evaluation already complete: $run_dir"
        return
    fi
    if test -n "$run_dir" -a -f "$run_dir/last.pt"; then
        echo "[RESUME] $run_dir/last.pt"
        resume_args=(--resume "$run_dir/last.pt")
    fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        python -u src/scripts/train_nakagami_cnndnn_fig72_amp.py \
        --data-dir "$DATASET" \
        --split-suffix "$SPLIT_SUFFIX" \
        --result-dir "$RESULT_ROOT" \
        --dataset-tag "$TAG" \
        --epochs 80 \
        --batch-size 200 \
        --lr 4e-5 \
        --num-workers 2 \
        --amp \
        --min-epochs 20 \
        --patience 10 \
        --seed "$SEED" \
        "${resume_args[@]}"
}

run_compare()
{
    local cnn_run cnn_eval gravnet_run gravnet_eval
    cnn_run=$(latest_run "$RESULT_ROOT" "*_CNNDNNFig72_${TAG}" || true)
    test -n "$cnn_run" || {
        echo "ERROR: CNN+DNN run not found in $RESULT_ROOT" >&2
        exit 1
    }
    cnn_eval="$cnn_run/evaluation_test"
    test -f "$cnn_eval/metrics.json" || {
        echo "ERROR: CNN+DNN evaluation is incomplete: $cnn_eval" >&2
        exit 1
    }

    gravnet_run=$(latest_run \
        "$GRAVNET_ROOT" \
        "*_GravNet_6b_h128_aohba_stopping_ablation_strict_track_200k_global_log_seed${SEED}" \
        || true)
    test -n "$gravnet_run" || {
        echo "ERROR: strict-track 200K GravNet run not found in $GRAVNET_ROOT" >&2
        exit 1
    }
    gravnet_eval="$gravnet_run/evaluation_test"
    test -f "$gravnet_eval/metrics.json" || {
        echo "ERROR: GravNet evaluation is incomplete: $gravnet_eval" >&2
        exit 1
    }

    python -u src/scripts/visual/compare_binary_eval.py \
        --item "GravNet, TreeRec graph" "$gravnet_eval" \
        --item "CNN+DNN, 10x12x12 + TOF" "$cnn_eval" \
        --out-dir "$COMPARE_DIR" \
        --x-min 0.90 \
        --y-max 20000 \
        --mark-efficiencies 0.90 0.95 0.98 0.99 1.00

    echo "CNN+DNN evaluation: $cnn_eval"
    echo "GravNet evaluation: $gravnet_eval"
    echo "comparison: $COMPARE_DIR/rejection_compare.png"
}

echo "phase          : $PHASE"
echo "selection      : strict track stop + truth top-trigger"
echo "source         : TreeRec graph cache (same 200K events and splits)"
echo "CNN input      : 10 x 12 x 12 Si(Li) hit-energy grid"
echo "DNN input      : 11-D TreeRec TOF features"
echo "beta input     : disabled"
echo "physical GPU   : $GPU"
echo "seed           : $SEED"
echo "dataset        : $DATASET"

case "$PHASE" in
    dataset) run_dataset ;;
    train) run_train ;;
    compare) run_compare ;;
    all)
        run_dataset
        run_train
        run_compare
        ;;
esac

echo "STRICT-TRACK TREERec 200K CNN+DNN $PHASE: COMPLETE"
