#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=${PROJECT:-"$HOME/HEP_Project/GAPS_Project"}
DATASET=${DATASET:-/mnt/aohba/aohba_treemc_fixedgrid_direct_200k}
GPU=${GPU:-0}
SEED=${SEED:-20260825}
TAG="aohba_treemc_fixedgrid_direct_200k_gravnet_seed${SEED}"

cd "$PROJECT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate naka

test -f "$DATASET/_SUCCESS" || {
    echo "ERROR: dataset is incomplete: $DATASET" >&2
    exit 1
}

echo "dataset : $DATASET"
echo "GPU     : $GPU"
echo "seed    : $SEED"
echo "beta input: disabled"
echo "TOF input : 172 zero paddle channels + 11 primary features"

before=$(mktemp)
find results -maxdepth 1 -type d -name "*_SparseVoxelGNN_${TAG}" -print \
    | sort >"$before"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
python -u src/scripts/train_aohba_sparse_voxel_gnn.py \
    --data-dir "$DATASET" \
    --dataset-tag "$TAG" \
    --model gravnet \
    --epochs 200 \
    --batch-size 512 \
    --num-workers 4 \
    --hidden 128 \
    --num-blocks 6 \
    --k 8 \
    --dropout 0.15 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --seed "$SEED" \
    --tof-mode paddles-primary \
    --early-stopping-patience 20 \
    --min-epochs 20 \
    --early-stopping-min-delta 1e-5

after=$(mktemp)
find results -maxdepth 1 -type d -name "*_SparseVoxelGNN_${TAG}" -print \
    | sort >"$after"
result=$(comm -13 "$before" "$after" | tail -1)
rm -f "$before" "$after"

if test -z "$result" || ! test -f "$result/best.pt"; then
    echo "ERROR: cannot identify the completed training result" >&2
    exit 1
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
python -u src/scripts/evaluate_sparse_voxel_gnn.py \
    --data-dir "$DATASET" \
    --model-path "$result/best.pt" \
    --output-dir "$result/evaluation_test" \
    --split test \
    --batch-size 512 \
    --num-workers 4

echo "result: $result"
echo "AOHBA TREEMC FIXED-GRID DIRECT 200K TRAINING: COMPLETE"
