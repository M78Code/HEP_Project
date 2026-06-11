#!/bin/bash
# 中上 4M CSV (csvFiles_Digitized) 前処理
# 中上自身の train_5cross / valid_5cross 划分をそのまま使用
# 実行: bash nakagami/bash/run_preprocess_4M.sh

CSV_ROOT=/mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Digitized
# 中上の IdentifywithNN.py で使われている正式パス (shuffled/)
# Train: 400 files × 4000 rows = 1,600,000 events (Dbar/Pbar 混合)
# Val  : 100 files × 4000 rows = 400,000 events
TRAIN_DIR=$CSV_ROOT/shuffled/train_5cross
VAL_DIR=$CSV_ROOT/shuffled/valid_5cross
OUT_DIR=/mnt/ynakagami3/nakagami_data/data_4M
SCRIPT=~/HEP_Project/GAPS_Project/nakagami/data_parse/preprocess_4M.py
LOG=~/HEP_Project/GAPS_Project/nakagami/bash/preprocess_4M.log

mkdir -p $OUT_DIR

echo "Start: $(date)" | tee $LOG
echo "Train dir: $TRAIN_DIR" | tee -a $LOG
echo "Val   dir: $VAL_DIR"   | tee -a $LOG
echo "Out   dir: $OUT_DIR"   | tee -a $LOG
echo "Workers  : 12 (of 20 cores)" | tee -a $LOG

python $SCRIPT \
    --train_dir   $TRAIN_DIR \
    --val_dir     $VAL_DIR \
    --out_dir     $OUT_DIR \
    --num_workers 12 \
    2>&1 | tee -a $LOG

echo "End: $(date)" | tee -a $LOG
