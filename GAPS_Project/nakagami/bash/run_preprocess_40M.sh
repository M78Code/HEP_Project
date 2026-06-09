#!/bin/bash
# 40Mデータの前処理スクリプト
# 実行方法: bash nakagami/bash/run_preprocess_40M.sh

CSV_DIR=/mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles
# OUT_DIR=~/HEP_Project/GAPS_Project/nakagami/data_40M  # test directory
OUT_DIR=/mnt/ynakagami3/nakagami_data/data_40M # real directory
SCRIPT=~/HEP_Project/GAPS_Project/nakagami/data_parse/preprocess_40M.py
LOG=~/HEP_Project/GAPS_Project/nakagami/bash/preprocess_40M.log

mkdir -p $OUT_DIR

echo "Start: $(date)" | tee $LOG

python $SCRIPT \
    --csv_dir $CSV_DIR \
    --out_dir $OUT_DIR \
    --train_ratio 0.8 \
    2>&1 | tee -a $LOG

echo "End: $(date)" | tee -a $LOG
