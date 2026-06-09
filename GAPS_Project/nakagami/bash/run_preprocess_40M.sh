#!/bin/bash
# 40Mデータ全量前処理（並列処理）
# 実行方法: nohup bash nakagami/bash/run_preprocess_40M.sh &

CSV_DIR=/mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles
OUT_DIR=/mnt/ynakagami3/nakagami_data/data_40M
SCRIPT=~/HEP_Project/GAPS_Project/nakagami/data_parse/preprocess_40M.py
LOG=~/HEP_Project/GAPS_Project/nakagami/bash/preprocess_40M.log

mkdir -p $OUT_DIR

echo "Start: $(date)" | tee $LOG
echo "CSV dir : $CSV_DIR" | tee -a $LOG
echo "Out dir : $OUT_DIR" | tee -a $LOG
echo "Workers : 12 (of 20 cores)" | tee -a $LOG

python $SCRIPT \
    --csv_dir    $CSV_DIR \
    --out_dir    $OUT_DIR \
    --train_ratio 0.8 \
    --num_workers 12 \
    2>&1 | tee -a $LOG

echo "End: $(date)" | tee -a $LOG
