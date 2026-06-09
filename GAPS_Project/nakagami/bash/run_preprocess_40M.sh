#!/bin/bash
# 中上40M CSV 前処理 - 平衡データ
# 目的: Dbar 20 + Pbar 20 ファイルでバランスのとれた訓練データを生成
# 実行: nohup bash nakagami/bash/run_preprocess_40M.sh &

OUT_DIR=/mnt/ynakagami3/nakagami_data/data_40M
SCRIPT=~/HEP_Project/GAPS_Project/nakagami/data_parse/preprocess_40M.py
LOG=~/HEP_Project/GAPS_Project/nakagami/bash/preprocess_40M.log

DBAR_DIR=/mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles
PBAR_DIR=/mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Mc

mkdir -p $OUT_DIR

echo "Start: $(date)" | tee $LOG
echo "Dbar dir: $DBAR_DIR" | tee -a $LOG
echo "Pbar dir: $PBAR_DIR" | tee -a $LOG
echo "Out dir : $OUT_DIR"  | tee -a $LOG
echo "max_files_per_class: 20" | tee -a $LOG
echo "Workers : 12 (of 20 cores)" | tee -a $LOG

python $SCRIPT \
    --csv_dirs $DBAR_DIR $PBAR_DIR \
    --out_dir  $OUT_DIR \
    --max_files_per_class 20 \
    --train_ratio 0.8 \
    --num_workers 12 \
    2>&1 | tee -a $LOG

echo "End: $(date)" | tee -a $LOG
