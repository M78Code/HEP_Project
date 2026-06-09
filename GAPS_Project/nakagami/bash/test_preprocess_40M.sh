#!/bin/bash
# 40Mデータ前処理テスト（少量ファイルで動作確認）
# 実行方法: bash nakagami/bash/test_preprocess_40M.sh

CSV_DIR=/mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles
OUT_DIR=~/HEP_Project/GAPS_Project/nakagami/data_40M
SCRIPT=~/HEP_Project/GAPS_Project/nakagami/data_parse/preprocess_40M.py
LOG=~/HEP_Project/GAPS_Project/nakagami/bash/test_preprocess_40M.log
TEST_DIR=/tmp/csv_test

mkdir -p $OUT_DIR $TEST_DIR

# CSVファイルを10個だけtmpにコピーしてテスト
ls $CSV_DIR/CNN*.csv | head -10 | xargs -I{} cp {} $TEST_DIR/

echo "Start: $(date)" | tee $LOG
echo "Test files:" | tee -a $LOG
ls $TEST_DIR/ | tee -a $LOG

python $SCRIPT \
    --csv_dir $TEST_DIR \
    --out_dir $OUT_DIR \
    --train_ratio 0.8 \
    2>&1 | tee -a $LOG

# 結果確認
python -c "
import numpy as np
import os
for f in ['train_40M.npz', 'val_40M.npz']:
    path = os.path.expanduser('~/HEP_Project/GAPS_Project/nakagami/data_40M/') + f
    if not os.path.exists(path):
        print(f'{f}: not found')
        continue
    d = np.load(path)
    print(f'{f}:')
    print(f'  si shape : {d[\"si\"].shape}')
    print(f'  tof shape: {d[\"tof\"].shape}')
    print(f'  label    : {d[\"label\"].shape}, unique={np.unique(d[\"label\"])}')
    print(f'  tof[0]   : {d[\"tof\"][0]}')
" 2>&1 | tee -a $LOG

# tmpクリーンアップ
rm -rf $TEST_DIR

echo "End: $(date)" | tee -a $LOG
