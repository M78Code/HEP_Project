"""
preprocess_40M.py
将40M CSV数据转换为npz格式，供CNNDNNHybrid（A.2 7.1.1架构）训练使用。

列结构（1631列，0-indexed）：
  Col 0       : randomSeed
  Col 1       : eventNumber
  Col 2       : label（1=dbar, 0=pbar）
  Col 3~1442  : Si edep（1440值 → 10×12×12）
  Col 1619~1629: 11维TOF聚合特征

用法：
  python preprocess_40M.py \
    --csv_dir /mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles \
    --out_dir ./data_40M \
    --train_ratio 0.8
"""
import argparse
import csv
import pathlib
import random
import numpy as np
from tqdm import tqdm

SI_START  = 3
SI_END    = 1443    # 1440値 → reshape(10,12,12)
TOF_START = 1619    # 11次元TOF特徴量
TOF_END   = 1630
LABEL_COL = 2


def parse_row(row):
    label = int(float(row[LABEL_COL]))

    si = np.zeros(1440, dtype=np.float32)
    for i in range(SI_START, SI_END):
        v = row[i]
        si[i - SI_START] = float(v) if v else 0.0

    tof = np.array([float(row[i]) for i in range(TOF_START, TOF_END)],
                   dtype=np.float32)
    return label, si.reshape(10, 12, 12), tof


def process_files(file_list, desc):
    si_list, tof_list, labels = [], [], []
    for path in tqdm(file_list, desc=desc):
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < TOF_END:
                    continue
                label, si, tof = parse_row(row)
                labels.append(label)
                si_list.append(si)
                tof_list.append(tof)
    return si_list, tof_list, labels


def save_npz(out_path, si_list, tof_list, labels):
    np.savez_compressed(
        out_path,
        si    = np.stack(si_list),    # (N, 10, 12, 12)
        tof   = np.stack(tof_list),   # (N, 11)
        label = np.array(labels, dtype=np.int8),
    )
    print(f"  saved → {out_path}  ({len(labels):,} events)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir",     required=True,
                        help="40MデータのcsvFilesディレクトリ")
    parser.add_argument("--out_dir",     required=True,
                        help="npz出力先ディレクトリ")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="訓練データの割合（デフォルト0.8）")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    file_list = sorted(pathlib.Path(args.csv_dir).glob("CNN*.csv"))
    print(f"Found {len(file_list)} CSV files")

    random.shuffle(file_list)
    split = int(len(file_list) * args.train_ratio)
    train_files = file_list[:split]
    val_files   = file_list[split:]
    print(f"Train: {len(train_files)} files, Val: {len(val_files)} files")

    print("=== Processing train ===")
    si_list, tof_list, labels = process_files(train_files, "train")
    save_npz(out / "train_40M.npz", si_list, tof_list, labels)

    print("=== Processing val ===")
    si_list, tof_list, labels = process_files(val_files, "val")
    save_npz(out / "val_40M.npz", si_list, tof_list, labels)

    print("Done.")
