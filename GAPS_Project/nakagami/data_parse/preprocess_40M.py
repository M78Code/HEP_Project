"""
preprocess_40M.py
中上40M CSV（1631列 Mc CSV）を読み込み、HybridDatasetFast（A.2 7.1.1架构）が
直接読めるnpz形式に変換する。

データソース（複数ディレクトリ対応）:
  Dbar (signal,    label=1): /mnt/ynakagami3/.../211209_*_40M/csvFiles
  Pbar (background, label=0): /mnt/ynakagami3/.../220104_4Mevents_isot_loose/csvFiles_Mc

列構造（1631列, 0-indexed）:
  Col 0       : randomSeed（ROOT文件名时间戳）
  Col 1       : ROOT entry index（注意：不是eventNumber_）
  Col 2       : label（1=dbar, 0=pbar）
  Col 4       : primaryBeta
  Col 3~1442  : Si edep（1440値 → 10×12×12）
  Col 1620~1630: 11次元TOF特徴量
      [outer_first_energy,
       inner_first_energy,
       inner_first_time - outer_first_time,
       inner_first_x, inner_first_y, inner_first_z,
       outer_first_x, outer_first_y, outer_first_z,
       stopping_x, stopping_y]

注意事項:
  - VolIDファイルは列構造が異なるため除外。
  - len(row) != 1631 の行は除外。
  - 出力npzのkey名はHybridDatasetFastに合わせる: voxels / tofs / labels。

用法:
  python preprocess_40M.py \
    --csv_dirs \
      /mnt/ynakagami3/SimulationData/211209_isot_0205_renewal_looseTrigger_40M/csvFiles \
      /mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Mc \
    --out_dir /mnt/ynakagami3/nakagami_data/data_40M \
    --max_files_per_class 20 \
    --train_ratio 0.8
"""
import argparse
import csv
import pathlib
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

SI_START  = 3
SI_END    = 1443    # 1440値 → reshape(10,12,12)
TOF_START = 1620    # 11次元TOF特徴量
TOF_END   = 1631
LABEL_COL = 2
N_COLS    = 1631    # 中上1631列Mc CSVの厳密な列数


def parse_row(row):
    label = int(float(row[LABEL_COL]))

    si = np.zeros(1440, dtype=np.float32)
    for i in range(SI_START, SI_END):
        v = row[i]
        si[i - SI_START] = float(v) if v else 0.0

    tof = np.array([float(row[i]) for i in range(TOF_START, TOF_END)],
                   dtype=np.float32)
    return label, si.reshape(10, 12, 12), tof


def process_one_file(path):
    si_list, tof_list, labels = [], [], []
    skipped = 0
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != N_COLS:    # 厳密に1631列のみ
                skipped += 1
                continue
            label, si, tof = parse_row(row)
            labels.append(label)
            si_list.append(si)
            tof_list.append(tof)
    return labels, si_list, tof_list, skipped


def process_files(file_list, desc, num_workers=12):
    import time
    all_si, all_tof, all_labels = [], [], []
    total_skipped = 0
    t0 = time.time()
    with Pool(processes=num_workers) as pool:
        for i, (labels, si_list, tof_list, skipped) in enumerate(tqdm(
            pool.imap(process_one_file, file_list),
            total=len(file_list), desc=desc,
            dynamic_ncols=True
        )):
            all_labels.extend(labels)
            all_si.extend(si_list)
            all_tof.extend(tof_list)
            total_skipped += skipped
            elapsed = time.time() - t0
            done = i + 1
            remain = len(file_list) - done
            eta = elapsed / done * remain
            print(f"  [{desc}] {done}/{len(file_list)} files | "
                  f"{len(all_labels):,} events (skipped {total_skipped}) | "
                  f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m",
                  flush=True)
    return all_si, all_tof, all_labels


def save_npz(out_path, si_list, tof_list, labels):
    # HybridDatasetFastが期待するkey名: voxels / tofs / labels
    np.savez_compressed(
        out_path,
        voxels = np.stack(si_list),                  # (N, 10, 12, 12) float32
        tofs   = np.stack(tof_list),                 # (N, 11) float32
        labels = np.array(labels, dtype=np.int64),   # (N,) int64
    )
    n_dbar = int(np.sum(np.array(labels) == 1))
    n_pbar = int(np.sum(np.array(labels) == 0))
    print(f"  saved → {out_path}  ({len(labels):,} events, "
          f"Dbar={n_dbar:,}, Pbar={n_pbar:,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dirs",          nargs="+", required=True,
                        help="複数のCSVディレクトリ（Dbar用とPbar用）")
    parser.add_argument("--out_dir",           required=True,
                        help="npz出力先ディレクトリ")
    parser.add_argument("--train_ratio",       type=float, default=0.8)
    parser.add_argument("--max_files_per_class", type=int, default=None,
                        help="各粒子種ごとの最大ファイル数（テスト用）")
    parser.add_argument("--num_workers",       type=int,   default=12)
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 複数ディレクトリから全CNN*.csvを収集（VolID除外）
    all_files = []
    for d in args.csv_dirs:
        d = pathlib.Path(d)
        files = sorted(
            p for p in d.glob("CNN*.csv")
            if "VolID" not in p.name
        )
        print(f"  {d}: {len(files)} non-VolID files")
        all_files.extend(files)
    print(f"Total CSV files: {len(all_files)}")

    # 粒子種ごとに分層
    dbar_files = [f for f in all_files if 'Dbar' in f.name]
    pbar_files = [f for f in all_files if 'Pbar' in f.name]
    print(f"  Dbar files: {len(dbar_files)}, Pbar files: {len(pbar_files)}")

    # テスト用に各クラスのファイル数を制限
    if args.max_files_per_class is not None:
        random.shuffle(dbar_files)
        random.shuffle(pbar_files)
        dbar_files = dbar_files[:args.max_files_per_class]
        pbar_files = pbar_files[:args.max_files_per_class]
        print(f"  After max_files_per_class={args.max_files_per_class}: "
              f"Dbar={len(dbar_files)}, Pbar={len(pbar_files)}")

    random.shuffle(dbar_files)
    random.shuffle(pbar_files)

    def split_files(files, ratio):
        n = int(len(files) * ratio)
        return files[:n], files[n:]

    dbar_train, dbar_val = split_files(dbar_files, args.train_ratio)
    pbar_train, pbar_val = split_files(pbar_files, args.train_ratio)

    train_files = dbar_train + pbar_train
    val_files   = dbar_val   + pbar_val
    random.shuffle(train_files)
    random.shuffle(val_files)
    print(f"Train: {len(train_files)} files (Dbar:{len(dbar_train)}, Pbar:{len(pbar_train)})")
    print(f"Val  : {len(val_files)} files (Dbar:{len(dbar_val)}, Pbar:{len(pbar_val)})")

    print("=== Processing train ===")
    si_list, tof_list, labels = process_files(train_files, "train", args.num_workers)
    save_npz(out / "train_hybrid_nakagami40M.npz", si_list, tof_list, labels)

    print("=== Processing val ===")
    si_list, tof_list, labels = process_files(val_files, "val", args.num_workers)
    save_npz(out / "val_hybrid_nakagami40M.npz", si_list, tof_list, labels)

    print("Done.")
