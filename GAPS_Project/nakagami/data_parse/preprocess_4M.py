"""
preprocess_4M.py
中上 4M CSV (csvFiles_Digitized) を読み込み、HybridDatasetFast (4M用) が
直接読める npz 形式に変換する。

データソース（中上 IdentifywithNN.py が使う shuffled/ ディレクトリ）:
  Train: csvFiles_Digitized/shuffled/train_5cross/
         400 files × 4000 rows (Dbar/Pbar 混合済み) = 1,600,000 events
  Val:   csvFiles_Digitized/shuffled/valid_5cross/
         100 files × 4000 rows                     =   400,000 events
  ★ 中上コードの numoftraining_data=1600000、numofvalidation_data=400000 と完全一致

列構造（1452列, 0-indexed）:
  Col 0       : randomSeed (ROOT 文件名时间戳)
  Col 1       : event index
  Col 2       : label (1=dbar, 0=pbar)
  Col 3~1442  : Si edep (1440値 → 10×12×12)
  Col 1443~1451: 9次元 TOF (各パドルのエネルギー損失)

注意:
  - 1452列以外の行はスキップ
  - 出力 npz の key名は HybridDatasetFast に合わせる: voxels / tofs / labels
  - 4M dataset は中上自身が train/val 划分済みなので、再划分しない

用法:
  python preprocess_4M.py \
    --train_dir /mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Digitized/shuffled/train_5cross \
    --val_dir   /mnt/ynakagami3/SimulationData/220104_4Mevents_isot_loose/csvFiles_Digitized/shuffled/valid_5cross \
    --out_dir   /mnt/ynakagami3/nakagami_data/data_4M \
    --num_workers 12
"""
import argparse
import csv
import pathlib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

SI_START  = 3
SI_END    = 1443    # 1440値 → reshape(10,12,12)
TOF_START = 1443    # 9次元 TOF
TOF_END   = 1452
LABEL_COL = 2
N_COLS    = 1452    # 1452列の厳密な列数


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
            if len(row) != N_COLS:
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
            total=len(file_list), desc=desc, dynamic_ncols=True
        )):
            all_labels.extend(labels)
            all_si.extend(si_list)
            all_tof.extend(tof_list)
            total_skipped += skipped
            elapsed = time.time() - t0
            done = i + 1
            remain = len(file_list) - done
            eta = elapsed / done * remain
            if done % 50 == 0 or done == len(file_list):
                print(f"  [{desc}] {done}/{len(file_list)} files | "
                      f"{len(all_labels):,} events (skipped {total_skipped}) | "
                      f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m",
                      flush=True)
    return all_si, all_tof, all_labels


def save_npz(out_path, si_list, tof_list, labels):
    np.savez_compressed(
        out_path,
        voxels = np.stack(si_list),                  # (N, 10, 12, 12) float32
        tofs   = np.stack(tof_list),                 # (N, 9) float32
        labels = np.array(labels, dtype=np.int64),   # (N,) int64
    )
    n_dbar = int(np.sum(np.array(labels) == 1))
    n_pbar = int(np.sum(np.array(labels) == 0))
    print(f"  saved → {out_path}  ({len(labels):,} events, "
          f"Dbar={n_dbar:,}, Pbar={n_pbar:,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",   required=True,
                        help="train_5cross ディレクトリ")
    parser.add_argument("--val_dir",     required=True,
                        help="valid_5cross ディレクトリ")
    parser.add_argument("--out_dir",     required=True,
                        help="npz 出力先ディレクトリ")
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # shuffled/ ディレクトリ内のファイル名は "220122_digitized_Atrest_shuffled_*.csv"
    # 直下の train_5cross/valid_5cross は "CNN220117_*.csv" だが shuffled/ には CNN なし
    # 両方サポートするため "*.csv" でマッチ
    train_files = sorted(pathlib.Path(args.train_dir).glob("*.csv"))
    val_files   = sorted(pathlib.Path(args.val_dir).glob("*.csv"))
    print(f"Train files: {len(train_files)}")
    print(f"Val   files: {len(val_files)}")

    print("=== Processing train ===")
    si_list, tof_list, labels = process_files(train_files, "train", args.num_workers)
    save_npz(out / "train_hybrid_nakagami4M.npz", si_list, tof_list, labels)

    print("=== Processing val ===")
    si_list, tof_list, labels = process_files(val_files, "val", args.num_workers)
    save_npz(out / "val_hybrid_nakagami4M.npz", si_list, tof_list, labels)

    print("Done.")
