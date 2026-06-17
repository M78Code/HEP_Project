"""
Preprocess Nakagami 4M onlyPrimary CSV for the ynakagami2 three-input model.

This matches /mnt/ynakagami2/DeepLearning/IdentifywithNN.py:
  source: /mnt/ynakagami2/SimulationData/210922_trigger/csvFiles_onlyPrimary/atrest_shuffled
  train: 32 files, 3,200,000 events
  valid:  8 files,   800,000 events

CSV layout (0-indexed, total 1631 columns):
  col 0          : file/random ID
  col 1          : event ID
  col 2          : label (0=pbar, 1=dbar)
  col 6:1446     : Si(Li) energy deposition, 1440 values -> (10, 12, 12)
  col 1446:1618  : TOF paddle energy deposition, 172 values
  col 1618:1629  : primary TOF physical quantities, 11 values
  col 1629:1631  : unused by IdentifywithNN.py

Implementation note:
  This script uses a two-pass reader. It first counts rows, then pre-allocates
  contiguous arrays and fills them directly. This avoids holding millions of
  small NumPy arrays in Python lists.
"""
import argparse
import csv
import pathlib
import time

import numpy as np
from tqdm import tqdm

N_COLS = 1631
LABEL_COL = 2
SI_START, SI_END = 6, 1446
TOF_PADDLE_START, TOF_PADDLE_END = 1446, 1618
TOF_PRIMARY_START, TOF_PRIMARY_END = 1618, 1629


def count_rows(files):
    total = 0
    bad = 0
    labels = {0: 0, 1: 0}
    for fp in tqdm(files, desc='count', dynamic_ncols=True):
        with open(fp) as f:
            for row in csv.reader(f):
                if len(row) != N_COLS:
                    bad += 1
                    continue
                label = int(float(row[LABEL_COL]))
                labels[label] += 1
                total += 1
    return total, bad, labels


def fill_arrays(files, n_events):
    voxels = np.empty((n_events, 10, 12, 12), dtype=np.float32)
    tof_paddles = np.empty((n_events, 172), dtype=np.float32)
    tof_primary = np.empty((n_events, 11), dtype=np.float32)
    labels = np.empty((n_events,), dtype=np.int64)

    idx = 0
    bad = 0
    t0 = time.time()
    for fp in tqdm(files, desc='fill', dynamic_ncols=True):
        with open(fp) as f:
            for row in csv.reader(f):
                if len(row) != N_COLS:
                    bad += 1
                    continue
                labels[idx] = int(float(row[LABEL_COL]))
                voxels[idx] = np.asarray(row[SI_START:SI_END], dtype=np.float32).reshape(10, 12, 12)
                tof_paddles[idx] = np.asarray(row[TOF_PADDLE_START:TOF_PADDLE_END], dtype=np.float32)
                tof_primary[idx] = np.asarray(row[TOF_PRIMARY_START:TOF_PRIMARY_END], dtype=np.float32)
                idx += 1

    elapsed = time.time() - t0
    print(f'  filled {idx:,} events in {elapsed/60:.1f} min')
    if idx != n_events:
        raise RuntimeError(f'filled events mismatch: expected {n_events}, got {idx}')
    return voxels, tof_paddles, tof_primary, labels, bad


def save_npz(out_path, voxels, tof_paddles, tof_primary, labels, compressed=False):
    saver = np.savez_compressed if compressed else np.savez
    saver(
        out_path,
        voxels=voxels,
        tof_paddles=tof_paddles,
        tof_primary=tof_primary,
        labels=labels,
    )
    n_pbar = int((labels == 0).sum())
    n_dbar = int((labels == 1).sum())
    size_gb = pathlib.Path(out_path).stat().st_size / 1e9
    print(f'  saved -> {out_path} ({size_gb:.2f} GB, {len(labels):,} events, Pbar={n_pbar:,}, Dbar={n_dbar:,})')


def process_split(split, split_dir, out_path, compressed=False, max_files=None):
    files = sorted(split_dir.glob('*.csv'))
    if max_files is not None:
        files = files[:max_files]
    print(f'=== {split} ===')
    print(f'dir: {split_dir}')
    print(f'files: {len(files)}')

    n_events, count_bad, label_counts = count_rows(files)
    print(f'  rows: {n_events:,}, bad_cols={count_bad}, labels={label_counts}')

    voxels, tof_paddles, tof_primary, labels, fill_bad = fill_arrays(files, n_events)
    if fill_bad:
        print(f'WARNING: skipped {fill_bad} bad-column rows during fill')
    save_npz(out_path, voxels, tof_paddles, tof_primary, labels, compressed=compressed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='/mnt/ynakagami2/SimulationData/210922_trigger/csvFiles_onlyPrimary/atrest_shuffled')
    parser.add_argument('--out-dir', default='/mnt/ynakagami3/nakagami_data/data_4M_onlyprimary')
    parser.add_argument('--compressed', action='store_true', help='use savez_compressed; slower and may use more memory')
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both')
    parser.add_argument('--max-files', type=int, default=None, help='limit files per split for smoke tests')
    args = parser.parse_args()

    base = pathlib.Path(args.base_dir)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.split in ('train', 'both'):
        process_split('train', base / 'train_5cross', out / 'train_onlyprimary_4M.npz', compressed=args.compressed, max_files=args.max_files)
    if args.split in ('val', 'both'):
        process_split('val', base / 'valid_5cross', out / 'val_onlyprimary_4M.npz', compressed=args.compressed, max_files=args.max_files)


if __name__ == '__main__':
    main()
