"""
新GAPS 5000万データのfile-level train/val/test split。
データをメモリに展開せず、pklファイルのパスリストを70/15/15で分割し
JSONマニフェストとして保存する。

antiD: 100 files → 70 train / 15 val / 15 test
antiP: 140 files → 98 train / 21 val / 21 test

使い方:
  python src/data_parse/split_aohba_files.py \
      --antiD-dir /mnt/ynakagami3/aohba_preprocess/antiD \
      --antiP-dir /mnt/ynakagami3/aohba_preprocess/antiP \
      --output-dir /mnt/ynakagami3/aohba_preprocess/split
"""
import argparse
import json
import random
from pathlib import Path


def split_files(files: list, train_ratio: float, val_ratio: float, seed: int):
    """ファイルリストをseed固定でshuffle → train/val/testに分割"""
    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--antiD-dir',    type=Path, required=True)
    ap.add_argument('--antiP-dir',    type=Path, required=True)
    ap.add_argument('--output-dir',   type=Path, required=True)
    ap.add_argument('--train-ratio',    type=float, default=0.70)
    ap.add_argument('--val-ratio',      type=float, default=0.15)
    ap.add_argument('--seed',           type=int,   default=42)
    ap.add_argument('--expected-antiD', type=int,   default=100)
    ap.add_argument('--expected-antiP', type=int,   default=140)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    antid_files = sorted(str(f) for f in args.antiD_dir.glob('*.pkl')
                         if not f.name.endswith('_summary.pkl'))
    antip_files = sorted(str(f) for f in args.antiP_dir.glob('*.pkl')
                         if not f.name.endswith('_summary.pkl'))

    print(f'antiD: {len(antid_files)} files')
    print(f'antiP: {len(antip_files)} files')

    if len(antid_files) != args.expected_antiD:
        raise RuntimeError(f'antiD files expected {args.expected_antiD}, got {len(antid_files)}')
    if len(antip_files) != args.expected_antiP:
        raise RuntimeError(f'antiP files expected {args.expected_antiP}, got {len(antip_files)}')

    antid_train, antid_val, antid_test = split_files(
        antid_files, args.train_ratio, args.val_ratio, args.seed)
    antip_train, antip_val, antip_test = split_files(
        antip_files, args.train_ratio, args.val_ratio, args.seed)

    manifest = {
        'seed':         args.seed,
        'train_ratio':  args.train_ratio,
        'val_ratio':    args.val_ratio,
        'stats': {
            'antiD': {'train': len(antid_train), 'val': len(antid_val), 'test': len(antid_test)},
            'antiP': {'train': len(antip_train), 'val': len(antip_val), 'test': len(antip_test)},
        },
        'train': {'antiD': antid_train, 'antiP': antip_train},
        'val':   {'antiD': antid_val,   'antiP': antip_val},
        'test':  {'antiD': antid_test,  'antiP': antip_test},
    }

    out_path = args.output_dir / 'split_manifest.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\n分割完了 → {out_path}')
    print(f'  train : antiD={len(antid_train):3d}, antiP={len(antip_train):3d}  '
          f'合計={len(antid_train)+len(antip_train)}ファイル')
    print(f'  val   : antiD={len(antid_val):3d}, antiP={len(antip_val):3d}  '
          f'合計={len(antid_val)+len(antip_val)}ファイル')
    print(f'  test  : antiD={len(antid_test):3d}, antiP={len(antip_test):3d}  '
          f'合計={len(antid_test)+len(antip_test)}ファイル')


if __name__ == '__main__':
    main()
