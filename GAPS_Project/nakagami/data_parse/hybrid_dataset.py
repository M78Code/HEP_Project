"""
HybridDatasetFast: 中上 4M (csvFiles_Digitized/shuffled/) または
40M 前処理 npz 用 Dataset。

期待する npz の key:
  voxels: (N, 10, 12, 12) float32
  tofs  : (N, D)          float32   (D=9 for 4M, D=11 for 40M)
  labels: (N,)            int64

TOF正規化:
  normalize_tof=False (デフォルト, 4M 厳密復元版):
    中上 IdentifywithNN.py に合わせて TOF を生値のまま使用。
    AMP は併用しない (FP32 で訓練すること)。
  normalize_tof=True (改良版):
    TOF を経験スケールで正規化、AMP との併用が可能。

NumPy 2.x の厳格な index 仕様に対応するため __getitem__ 内で int(idx) 変換。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


# ── TOF 9次元 (4M) のスケール（normalize_tof=True 用）──
# 中上 shuffled データで観測した値範囲 ±11000 から決定
TOF_SCALE_9 = np.full(9, 10000.0, dtype=np.float32)

# ── TOF 11次元 (40M) のスケール ──
TOF_SCALE_11 = np.array([
    50.0, 50.0,
    50.0,
    1500., 1500., 1500.,
    1500., 1500., 1500.,
    1500., 1500.,
], dtype=np.float32)


class HybridDatasetFast(Dataset):
    def __init__(self, npz_path, normalize_tof=False):
        data = np.load(npz_path)
        self.voxels = data['voxels']
        self.tofs   = data['tofs']
        self.labels = data['labels']
        self.normalize_tof = normalize_tof

        tof_dim = self.tofs.shape[1]
        if tof_dim == 9:
            self.tof_scale = TOF_SCALE_9
        elif tof_dim == 11:
            self.tof_scale = TOF_SCALE_11
        else:
            raise ValueError(f'未対応のTOF次元: {tof_dim}')

        norm_str = 'normalized' if normalize_tof else 'raw (no normalize)'
        print(f'  loaded {len(self.voxels):,} events from {npz_path} '
              f'(TOF dim={tof_dim}, {norm_str})')

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        idx = int(idx)
        tof = self.tofs[idx]
        if self.normalize_tof:
            tof = tof / self.tof_scale
        return (
            torch.from_numpy(self.voxels[idx]).unsqueeze(0),
            torch.from_numpy(tof),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
