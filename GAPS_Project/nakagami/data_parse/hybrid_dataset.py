"""
HybridDatasetFast: 中上40M前処理npzを読み込むためのDataset。

期待するnpzのkey:
  voxels: (N, 10, 12, 12) float32
  tofs  : (N, 11)         float32
  labels: (N,)            int64

TOF特徴量の正規化:
  TOF 11次元のうち、座標(±1700mm程度)と能量・時間(数十単位)で
  スケールが大きく異なるため、AMP使用時にFP16でオーバーフロー/
  NaNを起こしやすい。__getitem__内で正規化スケールで割って
  スケーリングする。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class HybridDatasetFast(Dataset):
    # ── TOF 11次元の正規化スケール ──
    # [outer_E, inner_E, time_diff,
    #  inner_x, inner_y, inner_z,
    #  outer_x, outer_y, outer_z,
    #  stop_x, stop_y]
    TOF_SCALE = np.array([
        50.0, 50.0,            # outer_E, inner_E (~10-50 MeV)
        50.0,                  # time_diff (~10-50 ns)
        1500., 1500., 1500.,   # inner_xyz (~±1500 mm)
        1500., 1500., 1500.,   # outer_xyz (~±1500 mm)
        1500., 1500.,          # stop_xy   (~±700 mm)
    ], dtype=np.float32)

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.voxels = data['voxels']   # (N, 10, 12, 12) float32
        self.tofs   = data['tofs']     # (N, 11)        float32
        self.labels = data['labels']   # (N,)           int64
        print(f'  loaded {len(self.voxels):,} events from {npz_path}')

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        # NumPy 2.x はtensor/0-d arrayをindexとして受け付けない厳格仕様。
        # WeightedRandomSamplerが返すtensor indexをintに変換。
        idx = int(idx)
        return (
            torch.from_numpy(self.voxels[idx]).unsqueeze(0),     # (1,10,12,12)
            torch.from_numpy(self.tofs[idx] / self.TOF_SCALE),    # (11,) 正規化
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
