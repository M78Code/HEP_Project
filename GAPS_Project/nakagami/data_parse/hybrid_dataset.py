"""
HybridDatasetFast: 中上40M前処理npzを読み込むためのDataset。

期待するnpzのkey:
  voxels: (N, 10, 12, 12) float32
  tofs  : (N, 11)         float32
  labels: (N,)            int64

注：元のGAPS_Project/src/data_parse/hybrid_dataset.pyからHybridDatasetFastのみを抜粋。
    HybridDataset（pklから再voxel化する版）はvoxelizer.pyに依存するため除外。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class HybridDatasetFast(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.voxels = data['voxels']   # (N, 10, 12, 12) float32
        self.tofs   = data['tofs']     # (N, 11)        float32
        self.labels = data['labels']   # (N,)           int64
        print(f'  loaded {len(self.voxels):,} events from {npz_path}')

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.voxels[idx]).unsqueeze(0),  # (1,10,12,12)
            torch.from_numpy(self.tofs[idx]),                  # (11,)
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
