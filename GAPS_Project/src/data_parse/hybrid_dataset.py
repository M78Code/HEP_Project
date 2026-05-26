import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from .voxelizer import build_sili_voxel, build_tof_features

ANTIDEUTERON = -1000010020


class HybridDatasetFast(Dataset):
  """从预计算的 npz 文件加载，训练速度远快于 HybridDataset。
  需先运行 src/scripts/preprocess_hybrid.py 生成 npz 文件。"""
  def __init__(self, npz_path):
      data = np.load(npz_path)
      self.voxels = data['voxels']   # (N, 10, 12, 12) float32
      self.tofs   = data['tofs']     # (N, 11) float32
      self.labels = data['labels']   # (N,) int64
      print(f'  loaded {len(self.voxels):,} events from {npz_path}')

  def __len__(self):
      return len(self.voxels)

  def __getitem__(self, idx):
      return (
          torch.from_numpy(self.voxels[idx]).unsqueeze(0),        # (1,10,12,12)
          torch.from_numpy(self.tofs[idx]),                        # (11,)
          torch.tensor(self.labels[idx], dtype=torch.long),
      )


class HybridDataset(Dataset):
  def __init__(self, pkl_path):
      with open(pkl_path, 'rb') as f:
          payload = pickle.load(f)
      self.data = payload['events']
      print(f'  loaded {len(self.data)} events from {pkl_path}')

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      ev     = self.data[idx]
      voxel  = build_sili_voxel(ev)       # (10,20,20)
      tof    = build_tof_features(ev)     # (11,)
      label  = 1 if ev['label'] == ANTIDEUTERON else 0

      return (
          torch.tensor(voxel, dtype=torch.float32).unsqueeze(0),  # (1,10,20,20)
          torch.tensor(tof,   dtype=torch.float32),                # (11,)
          torch.tensor(label, dtype=torch.long),
      )