import pickle
import torch
from torch.utils.data import Dataset
from .voxelizer import build_sili_voxel, build_tof_features

ANTIDEUTERON = -1000010020

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