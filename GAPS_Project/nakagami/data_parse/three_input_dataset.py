"""Dataset for Nakagami ynakagami2 three-input data."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ThreeInputDataset(Dataset):
    def __init__(self, data_path, normalize=False, mmap=True, max_events=None):
        data_path = Path(data_path)
        self.normalize = normalize

        if data_path.is_dir():
            mmap_mode = 'r' if mmap else None
            self.voxels = np.load(data_path / 'voxels.npy', mmap_mode=mmap_mode)
            self.tof_paddles = np.load(data_path / 'tof_paddles.npy', mmap_mode=mmap_mode)
            self.tof_primary = np.load(data_path / 'tof_primary.npy', mmap_mode=mmap_mode)
            self.labels = np.load(data_path / 'labels.npy', mmap_mode=mmap_mode)
            source = str(data_path)
        else:
            data = np.load(data_path)
            self.voxels = data['voxels']
            self.tof_paddles = data['tof_paddles']
            self.tof_primary = data['tof_primary']
            self.labels = data['labels']
            source = str(data_path)

        if max_events is not None:
            max_events = int(max_events)
            self.voxels = self.voxels[:max_events]
            self.tof_paddles = self.tof_paddles[:max_events]
            self.tof_primary = self.tof_primary[:max_events]
            self.labels = self.labels[:max_events]

        if self.tof_paddles.shape[1] != 172:
            raise ValueError(f'tof_paddles dim must be 172, got {self.tof_paddles.shape[1]}')
        if self.tof_primary.shape[1] != 11:
            raise ValueError(f'tof_primary dim must be 11, got {self.tof_primary.shape[1]}')

        print(
            f'  loaded {len(self.labels):,} events from {source} '
            f'(voxel={self.voxels.shape[1:]}, tof_paddles=172, tof_primary=11, '
            f'{"normalized" if normalize else "raw"})'
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        idx = int(idx)
        voxel = np.asarray(self.voxels[idx], dtype=np.float32)
        tof_paddle = np.asarray(self.tof_paddles[idx], dtype=np.float32)
        tof_primary = np.asarray(self.tof_primary[idx], dtype=np.float32)

        # Strict reproduction should keep raw values. The option is left for later stability tests.
        if self.normalize:
            tof_paddle = tof_paddle / 100.0
            scale = np.array([50, 50, 50, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500], dtype=np.float32)
            tof_primary = tof_primary / scale

        return (
            torch.from_numpy(voxel).unsqueeze(0),
            torch.from_numpy(tof_paddle),
            torch.from_numpy(tof_primary),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )
