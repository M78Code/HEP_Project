"""Dataset for Nakagami ynakagami2 three-input npz."""
import numpy as np
import torch
from torch.utils.data import Dataset


class ThreeInputDataset(Dataset):
    def __init__(self, npz_path, normalize=False):
        data = np.load(npz_path)
        self.voxels = data['voxels']
        self.tof_paddles = data['tof_paddles']
        self.tof_primary = data['tof_primary']
        self.labels = data['labels']
        self.normalize = normalize

        if self.tof_paddles.shape[1] != 172:
            raise ValueError(f'tof_paddles dim must be 172, got {self.tof_paddles.shape[1]}')
        if self.tof_primary.shape[1] != 11:
            raise ValueError(f'tof_primary dim must be 11, got {self.tof_primary.shape[1]}')

        print(
            f'  loaded {len(self.labels):,} events from {npz_path} '
            f'(voxel={self.voxels.shape[1:]}, tof_paddles=172, tof_primary=11, '
            f'{"normalized" if normalize else "raw"})'
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        idx = int(idx)
        voxel = self.voxels[idx]
        tof_paddle = self.tof_paddles[idx]
        tof_primary = self.tof_primary[idx]

        # Strict reproduction should keep raw values. The option is left for later stability tests.
        if self.normalize:
            tof_paddle = tof_paddle / 100.0
            scale = np.array([50, 50, 50, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500], dtype=np.float32)
            tof_primary = tof_primary / scale

        return (
            torch.from_numpy(voxel).unsqueeze(0),
            torch.from_numpy(tof_paddle.astype(np.float32, copy=False)),
            torch.from_numpy(tof_primary.astype(np.float32, copy=False)),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
