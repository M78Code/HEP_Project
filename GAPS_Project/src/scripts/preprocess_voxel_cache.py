"""
预计算 voxel 缓存：为每个 split 的 pkl 生成对应的 voxel npy 文件。
GapsDataset 加载时通过 mmap 读取，不占内存。

运行一次即可，输出：
  dataset/split/train_voxel_cache.npy
  dataset/split/val_voxel_cache.npy
  dataset/split/test_voxel_cache.npy
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import GAPS_Project
from GAPS_Project.src.data_parse.voxelizer import build_sili_voxel, GRID_Z, GRID_X, GRID_Y

PROJECT_ROOT = Path(GAPS_Project.__file__).parent
SPLIT_DIR = PROJECT_ROOT / 'dataset' / 'split'


def preprocess_voxel(pkl_path: Path, out_path: Path):
    print(f'\n处理: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)
    events = payload['events']
    N = len(events)
    print(f'  事例数: {N:,}')
    print(f'  voxel shape: ({GRID_Z}, {GRID_X}, {GRID_Y})')

    voxels = np.zeros((N, GRID_Z, GRID_X, GRID_Y), dtype=np.float32)
    for i, ev in enumerate(tqdm(events, desc='  computing voxels')):
        voxels[i] = build_sili_voxel(ev)

    np.save(out_path, voxels)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'  保存完成: {out_path} ({size_mb:.0f} MB)')


if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        preprocess_voxel(
            pkl_path=SPLIT_DIR / f'{split}.pkl',
            out_path=SPLIT_DIR / f'{split}_voxel_cache.npy',
        )
    print('\n全部 voxel 缓存生成完成。')
