"""
预处理脚本：将 train/val/test pkl 中的每个 event 提前计算 voxel 和 TOF 特征，
保存为压缩 npz 文件，供 HybridDatasetFast 直接加载，避免训练时重复计算。

运行一次即可，约需 30~60 分钟（取决于 CPU）。
输出文件：dataset/split/train_hybrid.npz, val_hybrid.npz, test_hybrid.npz
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import GAPS_Project
from GAPS_Project.src.data_parse.voxelizer import build_sili_voxel, build_tof_features

PROJECT_ROOT  = Path(GAPS_Project.__file__).parent
SPLIT_DIR     = PROJECT_ROOT / 'dataset' / 'split'
ANTIDEUTERON  = -1000010020


def preprocess_split(pkl_path: Path, out_path: Path):
    print(f'\n处理: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)
    events = payload['events']
    N = len(events)
    print(f'  事例数: {N:,}')

    voxels = np.zeros((N, 10, 20, 20), dtype=np.float32)
    tofs   = np.zeros((N, 11),         dtype=np.float32)
    labels = np.zeros(N,               dtype=np.int64)
    betas  = np.zeros(N,               dtype=np.float32)

    for i, ev in enumerate(tqdm(events, desc='  building features')):
        voxels[i] = build_sili_voxel(ev)
        tofs[i]   = build_tof_features(ev)
        labels[i] = 1 if ev['label'] == ANTIDEUTERON else 0
        betas[i]  = float(ev.get('beta', 0.0))

    print(f'  保存（压缩）→ {out_path}.npz ...')
    np.savez_compressed(out_path, voxels=voxels, tofs=tofs, labels=labels, betas=betas)
    size_mb = (out_path.parent / (out_path.name + '.npz')).stat().st_size / 1024 / 1024
    print(f'  完成，文件大小: {size_mb:.0f} MB')


if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        preprocess_split(
            pkl_path = SPLIT_DIR / f'{split}.pkl',
            out_path = SPLIT_DIR / f'{split}_hybrid_20x20',   # 自动加 .npz 后缀
        )
    print('\n全部预处理完成。')
