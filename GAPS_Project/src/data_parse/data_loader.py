# Step 2.3: DataLoader

import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
import GAPS_Project

from GAPS_Project.src.data_parse.gaps_dataset import GapsDataset

PROJECT_ROOT = Path(GAPS_Project.__file__).parent


def make_dataloaders(pkl_files: list, batch_size: int = 16, train_ratio: float = 0.7, val_ratio: float = 0.15,
                     shuffle: bool = True):
    """
    从pickle文件构建，train/val/test 三个DataLoader

    Args:
        pkl_files:      pickle文件路径列表
        batch_size:     每个batch的event数
        train_ratio:    训练集比例
        val_ratio:      验证集比例（剩余为test）
        shuffle:        是否打乱顺序

    Returns:
          train_loader, val_loader, test_loader
    """
    dataset = GapsDataset(pkl_files=pkl_files)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ── 快速测试 ────────────────────────────────────────────
if __name__ == "__main__":
    pkl_path = PROJECT_ROOT / 'dataset' / 'processed' / 'anti_deuteron_gaps_FTFP_BERT_1778138909.pkl'

    train_loader, val_loader, test_loader = make_dataloaders(pkl_files=[pkl_path], batch_size = 8)

    print(f"train batches: {len(train_loader)}")
    print(f"val   batches: {len(val_loader)}")
    print(f"test  batches: {len(test_loader)}")

    batch = next(iter(train_loader))
    print(f"\n取出第一个batch:")
    print(f"  x.shape:        {batch.x.shape}")
    print(f"  edge_index:     {batch.edge_index.shape}")
    print(f"  y:              {batch.y}")
    print(f"  batch向量:      {batch.batch}")


"""
train batches: 5
val   batches: 1
test  batches: 2

取出第一个batch:
  x.shape:        torch.Size([228, 5])
  edge_index:     torch.Size([2, 1824])
  y:              tensor([1, 1, 1, 1, 1, 1, 1, 1])
  batch向量:      tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])
"""

