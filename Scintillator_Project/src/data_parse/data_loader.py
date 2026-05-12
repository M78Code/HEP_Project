
from torch.utils.data import DataLoader
from pathlib import Path
from Scintillator_Project.src.data_parse.scintillator_dataset import ScintillatorDataset

def make_dataloaders(split_dir: str | Path, batch_size: int = 64, num_workers: int = 0):
    split_dir = Path(split_dir)

    train_ds = ScintillatorDataset(split_dir / "train.json")
    val_ds = ScintillatorDataset(split_dir / "val.json")
    test_ds = ScintillatorDataset(split_dir / "test.json")

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_loader

"""
 关键参数说明
  ┌─────────────┬───────┬──────────┬─────────────────────────────────────────────┐
  │    参数     │ train │ val/test │                    原因                     │
  ├─────────────┼───────┼──────────┼─────────────────────────────────────────────┤
  │ shuffle     │ True  │ False    │ 训练时打乱防止顺序偏差，验证/测试要复现结果 │
  ├─────────────┼───────┼──────────┼─────────────────────────────────────────────┤
  │ batch_size  │ 64    │ 64       │ 显存和精度的平衡，常用值                    │
  ├─────────────┼───────┼──────────┼─────────────────────────────────────────────┤
  │ num_workers │ 0     │ 0        │ Mac上0最安全，Linux可以改2/4                │
  └─────────────┴───────┴──────────┴─────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    # import os
    # print(os.getcwd())# 看PyCharm认为的工作目录在哪
    train_loader, val_loader, test_loader = make_dataloaders('../dataset/split')

    # 取一个batch看形状
    x, y = next(iter(train_loader))
    print(f"x shape: {x.shape}")  # 期望 [64, 2, 1024]
    print(f"y shape: {y.shape}")  # 期望 [64]
    print(f"y sample: {y[:5]}")  # 看一下前5个标签
    print(f"Train batches: {len(train_loader)}")  # 16514 / 64 ≈ 258


"""
Connected to pydev debugger (build 251.26094.141)
x shape: torch.Size([64, 2, 1024])
y shape: torch.Size([64])
y sample: tensor([ 45., 125., 125.,  15.,  80.])
Train batches: 259
"""