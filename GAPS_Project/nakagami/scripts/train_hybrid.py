"""
中上40M CSVから生成したnpzでCNN+DNN（A.2 7.1.1）を訓練するスクリプト。
GAPS_Project/src/scripts/train_hybrid.pyをベースに、importとデータパスを
nakagami配下に書き換えたもの。

クラス不均衡対策:
  Dbar (signal,    label=1): ~400K events
  Pbar (background, label=0): ~20K events
  比例 約 20:1 のため WeightedRandomSampler を使用して
  各 mini-batch を近似平衡化する（pos_weight は併用しない）。
"""
import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# nakagami配下のmodels/data_parseをimportするためsys.pathを追加
NAKAGAMI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NAKAGAMI_ROOT))

from data_parse.hybrid_dataset import HybridDatasetFast
from models.cnn_dnn_hybrid import CNNDNNHybrid

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE  = 200       # A.2 7.1.1 原版设定
EPOCHS      = 100
LR          = 4e-5      # A.2 7.1.1 原版设定
PATIENCE    = 10        # early stopping
NUM_WORKERS = 8
DATA_DIR    = Path('/mnt/ynakagami3/nakagami_data/data_40M')
SAVE_DIR    = NAKAGAMI_ROOT / 'results'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH   = SAVE_DIR / 'nakagami40M_cnndnn_best.pth'


def train():
  print(f'使用设备：{DEVICE}')
  train_set = HybridDatasetFast(DATA_DIR / 'train_hybrid_nakagami40M.npz')
  val_set   = HybridDatasetFast(DATA_DIR / 'val_hybrid_nakagami40M.npz')

  # ── WeightedRandomSampler でクラス不均衡を補償 ──
  train_labels = train_set.labels
  n_pbar = int((train_labels == 0).sum())
  n_dbar = int((train_labels == 1).sum())
  print(f'train label counts: Pbar={n_pbar:,}, Dbar={n_dbar:,}')

  sample_weights = np.where(
      train_labels == 0,
      1.0 / max(n_pbar, 1),
      1.0 / max(n_dbar, 1),
  )
  sampler = WeightedRandomSampler(
      weights=torch.as_tensor(sample_weights, dtype=torch.double),
      num_samples=len(sample_weights),
      replacement=True,
  )

  train_loader = DataLoader(
      train_set, batch_size=BATCH_SIZE,
      sampler=sampler, shuffle=False,
      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
  )
  # 検証集は元の分布を保持（samplerは使わない）
  val_loader = DataLoader(
      val_set, batch_size=BATCH_SIZE, shuffle=False,
      num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
  )

  model = CNNDNNHybrid(tof_dim=11).to(DEVICE)
  # 注: torch.compile はPyTorch 2.0+でのみ利用可能、PyTorch 1.10では使わない
  # 注: AMP は無効化（TOF特徴量の値範囲が±1700mmと広く、FP16でNaNを起こすため）
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  criterion = nn.BCEWithLogitsLoss()

  print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
  best_val_loss = float('inf')
  patience_counter = 0

  for epoch in range(1, EPOCHS + 1):
      epoch_start = time.time()
      model.train()
      t_loss = t_acc = 0.0
      train_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{EPOCHS} [train]', leave=False)
      for voxel, tof, label in train_bar:
          voxel, tof, label = voxel.to(DEVICE), tof.to(DEVICE), label.float().to(DEVICE)
          optimizer.zero_grad()
          out  = model(voxel, tof)
          loss = criterion(out, label)
          loss.backward()
          # 勾配クリッピングで爆発を防ぐ
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()
          t_loss += loss.item()
          t_acc  += ((out > 0).long() == label.long()).float().mean().item()
          train_bar.set_postfix(loss=f'{loss.item():.4f}')

      model.eval()
      v_loss = v_acc = 0.0
      with torch.no_grad():
          for voxel, tof, label in tqdm(val_loader, desc=f'Epoch {epoch:3d}/{EPOCHS} [val]  ', leave=False):
              voxel, tof, label = voxel.to(DEVICE), tof.to(DEVICE), label.float().to(DEVICE)
              out    = model(voxel, tof)
              v_loss += criterion(out, label).item()
              v_acc  += ((out > 0).long() == label.long()).float().mean().item()

      t_loss /= len(train_loader); t_acc /= len(train_loader)
      v_loss /= len(val_loader);   v_acc /= len(val_loader)
      elapsed = time.time() - epoch_start

      print(f'Epoch {epoch:3d}/{EPOCHS} | '
            f'train_loss: {t_loss:.4f}  train_acc: {t_acc:.4f} | '
            f'val_loss: {v_loss:.4f}  val_acc: {v_acc:.4f} | '
            f'{elapsed:.0f}s')

      if v_loss < best_val_loss:
          best_val_loss = v_loss
          patience_counter = 0
          raw = model._orig_mod if hasattr(model, '_orig_mod') else model
          torch.save(raw.state_dict(), SAVE_PATH)
          print(f'  → best model saved (val_loss={v_loss:.4f})')
      else:
          patience_counter += 1
          if patience_counter >= PATIENCE:
              print(f'  → early stopping at epoch {epoch} (patience={PATIENCE})')
              break

if __name__ == '__main__':
  train()
