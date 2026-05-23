from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import GAPS_Project

PROJECT_ROOT = Path(GAPS_Project.__file__).parent

from GAPS_Project.src.data_parse.hybrid_dataset import HybridDatasetFast
from GAPS_Project.src.models.cnn_dnn_hybrid import CNNDNNHybrid

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 200       # Nakagami A.2
EPOCHS     = 100
LR         = 4e-5      # Nakagami A.2: 0.00004
PATIENCE   = 10        # early stopping
NUM_WORKERS = 8
DATA_DIR   = PROJECT_ROOT / 'dataset' / 'split'
SAVE_PATH  = PROJECT_ROOT / 'results' / 'cnn_dnn_hybrid_best.pth'


def train():
  print(f'使用设备：{DEVICE}')
  train_set = HybridDatasetFast(DATA_DIR / 'train_hybrid.npz')
  val_set   = HybridDatasetFast(DATA_DIR / 'val_hybrid.npz')
  train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
  val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

  model     = CNNDNNHybrid(tof_dim=11).to(DEVICE)
  model     = torch.compile(model)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
  criterion = nn.BCEWithLogitsLoss()
  scaler    = torch.amp.GradScaler('cuda')   # 混合精度

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
          with torch.amp.autocast('cuda'):
              out  = model(voxel, tof)
              loss = criterion(out, label)
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
          t_loss += loss.item()
          t_acc  += ((out > 0).long() == label.long()).float().mean().item()
          train_bar.set_postfix(loss=f'{loss.item():.4f}')

      model.eval()
      v_loss = v_acc = 0.0
      with torch.no_grad():
          for voxel, tof, label in tqdm(val_loader, desc=f'Epoch {epoch:3d}/{EPOCHS} [val]  ', leave=False):
              voxel, tof, label = voxel.to(DEVICE), tof.to(DEVICE), label.float().to(DEVICE)
              with torch.amp.autocast('cuda'):
                  out    = model(voxel, tof)
                  v_loss += criterion(out, label).item()
              v_acc  += ((out > 0).long() == label.long()).float().mean().item()

      t_loss /= len(train_loader); t_acc /= len(train_loader)
      v_loss /= len(val_loader);   v_acc /= len(val_loader)
      scheduler.step()
      elapsed = time.time() - epoch_start

      print(f'Epoch {epoch:3d}/{EPOCHS} | '
            f'train_loss: {t_loss:.4f}  train_acc: {t_acc:.4f} | '
            f'val_loss: {v_loss:.4f}  val_acc: {v_acc:.4f} | '
            f'{elapsed:.0f}s')

      if v_loss < best_val_loss:
          best_val_loss = v_loss
          patience_counter = 0
          torch.save(model.state_dict(), SAVE_PATH)
          print(f'  → best model saved (val_loss={v_loss:.4f})')
      else:
          patience_counter += 1
          if patience_counter >= PATIENCE:
              print(f'  → early stopping at epoch {epoch} (patience={PATIENCE})')
              break

if __name__ == '__main__':
  train()