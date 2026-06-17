"""
中上 4M CSV (csvFiles_Digitized/shuffled/) を用いて NakagamiNet (A.1, 6.2節)
を厳密に復元するための訓練スクリプト。

中上 IdentifywithNN.py との対応:
  - batch_size = 200
  - epochs    = 50
  - LR        = 4e-5 (Adam)
  - dropout   = 0.1 (ResBlock 内)
  - EarlyStopping: patience=4, monitor=val_accuracy
  - Loss      : BCEWithLogitsLoss (≡ binary_crossentropy + sigmoid)
  - Activation: ReLU
  - Output    : sigmoid (推論時に外部で適用)
  - TOF normalize: なし (生値)
  - AMP       : なし (FP32)
  - 単GPU 訓練 (CUDA_VISIBLE_DEVICES=0 等で指定推奨)
"""
import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# nakagami配下のmodels/data_parseをimport
NAKAGAMI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NAKAGAMI_ROOT))

from data_parse.hybrid_dataset import HybridDatasetFast
from models.nakagami_model import NakagamiNet

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE  = 200       # 中上 IdentifywithNN.py 原版 (厳密復元)
EPOCHS      = 50        # 中上 IdentifywithNN.py 原版
LR          = 4e-5      # 中上 IdentifywithNN.py 原版 (Adam lr=0.00004)
PATIENCE    = 4         # 中上 EarlyStopping(patience=4)
NUM_WORKERS = 8
DROPOUT     = 0.1       # 中上 ResBlock 内 Dropout(rate=0.1)
# TOF normalize: True (PyTorch移植の安定化対策)
#   過滤異常事件後も座標値は ±16550 と大きく、生値だとforward激活が大爆発する
#   除以 10000 で範囲を [-1.7, +1.7] 程度に圧縮（中上Kerasにはない処理だが、
#   論文中で「PyTorch移植時の数値安定化」と明記する）
NORMALIZE_TOF = True
DATA_DIR    = Path('/mnt/ynakagami3/nakagami_data/data_4M')
SAVE_DIR    = NAKAGAMI_ROOT / 'results'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH   = SAVE_DIR / 'nakagami4M_cnndnn_best.pth'


def train():
    print(f'使用设备：{DEVICE}')
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')

    train_set = HybridDatasetFast(DATA_DIR / 'train_hybrid_nakagami4M.npz',
                                  normalize_tof=NORMALIZE_TOF)
    val_set   = HybridDatasetFast(DATA_DIR / 'val_hybrid_nakagami4M.npz',
                                  normalize_tof=NORMALIZE_TOF)

    # データが Dbar/Pbar 平衡なのでただ shuffle で OK
    train_labels = train_set.labels
    n_pbar = int((train_labels == 0).sum())
    n_dbar = int((train_labels == 1).sum())
    print(f'train label counts: Pbar={n_pbar:,}, Dbar={n_dbar:,}')

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    # ── NakagamiNet (A.1 架构, TOF 9次元, dropout 0.1) ──
    model = NakagamiNet(tof_dim=9, dropout=DROPOUT).to(DEVICE)
    # ★ 厳密復元のため DataParallel / torch.compile / AMP は使わない

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

    # ── EarlyStopping: monitor=val_acc, mode=max ──
    best_val_acc = -float('inf')
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

        # ── monitor=val_accuracy, mode=max ──
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  → best model saved (val_acc={v_acc:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'  → early stopping at epoch {epoch} (patience={PATIENCE})')
                break


if __name__ == '__main__':
    train()





