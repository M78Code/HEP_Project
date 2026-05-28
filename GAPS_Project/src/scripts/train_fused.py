"""CNN+GravNet 融合模型训练脚本"""

import time
import torch
from pathlib import Path
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import GAPS_Project

from GAPS_Project.src.data_parse.data_loader import make_data_loaders_from_split
from GAPS_Project.src.models.fused_model import FusedGravNet
from GAPS_Project.src.losses import FocalLoss

PROJECT_ROOT = Path(GAPS_Project.__file__).parent

# ── 设备 ──
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f'使用设备：{DEVICE}')

# ── 超参数 ──
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEP_SIZE = 15
GAMMA = 0.5
FOCAL_GAMMA = 1.5
LAZY_LOAD = False
NUM_WORKERS = 24

def train():
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    print('加载数据集...')
    train_loader, val_loader, _ = make_data_loaders_from_split(
        split_dir=split_dir, batch_size=BATCH_SIZE, lazy=LAZY_LOAD,
        num_workers=NUM_WORKERS, persistent_workers=True)
    print(f"train batches: {len(train_loader)}")
    print(f"val   batches: {len(val_loader)}")

    model = FusedGravNet(
        in_channels=8, hidden_dim=128, graph_feat_dim=45,
        num_blocks=6, cnn_feat_dim=128, dropout=0.3,
    ).to(DEVICE)

    print(f"\n模型: FusedGravNet (GravNet_6b_h128 + CNN encoder)")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / 'results' / f"{timestamp}_FusedGravNet"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float('inf')
    best_model_path = log_dir / f'{timestamp}_FusedGravNet_best.pth'

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{EPOCHS} [train]', leave=False)
        for batch in train_bar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            graph_feat = torch.cat([
                batch.n_hits.view(-1, 1),
                batch.total_energy.view(-1, 1),
                batch.sili_profile.view(-1, 16),
                batch.tof_profile.view(-1, 16),
                batch.tof_feat.view(-1, 11),
            ], dim=1)  # (B, 45)

            voxel = torch.log1p(batch.voxel.view(-1, 1, 10, 20, 20))  # log1p 标准化

            logits = model(batch.x, batch.edge_index, batch.batch,
                           graph_feat=graph_feat, voxel=voxel)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch.y.squeeze()).sum().item()
            total_samples += batch.num_graphs
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch:3d}/{EPOCHS} [val]  ', leave=False):
                batch = batch.to(DEVICE)
                graph_feat = torch.cat([
                    batch.n_hits.view(-1, 1),
                    batch.total_energy.view(-1, 1),
                    batch.sili_profile.view(-1, 16),
                    batch.tof_profile.view(-1, 16),
                    batch.tof_feat.view(-1, 11),
                ], dim=1)
                voxel = torch.log1p(batch.voxel.view(-1, 1, 10, 20, 20))  # log1p 标准化

                logits = model(batch.x, batch.edge_index, batch.batch,
                               graph_feat=graph_feat, voxel=voxel)
                loss = criterion(logits, batch.y.squeeze())
                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch.y.squeeze()).sum().item()
                val_samples += batch.num_graphs

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        scheduler.step()
        elapsed = time.time() - epoch_start

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  → best model saved (val_loss={best_val_loss:.4f})")

    writer.close()
    print(f"\n训练完成，最优模型: {best_model_path}")


if __name__ == '__main__':
    train()