# Step 4: 训练脚本

import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import GAPS_Project

from GAPS_Project.src.data_parse.data_loader import make_data_loaders_from_split
from GAPS_Project.src.models.gnn_base import GINClassifier
from GAPS_Project.src.models.gravnet import GravNetClassifier
from GAPS_Project.src.models.dgcnn import DGCNNClassifier
from GAPS_Project.src.losses import FocalLoss

PROJECT_ROOT = Path(GAPS_Project.__file__).parent

# ── 设备 ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f'使用设备：{DEVICE}')

# ── 超参数 ─────────────────────────────────────────────
MODEL_NAME = 'GravNet'  # 'GIN' | 'GravNet' | 'DGCNN'
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
STEP_SIZE = 15  # StepLR: 每10个epoch衰减一次
GAMMA = 0.5
FOCAL_GAMMA = 1.5  # Focal Loss γ（IceCube 2025论文最优值）

LAZY_LOAD = False  # 服务器设False，Mac设True

IN_CHANNEL = 9


def get_model(name: str):
    if name == 'GIN':
        return GINClassifier(in_channels=IN_CHANNEL, hidden_dim=64)
    elif name == 'GravNet':
        return GravNetClassifier(in_channels=IN_CHANNEL, hidden_dim=64, graph_feat_dim=2)
    elif name == 'DGCNN':
        return DGCNNClassifier(in_channels=IN_CHANNEL, hidden_dim=64, k=8, graph_feat_dim=2)
    else:
        raise ValueError(f'Unknown model: {name}')


def train():
    # ── 1. 数据加载 ────────────────────────────────────
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    print('加载数据集...')
    train_loader, val_loader, test_loader = make_data_loaders_from_split(split_dir=split_dir, batch_size=BATCH_SIZE,
                                                                         lazy=LAZY_LOAD)
    print(f"train batches: {len(train_loader)}")
    print(f"val   batches: {len(val_loader)}")
    print(f"test  batches: {len(test_loader)}")

    # ── 2. 模型 ────────────────────────────────────────
    model = get_model(MODEL_NAME).to(DEVICE)
    print(f"\n模型: {MODEL_NAME}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. 损失函数 / 优化器 / scheduler ───────────────
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # ── 4. TensorBoard ─────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / "results" / f"{timestamp}_{MODEL_NAME}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # ── 5. 训练循环 ────────────────────────────────────
    best_val_loss = float('inf')
    best_model_path = log_dir / f'{timestamp}_{MODEL_NAME}_best.pth'

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            graph_feat = torch.cat([batch.n_hits.view(-1, 1), batch.total_energy.view(-1, 1)], dim=1)
            logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch.y.squeeze()).sum().item()
            total_samples += batch.num_graphs

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Validation
        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                graph_feat = torch.cat([batch.n_hits.view(-1, 1), batch.total_energy.view(-1, 1)], dim=1)
                logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
                loss = criterion(logits, batch.y.squeeze())

                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch.y.squeeze()).sum().item()
                val_samples += batch.num_graphs

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples

        scheduler.step()

        # ── logging ────────────────────────────────────
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        # ── 保存最优模型 ────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  → best model saved (val_loss={best_val_loss:.4f})")

    writer.close()
    print(f"\n训练完成，最优模型: {best_model_path}")


def resume_train():
    RESUME_FROM = PROJECT_ROOT / 'results/20260517-070152_GravNet/20260517-070152_GravNet_best.pth'
    RESUME_EPOCH = 50
    TOTAL_EPOCHS = 80

    # ── 1. 数据加载 ────────────────────────────────────
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    print('加载数据集...')
    train_loader, val_loader, _ = make_data_loaders_from_split(
        split_dir=split_dir, batch_size=BATCH_SIZE, lazy=LAZY_LOAD)

    # ── 2. 模型 + 加载权重 ─────────────────────────────
    model = get_model(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(RESUME_FROM, map_location=DEVICE))
    print(f"已加载权重: {RESUME_FROM}")
    print(f"从 Epoch {RESUME_EPOCH + 1} 继续训练至 Epoch {TOTAL_EPOCHS}")

    # ── 3. 损失函数 / 优化器 / scheduler ───────────────
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # 恢复scheduler时需要手动设置initial_lr
    for group in optimizer.param_groups:
        group['initial_lr'] = LEARNING_RATE
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA, last_epoch=RESUME_EPOCH)

    # ── 4. TensorBoard ─────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / "results" / f"{timestamp}_{MODEL_NAME}_resume"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # ── 5. 训练循环 ────────────────────────────────────
    best_val_loss = float('inf')
    best_model_path = log_dir / f'{timestamp}_{MODEL_NAME}_resume_best.pth'

    for epoch in range(RESUME_EPOCH + 1, TOTAL_EPOCHS + 1):
        # Train
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            graph_feat = torch.cat([batch.n_hits.view(-1, 1), batch.total_energy.view(-1, 1)], dim=1)
            logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch.y.squeeze()).sum().item()
            total_samples += batch.num_graphs

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Validation
        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                graph_feat = torch.cat([batch.n_hits.view(-1, 1), batch.total_energy.view(-1, 1)], dim=1)
                logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
                loss = criterion(logits, batch.y.squeeze())

                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch.y.squeeze()).sum().item()
                val_samples += batch.num_graphs

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        scheduler.step()

        print(f"Epoch {epoch:3d}/{TOTAL_EPOCHS} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

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

if __name__ == "__main__":
    #train()
    resume_train()
