# Step 4: 训练脚本

import time
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import GAPS_Project

from GAPS_Project.src.data_parse.data_loader import make_data_loaders_from_split
from GAPS_Project.src.models.gnn_base import GINClassifier
from GAPS_Project.src.models.gravnet import GravNetClassifier
from GAPS_Project.src.models.dgcnn import DGCNNClassifier
from GAPS_Project.src.models.dnn_baseline import DNNBaseline
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
MODEL_NAME = 'DGCNN'  # 'GIN' | 'GravNet' | 'DGCNN'  ← DGCNN训练时用此配置
EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
STEP_SIZE = 15  # StepLR: 每10个epoch衰减一次
GAMMA = 0.5
FOCAL_GAMMA = 1.5  # Focal Loss γ（IceCube 2025论文最优值）

LAZY_LOAD = True
NUM_WORKERS = 0
PATIENCE = 10  # early stopping

IN_CHANNEL = 8


def get_model(name: str, num_blocks: int = 4):
    if name == 'GIN':
        return GINClassifier(in_channels=IN_CHANNEL, hidden_dim=64)
    elif name == 'GravNet':
        return GravNetClassifier(in_channels=IN_CHANNEL, hidden_dim=64, graph_feat_dim=45, num_blocks=num_blocks)
    elif name == 'DGCNN':
        return DGCNNClassifier(in_channels=IN_CHANNEL, hidden_dim=64, k=8, graph_feat_dim=45)
    else:
        raise ValueError(f'Unknown model: {name}')


def train():
    # ── 1. 数据加载 ────────────────────────────────────
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    print('加载数据集...')
    train_loader, val_loader, test_loader = make_data_loaders_from_split(
        split_dir=split_dir, batch_size=BATCH_SIZE, lazy=LAZY_LOAD,
        num_workers=NUM_WORKERS, persistent_workers=True, use_voxel=False)
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
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        # Train
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
            logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
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

        # Validation
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
                ], dim=1)  # (B, 45)
                logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
                loss = criterion(logits, batch.y.squeeze())

                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch.y.squeeze()).sum().item()
                val_samples += batch.num_graphs

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples

        scheduler.step()
        elapsed = time.time() - epoch_start

        # ── logging ────────────────────────────────────
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        # ── 保存最优模型 ────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  → early stopping: val_loss未改善已达{PATIENCE}个epoch")
                break

    writer.close()
    print(f"\n训练完成，最优模型: {best_model_path}")


def resume_train():
    RESUME_FROM = PROJECT_ROOT / 'results/20260517-235638_GravNet/20260517-235638_GravNet_best.pth'
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
            graph_feat = torch.cat([
                batch.n_hits.view(-1, 1),
                batch.total_energy.view(-1, 1),
                batch.sili_profile.view(-1, 16),
                batch.tof_profile.view(-1, 16),
                batch.tof_feat.view(-1, 11),
            ], dim=1)  # (B, 45)
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
                graph_feat = torch.cat([
                    batch.n_hits.view(-1, 1),
                    batch.total_energy.view(-1, 1),
                    batch.sili_profile.view(-1, 16),
                    batch.tof_profile.view(-1, 16),
                    batch.tof_feat.view(-1, 11),
                ], dim=1)  # (B, 45)
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


def train_narrow_beta():
    """在 β∈[0.335,0.340] 窄窗口数据上训练，与 Wada 2019 直接对比"""
    split_dir = PROJECT_ROOT / 'dataset' / 'split_narrow_beta'
    print('加载窄β数据集...')
    train_loader, val_loader, test_loader = make_data_loaders_from_split(
        split_dir=split_dir, batch_size=BATCH_SIZE, lazy=LAZY_LOAD)
    print(f"train batches: {len(train_loader)}")
    print(f"val   batches: {len(val_loader)}")

    model = get_model('GravNet').to(DEVICE)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / "results" / f"{timestamp}_GravNet_narrow_beta"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float('inf')
    best_model_path = log_dir / f'{timestamp}_GravNet_narrow_beta_best.pth'

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            graph_feat = torch.cat([
                batch.n_hits.view(-1, 1),
                batch.total_energy.view(-1, 1),
                batch.sili_profile.view(-1, 16),
                batch.tof_profile.view(-1, 16),
                batch.tof_feat.view(-1, 11),
            ], dim=1)  # (B, 45)
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

        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                graph_feat = torch.cat([
                    batch.n_hits.view(-1, 1),
                    batch.total_energy.view(-1, 1),
                    batch.sili_profile.view(-1, 16),
                    batch.tof_profile.view(-1, 16),
                    batch.tof_feat.view(-1, 11),
                ], dim=1)  # (B, 45)
                logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
                loss = criterion(logits, batch.y.squeeze())
                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch.y.squeeze()).sum().item()
                val_samples += batch.num_graphs

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  → best model saved (val_loss={best_val_loss:.4f})")

    writer.close()
    print(f"\n训练完成，最优模型: {best_model_path}")


def train_ablation():
    """消融实验：去掉MC stopping特征（6维），graph_feat_dim=45"""
    split_dir = PROJECT_ROOT / "dataset" / "split"
    print('加载数据集...')
    train_loader, val_loader, test_loader = make_data_loaders_from_split(
        split_dir=split_dir, batch_size=BATCH_SIZE, lazy=LAZY_LOAD
    )
    print(f'train batches: {len(train_loader)}')
    print(f"val   batches: {len(val_loader)}")
    model = GravNetClassifier(in_channels=IN_CHANNEL, hidden_dim=64, graph_feat_dim=45).to(DEVICE)
    print(f"\n模型: GravNet_ablation（graph_feat_dim=45）")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / "results" / f"{timestamp}_GravNet_ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float('inf')
    best_model_path = log_dir / f'{timestamp}_GravNet_ablation_best.pth'

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            graph_feat = torch.cat([
                batch.n_hits.view(-1, 1),
                batch.total_energy.view(-1, 1),
                batch.sili_profile.view(-1, 16),
                batch.tof_profile.view(-1, 16),
                batch.tof_feat.view(-1, 11),
            ], dim=1)  # (B, 45) — 无stopping_feat
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

        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                graph_feat = torch.cat([
                    batch.n_hits.view(-1, 1),
                    batch.total_energy.view(-1, 1),
                    batch.sili_profile.view(-1, 16),
                    batch.tof_profile.view(-1, 16),
                    batch.tof_feat.view(-1, 11),
                ], dim=1)  # (B, 45)
                logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
                loss = criterion(logits, batch.y.squeeze())
                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch.y.squeeze()).sum().item()
                val_samples += batch.num_graphs

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
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

def train_dnn_baseline():
    """DNN基线：仅使用45维graph_feat，无GNN图结构"""
    split_dir = PROJECT_ROOT / "dataset" / "split"
    print('加载数据集...')
    train_loader, val_loader, _ = make_data_loaders_from_split(
        split_dir=split_dir, batch_size=BATCH_SIZE, lazy=LAZY_LOAD)

    model = DNNBaseline(graph_feat_dim=45, hidden_dim=128).to(DEVICE)
    print(f"\n模型: DNNBaseline（无图结构，仅graph_feat 45维）")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / "results" / f"{timestamp}_DNNBaseline"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float('inf')
    best_model_path = log_dir / f'{timestamp}_DNNBaseline_best.pth'

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            graph_feat = torch.cat([
                batch.n_hits.view(-1, 1),
                batch.total_energy.view(-1, 1),
                batch.sili_profile.view(-1, 16),
                batch.tof_profile.view(-1, 16),
                batch.tof_feat.view(-1, 11),
            ], dim=1)  # (B, 45)
            logits = model(graph_feat=graph_feat)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch.y.squeeze()).sum().item()
            total_samples += batch.num_graphs

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                graph_feat = torch.cat([
                    batch.n_hits.view(-1, 1),
                    batch.total_energy.view(-1, 1),
                    batch.sili_profile.view(-1, 16),
                    batch.tof_profile.view(-1, 16),
                    batch.tof_feat.view(-1, 11),
                ], dim=1)  # (B, 45)
                logits = model(graph_feat=graph_feat)
                loss = criterion(logits, batch.y.squeeze())
                val_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch.y.squeeze()).sum().item()
                val_samples += batch.num_graphs

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
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

def train_deeper_gravnet(num_blocks: int = 6, hidden_dim: int = 64):
    """测试不同深度/宽度GravNet，num_blocks可选4/6/8，hidden_dim可选64/128"""
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    print('加载数据集...')
    train_loader, val_loader, _ = make_data_loaders_from_split(
        split_dir=split_dir, batch_size=BATCH_SIZE, lazy=LAZY_LOAD,
        num_workers=NUM_WORKERS, persistent_workers=True, use_voxel=False)

    exp_name = f"GravNet_{num_blocks}blocks_h{hidden_dim}"
    model = GravNetClassifier(in_channels=IN_CHANNEL, hidden_dim=hidden_dim, graph_feat_dim=45, num_blocks=num_blocks).to(DEVICE)
    print(f"\n模型: {exp_name}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / "results" / f"{timestamp}_{exp_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float('inf')
    best_model_path = log_dir / f'{timestamp}_{exp_name}_best.pth'
    patience_counter = 0

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
            logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
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
                ], dim=1)  # (B, 45)
                logits = model(batch.x, batch.edge_index, batch.batch, graph_feat=graph_feat)
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
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  → early stopping: val_loss未改善已达{PATIENCE}个epoch")
                break

    writer.close()
    print(f"\n训练完成，最优模型: {best_model_path}")
    return best_model_path


def train_all_combinations():
    """顺序训练所有待对比的深度/宽度组合"""
    combos = [
        (4, 128),   # 宽版v2：更宽但同深度
        (6, 128),   # 宽版6块
        (8, 64),    # 更深8块
    ]
    for num_blocks, hidden_dim in combos:
        print(f"\n{'=' * 60}")
        print(f"开始训练: num_blocks={num_blocks}, hidden_dim={hidden_dim}")
        print(f"{'=' * 60}")
        best_path = train_deeper_gravnet(num_blocks=num_blocks, hidden_dim=hidden_dim)
        print(f"✓ 完成: {best_path}\n")


if __name__ == "__main__":
    train()                                              # 1. DGCNN（MODEL_NAME='DGCNN'）
    # train_deeper_gravnet(num_blocks=6, hidden_dim=128)  # 2. GravNet_6b_h128（完成后自动继续）
    # resume_train()
    # train_narrow_beta()
    # train_ablation()
    # train_dnn_baseline()
    # train_all_combinations()
