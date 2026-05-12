import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.optim import Adam
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import Scintillator_Project
from Scintillator_Project.src.data_parse.data_loader import  make_dataloaders
from Scintillator_Project.src.models.resnet1d import ResNet1D

# 项目根目录（Scintillator_Project/），所有路径基于此
PROJECT_ROOT = Path(Scintillator_Project.__file__).parent


# 设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f'使用设备：{device}')

EPOCHS = 300
BATCH_SIZE = 64
STEP_SIZE = 75
LEARNING_RATE = 3e-4

def log(msg: str, log_file):
    """同时输出到终端和日志文件"""
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

def train():
    # ── 日志文件初始化 ────────────────────────
    logs_dir = PROJECT_ROOT / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = logs_dir / f'{timestamp}.log'

    with open(log_path, 'w', encoding='utf-8') as log_file:
        # 记录训练配置
        log(f'使用设备：{device}', log_file)
        log(f"EPOCHS={EPOCHS} | BATCH_SIZE={BATCH_SIZE} | LR={LEARNING_RATE} | "
            f"scheduler=StepLR(step={STEP_SIZE}, gamma=0.7)", log_file)
        log("", log_file)

        # wandb.init(project="scintillator", name='cnn1d_baseline')
        # DataLoader
        split_dir = PROJECT_ROOT / 'dataset' / 'split'
        train_loader, val_loader, _ = make_dataloaders(split_dir, batch_size=BATCH_SIZE)

        # 计算位置权重（稀疏位置权重更高）
        train_labels = train_loader.dataset.labels # numpy array of all train positions
        positions, counts = np.unique(train_labels, return_counts=True)
        max_count = float(counts.max())
        weight_map = {float(pos): max_count / float(cnt) for pos, cnt in zip(positions, counts)}

        # 模型
        model = ResNet1D().to(device)

        # ── 损失函数・优化器・学习率调度 ────────
        # MSELoss：输出单位 cm²，开根号后为实际误差（cm）
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
        # 每STEP_SIZE个epoch将学习率乘以0.7，避免过早停止学习
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.7)

        # ── 保存路径・TensorBoard ──────────────
        save_path = PROJECT_ROOT / 'results' / f'{timestamp}_best_model.pth'
        save_path.parent.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=PROJECT_ROOT / 'runs')
        best_val_loss = float('inf')

        for epoch in range(1, EPOCHS + 1):
            # ── 训练 ──────────────────────────────────
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1) # [batch] -> [batch, 1]

                optimizer.zero_grad()
                pred = model(x)
                w = torch.tensor([weight_map[float(yi)] for yi in y.squeeze().tolist()], dtype=torch.float32, device=device)
                loss = (w * (pred.squeeze(1) - y.squeeze(1)) ** 2).mean()
                # loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(x)

            train_loss /= len(train_loader.dataset)
            writer.add_scalar('Loss/train', train_loss, epoch)

            # ── 验证 ──────────────────────────────────
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device).unsqueeze(1)
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_loss += loss.item() * len(x)

            val_loss /= len(val_loader.dataset)
            writer.add_scalar('Loss/val', val_loss, epoch)

            # ── 学习率衰减 ────────────────────────────
            scheduler.step()

            # ── 最优模型保存 ──────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                mark = " ← 保存"
            else:
                mark = ""

            log(f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"lr: {scheduler.get_last_lr()[0]:.6f}{mark}", log_file)

        writer.close()
        log("", log_file)
        log(f"训练完成。最优 val_loss: {best_val_loss:.4f}", log_file)
        log(f"最优 RMSE: {best_val_loss ** 0.5:.4f} cm", log_file)
        log(f"模型保存于: {save_path}", log_file)
        log(f"日志保存于: {log_path}", log_file)



if __name__ == "__main__":
    train()
