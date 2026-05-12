import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from datetime import datetime


import Scintillator_Project
from Scintillator_Project.src.data_parse.data_loader import make_dataloaders
from Scintillator_Project.src.models.cnn1d import CNN1D

# 项目根目录
PROJECT_ROOT = Path(Scintillator_Project.__file__).parent

# 设备选择
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f'使用设备：{DEVICE}')

BATCH_SIZE = 64


def evaluate():
    # ── 1. 加载测试数据 ──────────────────────────────
    split_dir = PROJECT_ROOT / 'dataset' / 'split'      # 数据集划分目录（train/val/test）
    _, _, test_loader = make_dataloaders(split_dir, batch_size=BATCH_SIZE)  # 只取 test_loader

    # ── 2. 加载训练好的模型 ──────────────────────────
    model = CNN1D().to(DEVICE)  # 创建模型并放到 GPU/CPU
    # 自动找 results/ 下时间戳最新的 best_model.pth（只匹配数字时间戳格式），找不到则用旧模型
    import re
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--best', action='store_true', help='load best model result 4.78cm')
    args = parser.parse_args()

    candidates = sorted(
        [p for p in (PROJECT_ROOT / 'results').glob('*_best_model.pth')
         if re.match(r'^\d{8}-\d{6}_best_model\.pth$', p.name)]
    )

    if args.best:
        model_path = PROJECT_ROOT / 'results' / '20260506-142427_best_model.pth'
    else:
        model_path = candidates[-1] if candidates else PROJECT_ROOT / 'results' / 'best_model.pth'

    print(f'加载模型：{model_path.name}')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    # 加载模型参数（map_location 保证 GPU/CPU 兼容）

    model.eval()    # 切换为评估模式（关闭 dropout / 固定 BN）

    # ── 3. 推理（预测） ────────────────────────────────
    all_preds = []      # 保存所有预测结果
    all_labels = []     # 保存所有真实标签

    with torch.no_grad():   # 不计算梯度（节省显存 + 加速）
        for x, y in test_loader:    # 遍历测试集
            x = x.to(DEVICE)
            pred = model(x)   # 输入放到设备上（GPU/CPU）

            # squeeze(1): [batch,1] → [batch]
            # cpu(): GPU → CPU
            # numpy(): Tensor → numpy
            all_preds.append(pred.squeeze(1).cpu().numpy())

            # 标签一般在 CPU，直接转 numpy
            all_labels.append(y.numpy())

    # ── 4. 拼接所有 batch ────────────────────────────────
    all_preds = np.concatenate(all_preds) # shape: (N,)
    all_labels = np.concatenate(all_labels) # shape: (N,)

    # ── 5. 计算整体评估指标 ──────────────────────────────
    mae = np.mean(np.abs(all_preds - all_labels)) # 计算 MAE，平均绝对误差，预测值和真实值差多少，取绝对值，再求平均
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2)) # rmse: 均方根误差（对大误差更敏感）
    ss_res = np.sum((all_preds - all_labels) ** 2) # 残差平方和
    ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2) # 总方差
    r2 = 1 - (ss_res / ss_tot) # 拟合优度（越接近1越好）

    # ── 6. 打印整体结果 ──────────────────────────────
    print(f'\n===== Test Set 评估结果 =====')
    print(f'MAE: {mae:.4f} cm')
    print(f'RMSE: {rmse:.4f} cm')
    print(f'R²: {r2:.4f} ')

    # ── 7. 按“位置”统计误差 ────────────────────────────
    # 按每个位置单独统计 RMSE，找出哪些位置预测难
    print(f'\n===== 各位置误差 =====')
    positions = sorted(np.unique(all_labels))
    # 获取所有不同位置（例如 15cm, 25cm...）

    for pos in positions:
        mask = all_labels == pos
        # 筛选当前这个位置的数据（布尔掩码）

        pos_rmse = np.sqrt(np.mean((all_preds[mask] - all_labels[mask]) ** 2))
        # 只计算这个位置的 RMSE

        print(f" {pos:5.0f} cm | RMSE: {pos_rmse:.4f} cm | N: {mask.sum()}")
        # N 表示该位置样本数量

    # ── 8. 保存结果到文件 ──────────────────────────────
    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_path = results_dir / f"{timestamp}_test_results.txt"

    with open(result_path, 'w', encoding='utf-8') as f:
        # 写入整体指标
        f.write(f'===== Test Set 评估结果 =====\n')
        f.write(f"MAE: {mae:.4f} cm\n")
        f.write(f"RMSE: {rmse:.4f} cm\n")
        f.write(f"R²:   {r2:.4f}\n\n")

        # 写入各位置误差
        f.write(f'===== 各位置误差 =====\n')
        for pos in positions:
            mask = all_labels == pos
            pos_rmse = np.sqrt(np.mean((all_preds[mask] - all_labels[mask]) ** 2))
            f.write(f' {pos:5.0f} cm | RMSE: {pos_rmse:.4f} cm | N: {mask.sum()}\n')

    print(f'\n结果保存于：{result_path}')

if __name__ == '__main__':
    evaluate()


