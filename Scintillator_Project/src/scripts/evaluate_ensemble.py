import torch
import numpy as np
import re
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

# Ensemble 5个模型
ENSEMBLE_MODELS = [
    '20260508-123346_best_model.pth',  # seed=42
    '20260508-123517_best_model.pth',  # seed=123
    '20260508-123648_best_model.pth',  # seed=456
    '20260508-123820_best_model.pth',  # seed=789
    '20260508-123954_best_model.pth',  # seed=1024
]


def evaluate_ensemble():
    # ── 1. 加载测试数据 ──────────────────────────────
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    _, _, test_loader = make_dataloaders(split_dir, batch_size=BATCH_SIZE)

    # ── 2. 加载5个模型 ──────────────────────────────
    models = []
    for name in ENSEMBLE_MODELS:
        path = PROJECT_ROOT / 'results' / name
        m = CNN1D().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models.append(m)
        print(f'加载模型：{name}')

    # ── 3. 推理（多模型取平均） ────────────────────────────────
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            batch_preds = [m(x).squeeze(1).cpu().numpy() for m in models]
            avg_pred = np.mean(batch_preds, axis=0)  # 5个模型预测取平均
            all_preds.append(avg_pred)
            all_labels.append(y.numpy())

    # ── 4. 拼接所有 batch ────────────────────────────────
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ── 5. 计算整体评估指标 ──────────────────────────────
    mae    = np.mean(np.abs(all_preds - all_labels))
    rmse   = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    ss_res = np.sum((all_preds - all_labels) ** 2)
    ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
    r2     = 1 - (ss_res / ss_tot)

    # ── 6. 打印整体结果 ──────────────────────────────
    print(f'\n===== Ensemble Test Set 评估结果 =====')
    print(f'MAE: {mae:.4f} cm')
    print(f'RMSE: {rmse:.4f} cm')
    print(f'R²: {r2:.4f}')

    # ── 7. 按"位置"统计误差 ────────────────────────────
    print(f'\n===== 各位置误差 =====')
    positions = sorted(np.unique(all_labels))
    for pos in positions:
        mask = all_labels == pos
        pos_rmse = np.sqrt(np.mean((all_preds[mask] - all_labels[mask]) ** 2))
        print(f" {pos:5.0f} cm | RMSE: {pos_rmse:.4f} cm | N: {mask.sum()}")

    # ── 8. 保存结果到文件 ──────────────────────────────
    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_path = results_dir / f"{timestamp}_ensemble_test_results.txt"

    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f'Ensemble models: {len(ENSEMBLE_MODELS)}\n\n')
        f.write(f'===== Ensemble Test Set 评估结果 =====\n')
        f.write(f"MAE: {mae:.4f} cm\n")
        f.write(f"RMSE: {rmse:.4f} cm\n")
        f.write(f"R²:   {r2:.4f}\n\n")
        f.write(f'===== 各位置误差 =====\n')
        for pos in positions:
            mask = all_labels == pos
            pos_rmse = np.sqrt(np.mean((all_preds[mask] - all_labels[mask]) ** 2))
            f.write(f' {pos:5.0f} cm | RMSE: {pos_rmse:.4f} cm | N: {mask.sum()}\n')

    print(f'\n结果保存于：{result_path}')


if __name__ == '__main__':
    evaluate_ensemble()
