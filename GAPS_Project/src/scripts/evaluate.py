# Step 5: 性能评估

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import platform

# 自动判断系统并设置字体
system_name = platform.system()

if system_name == "Darwin":  # macOS
    plt.rcParams['font.sans-serif'] = [
        'PingFang SC',
        'Heiti TC',
        'Microsoft YaHei',
        'SimHei',
        'Arial Unicode MS',
        'DejaVu Sans'
    ]

elif system_name == "Linux":  # Ubuntu / Linux
    plt.rcParams['font.sans-serif'] = [
        'Noto Sans CJK JP',
        'Noto Sans CJK SC',
        'WenQuanYi Zen Hei',
        'DejaVu Sans'
    ]

elif system_name == "Windows":  # Windows
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei',
        'SimHei',
        'Arial Unicode MS',
        'DejaVu Sans'
    ]

else:  # 其他系统兜底
    plt.rcParams['font.sans-serif'] = [
        'DejaVu Sans'
    ]

# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 全局字体大小
plt.rcParams['font.size'] = 12

import GAPS_Project

from GAPS_Project.src.data_parse.data_loader import make_data_loaders_from_split
from GAPS_Project.src.models.gnn_base import GINClassifier
from GAPS_Project.src.models.gravnet import GravNetClassifier
from GAPS_Project.src.models.dgcnn import DGCNNClassifier

PROJECT_ROOT = Path(GAPS_Project.__file__).parent

# ── 配置 ──────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f'使用设备：{DEVICE}')

BATCH_SIZE = 256
LAZY_LOAD = False

# 要评估的模型列表：(模型名, 权重路径)
EVAL_MODELS = [
    ('GIN', PROJECT_ROOT / 'results/20260513-140442_GIN/20260513-140442_GIN_best.pth'),
    ('GravNet', PROJECT_ROOT / 'results/20260513-214452_GravNet/20260513-214452_GravNet_best.pth'),
    ('DGCNN', PROJECT_ROOT / 'results/20260514-104133_DGCNN/20260514-104133_DGCNN_best.pth'),
]


def get_model(name: str):
    if name == 'GIN':
        return GINClassifier(in_channels=5, hidden_dim=64)
    elif name == 'GravNet':
        return GravNetClassifier(in_channels=5, hidden_dim=64)
    elif name == 'DGCNN':
        return DGCNNClassifier(in_channels=5, hidden_dim=64, k=8)
    else:
        raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def run_inference(model, loader, device):
    """返回（all_labels, all_probs）numpy arrays"""
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        batch = batch.to(device=device)
        logits = model(batch.x, batch.edge_index, batch.batch) # [batch, 2] 原始分数
        probs = torch.softmax(logits, dim=1)[:, 1]  # 类别1（antiD）的概率
        all_labels.append(batch.y.squeeze().cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def print_metrics(name, labels, probs):
    """计算评价指标"""
    preds = (probs >= 0.5).astype(int) # 如果antiD概率 >= 50%, 就判断它是antiD, 否则为antiP
    acc = accuracy_score(labels, preds) # 总体正确率
    prec = precision_score(labels, preds, zero_division=0) # 你说是antiD的里面，到底有多少是真的，即纯度（Purity）
    rec = recall_score(labels, preds, zero_division=0) # 真正的antiD，你找回来了多少，即信号效率（Signal Efficiency）
    f1 = f1_score(labels, preds, zero_division=0) # 综合Precision + Recall，平衡指标
    auc = roc_auc_score(labels, probs) # 模型整体分类能力
    cm = confusion_matrix(labels, preds) # 混淆矩阵（Confusion Matrix）
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'=' * 50}")
    print(f"模型: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}  (antiD识别精确率)")
    print(f"  Recall   : {rec:.4f}  (antiD信号效率)")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {auc:.4f}")
    print(f"  混淆矩阵: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  背景抑制率(Rejection): {tn / (tn + fp):.4f}  ({tn + fp}个antiP中正确拒绝{tn}个)")
    return auc


def plot_roc_curves(results, save_path):
    """results: list of (name, labels, probs)"""
    plt.figure(figsize=(7, 6))
    for name, labels, probs in results:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('False Positive Rate (antiP误识别率)')
    plt.ylabel('True Positive Rate (antiD信号效率)')
    plt.title('ROC Curve - antiD vs antiP')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'\nROC曲线已保存：{save_path}')


def plot_rejection_curve(results, save_path):
    """
    HEP标准：x轴=信号效率（antiD TPR）, y轴=背景抑制率（1/FPR）
    results：list of (name, labels, probs)
    """
    plt.figure(figsize=(7, 6))
    for name, labels, probs in results:
        fpr, tpr, _ = roc_curve(labels, probs)
        # 避免除以0
        fpr_safe = np.where(fpr == 0, 1e-10, fpr)
        rejection = 1.0 / fpr_safe
        plt.semilogy(tpr, rejection, label=name)

    plt.xlabel('Signal Efficiency (antiD recall)')
    plt.ylabel('Background Rejection (1 / FPR)')
    plt.title('Rejection Curve - antiD Signal vs antiP Background')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Rejection曲线已保存: {save_path}")


def print_rejection_at_efficiency(results, signal_efficiencies=(0.50, 0.80, 0.90, 0.95, 0.98, 0.99)):
    """输出指定信号效率下各模型的背景抑制率"""
    print(f"\n{'='*60}")
    print('各信号效率下的背景抑制率 （Background Rejection = 1/FPR）')
    print(f"{'Signal Eff':>12}", end="")
    for name, _, _ in results:
        print(f' {name:>12}', end="")
    print()
    print('-' * 60)

    for target_eff in signal_efficiencies:
        print(f' {target_eff:>12.2f}', end="")
        for name, labels, probs in results:
            fpr, tpr, _ = roc_curve(labels, probs)
            # 找最接近 target_eff 的 tpr对应的fpr
            idx = np.argmin(np.abs(tpr - target_eff))
            fpr_at_etf = fpr[idx]
            if fpr_at_etf == 0:
                rejection_str = '>1e10'
            else:
                rejection = 1.0 / fpr_at_etf
                rejection_str = f'{rejection:.2e}'
            print(f"  {rejection_str:>12}", end="")
        print()


def evaluate():
    # 数据加载（只用test集）
    split_dir = PROJECT_ROOT / 'dataset' / 'split'
    print('加载test数据集...')
    _, _, test_loader = make_data_loaders_from_split(split_dir=split_dir, batch_size=BATCH_SIZE, lazy=LAZY_LOAD)
    print(f'test batches: {len(test_loader)}')

    # 输出目录
    out_dir = PROJECT_ROOT / 'results' / 'evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, weight_path in EVAL_MODELS:
        print(f'\n加载模型 {name}: {weight_path}')
        model = get_model(name).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

        labels, probs = run_inference(model, test_loader, DEVICE)
        print_metrics(name, labels, probs)
        results.append((name, labels, probs))

    # 绘图
    plot_roc_curves(results, out_dir / "roc_curves.png")
    plot_rejection_curve(results, out_dir / "rejection_curve.png")
    print(f"\n所有结果保存至: {out_dir}")

    plot_rejection_curve(results)


if __name__ == '__main__':
    evaluate()
