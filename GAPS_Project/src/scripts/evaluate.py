# Step 5: 性能评估

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, precision_recall_curve)
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

# 要评估的模型列表：(模型名, 权重路径, in_channels)
EVAL_MODELS = [
    ('GIN',      PROJECT_ROOT / 'results/20260513-140442_GIN/20260513-140442_GIN_best.pth',          5),
    ('GravNet',  PROJECT_ROOT / 'results/20260513-214452_GravNet/20260513-214452_GravNet_best.pth',  5),
    ('DGCNN',    PROJECT_ROOT / 'results/20260514-104133_DGCNN/20260514-104133_DGCNN_best.pth',      5),
    ('DGCNN_v2', PROJECT_ROOT / 'results/20260515-151601_DGCNN/20260515-151601_DGCNN_best.pth',      6),
]


def get_model(name: str, in_channels: int = 6):
    base_name = name.replace('_v2', '')
    if base_name == 'GIN':
        return GINClassifier(in_channels=in_channels, hidden_dim=64)
    elif base_name == 'GravNet':
        return GravNetClassifier(in_channels=in_channels, hidden_dim=64)
    elif base_name == 'DGCNN':
        return DGCNNClassifier(in_channels=in_channels, hidden_dim=64, k=8)
    else:
        raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def run_inference(model, loader, device):
    """返回（all_labels, all_probs）numpy arrays"""
    model.eval()
    all_labels, all_probs, all_betas = [], [], []
    for batch in loader:
        batch = batch.to(device=device)
        logits = model(batch.x, batch.edge_index, batch.batch) # [batch, 2] 原始分数
        probs = torch.softmax(logits, dim=1)[:, 1]  # 类别1（antiD）的概率
        all_labels.append(batch.y.squeeze().cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        # 取每个图的第一个节点的第5列（beta列，index=5）
        beta_per_graph = []
        ptr = batch.ptr.cpu().numpy()
        for i in range(len(ptr) - 1):
            beta_per_graph.append(batch.x[ptr[i], 5].item())
        all_betas.append(np.array(beta_per_graph))
    return (np.concatenate(all_labels),
            np.concatenate(all_probs),
            np.concatenate(all_betas))


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
    for name, weight_path, in_ch in EVAL_MODELS:
        print(f'\n加载模型 {name}: {weight_path}')
        model = get_model(name, in_channels=in_ch).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

        labels, probs, betas = run_inference(model, test_loader, DEVICE)
        print_metrics(name, labels, probs)
        results.append((name, labels, probs))

        # 保存推理结果，供后续单独分析
        np.save(out_dir / f"{name}_labels.npy", labels)
        np.save(out_dir / f"{name}_probs.npy", probs)
        np.save(out_dir / f"{name}_betas.npy", betas)

    # 绘图
    plot_roc_curves(results, out_dir / "roc_curves.png")
    plot_rejection_curve(results, out_dir / "rejection_curve.png")
    print(f"\n所有结果保存至: {out_dir}")


def analyze_only():
    """跳过推理，直接从已保存的npy文件读取结果"""
    out_dir = PROJECT_ROOT / 'results' / 'evaluation'
    results = []
    for name, _, _in_ch in EVAL_MODELS:
        labels = np.load(out_dir / f"{name}_labels.npy")
        probs = np.load(out_dir / f"{name}_probs.npy")
        results.append((name, labels, probs))
        print(f'已加载 {name}: {len(labels)} samples')

    print_rejection_at_efficiency(results)

def analyze_threshold():
    """阈值优化分析，从已保存的npy文件读取，无需重新推理"""
    out_dir = PROJECT_ROOT / 'results' / 'evaluation'

    # 加载已保存的推理结果
    results = []
    for name, _, _in_ch in EVAL_MODELS:
        labels = np.load(out_dir / f"{name}_labels.npy")
        probs = np.load(out_dir / f"{name}_probs.npy")
        results.append((name, labels, probs))
        print(f"已加载 {name}: {len(labels)} samples")

    # ── 1. 各模型最优阈值搜索 ──────────────────────────
    print(f"\n{'=' * 70}")
    print("最优阈值搜索（按 F1 最大 / Rejection最大@Signal≥0.98）")
    print(f"{'Model':>10} {'Best_T(F1)':>12} {'F1':>8} {'Recall':>8} "
          f"{'Precision':>10} {'Rejection':>12}")
    print("-" * 70)

    for name, labels, probs in results:
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_f1, best_t_f1 = 0, 0.5
        best_rej, best_t_rej = 0, 0.5

        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()

            recall = tp / (tp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            rejection = tn / (tn + fp + 1e-10) if (tn + fp) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_t_f1 = t
                best_recall_f1 = recall
                best_prec_f1 = precision
                best_rej_f1 = rejection

        print(f"{name:>10} {best_t_f1:>12.2f} {best_f1:>8.4f} "
              f"{best_recall_f1:>8.4f} {best_prec_f1:>10.4f} {best_rej_f1:>12.4f}")

    # ── 2. DGCNN 详细阈值扫描表 ────────────────────────
    print(f'\n{'=' * 70}')
    print("DGCNN 阈值扫描（Signal Efficiency ≥ 0.98 区间重点展示）")
    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} "
          f"{'F1':>8} {'Rejection(TN/N_antiP)':>22} {'Rej_Power(1/FPR)':>18}")
    print("-" * 70)

    name_dgcnn = 'DGCNN'
    labels_d = next(l for n, l, _ in results if n == name_dgcnn)
    probs_d = next(p for n, _, p in results if n == name_dgcnn)

    for t in np.arange(0.01, 0.60, 0.02):
        preds = (probs_d >= t).astype(int)
        tp = ((preds == 1) & (labels_d == 1)).sum()
        fp = ((preds == 1) & (labels_d == 0)).sum()
        fn = ((preds == 0) & (labels_d == 1)).sum()
        tn = ((preds == 0) & (labels_d == 0)).sum()

        recall = tp / (tp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        rej_rate = tn / (tn + fp + 1e-10)
        fpr = fp / (fp + tn + 1e-10)
        rej_power = 1 / fpr if fpr > 0 else float('inf')

        print(f"{t:>10.2f} {recall:>8.4f} {precision:>10.4f} "
              f"{f1:>8.4f} {rej_rate:>22.4f} {rej_power:>18.2f}")

    # ── 3. Precision-Recall 曲线图 ─────────────────────
    plt.figure(figsize=(7, 6))
    for name, labels, probs in results:
        precision_arr, recall_arr, _ = precision_recall_curve(labels, probs)
        plt.plot(recall_arr, precision_arr, label=name)
    plt.xlabel("Recall (Signal Efficiency)")
    plt.ylabel("Precision (antiD Purity)")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=150)
    print(f"\nPR曲线已保存: {out_dir / 'precision_recall_curve.png'}")

    # ── 4. Rejection Power vs Signal Efficiency 曲线图 ─
    plt.figure(figsize=(7, 6))
    for name, labels, probs in results:
        fpr, tpr, _ = roc_curve(labels, probs)
        fpr_safe = np.where(fpr == 0, 1e-10, fpr)
        rej_power = 1.0 / fpr_safe
        plt.semilogy(tpr, rej_power, label=name)
    plt.axvline(x=0.98, color='red', linestyle='--', linewidth=0.8, label='Signal Eff=0.98')
    plt.xlabel("Signal Efficiency (Recall)")
    plt.ylabel("Rejection Power (1/FPR)")
    plt.title("Rejection Power vs Signal Efficiency")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "rejection_power_curve.png", dpi=150)
    print(f"Rejection Power曲线已保存: {out_dir / 'rejection_power_curve.png'}")


if __name__ == '__main__':
   evaluate()        # 完整推理+评估（第一次运行）
    # analyze_only()    # 只输出各效率下的Rejection表
    # analyze_threshold()  # 阈值优化分析（无需重新推理）

