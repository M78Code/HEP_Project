import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

import GAPS_Project
from GAPS_Project.src.data_parse.hybrid_dataset import HybridDatasetFast
from GAPS_Project.src.models.cnn_dnn_hybrid import CNNDNNHybrid

PROJECT_ROOT = Path(GAPS_Project.__file__).parent

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512
DATA_DIR   = PROJECT_ROOT / 'dataset' / 'split'
MODEL_PATH = PROJECT_ROOT / 'results' / 'cnn_dnn_hybrid_best.pth'
OUT_DIR    = PROJECT_ROOT / 'results' / 'evaluation'


@torch.no_grad()
def run_inference(model, loader, device):
    """返回 (labels, probs) numpy arrays"""
    model.eval()
    all_labels, all_probs = [], []
    for voxel, tof, label in tqdm(loader, desc='CNN+DNN eval'):
        voxel, tof = voxel.to(device), tof.to(device)
        logits = model(voxel, tof)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(label.numpy())
        all_probs.append(probs)
    return np.concatenate(all_labels), np.concatenate(all_probs)


def print_metrics(name, labels, probs):
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(labels, preds)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    f1    = f1_score(labels, preds, zero_division=0)
    auc   = roc_auc_score(labels, probs)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'=' * 50}")
    print(f"模型: {name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}  (antiD识别精确率)")
    print(f"  Recall   : {rec:.4f}  (antiD信号效率)")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {auc:.4f}")
    print(f"  混淆矩阵 : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  背景抑制率: {tn / (tn + fp):.4f}  ({tn + fp}个antiP中正确拒绝{tn}个)")
    return auc


def print_rejection_at_efficiency(results, signal_efficiencies=(0.50, 0.80, 0.90, 0.95, 0.98, 0.99)):
    print(f"\n{'=' * 60}")
    print('各信号效率下的背景抑制率 （Background Rejection = 1/FPR）')
    print(f"{'Signal Eff':>12}", end="")
    for name, _, _ in results:
        print(f' {name:>20}', end="")
    print()
    print('-' * 60)
    for target_eff in signal_efficiencies:
        print(f' {target_eff:>12.2f}', end="")
        for name, labels, probs in results:
            fpr, tpr, _ = roc_curve(labels, probs)
            idx = np.argmin(np.abs(tpr - target_eff))
            f = fpr[idx]
            rej_str = f'{1.0 / f:.2e}' if f > 0 else '>1e10'
            print(f"  {rej_str:>20}", end="")
        print()


def plot_rejection_curve(results, save_path):
    plt.figure(figsize=(7, 6))
    for name, labels, probs in results:
        fpr, tpr, _ = roc_curve(labels, probs)
        fpr_safe = np.where(fpr == 0, 1e-10, fpr)
        plt.semilogy(tpr, 1.0 / fpr_safe, label=name)
    plt.xlabel('Signal Efficiency (antiD recall)')
    plt.ylabel('Background Rejection (1 / FPR)')
    plt.legend()
    plt.xlim(0.5, 1.0)
    plt.ylim(1, 2e4)
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    def _log_fmt(x, _):
        if x == 1:  return '1'
        if x == 10: return '10'
        exp = int(round(np.log10(x)))
        return f'$10^{{{exp}}}$'
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(_log_fmt))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Rejection曲线已保存: {save_path}")


def evaluate():
    """CNN+DNN baseline 评估（从预处理npz加载）"""
    print(f'使用设备：{DEVICE}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 加载测试数据 ──
    # 评估12×12模型时改为 test_hybrid.npz
    # 评估20×20模型时改为 test_hybrid_20x20.npz
    test_npz = DATA_DIR / 'test_hybrid.npz'
    test_set    = HybridDatasetFast(test_npz)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)
    print(f'test events: {len(test_set)}')

    # ── 加载模型（处理 torch.compile 权重前缀）──
    model = CNNDNNHybrid(tof_dim=11).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state)
    print(f'已加载模型: {MODEL_PATH}')

    # ── 推理 ──
    labels, probs = run_inference(model, test_loader, DEVICE)
    betas = np.load(test_npz)['betas']

    # ── 评价指标 ──
    tag = 'CNN_DNN_12x12'   # 评估20×20时改为 CNN_DNN_20x20
    print_metrics(tag, labels, probs)

    np.save(OUT_DIR / f'{tag}_labels.npy', labels)
    np.save(OUT_DIR / f'{tag}_probs.npy',  probs)
    np.save(OUT_DIR / f'{tag}_betas.npy',  betas)
    print(f'推理结果已保存至: {OUT_DIR}')

    # ── 与 GNN 对比 ──
    results = [(tag, labels, probs)]
    for name in ['GravNet_6b_h128_rec', 'DGCNN_rec']:
        lbl_path = OUT_DIR / f'{name}_labels.npy'
        prb_path = OUT_DIR / f'{name}_probs.npy'
        if lbl_path.exists():
            lbl = np.load(lbl_path)
            prb = np.load(prb_path)
            results.append((name, lbl, prb))

    print_rejection_at_efficiency(results)
    plot_rejection_curve(results, OUT_DIR / f'rejection_{tag}_vs_gnn.png')


if __name__ == '__main__':
    evaluate()
