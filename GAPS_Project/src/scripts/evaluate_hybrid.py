import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import GAPS_Project
from GAPS_Project.src.data_parse.hybrid_dataset import HybridDataset
from GAPS_Project.src.models.cnn_dnn_hybrid import CNNDNNHybrid

PROJECT_ROOT = Path(GAPS_Project.__file__).parent

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 200
DATA_DIR   = PROJECT_ROOT / 'dataset' / 'split'
MODEL_PATH = PROJECT_ROOT / 'results' / 'cnn_dnn_hybrid_best.pth'
OUT_DIR    = PROJECT_ROOT / 'results' / 'evaluation'


@torch.no_grad()
def run_inference(model, loader, dataset, device):
    """返回 (labels, probs, betas) numpy arrays"""
    model.eval()
    all_labels, all_probs, all_betas = [], [], []
    idx = 0
    for voxel, tof, label in loader:
        voxel, tof = voxel.to(device), tof.to(device)
        logits = model(voxel, tof)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(label.numpy())
        all_probs.append(probs)
        # beta 从原始 event dict 读取
        batch_size = label.shape[0]
        betas = [float(dataset.data[i].get('beta', 0.0)) for i in range(idx, idx + batch_size)]
        all_betas.append(np.array(betas, dtype=np.float32))
        idx += batch_size
    return (np.concatenate(all_labels),
            np.concatenate(all_probs),
            np.concatenate(all_betas))


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
    print(f'使用设备：{DEVICE}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_set    = HybridDataset(DATA_DIR / 'test.pkl')
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f'test events: {len(test_set)}')

    model = CNNDNNHybrid(tof_dim=11).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f'已加载模型: {MODEL_PATH}')

    labels, probs, betas = run_inference(model, test_loader, test_set, DEVICE)
    print_metrics('CNN+DNN (Nakagami A.2)', labels, probs)

    np.save(OUT_DIR / 'CNNDNNHybrid_labels.npy', labels)
    np.save(OUT_DIR / 'CNNDNNHybrid_probs.npy',  probs)
    np.save(OUT_DIR / 'CNNDNNHybrid_betas.npy',  betas)
    print(f'推理结果已保存至: {OUT_DIR}')

    results = [('CNN+DNN (Nakagami A.2)', labels, probs)]
    print_rejection_at_efficiency(results)
    plot_rejection_curve(results, OUT_DIR / 'hybrid_rejection_curve.png')


if __name__ == '__main__':
    evaluate()
