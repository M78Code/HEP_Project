"""
中上 4M (csvFiles_Digitized) で NakagamiNet (A.1) を評価。
val_hybrid_nakagami4M.npz を test set として使用。
"""
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

NAKAGAMI_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NAKAGAMI_ROOT))

from data_parse.hybrid_dataset import HybridDatasetFast
from models.nakagami_model import NakagamiNet

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512
DATA_DIR   = Path('/mnt/ynakagami3/nakagami_data/data_4M')
MODEL_PATH = NAKAGAMI_ROOT / 'results' / 'nakagami4M_cnndnn_best.pth'
OUT_DIR    = NAKAGAMI_ROOT / 'results' / 'evaluation'


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for voxel, tof, label in tqdm(loader, desc='NakagamiNet eval'):
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
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {auc:.4f}")
    print(f"  混淆矩阵 : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    if (tn + fp) > 0:
        print(f"  背景抑制率: {tn / (tn + fp):.4f}")
    return auc


def print_rejection_at_efficiency(labels, probs,
                                  signal_efficiencies=(0.50, 0.80, 0.90, 0.95, 0.98, 0.99)):
    print(f"\n{'=' * 60}")
    print('各信号效率下的背景抑制率 (Background Rejection = 1/FPR)')
    print(f"{'Signal Eff':>12}  {'Rejection':>15}")
    print('-' * 30)
    fpr, tpr, _ = roc_curve(labels, probs)
    for target_eff in signal_efficiencies:
        idx = np.argmin(np.abs(tpr - target_eff))
        f = fpr[idx]
        rej_str = f'{1.0 / f:.2e}' if f > 0 else '>1e10'
        print(f"  {target_eff:>10.2f}    {rej_str:>15}")


def plot_rejection_curve(labels, probs, save_path):
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(labels, probs)
    fpr_safe = np.where(fpr == 0, 1e-10, fpr)
    plt.semilogy(tpr, 1.0 / fpr_safe, label='Nakagami4M (A.1)')
    plt.xlabel('Signal Efficiency (antiD recall)')
    plt.ylabel('Background Rejection (1 / FPR)')
    plt.legend()
    plt.xlim(0.5, 1.0)
    plt.ylim(1, 1e6)
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

    test_npz    = DATA_DIR / 'val_hybrid_nakagami4M.npz'
    test_set    = HybridDatasetFast(test_npz)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=8, pin_memory=True)
    print(f'test events: {len(test_set)}')

    model = NakagamiNet(tof_dim=9).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state)
    print(f'已加载模型: {MODEL_PATH}')

    labels, probs = run_inference(model, test_loader, DEVICE)

    tag = 'Nakagami4M_A1'
    print_metrics(tag, labels, probs)

    np.save(OUT_DIR / f'{tag}_labels.npy', labels)
    np.save(OUT_DIR / f'{tag}_probs.npy',  probs)
    print(f'推理结果已保存至: {OUT_DIR}')

    print_rejection_at_efficiency(labels, probs)
    plot_rejection_curve(labels, probs, OUT_DIR / f'rejection_{tag}.png')


if __name__ == '__main__':
    evaluate()
