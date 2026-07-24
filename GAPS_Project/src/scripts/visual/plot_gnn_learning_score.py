#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+"
    r"train_loss=([0-9.eE+-]+)\s+train_acc=([0-9.eE+-]+)\s+\|\s+"
    r"val_loss=([0-9.eE+-]+)\s+val_acc=([0-9.eE+-]+)\s+val_auc=([0-9.eE+-]+)"
)


def parse_log(path: Path) -> dict[str, np.ndarray]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_acc": float(m.group(3)),
                "val_loss": float(m.group(4)),
                "val_acc": float(m.group(5)),
                "val_auc": float(m.group(6)),
            }
        )
    if not rows:
        raise RuntimeError(f"no epoch rows parsed from {path}")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def rejection_curve(eval_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.load(eval_dir / "labels.npy")
    scores = np.load(eval_dir / "scores.npy")
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=True)
    n_background = int((labels == 0).sum())
    fpr_floor = 1.0 / n_background
    auc = float(roc_auc_score(labels, scores))
    return tpr, 1.0 / np.maximum(fpr, fpr_floor), auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--cnn-eval-dir", type=Path, default=None)
    ap.add_argument("--gnn-label", default="GNN")
    ap.add_argument("--cnn-label", default="CNN+DNN")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--x-min", type=float, default=0.5)
    ap.add_argument("--y-max", type=float, default=1e6)
    args = ap.parse_args()

    hist = parse_log(args.log)
    labels = np.load(args.eval_dir / "labels.npy")
    scores = np.load(args.eval_dir / "scores.npy")

    anti_p = scores[labels == 0]
    anti_d = scores[labels == 1]

    out_stem = args.out.with_suffix("")
    out_score = out_stem.with_name(out_stem.name + "_validation_score").with_suffix(args.out.suffix)
    out_rej = out_stem.with_name(out_stem.name + "_rejection_curve").with_suffix(args.out.suffix)

    fig_vs, (ax_auc, ax_score) = plt.subplots(1, 2, figsize=(12.4, 4.2), dpi=args.dpi)
    fig_vs.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.88, wspace=0.28)

    ax_auc.plot(hist["epoch"], hist["val_auc"], linewidth=1.8)
    ax_auc.set_title("(a) GNN validation ROC-AUC")
    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("ROC-AUC")
    ax_auc.grid(True, alpha=0.3)

    bins = np.linspace(0.0, 1.0, 101)
    ax_score.hist(
        anti_p,
        bins=bins,
        histtype="step",
        linewidth=1.9,
        color="#d95f02",
        label="anti-proton",
    )
    ax_score.hist(
        anti_d,
        bins=bins,
        histtype="step",
        linewidth=1.9,
        color="#1b9e77",
        label="anti-deuteron",
    )
    ax_score.set_title("(b) GNN score distribution")
    ax_score.set_xlabel("Classification score for anti-deuteron")
    ax_score.set_ylabel("Counts")
    ax_score.set_yscale("log")
    ax_score.set_xlim(-0.025, 1.025)
    ax_score.set_ylim(bottom=1.0)
    ax_score.grid(True, which="both", alpha=0.3)
    ax_score.legend()

    fig_rej, ax_rej = plt.subplots(1, 1, figsize=(7.2, 4.8), dpi=args.dpi)
    fig_rej.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.92)

    if args.cnn_eval_dir is not None:
        eff, rej, auc = rejection_curve(args.cnn_eval_dir)
        ax_rej.plot(eff, rej, label=f"{args.cnn_label} AUC={auc:.4f}", linewidth=1.8)
    eff, rej, auc = rejection_curve(args.eval_dir)
    ax_rej.plot(eff, rej, label=f"{args.gnn_label} AUC={auc:.4f}", linewidth=1.8)
    ax_rej.set_title("Rejection curve")
    ax_rej.set_xlabel("Signal efficiency")
    ax_rej.set_ylabel("Background rejection")
    ax_rej.set_yscale("log")
    ax_rej.set_xlim(args.x_min, 1.0)
    ax_rej.set_ylim(1.0, args.y_max)
    ax_rej.grid(True, which="both", alpha=0.3)
    ax_rej.legend(loc="upper right", fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig_vs.savefig(out_score)
    fig_rej.savefig(out_rej)
    print(f"saved: {out_score}")
    print(f"saved: {out_rej}")


if __name__ == "__main__":
    main()
