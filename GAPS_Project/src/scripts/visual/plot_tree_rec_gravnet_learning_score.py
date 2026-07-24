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
    r"train_loss[:=]\s*([0-9.eE+-]+)\s+train_acc[:=]\s*([0-9.eE+-]+)\s+\|\s+"
    r"val_loss[:=]\s*([0-9.eE+-]+)\s+val_acc[:=]\s*([0-9.eE+-]+)"
    r"(?:\s+val_auc[:=]\s*([0-9.eE+-]+))?"
)


def parse_loss_log(path: Path) -> dict[str, np.ndarray]:
    by_epoch = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        epoch = int(m.group(1))
        by_epoch[epoch] = {
            "epoch": epoch,
            "train_loss": float(m.group(2)),
            "train_acc": float(m.group(3)),
            "val_loss": float(m.group(4)),
            "val_acc": float(m.group(5)),
            "val_auc": float(m.group(6)) if m.group(6) is not None else np.nan,
        }
    rows = [by_epoch[k] for k in sorted(by_epoch)]
    if not rows:
        raise RuntimeError(f"no epoch rows parsed from {path}")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def load_scores(eval_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    labels_path = eval_dir / "labels.npy"
    scores_path = eval_dir / "scores.npy"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    labels = np.load(labels_path).astype(int)
    scores = np.load(scores_path).astype(float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(f"{eval_dir}: labels/scores length mismatch")
    return labels, scores


def rejection_curve(eval_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    labels, scores = load_scores(eval_dir)
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=True)
    n_background = int((labels == 0).sum())
    fpr_floor = 1.0 / n_background
    auc = float(roc_auc_score(labels, scores))
    return tpr, 1.0 / np.maximum(fpr, fpr_floor), auc


def plot_validation(ax, hist: dict[str, np.ndarray], title: str) -> None:
    if np.isfinite(hist["val_auc"]).any():
        ax.plot(hist["epoch"], hist["val_auc"], linewidth=1.8)
        ax.set_ylabel("ROC-AUC")
    else:
        ax.plot(hist["epoch"], hist["val_acc"], linewidth=1.8)
        ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)


def plot_score(ax, eval_dir: Path, title: str) -> None:
    labels, scores = load_scores(eval_dir)
    anti_p = scores[labels == 0]
    anti_d = scores[labels == 1]
    bins = np.linspace(0.0, 1.0, 101)
    ax.hist(
        anti_p,
        bins=bins,
        histtype="step",
        linewidth=1.9,
        color="#d95f02",
        label="anti-proton",
    )
    ax.hist(
        anti_d,
        bins=bins,
        histtype="step",
        linewidth=1.9,
        color="#1b9e77",
        label="anti-deuteron",
    )
    ax.set_title(title)
    ax.set_xlabel("Classification score for anti-deuteron")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(bottom=1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def plot_validation_score(
    log_path: Path,
    eval_dir: Path,
    title_prefix: str,
    out: Path,
    dpi: int,
) -> None:
    hist = parse_loss_log(log_path)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.2), dpi=dpi)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.88, wspace=0.28)
    plot_validation(axs[0], hist, f"(a) {title_prefix} validation")
    plot_score(axs[1], eval_dir, f"(b) {title_prefix} score distribution")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"saved: {out}")


def plot_rejection(eval_dir: Path, title_prefix: str, out: Path, dpi: int, x_min: float, y_max: float) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), dpi=dpi)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.92)
    eff, rej, auc = rejection_curve(eval_dir)
    ax.plot(eff, rej, label=f"{title_prefix} AUC={auc:.4f}", linewidth=1.8)
    ax.set_title("Rejection curve")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_yscale("log")
    ax.set_xlim(x_min, 1.0)
    ax.set_ylim(1.0, y_max)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-4m", required=True, type=Path)
    ap.add_argument("--eval-4m", required=True, type=Path)
    ap.add_argument("--log-50m", required=True, type=Path)
    ap.add_argument("--eval-50m", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--x-min", type=float, default=0.5)
    ap.add_argument("--y-max", type=float, default=1e6)
    args = ap.parse_args()

    out_stem = args.out.with_suffix("")
    suffix = args.out.suffix
    plot_validation_score(
        args.log_4m,
        args.eval_4m,
        "GravNet 4M",
        out_stem.with_name(out_stem.name + "_4m_validation_score").with_suffix(suffix),
        args.dpi,
    )
    plot_rejection(
        args.eval_4m,
        "GravNet 4M",
        out_stem.with_name(out_stem.name + "_4m_rejection_curve").with_suffix(suffix),
        args.dpi,
        args.x_min,
        args.y_max,
    )
    plot_validation_score(
        args.log_50m,
        args.eval_50m,
        "GravNet 50M",
        out_stem.with_name(out_stem.name + "_50m_validation_score").with_suffix(suffix),
        args.dpi,
    )
    plot_rejection(
        args.eval_50m,
        "GravNet 50M",
        out_stem.with_name(out_stem.name + "_50m_rejection_curve").with_suffix(suffix),
        args.dpi,
        args.x_min,
        args.y_max,
    )


if __name__ == "__main__":
    main()
