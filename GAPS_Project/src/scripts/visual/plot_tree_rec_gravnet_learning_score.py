#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+"
    r"train_loss[:=]\s*([0-9.eE+-]+)\s+train_acc[:=]\s*([0-9.eE+-]+)\s+\|\s+"
    r"val_loss[:=]\s*([0-9.eE+-]+)\s+val_acc[:=]\s*([0-9.eE+-]+)"
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
            "val_loss": float(m.group(4)),
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


def plot_loss(ax, hist: dict[str, np.ndarray], title: str) -> None:
    ax.plot(hist["epoch"], hist["train_loss"], label="train", linewidth=1.8)
    ax.plot(hist["epoch"], hist["val_loss"], label="validation", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-4m", required=True, type=Path)
    ap.add_argument("--eval-4m", required=True, type=Path)
    ap.add_argument("--log-50m", required=True, type=Path)
    ap.add_argument("--eval-50m", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    hist_4m = parse_loss_log(args.log_4m)
    hist_50m = parse_loss_log(args.log_50m)

    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.2), dpi=args.dpi)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.94,
        wspace=0.28,
        hspace=0.30,
    )

    plot_loss(axs[0, 0], hist_4m, "(a) GravNet 4M loss")
    plot_score(axs[0, 1], args.eval_4m, "(b) GravNet 4M score distribution")
    plot_loss(axs[1, 0], hist_50m, "(c) GravNet 50M loss")
    plot_score(axs[1, 1], args.eval_50m, "(d) GravNet 50M score distribution")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
