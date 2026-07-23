#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    hist = parse_log(args.log)
    labels = np.load(args.eval_dir / "labels.npy")
    scores = np.load(args.eval_dir / "scores.npy")

    anti_p = scores[labels == 0]
    anti_d = scores[labels == 1]

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.2),
        dpi=args.dpi,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    ax_loss, ax_score = axs[0]
    ax_auc = axs[1, 0]
    axs[1, 1].axis("off")

    ax_loss.plot(hist["epoch"], hist["train_loss"], label="train", linewidth=1.8)
    ax_loss.plot(hist["epoch"], hist["val_loss"], label="validation", linewidth=1.8)
    ax_loss.set_title("(a) GNN loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_auc.plot(hist["epoch"], hist["val_auc"], linewidth=1.8)
    ax_auc.set_title("(b) GNN validation ROC-AUC")
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
    ax_score.set_title("(c) GNN score distribution")
    ax_score.set_xlabel("Classification score for anti-deuteron")
    ax_score.set_ylabel("Counts")
    ax_score.set_yscale("log")
    ax_score.set_xlim(0.0, 1.0)
    ax_score.set_ylim(bottom=1.0)
    ax_score.grid(True, which="both", alpha=0.3)
    ax_score.legend()

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
