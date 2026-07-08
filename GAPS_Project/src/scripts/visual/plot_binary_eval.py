#!/usr/bin/env python3
"""Plot binary-classification evaluation outputs.

This script is intentionally generic: it only needs a result directory
containing labels.npy and scores.npy (or probs.npy). It is useful for the
Nakagami old CSV, local430 old-style graphs, and 50M GravNet evaluations.

Examples:
  python src/scripts/visual/plot_binary_eval.py \
    --result-dir results/local430_oldstyle_dgcnn_balanced_timelog \
    --train-log ~/train_local430_oldstyle_dgcnn_balanced_timelog.log

  python src/scripts/visual/plot_binary_eval.py \
    --result-dir results/evaluation_aohba50M_gravnet_on_4Mtest_epoch64
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--train-log", type=Path, default=None)
    p.add_argument("--label", default=None, help="legend/title label")
    p.add_argument("--score-file", default=None, help="default: scores.npy or probs.npy")
    p.add_argument("--labels-file", default="labels.npy")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--x-min", type=float, default=0.45)
    p.add_argument("--y-max", type=float, default=1e5)
    return p.parse_args()


def load_arrays(result_dir: Path, labels_file: str, score_file: str | None):
    labels_path = result_dir / labels_file
    if score_file is not None:
        scores_path = result_dir / score_file
    else:
        scores_path = result_dir / "scores.npy"
        if not scores_path.exists():
            scores_path = result_dir / "probs.npy"

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)

    labels = np.load(labels_path).astype(int)
    scores = np.load(scores_path).astype(float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(
            f"labels/scores length mismatch: {labels.shape[0]} vs {scores.shape[0]}"
        )
    return labels, scores, labels_path, scores_path


def rejection_at(labels: np.ndarray, scores: np.ndarray, target: float):
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    idxs = np.flatnonzero(tpr >= target)
    if len(idxs) == 0:
        return math.nan, math.nan, math.nan, math.nan
    idx = idxs[np.argmin(fpr[idxs])]
    rej = math.inf if fpr[idx] == 0 else 1.0 / fpr[idx]
    return float(rej), float(tpr[idx]), float(fpr[idx]), float(thr[idx])


def compute_metrics(labels: np.ndarray, scores: np.ndarray):
    metrics = {
        "n_events": int(labels.size),
        "label_counts": {
            "0": int((labels == 0).sum()),
            "1": int((labels == 1).sum()),
        },
        "accuracy_at_0.5": float(accuracy_score(labels, scores >= 0.5)),
        "auc": float(roc_auc_score(labels, scores)),
        "score_min": float(np.nanmin(scores)),
        "score_max": float(np.nanmax(scores)),
        "rejection": [],
    }
    for target in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]:
        rej, eff, fpr, threshold = rejection_at(labels, scores, target)
        metrics["rejection"].append(
            {
                "target_efficiency": target,
                "actual_efficiency": eff,
                "fpr": fpr,
                "rejection": rej,
                "threshold": threshold,
            }
        )
    return metrics


def parse_training_log(path: Path):
    if path is None or not path.exists():
        return None

    patterns = [
        # train_aohba.py:
        re.compile(
            r"Epoch\s+(\d+)/\d+\s+\|\s+train_loss:\s+([0-9.]+)\s+"
            r"train_acc:\s+([0-9.]+)\s+\|\s+val_loss:\s+([0-9.]+)\s+"
            r"val_acc:\s+([0-9.]+)"
        ),
        # train_nakagami_volid_dgcnn.py:
        re.compile(
            r"Epoch\s+(\d+)/\d+\s+\|\s+train_loss:\s+([0-9.]+)\s+"
            r"train_acc:\s+([0-9.]+)\s+\|\s+val_acc:\s+([0-9.]+)\s+"
            r"val_auc:\s+([0-9.]+)"
        ),
    ]

    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        matched = False
        for idx, pat in enumerate(patterns):
            m = pat.search(line)
            if not m:
                continue
            if idx == 0:
                rows.append(
                    {
                        "epoch": int(m.group(1)),
                        "train_loss": float(m.group(2)),
                        "train_acc": float(m.group(3)),
                        "val_loss": float(m.group(4)),
                        "val_acc": float(m.group(5)),
                        "val_auc": math.nan,
                    }
                )
            else:
                rows.append(
                    {
                        "epoch": int(m.group(1)),
                        "train_loss": float(m.group(2)),
                        "train_acc": float(m.group(3)),
                        "val_loss": math.nan,
                        "val_acc": float(m.group(4)),
                        "val_auc": float(m.group(5)),
                    }
                )
            matched = True
            break
        if matched:
            continue

    if not rows:
        return None
    return rows


def plot_score_distribution(labels, scores, out: Path, title: str, dpi: int):
    plt.figure(figsize=(7.0, 5.0), dpi=dpi)
    bins = np.linspace(0.0, 1.0, 101)
    plt.hist(
        scores[labels == 0],
        bins=bins,
        histtype="step",
        linewidth=2.0,
        label="anti-proton",
    )
    plt.hist(
        scores[labels == 1],
        bins=bins,
        histtype="step",
        linewidth=2.0,
        label="anti-deuteron",
    )
    plt.yscale("log")
    plt.xlabel("Classification score for anti-deuteron")
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_rejection_curve(labels, scores, out: Path, label: str, x_min: float, y_max: float, dpi: int):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=True)
    mask = fpr > 0
    plt.figure(figsize=(7.0, 5.0), dpi=dpi)
    plt.plot(tpr[mask], 1.0 / fpr[mask], linewidth=2.0, label=label)
    plt.yscale("log")
    plt.xlim(x_min, 1.0)
    plt.ylim(1, y_max)
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_learning_curve(rows: list[dict], out: Path, dpi: int):
    epochs = np.array([r["epoch"] for r in rows])
    train_acc = np.array([r["train_acc"] for r in rows])
    val_acc = np.array([r["val_acc"] for r in rows])
    val_auc = np.array([r["val_auc"] for r in rows])

    plt.figure(figsize=(7.0, 5.0), dpi=dpi)
    plt.plot(epochs, train_acc, linewidth=2.0, label="train acc")
    plt.plot(epochs, val_acc, linewidth=2.0, label="val acc")
    if np.isfinite(val_auc).any():
        plt.plot(epochs, val_auc, linewidth=2.0, label="val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    ymin = max(0.0, min(train_acc.min(), val_acc.min()) - 0.03)
    plt.ylim(ymin, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else result_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, scores, labels_path, scores_path = load_arrays(
        result_dir, args.labels_file, args.score_file
    )
    name = args.label or result_dir.name

    metrics = compute_metrics(labels, scores)
    metrics["labels_path"] = str(labels_path)
    metrics["scores_path"] = str(scores_path)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    title = f"{name} (AUC={metrics['auc']:.4f})"
    plot_score_distribution(labels, scores, out_dir / "score_distribution.png", title, args.dpi)
    plot_rejection_curve(
        labels,
        scores,
        out_dir / "rejection_curve.png",
        name,
        args.x_min,
        args.y_max,
        args.dpi,
    )

    rows = parse_training_log(args.train_log)
    if rows is not None:
        plot_learning_curve(rows, out_dir / "learning_curve.png", args.dpi)
        (out_dir / "learning_curve.json").write_text(
            json.dumps(rows, indent=2, allow_nan=True),
            encoding="utf-8",
        )
    else:
        print(f"[skip] no parseable train log: {args.train_log}")

    print("result:", result_dir)
    print("events:", metrics["n_events"], metrics["label_counts"])
    print(f"accuracy@0.5: {metrics['accuracy_at_0.5']:.6f}")
    print(f"AUC: {metrics['auc']:.6f}")
    for item in metrics["rejection"]:
        target = item["target_efficiency"]
        rej = item["rejection"]
        fpr = item["fpr"]
        eff = item["actual_efficiency"]
        if math.isinf(rej):
            rej_text = "inf"
        else:
            rej_text = f"{rej:.3f}"
        print(f"Rej@{target:.2f}: {rej_text}  FPR={fpr:.8g}  eff={eff:.6f}")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
