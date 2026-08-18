#!/usr/bin/env python3
"""Compare binary-classification result directories.

Each result directory must contain labels.npy and scores.npy (or probs.npy).
The rejection curve shows FPR=0 points with a finite-statistics cap by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--item",
        action="append",
        nargs=2,
        metavar=("LABEL", "RESULT_DIR"),
        required=True,
        help="Model label and result directory. Repeat for multiple models.",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--x-min", type=float, default=0.45)
    p.add_argument("--y-max", type=float, default=1e5)
    p.add_argument("--zero-fpr-mode", choices=["cap", "drop"], default="cap")
    p.add_argument(
        "--mark-efficiencies",
        nargs="*",
        type=float,
        default=[0.95, 0.98],
        metavar="EFF",
        help="Signal efficiencies to show with vertical guides and curve markers.",
    )
    return p.parse_args()


def load_arrays(result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    labels_path = result_dir / "labels.npy"
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
        raise ValueError(f"{result_dir}: labels/scores length mismatch")
    return labels, scores


def rejection_at(labels: np.ndarray, scores: np.ndarray, target: float) -> dict[str, float]:
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    idxs = np.flatnonzero(tpr >= target)
    if len(idxs) == 0:
        return {
            "target_efficiency": target,
            "actual_efficiency": math.nan,
            "fpr": math.nan,
            "rejection": math.nan,
            "threshold": math.nan,
        }
    idx = idxs[np.argmin(fpr[idxs])]
    return {
        "target_efficiency": target,
        "actual_efficiency": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "rejection": math.inf if fpr[idx] == 0 else float(1.0 / fpr[idx]),
        "threshold": float(thr[idx]),
    }


def rejection_curve(labels: np.ndarray, scores: np.ndarray, zero_fpr_mode: str):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=True)
    if zero_fpr_mode == "cap":
        n_background = int((labels == 0).sum())
        fpr_floor = 1.0 / n_background
        return tpr, 1.0 / np.maximum(fpr, fpr_floor)
    mask = fpr > 0
    return tpr[mask], 1.0 / fpr[mask]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=args.dpi)

    for label, result_dir_text in args.item:
        result_dir = Path(result_dir_text)
        labels, scores = load_arrays(result_dir)
        auc = float(roc_auc_score(labels, scores))
        acc = float(accuracy_score(labels, scores >= 0.5))
        tpr, rejection = rejection_curve(labels, scores, args.zero_fpr_mode)
        (line,) = ax.plot(tpr, rejection, label=f"{label} AUC={auc:.4f}")

        row = {
            "label": label,
            "result_dir": str(result_dir),
            "n_events": int(labels.size),
            "label_counts": {
                "0": int((labels == 0).sum()),
                "1": int((labels == 1).sum()),
            },
            "accuracy_at_0.5": acc,
            "auc": auc,
            "rejection": [
                rejection_at(labels, scores, target)
                for target in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
            ],
        }
        summary.append(row)

        n_background = int((labels == 0).sum())
        for point in row["rejection"]:
            target = point["target_efficiency"]
            if target not in args.mark_efficiencies or not np.isfinite(point["fpr"]):
                continue
            rejection_value = 1.0 / max(point["fpr"], 1.0 / n_background)
            ax.plot(
                point["actual_efficiency"],
                rejection_value,
                marker="o",
                markersize=4.5,
                color=line.get_color(),
                zorder=3,
            )

    for efficiency in args.mark_efficiencies:
        ax.axvline(efficiency, color="0.45", linestyle="--", linewidth=0.8, alpha=0.7)

    base_ticks = [round(tick, 1) for tick in np.arange(0.0, 1.01, 0.1)]
    x_ticks = sorted(set(base_ticks).union(args.mark_efficiencies))
    x_ticks = [tick for tick in x_ticks if args.x_min <= tick <= 1.0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [
            f"{tick:.2f}" if any(np.isclose(tick, mark) for mark in args.mark_efficiencies)
            else f"{tick:.1f}"
            for tick in x_ticks
        ],
        fontsize=11,
    )
    for tick, label in zip(x_ticks, ax.get_xticklabels()):
        if any(np.isclose(tick, mark) for mark in args.mark_efficiencies):
            label.set_fontsize(9)
            label.set_rotation(45)
            label.set_horizontalalignment("right")
    ax.set_yscale("log")
    ax.set_xlim(args.x_min, 1.0)
    ax.set_ylim(1, args.y_max)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "rejection_compare.png")
    plt.close(fig)

    (args.out_dir / "metrics_compare.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    header = "model, accuracy, auc, rej50, rej70, rej80, rej90, rej95, rej98, rej99"
    print(header)
    for row in summary:
        values = {r["target_efficiency"]: r["rejection"] for r in row["rejection"]}
        print(
            f"{row['label']}, {row['accuracy_at_0.5']:.6f}, {row['auc']:.6f}, "
            f"{values[0.50]:.3f}, {values[0.70]:.3f}, {values[0.80]:.3f}, "
            f"{values[0.90]:.3f}, {values[0.95]:.3f}, {values[0.98]:.3f}, "
            f"{values[0.99]:.3f}"
        )
    print("saved:", args.out_dir)


if __name__ == "__main__":
    main()
