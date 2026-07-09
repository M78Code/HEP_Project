#!/usr/bin/env python3
"""Make summary plots/tables for local430 selection-ablation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


TARGETS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--item",
        action="append",
        nargs=3,
        metavar=("SELECTION", "FEATURE", "RESULT_DIR"),
        required=True,
        help="Selection name, feature mode, and result directory. Repeat this option.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--title", default="Selection ablation")
    parser.add_argument(
        "--auc-y-min",
        type=float,
        default=None,
        help="Lower y-axis limit for the AUC panel. Default: min(AUC)-0.02.",
    )
    return parser.parse_args()


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


def rejection_at(labels: np.ndarray, scores: np.ndarray, target: float) -> dict[str, float | bool]:
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    idxs = np.flatnonzero(tpr >= target)
    if len(idxs) == 0:
        return {
            "target_efficiency": target,
            "actual_efficiency": math.nan,
            "fpr": math.nan,
            "rejection": math.nan,
            "zero_fpr": False,
            "threshold": math.nan,
        }
    idx = idxs[np.argmin(fpr[idxs])]
    zero_fpr = bool(fpr[idx] == 0)
    return {
        "target_efficiency": target,
        "actual_efficiency": float(tpr[idx]),
        "fpr": float(fpr[idx]),
        "rejection": math.inf if zero_fpr else float(1.0 / fpr[idx]),
        "zero_fpr": zero_fpr,
        "threshold": float(thr[idx]),
    }


def summarize(selection: str, feature: str, result_dir: Path) -> dict:
    labels, scores = load_arrays(result_dir)
    n_background = int((labels == 0).sum())
    rejections = {target: rejection_at(labels, scores, target) for target in TARGETS}
    return {
        "selection": selection,
        "feature": feature,
        "result_dir": str(result_dir),
        "n_events": int(labels.size),
        "n_background": n_background,
        "n_signal": int((labels == 1).sum()),
        "accuracy": float(accuracy_score(labels, scores >= 0.5)),
        "auc": float(roc_auc_score(labels, scores)),
        "rejection": rejections,
    }


def finite_rejection_for_plot(row: dict, target: float) -> float:
    value = float(row["rejection"][target]["rejection"])
    if math.isinf(value):
        return float(row["n_background"])
    return value


def rejection_text(row: dict, target: float) -> str:
    rejection = row["rejection"][target]
    value = float(rejection["rejection"])
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return f">{row['n_background']}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "selection",
        "feature",
        "n_events",
        "n_background",
        "n_signal",
        "accuracy",
        "auc",
        "rej50",
        "rej70",
        "rej80",
        "rej90",
        "rej95",
        "rej98",
        "result_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "selection": row["selection"],
                    "feature": row["feature"],
                    "n_events": row["n_events"],
                    "n_background": row["n_background"],
                    "n_signal": row["n_signal"],
                    "accuracy": f"{row['accuracy']:.6f}",
                    "auc": f"{row['auc']:.6f}",
                    "rej50": rejection_text(row, 0.50),
                    "rej70": rejection_text(row, 0.70),
                    "rej80": rejection_text(row, 0.80),
                    "rej90": rejection_text(row, 0.90),
                    "rej95": rejection_text(row, 0.95),
                    "rej98": rejection_text(row, 0.98),
                    "result_dir": row["result_dir"],
                }
            )


def make_bar_plot(rows: list[dict], path: Path, title: str, dpi: int, auc_y_min: float | None) -> None:
    labels = [f"{row['selection']}\n{row['feature']}" for row in rows]
    aucs = [row["auc"] for row in rows]
    rej90 = [finite_rejection_for_plot(row, 0.90) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=dpi, constrained_layout=True)

    x = np.arange(len(rows))
    axes[0].bar(x, aucs, color="#4c78a8")
    y_min = max(0.0, min(aucs) - 0.02) if auc_y_min is None else auc_y_min
    y_max = 1.003
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Classification AUC")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)
    for idx, value in enumerate(aucs):
        y_text = min(value + 0.0006, y_max - 0.0005)
        axes[0].text(idx, y_text, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, rej90, color="#f58518")
    axes[1].set_yscale("log")
    axes[1].set_ylim(max(1.0, min(rej90) * 0.6), max(rej90) * 2.2)
    axes[1].set_ylabel("Background rejection at 0.90 signal efficiency")
    axes[1].set_title("Rej@0.90")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].grid(True, axis="y", which="both", linestyle=":", alpha=0.5)
    for idx, row in enumerate(rows):
        value = rej90[idx]
        axes[1].text(idx, value * 1.08, rejection_text(row, 0.90), ha="center", va="bottom", fontsize=8)

    fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)


def make_table_plot(rows: list[dict], path: Path, dpi: int) -> None:
    table_rows = [
        [
            row["selection"],
            row["feature"],
            f"{row['auc']:.6f}",
            rejection_text(row, 0.90),
            rejection_text(row, 0.95),
        ]
        for row in rows
    ]
    columns = ["selection", "feature", "AUC", "Rej@0.90", "Rej@0.95"]

    fig, ax = plt.subplots(figsize=(8.5, 0.55 + 0.42 * len(rows)), dpi=dpi)
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize(selection, feature, Path(result_dir)) for selection, feature, result_dir in args.item]

    serializable_rows = []
    for row in rows:
        copied = dict(row)
        copied["rejection"] = {str(k): v for k, v in row["rejection"].items()}
        serializable_rows.append(copied)

    (args.out_dir / "selection_ablation_summary.json").write_text(
        json.dumps(serializable_rows, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    write_csv(rows, args.out_dir / "selection_ablation_summary.csv")
    make_bar_plot(rows, args.out_dir / "selection_ablation_auc_rej90.png", args.title, args.dpi, args.auc_y_min)
    make_table_plot(rows, args.out_dir / "selection_ablation_table.png", args.dpi)

    print("selection, feature, auc, rej90, rej95, result_dir")
    for row in rows:
        print(
            f"{row['selection']}, {row['feature']}, {row['auc']:.6f}, "
            f"{rejection_text(row, 0.90)}, {rejection_text(row, 0.95)}, {row['result_dir']}"
        )
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
