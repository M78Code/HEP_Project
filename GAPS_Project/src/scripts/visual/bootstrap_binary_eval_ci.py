#!/usr/bin/env python3
"""Bootstrap confidence intervals for high-efficiency binary rejection points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", action="append", nargs=2, required=True,
                        metavar=("LABEL", "RESULT_DIR"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--targets", type=float, nargs="+",
                        default=[0.90, 0.95, 0.98, 0.99])
    return parser.parse_args()


def load_arrays(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    labels = np.load(directory / "labels.npy").astype(np.int8)
    score_path = directory / "scores.npy"
    if not score_path.is_file():
        score_path = directory / "probs.npy"
    scores = np.load(score_path).astype(np.float64)
    if len(labels) != len(scores) or not np.array_equal(np.unique(labels), [0, 1]):
        raise RuntimeError(f"invalid binary evaluation arrays: {directory}")
    return labels, scores


def capped_rejection_at(
    labels: np.ndarray, scores: np.ndarray, target: float,
) -> tuple[float, float, int]:
    """Return FPR, capped rejection, and passed-background count."""
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero(tpr >= target)
    if not len(candidates):
        raise RuntimeError(f"no ROC point reaches target efficiency {target}")
    index = candidates[np.argmin(fpr[candidates])]
    n_background = int(np.count_nonzero(labels == 0))
    point_fpr = float(fpr[index])
    passed = int(round(point_fpr * n_background))
    return point_fpr, float(1.0 / max(point_fpr, 1.0 / n_background)), passed


def summarize(values: np.ndarray, estimate: float) -> dict[str, float | list[float]]:
    low, median, high = np.percentile(values, [2.5, 50.0, 97.5])
    return {
        "estimate": float(estimate),
        "bootstrap_median": float(median),
        "ci95": [float(low), float(high)],
    }


def bootstrap_item(
    label: str, directory: Path, repeats: int, seed: int, targets: list[float],
) -> dict:
    labels, scores = load_arrays(directory)
    signal = np.flatnonzero(labels == 1)
    background = np.flatnonzero(labels == 0)
    observed = {
        target: capped_rejection_at(labels, scores, target) for target in targets
    }
    print(f"[START] {label}: {len(signal):,} signal, {len(background):,} background, "
          f"{repeats:,} bootstrap repeats", flush=True)
    samples = {target: np.empty(repeats, dtype=np.float64) for target in targets}
    auc_samples = np.empty(repeats, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        indices = np.concatenate((
            rng.choice(signal, size=len(signal), replace=True),
            rng.choice(background, size=len(background), replace=True),
        ))
        boot_labels, boot_scores = labels[indices], scores[indices]
        auc_samples[repeat] = roc_auc_score(boot_labels, boot_scores)
        for target in targets:
            _, rejection, _ = capped_rejection_at(boot_labels, boot_scores, target)
            samples[target][repeat] = rejection
        if (repeat + 1) % 1000 == 0 or repeat + 1 == repeats:
            print(f"[PROGRESS] {label}: {repeat + 1:,}/{repeats:,}", flush=True)

    result = {
        "label": label,
        "result_dir": str(directory),
        "n_signal": int(len(signal)),
        "n_background": int(len(background)),
        "auc": summarize(auc_samples, float(roc_auc_score(labels, scores))),
        "rejection": [],
    }
    for target in targets:
        fpr, rejection, passed = observed[target]
        row = summarize(samples[target], rejection)
        row.update({
            "target_efficiency": target,
            "observed_fpr": fpr,
            "observed_background_passed": passed,
            "finite_sample_cap": float(len(background)),
        })
        result["rejection"].append(row)
    print(f"[DONE] {label}", flush=True)
    return result


def cell(value: dict[str, float | list[float]], decimals: int) -> str:
    low, high = value["ci95"]
    return (f"{value['estimate']:.{decimals}f}\n"
            f"[{low:.{decimals}f}, {high:.{decimals}f}]")


def make_table(results: list[dict], targets: list[float], output: Path) -> None:
    columns = ["Input", "AUC"] + [f"Rej@{target:.2f}" for target in targets]
    rows = []
    for result in results:
        by_target = {row["target_efficiency"]: row for row in result["rejection"]}
        rows.append([
            result["label"],
            cell(result["auc"], 4),
            *[cell(by_target[target], 1) for target in targets],
        ])
    fig, ax = plt.subplots(figsize=(17.0, 4.6), dpi=220)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center",
                     colWidths=[0.34, 0.13] + [0.1325] * len(targets))
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 2.6)
    for index in range(len(columns)):
        table[(0, index)].set_facecolor("#216da8")
        table[(0, index)].set_text_props(color="white", weight="bold")
    for row_index in range(1, len(rows) + 1):
        table[(row_index, 0)].set_text_props(weight="bold")
        if row_index % 2 == 0:
            for column_index in range(len(columns)):
                table[(row_index, column_index)].set_facecolor("#f2f6f8")
    fig.suptitle("Four-way high-efficiency rejection: 95% stratified-bootstrap CI",
                 fontsize=15, weight="bold", y=0.94)
    fig.text(0.5, 0.04,
             "Each group: 2,000 signal and 2,000 background test events. "
             "Rejection uses the finite-sample cap 1/Nbackground for zero-FPR resamples.",
             ha="center", fontsize=8.5, color="0.30")
    fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.90))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.repeats < 100:
        raise ValueError("--repeats must be at least 100")
    if any(not 0.0 < target <= 1.0 for target in args.targets):
        raise ValueError("targets must be in (0, 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("===== Four-way high-efficiency bootstrap CI =====", flush=True)
    print(f"repeats: {args.repeats:,}; stratified by true class", flush=True)
    results = [
        bootstrap_item(label, Path(directory), args.repeats, args.seed + index,
                       args.targets)
        for index, (label, directory) in enumerate(args.item)
    ]
    payload = {
        "method": "stratified nonparametric bootstrap over signal and background test events",
        "repeats": args.repeats,
        "seed": args.seed,
        "targets": args.targets,
        "zero_fpr_handling": "finite-sample rejection cap of N_background",
        "items": results,
    }
    json_path = args.out_dir / "bootstrap_high_efficiency_ci.json"
    json_path.write_text(json.dumps(payload, indent=2))
    figure_path = args.out_dir / "bootstrap_high_efficiency_ci.png"
    make_table(results, args.targets, figure_path)

    print("\n===== Four-way high-efficiency bootstrap CI =====")
    print(f"repeats: {args.repeats:,}; stratified by true class")
    for result in results:
        line = [f"{result['label']}: AUC={result['auc']['estimate']:.6f}"]
        for row in result["rejection"]:
            low, high = row["ci95"]
            line.append(f"Rej@{row['target_efficiency']:.2f}="
                        f"{row['estimate']:.1f} [{low:.1f}, {high:.1f}]")
        print(" | ".join(line))
    print(f"figure : {figure_path}")
    print(f"metrics: {json_path}")
    print("BOOTSTRAP HIGH-EFFICIENCY CI: COMPLETE")


if __name__ == "__main__":
    main()
