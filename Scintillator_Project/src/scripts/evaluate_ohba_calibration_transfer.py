"""Evaluate small-reference-event offset calibration for held-out positions/runs.

The frozen predictions from complete position/run holdout are used directly.
For each unseen position, only its earliest K labelled reference events estimate
one constant residual offset.  The remaining events are the evaluation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import Scintillator_Project
from Scintillator_Project.src.scripts.train_ohba_quality_selected_hybrid import gaussian_metrics


PROJECT_ROOT = Path(Scintillator_Project.__file__).resolve().parent


def parse_counts(text: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("calibration counts must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("provide unique non-negative calibration counts")
    return sorted(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results/ohba_all_usable_leave_one_position_out",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "dataset/cache/ohba_all_usable_events_roi450_800.npz",
    )
    parser.add_argument("--calibration-counts", type=parse_counts, default=[0, 10, 30, 100])
    return parser.parse_args()


def fold_prediction_data(
    results_root: Path,
    cache: dict[str, np.ndarray],
    fold_definition: list[dict[str, object]],
    model_seed: int,
) -> list[dict[str, np.ndarray | int | list[float]]]:
    folds = []
    for definition in fold_definition:
        fold = int(definition["fold"])
        positions = definition["test_positions_cm"]
        indices = np.flatnonzero(np.isin(cache["labels"], positions))
        path = results_root / f"fold_{fold}_model_{model_seed}" / "predictions.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing predictions: {path}")
        with np.load(path) as prediction:
            labels = prediction["labels"]
            traditional = prediction["traditional_baseline"]
            hybrid = prediction["hybrid_prediction"]
        expected_labels = cache["labels"][indices]
        if labels.shape != expected_labels.shape or not np.allclose(labels, expected_labels):
            raise RuntimeError(f"prediction order does not match reconstructed fold {fold}")
        folds.append({
            "fold": fold,
            "positions": positions,
            "labels": labels,
            "traditional": traditional,
            "hybrid": hybrid,
            "event_ids": cache["event_ids"][indices],
        })
    return folds


def calibrated_metrics(
    folds: list[dict[str, np.ndarray | int | list[float]]], count: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    labels_all, traditional_all, hybrid_all = [], [], []
    offsets: list[dict[str, object]] = []
    for fold_data in folds:
        labels = fold_data["labels"]
        traditional = fold_data["traditional"]
        hybrid = fold_data["hybrid"]
        event_ids = fold_data["event_ids"]
        positions = np.asarray(fold_data["positions"], dtype=np.float32)
        corrected_traditional = traditional.copy()
        corrected_hybrid = hybrid.copy()
        evaluation = np.ones(labels.size, dtype=bool)
        for position in positions:
            position_indices = np.flatnonzero(labels == position)
            ordered = position_indices[np.argsort(event_ids[position_indices], kind="stable")]
            if count >= ordered.size:
                raise ValueError(
                    f"requested {count} calibration events but position {position:.0f} has {ordered.size}"
                )
            calibration = ordered[:count]
            if count:
                traditional_offset = float(np.mean(traditional[calibration] - labels[calibration]))
                hybrid_offset = float(np.mean(hybrid[calibration] - labels[calibration]))
                corrected_traditional[position_indices] -= traditional_offset
                corrected_hybrid[position_indices] -= hybrid_offset
            else:
                traditional_offset = 0.0
                hybrid_offset = 0.0
            evaluation[calibration] = False
            offsets.append({
                "fold": int(fold_data["fold"]),
                "position_cm": float(position),
                "calibration_events": count,
                "traditional_offset_cm": traditional_offset,
                "hybrid_offset_cm": hybrid_offset,
                "evaluation_events": int(evaluation[position_indices].sum()),
            })
        labels_all.append(labels[evaluation])
        traditional_all.append(corrected_traditional[evaluation])
        hybrid_all.append(corrected_hybrid[evaluation])
    labels = np.concatenate(labels_all)
    traditional = np.concatenate(traditional_all)
    hybrid = np.concatenate(hybrid_all)
    return {
        "evaluation_events": int(labels.size),
        "traditional": gaussian_metrics(labels, traditional),
        "hybrid": gaussian_metrics(labels, hybrid),
    }, offsets


def plot_transfer(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["calibration_events_per_position"]) for row in rows]
    traditional_sigma = [row["traditional"]["gaussian_fit"]["sigma_cm"] for row in rows]
    hybrid_sigma = [row["hybrid"]["gaussian_fit"]["sigma_cm"] for row in rows]
    traditional_rmse = [row["traditional"]["rmse_cm"] for row in rows]
    hybrid_rmse = [row["hybrid"]["rmse_cm"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].plot(x, traditional_sigma, "o-", label="Traditional")
    axes[0].plot(x, hybrid_sigma, "o-", label="Hybrid")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Known reference events per held-out position")
    axes[0].set_ylabel("Gaussian core sigma [cm]")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(x, traditional_rmse, "o-", label="Traditional")
    axes[1].plot(x, hybrid_rmse, "o-", label="Hybrid")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Known reference events per held-out position")
    axes[1].set_ylabel("RMSE [cm]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle("Held-out position/run: fixed-offset calibration transfer")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = args.results_root / "position_holdout_summary.json"
    definition_path = args.results_root / "fold_definition.json"
    if not summary_path.is_file() or not definition_path.is_file():
        raise FileNotFoundError("leave-one-position-out result files are missing")
    with np.load(args.cache) as data:
        cache = {name: data[name] for name in data.files}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fold_definition = json.loads(definition_path.read_text(encoding="utf-8"))
    folds = fold_prediction_data(args.results_root, cache, fold_definition, int(summary["model_seed"]))
    output_dir = args.results_root / "calibration_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== Held-out position/run calibration transfer =====", flush=True)
    print("Frozen leave-one-position models; no retraining or test-event tuning.", flush=True)
    print("Each method receives the same earliest labelled reference events per position.", flush=True)
    rows = []
    all_offsets = []
    for count in args.calibration_counts:
        row, offsets = calibrated_metrics(folds, count)
        row["calibration_events_per_position"] = count
        row["sigma_improvement_cm"] = (
            row["traditional"]["gaussian_fit"]["sigma_cm"]
            - row["hybrid"]["gaussian_fit"]["sigma_cm"]
        )
        row["rmse_improvement_cm"] = row["traditional"]["rmse_cm"] - row["hybrid"]["rmse_cm"]
        rows.append(row)
        all_offsets.extend(offsets)
        print(
            f"K={count:>3} | eval={row['evaluation_events']:>5} | traditional sigma="
            f"{row['traditional']['gaussian_fit']['sigma_cm']:.3f} cm | hybrid sigma="
            f"{row['hybrid']['gaussian_fit']['sigma_cm']:.3f} cm | improvement="
            f"{row['sigma_improvement_cm']:.3f} cm | traditional/hybrid RMSE="
            f"{row['traditional']['rmse_cm']:.3f}/{row['hybrid']['rmse_cm']:.3f}",
            flush=True,
        )
    result = {
        "protocol": "per-held-position constant offset from earliest K labelled reference events",
        "rows": rows,
        "per_position_offsets": all_offsets,
    }
    output = output_dir / "calibration_transfer.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    figure = output_dir / "calibration_transfer.png"
    plot_transfer(figure, rows)
    print(f"figure : {figure}", flush=True)
    print(f"metrics: {output}", flush=True)
    print("OHBA CALIBRATION TRANSFER: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
