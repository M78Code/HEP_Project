"""Cross-validate reconstruction on positions and runs never seen in training.

Each fold holds out complete measured positions, which also holds out their
corresponding acquisition files.  Remaining positions are split by event ID:
the earliest 85% train the calibration/model and the latest 15% validate early
stopping.  Every usable event appears once in a held-out-position test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import Scintillator_Project
from Scintillator_Project.src.scripts.repeat_ohba_all_usable_splits import (
    ensure_cache,
    gaussian_metrics,
    source_files,
    train_trial,
)


PROJECT_ROOT = Path(Scintillator_Project.__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "dataset/split")
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "dataset/cache/ohba_all_usable_events_roi450_800.npz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/ohba_all_usable_position_holdout",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model-seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def balanced_position_groups(labels: np.ndarray, folds: int) -> list[list[float]]:
    positions, counts = np.unique(labels, return_counts=True)
    if folds < 2 or folds > positions.size:
        raise ValueError(f"--folds must be between 2 and {positions.size}")
    groups: list[list[float]] = [[] for _ in range(folds)]
    totals = np.zeros(folds, dtype=np.int64)
    for position, count in sorted(zip(positions, counts), key=lambda item: -item[1]):
        index = int(np.argmin(totals))
        groups[index].append(float(position))
        totals[index] += int(count)
    return groups


def make_fold_split(
    labels: np.ndarray,
    event_ids: np.ndarray,
    test_positions: list[float],
) -> dict[str, np.ndarray]:
    test_mask = np.isin(labels, test_positions)
    train_parts, val_parts = [], []
    for position in np.unique(labels[~test_mask]):
        indices = np.flatnonzero(labels == position)
        ordered = indices[np.argsort(event_ids[indices], kind="stable")]
        train_stop = int(ordered.size * 0.85)
        train_parts.append(ordered[:train_stop])
        val_parts.append(ordered[train_stop:])
    return {
        "train": np.concatenate(train_parts),
        "val": np.concatenate(val_parts),
        "test": np.flatnonzero(test_mask),
    }


def serialize_groups(groups: list[list[float]], labels: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for fold, positions in enumerate(groups):
        mask = np.isin(labels, positions)
        rows.append({
            "fold": fold,
            "test_positions_cm": positions,
            "test_events": int(mask.sum()),
        })
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_cache(source_files(args.split_dir), args.cache)
    with np.load(args.cache) as data:
        arrays = {name: data[name] for name in data.files}

    groups = balanced_position_groups(arrays["labels"], args.folds)
    fold_audit = serialize_groups(groups, arrays["labels"])
    (args.output_root / "fold_definition.json").write_text(
        json.dumps(fold_audit, indent=2) + "\n", encoding="utf-8"
    )
    print("===== Position/run holdout benchmark =====", flush=True)
    print("Test positions are unseen by calibration, training, and validation.", flush=True)
    print("Development positions: earliest 85% train | latest 15% validation", flush=True)
    for row in fold_audit:
        positions = ", ".join(f"{value:.0f}" for value in row["test_positions_cm"])
        print(f"fold={row['fold']} | test positions=[{positions}] cm | events={row['test_events']}", flush=True)

    pooled_labels, pooled_traditional, pooled_hybrid = [], [], []
    results = []
    for row in fold_audit:
        fold = int(row["fold"])
        split = make_fold_split(arrays["labels"], arrays["event_ids"], row["test_positions_cm"])
        trial_dir = args.output_root / f"fold_{fold}_model_{args.model_seed}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[FOLD START] fold={fold} | train/val/test="
            f"{split['train'].size}/{split['val'].size}/{split['test'].size}",
            flush=True,
        )
        report = train_trial(arrays, split, args, trial_dir)
        with np.load(trial_dir / "predictions.npz") as prediction:
            pooled_labels.append(prediction["labels"])
            pooled_traditional.append(prediction["traditional_baseline"])
            pooled_hybrid.append(prediction["hybrid_prediction"])
        traditional = report["traditional"]
        hybrid = report["hybrid"]
        result = {
            **row,
            "traditional_sigma_cm": traditional["gaussian_fit"]["sigma_cm"],
            "hybrid_sigma_cm": hybrid["gaussian_fit"]["sigma_cm"],
            "improvement_cm": traditional["gaussian_fit"]["sigma_cm"] - hybrid["gaussian_fit"]["sigma_cm"],
            "traditional_rmse_cm": traditional["rmse_cm"],
            "hybrid_rmse_cm": hybrid["rmse_cm"],
        }
        results.append(result)
        print(
            f"[FOLD DONE] fold={fold} | traditional={result['traditional_sigma_cm']:.3f} cm | "
            f"hybrid={result['hybrid_sigma_cm']:.3f} cm | improvement={result['improvement_cm']:.3f} cm",
            flush=True,
        )

    labels = np.concatenate(pooled_labels)
    traditional = np.concatenate(pooled_traditional)
    hybrid = np.concatenate(pooled_hybrid)
    if labels.size != arrays["labels"].size:
        raise RuntimeError(f"pooled test coverage is {labels.size}, expected {arrays['labels'].size}")
    pooled = {
        "events": int(labels.size),
        "traditional": gaussian_metrics(labels, traditional),
        "hybrid": gaussian_metrics(labels, hybrid),
    }
    pooled["improvement_cm"] = (
        pooled["traditional"]["gaussian_fit"]["sigma_cm"]
        - pooled["hybrid"]["gaussian_fit"]["sigma_cm"]
    )
    improvements = np.asarray([row["improvement_cm"] for row in results])
    summary = {
        "protocol": (
            f"{args.folds}-fold complete-position/run holdout; "
            "chronological development split"
        ),
        "model_seed": args.model_seed,
        "folds": results,
        "pooled_out_of_position": pooled,
        "fold_improvement_mean_cm": float(improvements.mean()),
        "fold_improvement_sample_std_cm": float(improvements.std(ddof=1)),
        "all_folds_hybrid_better": bool(np.all(improvements > 0.0)),
    }
    output = args.output_root / "position_holdout_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("===== Position/run holdout summary =====", flush=True)
    for row in results:
        print(
            f"fold={row['fold']} | traditional={row['traditional_sigma_cm']:.3f} | "
            f"hybrid={row['hybrid_sigma_cm']:.3f} | improvement={row['improvement_cm']:.3f}",
            flush=True,
        )
    print(
        f"pooled unseen-position test: traditional="
        f"{pooled['traditional']['gaussian_fit']['sigma_cm']:.3f} cm | hybrid="
        f"{pooled['hybrid']['gaussian_fit']['sigma_cm']:.3f} cm | improvement="
        f"{pooled['improvement_cm']:.3f} cm",
        flush=True,
    )
    print(
        f"mean fold improvement={summary['fold_improvement_mean_cm']:.3f} +/- "
        f"{summary['fold_improvement_sample_std_cm']:.3f} cm | "
        f"all hybrid better={summary['all_folds_hybrid_better']}",
        flush=True,
    )
    print(f"saved: {output}", flush=True)
    print("OHBA POSITION/RUN HOLDOUT BENCHMARK: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
