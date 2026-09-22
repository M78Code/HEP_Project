"""Quantify reference-event selection sensitivity for calibration transfer.

Frozen leave-one-position/run predictions are reused.  For each held-out
position, a random subset of its first reference-pool events estimates one
constant offset; only later events are evaluated.  Repeats therefore measure
the stability of a practical small-reference calibration step, not retraining
variance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import Scintillator_Project
from Scintillator_Project.src.scripts.evaluate_ohba_calibration_transfer import fold_prediction_data
from Scintillator_Project.src.scripts.reproduce_ohba_traditional import fit_residual_sigma


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
    parser.add_argument("--reference-pool", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def evaluate_once(
    folds: list[dict[str, np.ndarray | int | list[float]]],
    count: int,
    pool_size: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    labels_all, traditional_all, hybrid_all = [], [], []
    for fold_data in folds:
        labels = fold_data["labels"]
        traditional = fold_data["traditional"]
        hybrid = fold_data["hybrid"]
        event_ids = fold_data["event_ids"]
        positions = np.asarray(fold_data["positions"], dtype=np.float32)
        for position in positions:
            indices = np.flatnonzero(labels == position)
            ordered = indices[np.argsort(event_ids[indices], kind="stable")]
            if ordered.size <= pool_size or count > pool_size:
                raise ValueError(
                    f"position {position:.0f} has {ordered.size} events; need more than pool size {pool_size}"
                )
            calibration_pool = ordered[:pool_size]
            evaluation = ordered[pool_size:]
            if count:
                calibration = rng.choice(calibration_pool, size=count, replace=False)
                traditional_offset = np.mean(traditional[calibration] - labels[calibration])
                hybrid_offset = np.mean(hybrid[calibration] - labels[calibration])
            else:
                traditional_offset = 0.0
                hybrid_offset = 0.0
            labels_all.append(labels[evaluation])
            traditional_all.append(traditional[evaluation] - traditional_offset)
            hybrid_all.append(hybrid[evaluation] - hybrid_offset)
    labels = np.concatenate(labels_all)
    traditional = np.concatenate(traditional_all)
    hybrid = np.concatenate(hybrid_all)
    traditional_residual = traditional - labels
    hybrid_residual = hybrid - labels
    traditional_sigma = fit_residual_sigma(traditional_residual, -40.0, 40.0, 0.5)["sigma_cm"]
    hybrid_sigma = fit_residual_sigma(hybrid_residual, -40.0, 40.0, 0.5)["sigma_cm"]
    traditional_rmse = float(np.sqrt(np.mean(traditional_residual ** 2)))
    hybrid_rmse = float(np.sqrt(np.mean(hybrid_residual ** 2)))
    return traditional_sigma, hybrid_sigma, traditional_rmse, hybrid_rmse


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]


def main() -> None:
    args = parse_args()
    if args.reference_pool < max(args.calibration_counts) or args.repeats < 1:
        raise ValueError("reference pool must cover calibration counts and repeats must be positive")
    summary_path = args.results_root / "position_holdout_summary.json"
    definition_path = args.results_root / "fold_definition.json"
    with np.load(args.cache) as data:
        cache = {name: data[name] for name in data.files}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    folds = fold_prediction_data(args.results_root, cache, definition, int(summary["model_seed"]))
    output_dir = args.results_root / "calibration_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== Calibration-reference resampling =====", flush=True)
    print(
        f"Reference pool: earliest {args.reference_pool} events per held-out position; "
        "evaluation starts after that pool.",
        flush=True,
    )
    print(f"repeats={args.repeats}; seed={args.seed}; frozen leave-one-position models", flush=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for count in args.calibration_counts:
        values = np.empty((1 if count == 0 else args.repeats, 4), dtype=np.float64)
        for repeat in range(values.shape[0]):
            values[repeat] = evaluate_once(folds, count, args.reference_pool, rng)
            if values.shape[0] >= 100 and (repeat + 1) % 100 == 0:
                print(f"[PROGRESS] K={count}: {repeat + 1}/{values.shape[0]}", flush=True)
        improvement = values[:, 0] - values[:, 1]
        rmse_improvement = values[:, 2] - values[:, 3]
        row = {
            "calibration_events_per_position": count,
            "evaluation_events": int(sum(
                np.asarray(fold["labels"]).size - args.reference_pool * len(fold["positions"])
                for fold in folds
            )),
            "traditional_sigma_cm": percentile_interval(values[:, 0]),
            "hybrid_sigma_cm": percentile_interval(values[:, 1]),
            "sigma_improvement_cm": percentile_interval(improvement),
            "traditional_rmse_cm": percentile_interval(values[:, 2]),
            "hybrid_rmse_cm": percentile_interval(values[:, 3]),
            "rmse_improvement_cm": percentile_interval(rmse_improvement),
            "probability_hybrid_sigma_better": float(np.mean(improvement > 0.0)),
        }
        rows.append(row)
        delta = row["sigma_improvement_cm"]
        print(
            f"K={count:>3} | traditional sigma={row['traditional_sigma_cm'][1]:.3f} "
            f"[{row['traditional_sigma_cm'][0]:.3f}, {row['traditional_sigma_cm'][2]:.3f}] | "
            f"hybrid={row['hybrid_sigma_cm'][1]:.3f} "
            f"[{row['hybrid_sigma_cm'][0]:.3f}, {row['hybrid_sigma_cm'][2]:.3f}] | "
            f"improvement={delta[1]:.3f} [{delta[0]:.3f}, {delta[2]:.3f}] | "
            f"P(hybrid better)={row['probability_hybrid_sigma_better']:.3f}",
            flush=True,
        )
    result = {
        "protocol": "random reference subsets from the first fixed event pool; later events only for evaluation",
        "reference_pool_events_per_position": args.reference_pool,
        "repeats": args.repeats,
        "seed": args.seed,
        "rows": rows,
    }
    output = output_dir / "calibration_reference_resampling.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"metrics: {output}", flush=True)
    print("OHBA CALIBRATION-REFERENCE RESAMPLING: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
