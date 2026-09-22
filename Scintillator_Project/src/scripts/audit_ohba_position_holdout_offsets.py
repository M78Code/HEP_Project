"""Diagnose fold-dependent absolute-position offsets in position holdout tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import Scintillator_Project
from Scintillator_Project.src.scripts.train_ohba_quality_selected_hybrid import gaussian_metrics


PROJECT_ROOT = Path(Scintillator_Project.__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results/ohba_all_usable_leave_one_position_out",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.results_root / "position_holdout_summary.json"
    folds_path = args.results_root / "fold_definition.json"
    if not summary_path.is_file() or not folds_path.is_file():
        raise FileNotFoundError("position-holdout summary or fold definition is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fold_definition = json.loads(folds_path.read_text(encoding="utf-8"))
    model_seed = int(summary["model_seed"])

    rows = []
    traditional_residuals, hybrid_residuals = [], []
    centered_traditional, centered_hybrid = [], []
    print("===== Position-holdout offset audit =====", flush=True)
    print("Each fold uses an unseen position/run. Centered values are an oracle diagnostic only.", flush=True)
    print(
        "fold  position(s)       N  trad bias  hybrid bias  trad sigma  hybrid sigma",
        flush=True,
    )
    for definition in fold_definition:
        fold = int(definition["fold"])
        path = args.results_root / f"fold_{fold}_model_{model_seed}" / "predictions.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing fold predictions: {path}")
        with np.load(path) as data:
            labels = data["labels"]
            traditional = data["traditional_baseline"]
            hybrid = data["hybrid_prediction"]
        residual_traditional = traditional - labels
        residual_hybrid = hybrid - labels
        traditional_metric = gaussian_metrics(labels, traditional)
        hybrid_metric = gaussian_metrics(labels, hybrid)
        positions = ",".join(f"{position:.0f}" for position in definition["test_positions_cm"])
        row = {
            "fold": fold,
            "test_positions_cm": definition["test_positions_cm"],
            "events": int(labels.size),
            "traditional_bias_cm": traditional_metric["bias_cm"],
            "hybrid_bias_cm": hybrid_metric["bias_cm"],
            "traditional_sigma_cm": traditional_metric["gaussian_fit"]["sigma_cm"],
            "hybrid_sigma_cm": hybrid_metric["gaussian_fit"]["sigma_cm"],
            "traditional_rmse_cm": traditional_metric["rmse_cm"],
            "hybrid_rmse_cm": hybrid_metric["rmse_cm"],
        }
        rows.append(row)
        print(
            f"{fold:>4}  {positions:<15} {labels.size:>5} "
            f"{row['traditional_bias_cm']:>10.3f} {row['hybrid_bias_cm']:>12.3f} "
            f"{row['traditional_sigma_cm']:>11.3f} {row['hybrid_sigma_cm']:>13.3f}",
            flush=True,
        )
        traditional_residuals.append(residual_traditional)
        hybrid_residuals.append(residual_hybrid)
        centered_traditional.append(residual_traditional - residual_traditional.mean())
        centered_hybrid.append(residual_hybrid - residual_hybrid.mean())

    raw_traditional = np.concatenate(traditional_residuals)
    raw_hybrid = np.concatenate(hybrid_residuals)
    oracle_traditional = np.concatenate(centered_traditional)
    oracle_hybrid = np.concatenate(centered_hybrid)
    zeros = np.zeros(raw_traditional.size, dtype=np.float64)
    raw = {
        "traditional": gaussian_metrics(zeros, raw_traditional),
        "hybrid": gaussian_metrics(zeros, raw_hybrid),
    }
    oracle = {
        "traditional": gaussian_metrics(zeros, oracle_traditional),
        "hybrid": gaussian_metrics(zeros, oracle_hybrid),
    }
    result = {"per_fold": rows, "pooled_raw": raw, "pooled_oracle_centered": oracle}
    output = args.results_root / "position_holdout_offset_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("===== Pooled result =====", flush=True)
    print(
        f"raw: traditional sigma={raw['traditional']['gaussian_fit']['sigma_cm']:.3f} cm | "
        f"hybrid sigma={raw['hybrid']['gaussian_fit']['sigma_cm']:.3f} cm",
        flush=True,
    )
    print(
        f"oracle per-fold centered: traditional sigma="
        f"{oracle['traditional']['gaussian_fit']['sigma_cm']:.3f} cm | hybrid sigma="
        f"{oracle['hybrid']['gaussian_fit']['sigma_cm']:.3f} cm",
        flush=True,
    )
    print(f"saved: {output}", flush=True)
    print("OHBA POSITION-HOLDOUT OFFSET AUDIT: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
