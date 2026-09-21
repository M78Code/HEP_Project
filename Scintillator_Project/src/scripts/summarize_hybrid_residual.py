import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    reports = []
    for path in sorted(args.result_root.glob("seed_*/metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            reports.append(json.load(handle))
    if not reports:
        raise RuntimeError(f"no seed metrics found under {args.result_root}")

    print("\n===== Scintillator hybrid residual summary =====")
    print("seed       baseline RMSE/sigma    hybrid RMSE/sigma    sigma@95% coverage")
    for report in reports:
        baseline = report["physics_baseline"]
        hybrid = report["hybrid_all_events"]
        selected = report["hybrid_by_validation_uncertainty"]["coverage_0.95"]
        print(
            f"{report['seed']:<10d} "
            f"{baseline['rmse_cm']:.3f}/{baseline['residual_std_cm']:.3f}          "
            f"{hybrid['rmse_cm']:.3f}/{hybrid['residual_std_cm']:.3f}          "
            f"{selected['residual_std_cm']:.3f} "
            f"(kept {selected['actual_test_coverage'] * 100:.1f}%)"
        )

    summary = {}
    for name, getter in {
        "hybrid_rmse_cm": lambda r: r["hybrid_all_events"]["rmse_cm"],
        "hybrid_residual_std_cm": lambda r: r["hybrid_all_events"]["residual_std_cm"],
        "hybrid_residual_std_at_95pct_cm": lambda r: r["hybrid_by_validation_uncertainty"]["coverage_0.95"]["residual_std_cm"],
    }.items():
        values = np.asarray([getter(report) for report in reports])
        summary[name] = {"mean": float(values.mean()), "std": float(values.std())}
        print(f"{name}: {values.mean():.4f} +/- {values.std():.4f}")

    with (args.result_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"summary: {args.result_root / 'summary.json'}")


if __name__ == "__main__":
    main()
