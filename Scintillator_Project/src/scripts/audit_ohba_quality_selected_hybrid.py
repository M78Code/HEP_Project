"""Audit core resolution, tails, and bootstrap uncertainty for one ML run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Scintillator_Project.src.scripts.reproduce_ohba_traditional import fit_residual_sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def metric(labels: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    residual = prediction - labels
    fit = fit_residual_sigma(residual, -40.0, 40.0, 0.5)
    return {
        "events": int(residual.size),
        "gaussian_sigma_cm": fit["sigma_cm"],
        "gaussian_sigma_error_cm": fit["sigma_error_cm"],
        "fit_range_retained_fraction": float((np.abs(residual) <= 40.0).mean()),
        "rmse_cm": float(np.sqrt(np.mean(residual ** 2))),
        "mae_cm": float(np.mean(np.abs(residual))),
        "bias_cm": float(np.mean(residual)),
        "raw_std_cm": float(np.std(residual)),
        "absolute_residual_fraction_gt_5cm": float((np.abs(residual) > 5.0).mean()),
        "absolute_residual_fraction_gt_10cm": float((np.abs(residual) > 10.0).mean()),
        "absolute_residual_fraction_gt_20cm": float((np.abs(residual) > 20.0).mean()),
        "absolute_residual_fraction_gt_40cm": float((np.abs(residual) > 40.0).mean()),
    }


def stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    for position in np.unique(labels):
        indices = np.flatnonzero(labels == position)
        pieces.append(rng.choice(indices, size=indices.size, replace=True))
    return np.concatenate(pieces)


def bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    model: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    base_sigma = np.empty(repeats, dtype=np.float64)
    model_sigma = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = stratified_indices(labels, rng)
        base_sigma[index] = fit_residual_sigma(baseline[sampled] - labels[sampled], -40.0, 40.0, 0.5)["sigma_cm"]
        model_sigma[index] = fit_residual_sigma(model[sampled] - labels[sampled], -40.0, 40.0, 0.5)["sigma_cm"]
        if (index + 1) % 250 == 0 or index + 1 == repeats:
            print(f"[BOOTSTRAP] {index + 1}/{repeats}", flush=True)
    improvement = base_sigma - model_sigma
    return {
        "repeats": repeats,
        "stratification": "true position",
        "baseline_sigma_95ci_cm": np.quantile(base_sigma, [0.025, 0.975]).tolist(),
        "hybrid_sigma_95ci_cm": np.quantile(model_sigma, [0.025, 0.975]).tolist(),
        "improvement_cm": float(np.mean(improvement)),
        "improvement_95ci_cm": np.quantile(improvement, [0.025, 0.975]).tolist(),
        "probability_hybrid_better": float((improvement > 0.0).mean()),
        "samples": {
            "baseline_sigma_cm": base_sigma.tolist(),
            "hybrid_sigma_cm": model_sigma.tolist(),
        },
    }


def per_position(labels: np.ndarray, baseline: np.ndarray, model: np.ndarray) -> dict[str, object]:
    result = {}
    for position in np.unique(labels):
        mask = labels == position
        try:
            result[str(int(position))] = {
                "events": int(mask.sum()),
                "traditional": metric(labels[mask], baseline[mask]),
                "hybrid": metric(labels[mask], model[mask]),
            }
        except RuntimeError as exc:
            result[str(int(position))] = {"events": int(mask.sum()), "error": str(exc)}
    return result


def make_figure(path: Path, labels: np.ndarray, baseline: np.ndarray, model: np.ndarray) -> None:
    residuals = {"Traditional": baseline - labels, "Hybrid": model - labels}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bins = np.arange(-40.0, 40.5, 0.5)
    for name, residual in residuals.items():
        axes[0].hist(residual, bins=bins, histtype="step", linewidth=1.6, label=name)
    axes[0].set_xlabel("Position residual [cm]")
    axes[0].set_ylabel("Events / 0.5 cm")
    axes[0].set_title("Full residual distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    positions = np.unique(labels)
    traditional_rmse = [np.sqrt(np.mean((baseline[labels == pos] - pos) ** 2)) for pos in positions]
    hybrid_rmse = [np.sqrt(np.mean((model[labels == pos] - pos) ** 2)) for pos in positions]
    axes[1].plot(positions, traditional_rmse, marker="o", label="Traditional")
    axes[1].plot(positions, hybrid_rmse, marker="o", label="Hybrid")
    axes[1].set_xlabel("True position [cm]")
    axes[1].set_ylabel("Per-position RMSE [cm]")
    axes[1].set_title("Position-dependent tails")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats <= 0:
        raise ValueError("--bootstrap-repeats must be positive")
    data = np.load(args.predictions)
    labels = data["labels"].astype(np.float64)
    baseline = data["traditional_baseline"].astype(np.float64)
    model = data["hybrid_prediction"].astype(np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("===== Quality-selected hybrid audit =====", flush=True)
    print(f"predictions: {args.predictions}", flush=True)
    print(f"events: {labels.size}; bootstrap repeats: {args.bootstrap_repeats}", flush=True)
    report = {
        "traditional": metric(labels, baseline),
        "hybrid": metric(labels, model),
        "per_position": per_position(labels, baseline, model),
    }
    print(
        f"traditional sigma={report['traditional']['gaussian_sigma_cm']:.3f} cm | "
        f"hybrid sigma={report['hybrid']['gaussian_sigma_cm']:.3f} cm",
        flush=True,
    )
    print(
        f"traditional/hybrid RMSE={report['traditional']['rmse_cm']:.3f}/"
        f"{report['hybrid']['rmse_cm']:.3f} cm | "
        f"tail >20 cm={report['traditional']['absolute_residual_fraction_gt_20cm']:.3%}/"
        f"{report['hybrid']['absolute_residual_fraction_gt_20cm']:.3%}",
        flush=True,
    )
    report["bootstrap"] = bootstrap(labels, baseline, model, args.bootstrap_repeats, args.seed)
    bootstrap_summary = report["bootstrap"]
    print(
        f"paired improvement={bootstrap_summary['improvement_cm']:.3f} cm | "
        f"95% CI={bootstrap_summary['improvement_95ci_cm']} | "
        f"P(hybrid better)={bootstrap_summary['probability_hybrid_better']:.4f}",
        flush=True,
    )
    (args.output_dir / "quality_selected_hybrid_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(args.output_dir / "quality_selected_hybrid_audit.png", labels, baseline, model)
    print(f"output: {args.output_dir}", flush=True)
    print("QUALITY-SELECTED HYBRID AUDIT: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
