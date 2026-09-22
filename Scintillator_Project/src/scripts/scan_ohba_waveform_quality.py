"""Select waveform-quality cuts for a leakage-free Ohba-style baseline.

The unpublished 4.72 cm result is known to use waveform-quality selection,
but its exact cuts were not preserved.  This scan therefore does not claim to
reproduce that number.  It tests a transparent, physically motivated proxy:
the smaller CH0/CH1 pulse signal-to-noise ratio, with both peaks required to
fall in the verified 450:800 pulse window.

Calibration and fusion weights are fitted on train only.  Candidate cuts are
chosen on validation only; the selected configuration is evaluated once on
the held-out test split.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import Scintillator_Project
from Scintillator_Project.src.scripts.reproduce_ohba_traditional import (
    cfd_time_ns,
    fit_residual_sigma,
    modal_calibration,
)


PROJECT_ROOT = Path(Scintillator_Project.__file__).resolve().parent


@dataclass(frozen=True)
class QualityConfig:
    baseline_stop: int = 400
    roi_start: int = 450
    roi_stop: int = 800
    cfd_fraction: float = 0.20
    mode_bins: int = 120
    residual_min: float = -40.0
    residual_max: float = 40.0
    residual_bin_width: float = 0.5
    fusion_weight_step: float = 0.01


def parse_retentions(text: str) -> list[float]:
    try:
        values = [float(value) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("retentions must be comma-separated numbers") from exc
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("each retention must satisfy 0 < value <= 1")
    return sorted(set(values), reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir", type=Path, default=PROJECT_ROOT / "dataset/split",
        help="Existing stratified train/val/test JSON directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results/ohba_quality_selection",
    )
    parser.add_argument(
        "--retentions", type=parse_retentions,
        default=parse_retentions("1.00,0.99,0.98,0.95,0.90,0.85,0.80"),
        help="Train-SNR retention candidates, selected on validation.",
    )
    parser.add_argument(
        "--minimum-validation-retention", type=float, default=0.80,
        help="Do not select a cut retaining less than this validation fraction.",
    )
    parser.add_argument("--minimum-train-events-per-position", type=int, default=80)
    return parser.parse_args()


def extract_split_features(path: Path, config: QualityConfig) -> dict[str, np.ndarray]:
    """Extract traditional observables and label-free waveform quality values."""
    with path.open("r", encoding="utf-8") as handle:
        events = json.load(handle)["events"]

    positions: list[float] = []
    charge_ratios: list[float] = []
    time_differences: list[float] = []
    snrs: list[float] = []
    peak_inside: list[bool] = []

    for event in events:
        left_raw = np.asarray(event["CH0"], dtype=np.float64)
        right_raw = np.asarray(event["CH1"], dtype=np.float64)
        left_baseline = float(np.median(left_raw[:config.baseline_stop]))
        right_baseline = float(np.median(right_raw[:config.baseline_stop]))
        left_noise = float(np.std(left_raw[:config.baseline_stop]))
        right_noise = float(np.std(right_raw[:config.baseline_stop]))
        left_pulse = np.maximum(left_baseline - left_raw, 0.0)
        right_pulse = np.maximum(right_baseline - right_raw, 0.0)

        left_peak = int(np.argmax(left_pulse))
        right_peak = int(np.argmax(right_pulse))
        left_charge = float(left_pulse[config.roi_start:config.roi_stop].sum())
        right_charge = float(right_pulse[config.roi_start:config.roi_stop].sum())
        left_time = cfd_time_ns(left_pulse, config.cfd_fraction)
        right_time = cfd_time_ns(right_pulse, config.cfd_fraction)

        if (
            left_charge <= 0.0
            or right_charge <= 0.0
            or not np.isfinite(left_time)
            or not np.isfinite(right_time)
        ):
            continue

        positions.append(float(event["position_label"]))
        charge_ratios.append(float(np.log(right_charge / left_charge)))
        time_differences.append(left_time - right_time)
        snrs.append(float(min(
            left_pulse[left_peak] / max(left_noise, 1e-12),
            right_pulse[right_peak] / max(right_noise, 1e-12),
        )))
        peak_inside.append(
            config.roi_start <= left_peak < config.roi_stop
            and config.roi_start <= right_peak < config.roi_stop
        )

    del events
    gc.collect()
    return {
        "position": np.asarray(positions, dtype=np.float64),
        "log_charge_ratio": np.asarray(charge_ratios, dtype=np.float64),
        "time_difference_ns": np.asarray(time_differences, dtype=np.float64),
        "min_channel_snr": np.asarray(snrs, dtype=np.float64),
        "peak_inside_roi": np.asarray(peak_inside, dtype=bool),
    }


def calibration_from_train(
    values: dict[str, np.ndarray], mask: np.ndarray, config: QualityConfig
) -> dict[str, float]:
    position = values["position"][mask]
    _, _, charge_intercept, charge_slope = modal_calibration(
        values["log_charge_ratio"][mask], position, config.mode_bins
    )
    _, _, time_intercept, time_slope = modal_calibration(
        values["time_difference_ns"][mask], position, config.mode_bins
    )
    return {
        "charge_intercept": charge_intercept,
        "charge_slope": charge_slope,
        "time_intercept": time_intercept,
        "time_slope": time_slope,
    }


def residuals(
    values: dict[str, np.ndarray], mask: np.ndarray, calibration: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    position = values["position"][mask]
    charge_position = (
        values["log_charge_ratio"][mask] - calibration["charge_intercept"]
    ) / calibration["charge_slope"]
    time_position = (
        values["time_difference_ns"][mask] - calibration["time_intercept"]
    ) / calibration["time_slope"]
    return charge_position - position, time_position - position


def best_fusion(
    charge_residual: np.ndarray,
    time_residual: np.ndarray,
    config: QualityConfig,
) -> tuple[float, dict[str, object]]:
    best_weight: float | None = None
    best_metric: dict[str, object] | None = None
    weights = np.arange(0.0, 1.0 + config.fusion_weight_step / 2.0, config.fusion_weight_step)
    for weight in weights:
        metric = fit_residual_sigma(
            weight * charge_residual + (1.0 - weight) * time_residual,
            config.residual_min,
            config.residual_max,
            config.residual_bin_width,
        )
        if best_metric is None or metric["sigma_cm"] < best_metric["sigma_cm"]:
            best_weight, best_metric = float(weight), metric
    assert best_weight is not None and best_metric is not None
    return best_weight, best_metric


def compact_metrics(
    charge_residual: np.ndarray,
    time_residual: np.ndarray,
    weight: float,
    config: QualityConfig,
    include_histogram: bool = False,
) -> dict[str, object]:
    fusion_residual = weight * charge_residual + (1.0 - weight) * time_residual
    metrics = {
        "charge": fit_residual_sigma(
            charge_residual, config.residual_min, config.residual_max, config.residual_bin_width
        ),
        "cfd": fit_residual_sigma(
            time_residual, config.residual_min, config.residual_max, config.residual_bin_width
        ),
        "fusion": fit_residual_sigma(
            fusion_residual, config.residual_min, config.residual_max, config.residual_bin_width
        ),
    }
    metrics["fusion"]["charge_weight"] = weight
    if include_histogram:
        return metrics
    for metric in metrics.values():
        metric.pop("histogram_edges_cm", None)
        metric.pop("histogram_counts", None)
        metric.pop("fit_curve_counts", None)
    return metrics


def make_figure(path: Path, candidates: list[dict[str, object]], selected: dict[str, object]) -> None:
    valid = [row for row in candidates if row["eligible"]]
    retention = [row["validation_retention"] * 100.0 for row in valid]
    sigma = [row["validation_metrics"]["fusion"]["sigma_cm"] for row in valid]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(retention, sigma, marker="o", color="#2878b5")
    ax.scatter(
        [selected["validation_retention"] * 100.0],
        [selected["validation_metrics"]["fusion"]["sigma_cm"]],
        s=70,
        color="#d8572a",
        zorder=3,
        label="validation-selected cut",
    )
    ax.set_xlabel("Validation events retained [%]")
    ax.set_ylabel("CFD + charge Gaussian sigma [cm]")
    ax.set_title("Waveform-quality selection scan")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.minimum_validation_retention <= 1.0:
        raise ValueError("--minimum-validation-retention must satisfy 0 < value <= 1")
    config = QualityConfig()
    split_paths = {name: args.split_dir / f"{name}.json" for name in ("train", "val", "test")}
    if missing := [str(path) for path in split_paths.values() if not path.is_file()]:
        raise FileNotFoundError("missing split files: " + ", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("===== Ohba waveform-quality selection scan =====", flush=True)
    print("Calibration: train only | cut selection: validation only | final metric: test only", flush=True)
    print("Quality proxy: min(CH0, CH1) pulse SNR; both pulse peaks must be in 450:800", flush=True)
    values = {}
    for name, path in split_paths.items():
        print(f"[LOAD] {name}: {path}", flush=True)
        values[name] = extract_split_features(path, config)
        print(
            f"[READY] {name}: {values[name]['position'].size} valid events | "
            f"peak-in-ROI={values[name]['peak_inside_roi'].mean() * 100.0:.3f}%",
            flush=True,
        )

    train_base = values["train"]["peak_inside_roi"]
    train_scores = values["train"]["min_channel_snr"][train_base]
    candidates: list[dict[str, object]] = []
    for target_retention in args.retentions:
        threshold = (
            float(np.nextafter(train_scores.min(), -np.inf))
            if target_retention == 1.0
            else float(np.quantile(train_scores, 1.0 - target_retention))
        )
        masks = {
            name: values[name]["peak_inside_roi"] & (values[name]["min_channel_snr"] >= threshold)
            for name in values
        }
        train_counts = {
            str(int(position)): int(((values["train"]["position"] == position) & masks["train"]).sum())
            for position in np.unique(values["train"]["position"])
        }
        eligible = (
            min(train_counts.values()) >= args.minimum_train_events_per_position
            and float(masks["val"].mean()) >= args.minimum_validation_retention
        )
        row: dict[str, object] = {
            "target_train_retention": target_retention,
            "snr_threshold": threshold,
            "train_retention": float(masks["train"].mean()),
            "validation_retention": float(masks["val"].mean()),
            "test_retention": float(masks["test"].mean()),
            "train_events_per_position": train_counts,
            "eligible": eligible,
        }
        if eligible:
            calibration = calibration_from_train(values["train"], masks["train"], config)
            train_charge, train_time = residuals(values["train"], masks["train"], calibration)
            weight, _ = best_fusion(train_charge, train_time, config)
            val_charge, val_time = residuals(values["val"], masks["val"], calibration)
            row["calibration"] = calibration
            row["charge_weight"] = weight
            row["validation_metrics"] = compact_metrics(val_charge, val_time, weight, config)
        else:
            row["reason"] = "insufficient per-position train support or validation retention"
        candidates.append(row)
        message = (
            f"retention target={target_retention:.2f} | val retained={row['validation_retention']:.3f} | "
            f"eligible={eligible}"
        )
        if eligible:
            message += (
                f" | val fusion sigma={row['validation_metrics']['fusion']['sigma_cm']:.3f} cm"
                f" | w={weight:.2f}"
            )
        print(message, flush=True)

    eligible_candidates = [row for row in candidates if row["eligible"]]
    if not eligible_candidates:
        raise RuntimeError("no quality cut satisfies the support and retention constraints")
    selected = min(eligible_candidates, key=lambda row: row["validation_metrics"]["fusion"]["sigma_cm"])
    threshold = float(selected["snr_threshold"])
    selected_masks = {
        name: values[name]["peak_inside_roi"] & (values[name]["min_channel_snr"] >= threshold)
        for name in values
    }
    calibration = selected["calibration"]
    train_charge, train_time = residuals(values["train"], selected_masks["train"], calibration)
    weight = float(selected["charge_weight"])
    test_charge, test_time = residuals(values["test"], selected_masks["test"], calibration)
    final = {
        "protocol": "train calibration, validation cut selection, one held-out test evaluation",
        "quality_proxy": "min channel pulse SNR with both peak indices inside ROI",
        "config": asdict(config),
        "selection_constraints": {
            "minimum_validation_retention": args.minimum_validation_retention,
            "minimum_train_events_per_position": args.minimum_train_events_per_position,
        },
        "candidates": candidates,
        "selected": {
            **selected,
            "test_metrics": compact_metrics(test_charge, test_time, weight, config, include_histogram=True),
            "test_events": int(selected_masks["test"].sum()),
            "test_retention": float(selected_masks["test"].mean()),
            "train_metrics": compact_metrics(train_charge, train_time, weight, config),
        },
    }
    output_json = args.output_dir / "quality_selection_results.json"
    output_json.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    make_figure(args.output_dir / "quality_selection_validation_scan.png", candidates, final["selected"])

    test_fusion = final["selected"]["test_metrics"]["fusion"]
    print("===== Selected quality cut: held-out test =====", flush=True)
    print(
        f"train target retention={selected['target_train_retention']:.2f} | "
        f"SNR threshold={threshold:.3f} | test retained={final['selected']['test_retention']:.3f}",
        flush=True,
    )
    print(
        f"test charge sigma={final['selected']['test_metrics']['charge']['sigma_cm']:.3f} cm | "
        f"test CFD sigma={final['selected']['test_metrics']['cfd']['sigma_cm']:.3f} cm | "
        f"test fusion sigma={test_fusion['sigma_cm']:.3f} cm | w_charge={weight:.2f}",
        flush=True,
    )
    print(f"saved: {output_json}", flush=True)
    print("OHBA QUALITY-SELECTION SCAN: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
