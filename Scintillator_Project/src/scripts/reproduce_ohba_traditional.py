"""Reproduce Ohba's traditional TOF position-reconstruction protocol.

This is deliberately separate from ``evaluate_traditional.py``.  Ohba's
thesis calibrates on all measured positions using the modal charge-ratio and
time-difference values, then evaluates the combined residual distribution with
a Gaussian-fit sigma.  It is therefore an in-sample protocol reproduction,
not a train/test generalisation measurement.

The thesis does not publish every waveform-extraction constant.  Defaults are
read from its Figure 5 as closely as the published figure permits; every such
choice is saved alongside the results and can be scanned explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import Scintillator_Project


PROJECT_ROOT = Path(Scintillator_Project.__file__).resolve().parent
SAMPLE_PERIOD_NS = 0.3125
POSITION_RE = re.compile(r"run\d+_(\d+)\.dat$")
EVENT_RE = re.compile(r"EVENT\s+(\d+)")
CHANNEL_RE = re.compile(r"CH:\s*(\d+)")


@dataclass(frozen=True)
class ProtocolConfig:
    baseline_start: int
    baseline_stop: int
    integration_start: int
    integration_stop: int
    cfd_fraction: float
    mode_bins: int
    residual_min: float
    residual_max: float
    residual_bin_width: float
    weight_step: float


def parse_range(text: str) -> tuple[int, int]:
    try:
        start, stop = (int(part) for part in text.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranges must look like START:STOP") from exc
    if start < 0 or stop <= start or stop > 1024:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START < STOP <= 1024")
    return start, stop


def parse_fractions(text: str) -> list[float]:
    try:
        values = [float(value) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fractions must be comma-separated floats") from exc
    if not values or any(value <= 0.0 or value >= 1.0 for value in values):
        raise argparse.ArgumentTypeError("each CFD fraction must be between 0 and 1")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path,
        default=PROJECT_ROOT / "dataset/tar_zip/A",
        help="Directory containing the verified run*.dat files.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results/ohba_traditional_reproduction",
    )
    parser.add_argument(
        "--baseline-range", type=parse_range, default=(100, 200),
        help="Baseline samples, default 100:200 from the published waveform figure.",
    )
    parser.add_argument(
        "--integration-range", type=parse_range, default=(600, 1000),
        help="Charge-integration samples, default 600:1000 from the published waveform figure.",
    )
    parser.add_argument(
        "--cfd-fractions", type=parse_fractions, default=[0.10, 0.20, 0.30, 0.40, 0.50],
        help="Comma-separated CFD fractions to evaluate.",
    )
    parser.add_argument(
        "--mode-bins", type=int, default=120,
        help="Shared histogram bins used to estimate each position's feature mode.",
    )
    parser.add_argument("--residual-range", type=float, nargs=2, default=(-40.0, 40.0))
    parser.add_argument("--residual-bin-width", type=float, default=0.5)
    parser.add_argument(
        "--weight-step", type=float, default=0.01,
        help="Charge-fusion scan step. Use 0.01 for the baseline scan, then refine locally if needed.",
    )
    return parser.parse_args()


def infer_position(path: Path) -> float:
    match = POSITION_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot infer a position from {path.name}")
    return float(match.group(1))


def yield_raw_events(raw_dir: Path) -> Iterator[tuple[str, int, float, np.ndarray, np.ndarray]]:
    """Stream CH0/CH1 directly from text .dat files without retaining waveforms."""
    for path in sorted(raw_dir.glob("run*_*.dat"), key=lambda item: infer_position(item)):
        position = infer_position(path)
        event_id: int | None = None
        channel: int | None = None
        channels: dict[int, list[float]] = {}

        def complete_event() -> tuple[str, int, float, np.ndarray, np.ndarray] | None:
            if event_id is None or 0 not in channels or 1 not in channels:
                return None
            ch0 = np.asarray(channels[0], dtype=np.float64)
            ch1 = np.asarray(channels[1], dtype=np.float64)
            if ch0.size != 1024 or ch1.size != 1024:
                return None
            return path.stem, event_id, position, ch0, ch1

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if "=== EVENT" in line:
                    previous = complete_event()
                    if previous is not None:
                        yield previous
                    match = EVENT_RE.search(line)
                    event_id = int(match.group(1)) if match else None
                    channel = None
                    channels = {}
                    continue
                if "=== CH:" in line:
                    match = CHANNEL_RE.search(line)
                    channel = int(match.group(1)) if match else None
                    if channel in (0, 1):
                        channels[channel] = []
                    continue
                if channel not in (0, 1) or line.startswith("==="):
                    continue
                try:
                    channels[channel].extend(float(value) for value in line.split())
                except ValueError:
                    # Header lines are not waveform samples.
                    continue

        previous = complete_event()
        if previous is not None:
            yield previous


def positive_pulse(waveform: np.ndarray, baseline_start: int, baseline_stop: int) -> np.ndarray:
    baseline = float(np.median(waveform[baseline_start:baseline_stop]))
    positive = waveform - baseline
    negative = baseline - waveform
    # The archived waveforms may use either electronics polarity.  Preserve the
    # physical pulse magnitude while reporting the chosen polarity separately.
    excursion = positive if np.max(positive) >= np.max(negative) else negative
    return np.maximum(excursion, 0.0)


def cfd_time_ns(pulse: np.ndarray, fraction: float) -> float:
    peak_index = int(np.argmax(pulse))
    peak = float(pulse[peak_index])
    if peak <= 0.0:
        return float("nan")
    threshold = fraction * peak
    indices = np.flatnonzero(pulse[: peak_index + 1] >= threshold)
    if not indices.size:
        return float("nan")
    right = int(indices[0])
    if right == 0:
        return 0.0
    left = right - 1
    y0, y1 = float(pulse[left]), float(pulse[right])
    crossing = float(right) if y1 == y0 else left + (threshold - y0) / (y1 - y0)
    return crossing * SAMPLE_PERIOD_NS


def extract_features(
    raw_dir: Path,
    config: ProtocolConfig,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    run_names: list[str] = []
    event_ids: list[int] = []
    positions: list[float] = []
    charge_ratios: list[float] = []
    time_differences: list[float] = []
    invalid = Counter()

    for run_name, event_id, position, ch0, ch1 in yield_raw_events(raw_dir):
        left = positive_pulse(ch0, config.baseline_start, config.baseline_stop)
        right = positive_pulse(ch1, config.baseline_start, config.baseline_stop)
        q_left = float(left[config.integration_start:config.integration_stop].sum())
        q_right = float(right[config.integration_start:config.integration_stop].sum())
        if q_left <= 0.0 or q_right <= 0.0:
            invalid["nonpositive_charge"] += 1
            continue

        t_left = cfd_time_ns(left, config.cfd_fraction)
        t_right = cfd_time_ns(right, config.cfd_fraction)
        if not np.isfinite(t_left) or not np.isfinite(t_right):
            invalid["invalid_cfd_crossing"] += 1
            continue

        run_names.append(run_name)
        event_ids.append(event_id)
        positions.append(position)
        charge_ratios.append(float(np.log(q_right / q_left)))
        # Thesis convention: Delta t = t_L - t_R.
        time_differences.append(t_left - t_right)

    values = {
        "run": np.asarray(run_names, dtype=object),
        "event_id": np.asarray(event_ids, dtype=np.int32),
        "position": np.asarray(positions, dtype=np.float64),
        "log_charge_ratio": np.asarray(charge_ratios, dtype=np.float64),
        "time_difference_ns": np.asarray(time_differences, dtype=np.float64),
    }
    return values, dict(invalid)


def modal_calibration(feature: np.ndarray, position: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit feature_mode = intercept + slope * position using shared bins."""
    unique_positions = np.unique(position)
    edges = np.linspace(float(feature.min()), float(feature.max()), bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    modes = []
    for value in unique_positions:
        counts, _ = np.histogram(feature[position == value], bins=edges)
        modes.append(float(centers[int(np.argmax(counts))]))
    modes_array = np.asarray(modes)
    slope, intercept = np.polyfit(unique_positions, modes_array, deg=1)
    if abs(slope) < 1e-12:
        raise RuntimeError("degenerate calibration slope")
    return unique_positions, modes_array, float(intercept), float(slope)


def gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_residual_sigma(
    residual: np.ndarray,
    minimum: float,
    maximum: float,
    width: float,
) -> dict[str, object]:
    edges = np.arange(minimum, maximum + width, width)
    counts, _ = np.histogram(residual, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    nonzero = counts > 0
    if nonzero.sum() < 4:
        raise RuntimeError("not enough populated residual bins for a Gaussian fit")
    initial = (float(counts.max()), float(np.mean(residual)), float(np.std(residual)))
    params, covariance = curve_fit(
        gaussian,
        centers[nonzero],
        counts[nonzero],
        p0=initial,
        bounds=([0.0, minimum, 0.05], [np.inf, maximum, maximum - minimum]),
        maxfev=20_000,
    )
    uncertainty = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    return {
        "sigma_cm": float(params[2]),
        "sigma_error_cm": float(uncertainty[2]),
        "mean_cm": float(params[1]),
        "mean_error_cm": float(uncertainty[1]),
        "amplitude": float(params[0]),
        "raw_std_cm": float(np.std(residual)),
        "raw_rmse_cm": float(np.sqrt(np.mean(residual ** 2))),
        "histogram_edges_cm": edges.tolist(),
        "histogram_counts": counts.astype(int).tolist(),
        "fit_curve_counts": gaussian(centers, *params).tolist(),
    }


def write_calibration_csv(
    path: Path,
    positions: np.ndarray,
    charge_modes: np.ndarray,
    time_modes: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position_cm", "log_charge_ratio_mode", "time_difference_mode_ns"])
        for row in zip(positions, charge_modes, time_modes):
            writer.writerow(row)


def make_figure(path: Path, result: dict[str, object]) -> None:
    calibration = result["calibration"]
    metrics = result["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    positions = np.asarray(calibration["positions_cm"])
    q_modes = np.asarray(calibration["charge_ratio"]["modes"])
    q_intercept = calibration["charge_ratio"]["intercept"]
    q_slope = calibration["charge_ratio"]["slope"]
    axes[0].scatter(positions, q_modes, color="#2878b5", label="per-position mode")
    axes[0].plot(positions, q_intercept + q_slope * positions, color="#d8572a", label="linear calibration")
    axes[0].set_xlabel("True position [cm]")
    axes[0].set_ylabel("ln(Q_R / Q_L)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    fusion = metrics["fusion"]
    edges = np.asarray(fusion["histogram_edges_cm"])
    centers = (edges[:-1] + edges[1:]) / 2.0
    axes[1].step(centers, fusion["histogram_counts"], where="mid", color="#2878b5", label="all-position residuals")
    axes[1].plot(centers, fusion["fit_curve_counts"], color="#d8572a", label="Gaussian fit")
    axes[1].set_xlabel("x_hat - x_true [cm]")
    axes[1].set_ylabel("Events / bin")
    axes[1].set_title(f"CFD + charge fusion: sigma = {fusion['sigma_cm']:.3f} cm")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Ohba-style traditional reconstruction on the delivered raw data")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.mode_bins < 10:
        raise ValueError("--mode-bins must be at least 10")
    if args.residual_bin_width <= 0.0 or args.weight_step <= 0.0:
        raise ValueError("bin width and weight step must be positive")
    if not args.raw_dir.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {args.raw_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("===== Ohba traditional-method reproduction =====", flush=True)
    print("Protocol: all-position modal calibration + all-event Gaussian sigma", flush=True)
    print("This is a reproduction protocol, not a held-out ML evaluation.", flush=True)
    print(f"raw data: {args.raw_dir}", flush=True)

    for fraction in args.cfd_fractions:
        config = ProtocolConfig(
            baseline_start=args.baseline_range[0], baseline_stop=args.baseline_range[1],
            integration_start=args.integration_range[0], integration_stop=args.integration_range[1],
            cfd_fraction=fraction, mode_bins=args.mode_bins,
            residual_min=args.residual_range[0], residual_max=args.residual_range[1],
            residual_bin_width=args.residual_bin_width, weight_step=args.weight_step,
        )
        values, invalid = extract_features(args.raw_dir, config)
        if not values["position"].size:
            raise RuntimeError("no valid events after waveform feature extraction")

        positions, q_modes, q_intercept, q_slope = modal_calibration(
            values["log_charge_ratio"], values["position"], config.mode_bins
        )
        time_positions, t_modes, t_intercept, t_slope = modal_calibration(
            values["time_difference_ns"], values["position"], config.mode_bins
        )
        if not np.array_equal(positions, time_positions):
            raise RuntimeError("charge and time calibration positions differ")

        x_charge = (values["log_charge_ratio"] - q_intercept) / q_slope
        x_time = (values["time_difference_ns"] - t_intercept) / t_slope
        residual_charge = x_charge - values["position"]
        residual_time = x_time - values["position"]
        charge_fit = fit_residual_sigma(residual_charge, config.residual_min, config.residual_max, config.residual_bin_width)
        time_fit = fit_residual_sigma(residual_time, config.residual_min, config.residual_max, config.residual_bin_width)

        best_weight = None
        best_fusion = None
        weight_rows = []
        weights = np.arange(0.0, 1.0 + config.weight_step / 2.0, config.weight_step)
        for weight in weights:
            residual = weight * residual_charge + (1.0 - weight) * residual_time
            fitted = fit_residual_sigma(residual, config.residual_min, config.residual_max, config.residual_bin_width)
            weight_rows.append((float(weight), fitted["sigma_cm"], fitted["sigma_error_cm"]))
            if best_fusion is None or fitted["sigma_cm"] < best_fusion["sigma_cm"]:
                best_weight = float(weight)
                best_fusion = fitted

        assert best_fusion is not None and best_weight is not None
        prefix = f"cfd_{fraction:.3f}".replace(".", "p")
        result = {
            "protocol": "Ohba thesis-style all-position modal calibration and Gaussian-fit sigma",
            "data_version": "delivered raw .dat archive; 23,602 verified events expected",
            "config": asdict(config),
            "event_count": int(values["position"].size),
            "events_per_position": {str(int(position)): int((values["position"] == position).sum()) for position in positions},
            "invalid_events_excluded": invalid,
            "calibration": {
                "positions_cm": positions.tolist(),
                "charge_ratio": {"modes": q_modes.tolist(), "intercept": q_intercept, "slope": q_slope},
                "time_difference": {"modes": t_modes.tolist(), "intercept": t_intercept, "slope": t_slope},
            },
            "metrics": {"charge": charge_fit, "cfd": time_fit, "fusion": {**best_fusion, "charge_weight": best_weight}},
        }
        output_json = args.output_dir / f"{prefix}_metrics.json"
        output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        write_calibration_csv(args.output_dir / f"{prefix}_calibration_modes.csv", positions, q_modes, t_modes)
        with (args.output_dir / f"{prefix}_fusion_weight_scan.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["charge_weight", "gaussian_sigma_cm", "gaussian_sigma_error_cm"])
            writer.writerows(weight_rows)
        make_figure(args.output_dir / f"{prefix}_calibration_and_fusion.png", result)

        print(
            f"CFD fraction={fraction:.3f} | events={result['event_count']} | "
            f"charge sigma={charge_fit['sigma_cm']:.3f} cm | "
            f"CFD sigma={time_fit['sigma_cm']:.3f} cm | "
            f"fusion sigma={best_fusion['sigma_cm']:.3f} cm at w_charge={best_weight:.3f}",
            flush=True,
        )
        print(f"saved: {output_json}", flush=True)

    print("OHBA TRADITIONAL REPRODUCTION: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
