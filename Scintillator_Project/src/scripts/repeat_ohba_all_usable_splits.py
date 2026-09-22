"""Repeat the all-usable traditional-vs-hybrid benchmark over event splits.

Each trial creates a new 70/15/15 split within every measured position.  The
traditional modal calibration is fitted on that trial's train events only;
the neural network uses the same train/validation/test populations.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import Scintillator_Project
from Scintillator_Project.src.models.hybrid_residual import HybridResidualNet
from Scintillator_Project.src.scripts.reproduce_ohba_traditional import (
    cfd_time_ns,
    fit_residual_sigma,
    modal_calibration,
)
from Scintillator_Project.src.scripts.train_ohba_quality_selected_hybrid import (
    CFD_FRACTIONS,
    SelectedWaveformDataset,
    choose_device,
    gaussian_metrics,
    make_loader,
    plot_result,
    predict,
    seed_everything,
    train_epoch,
)


PROJECT_ROOT = Path(Scintillator_Project.__file__).resolve().parent


def parse_split_seeds(text: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split seeds must be comma-separated integers") from exc
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("provide one or more unique split seeds")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "dataset/split")
    parser.add_argument(
        "--cache", type=Path,
        default=PROJECT_ROOT / "dataset/cache/ohba_all_usable_events_roi450_800.npz",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "results/ohba_all_usable_repeated_splits",
    )
    parser.add_argument("--split-seeds", type=parse_split_seeds, default=[42, 31415, 271828])
    parser.add_argument("--model-seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def source_files(split_dir: Path) -> list[Path]:
    files = [split_dir / f"{name}.json" for name in ("train", "val", "test")]
    if missing := [str(path) for path in files if not path.is_file()]:
        raise FileNotFoundError("missing source splits: " + ", ".join(missing))
    return files


def build_cache(source: list[Path], target: Path) -> None:
    waveforms: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[float] = []
    event_ids: list[int] = []
    charge_ratio: list[float] = []
    time_difference: list[float] = []
    excluded = 0
    for path in source:
        print(f"[CACHE LOAD] {path}", flush=True)
        with path.open("r", encoding="utf-8") as handle:
            events = json.load(handle)["events"]
        for event in events:
            raw = np.asarray([event["CH0"], event["CH1"]], dtype=np.float64)
            baseline = np.median(raw[:, :400], axis=1, keepdims=True)
            noise = np.std(raw[:, :400], axis=1)
            pulse = np.maximum(baseline - raw, 0.0)
            peaks = pulse.max(axis=1)
            peak_indices = pulse.argmax(axis=1)
            if not np.all((peak_indices >= 450) & (peak_indices < 800)):
                excluded += 1
                continue
            charge = pulse[:, 450:800].sum(axis=1)
            crossings = np.asarray(
                [[cfd_time_ns(pulse[channel], fraction) for fraction in CFD_FRACTIONS]
                 for channel in range(2)], dtype=np.float64
            )
            if np.any(charge <= 0.0) or not np.isfinite(crossings).all():
                excluded += 1
                continue
            ratio = float(np.log(charge[1] / charge[0]))
            time_diff = float(crossings[0, 1] - crossings[1, 1])
            roi = pulse[:, 450:800]
            waveform = roi / float(roi.max())
            total_charge = float(charge.sum())
            rise = crossings[:, -1] - crossings[:, 0]
            snr = float(np.min(peaks / np.maximum(noise, 1e-12)))
            feature = np.asarray(
                [
                    ratio,
                    float((charge[1] - charge[0]) / total_charge),
                    float(np.log(total_charge)),
                    float(np.log(charge[0])),
                    float(np.log(charge[1])),
                    float(np.log(peaks[1] / peaks[0])),
                    float(np.log(peaks.sum())),
                    float(peak_indices[1] - peak_indices[0]),
                    *(crossings[1] - crossings[0]).tolist(),
                    float(rise[0]),
                    float(rise[1]),
                    float(rise[1] - rise[0]),
                    snr,
                ],
                dtype=np.float32,
            )
            waveforms.append(waveform.astype(np.float32))
            features.append(feature)
            labels.append(float(event["position_label"]))
            event_ids.append(int(event["event_id"]))
            charge_ratio.append(ratio)
            time_difference.append(time_diff)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        waveforms=np.asarray(waveforms, dtype=np.float32),
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        event_ids=np.asarray(event_ids, dtype=np.int32),
        charge_ratio=np.asarray(charge_ratio, dtype=np.float64),
        time_difference=np.asarray(time_difference, dtype=np.float64),
    )
    print(f"[CACHE DONE] events={len(labels)} excluded={excluded} path={target}", flush=True)


def ensure_cache(source: list[Path], target: Path) -> None:
    newest_source = max(path.stat().st_mtime for path in source)
    if target.is_file() and target.stat().st_mtime >= newest_source:
        with np.load(target) as existing:
            if "event_ids" in existing.files:
                print(f"[CACHE USE] {target}", flush=True)
                return
    build_cache(source, target)


def make_split(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pieces = {"train": [], "val": [], "test": []}
    for position in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == position))
        train_stop = int(indices.size * 0.70)
        val_stop = train_stop + int(indices.size * 0.15)
        pieces["train"].append(indices[:train_stop])
        pieces["val"].append(indices[train_stop:val_stop])
        pieces["test"].append(indices[val_stop:])
    return {name: np.concatenate(parts) for name, parts in pieces.items()}


def make_chronological_split(labels: np.ndarray, event_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Split every position into earliest train, middle validation, latest test."""
    pieces = {"train": [], "val": [], "test": []}
    for position in np.unique(labels):
        indices = np.flatnonzero(labels == position)
        ordered = indices[np.argsort(event_ids[indices], kind="stable")]
        train_stop = int(ordered.size * 0.70)
        val_stop = train_stop + int(ordered.size * 0.15)
        pieces["train"].append(ordered[:train_stop])
        pieces["val"].append(ordered[train_stop:val_stop])
        pieces["test"].append(ordered[val_stop:])
    return {name: np.concatenate(parts) for name, parts in pieces.items()}


def traditional_predictions(
    charge_ratio: np.ndarray,
    time_difference: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    train_labels = labels[train_indices]
    _, _, charge_intercept, charge_slope = modal_calibration(
        charge_ratio[train_indices], train_labels, 120
    )
    _, _, time_intercept, time_slope = modal_calibration(
        time_difference[train_indices], train_labels, 120
    )
    charge_position = (charge_ratio[target_indices] - charge_intercept) / charge_slope
    time_position = (time_difference[target_indices] - time_intercept) / time_slope

    train_charge = (charge_ratio[train_indices] - charge_intercept) / charge_slope - train_labels
    train_time = (time_difference[train_indices] - time_intercept) / time_slope - train_labels
    weights = np.arange(0.0, 1.01, 0.01)
    # Match the traditional benchmark: select the mixture by the fitted
    # Gaussian core width on the train partition, rather than raw variance.
    best_weight = min(
        weights,
        key=lambda weight: fit_residual_sigma(
            weight * train_charge + (1.0 - weight) * train_time,
            -40.0,
            40.0,
            0.5,
        )["sigma_cm"],
    )
    return best_weight * charge_position + (1.0 - best_weight) * time_position


def train_trial(
    arrays: dict[str, np.ndarray], split: dict[str, np.ndarray], args: argparse.Namespace, trial_dir: Path
) -> dict[str, object]:
    seed_everything(args.model_seed)
    device = choose_device()
    train_indices = split["train"]
    baselines = {
        name: traditional_predictions(
            arrays["charge_ratio"], arrays["time_difference"], arrays["labels"], train_indices, indices
        ).astype(np.float32)
        for name, indices in split.items()
    }
    feature_mean = arrays["features"][train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = arrays["features"][train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-6)
    datasets = {
        name: SelectedWaveformDataset(
            arrays["waveforms"][indices], arrays["features"][indices], baselines[name],
            arrays["labels"][indices], feature_mean, feature_std,
        )
        for name, indices in split.items()
    }
    loaders = {
        "train": make_loader(datasets["train"], args.batch_size, args.num_workers, True),
        "val": make_loader(datasets["val"], args.batch_size, args.num_workers, False),
        "test": make_loader(datasets["test"], args.batch_size, args.num_workers, False),
    }
    model = HybridResidualNet(arrays["features"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    checkpoint = trial_dir / "best_model.pth"
    best_rmse = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, device)
        val_prediction, _, val_labels = predict(model, loaders["val"], device)
        val_rmse = float(np.sqrt(np.mean((val_prediction - val_labels) ** 2)))
        scheduler.step(val_rmse)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "train_loss": float(loss),
            "validation_rmse_cm": val_rmse,
            "learning_rate": learning_rate,
        })
        if val_rmse < best_rmse - 1e-4:
            best_rmse, stale = val_rmse, 0
            torch.save(model.state_dict(), checkpoint)
            marker = " best"
        else:
            stale += 1
            marker = ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} loss={loss:.5f} val_rmse={val_rmse:.4f}"
            f" lr={learning_rate:.2e}{marker}", flush=True
        )
        if epoch >= 50 and stale >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_prediction, test_baseline, test_labels = predict(model, loaders["test"], device)
    report = {
        "model_seed": args.model_seed,
        "device": str(device),
        "split_events": {name: int(indices.size) for name, indices in split.items()},
        "traditional": gaussian_metrics(test_labels, test_baseline),
        "hybrid": gaussian_metrics(test_labels, test_prediction),
    }
    np.savez_compressed(
        trial_dir / "predictions.npz", labels=test_labels,
        traditional_baseline=test_baseline, hybrid_prediction=test_prediction,
    )
    (trial_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (trial_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    plot_result(trial_dir / "test_residual_comparison.png", test_labels, test_baseline, test_prediction)
    return report


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_cache(source_files(args.split_dir), args.cache)
    data = np.load(args.cache)
    arrays = {name: data[name] for name in data.files}
    print("===== Repeated all-usable split benchmark =====", flush=True)
    print(f"events={arrays['labels'].size} | split seeds={args.split_seeds} | model seed={args.model_seed}", flush=True)
    results = []
    for split_seed in args.split_seeds:
        print(f"[TRIAL START] split_seed={split_seed}", flush=True)
        trial_dir = args.output_root / f"split_{split_seed}_model_{args.model_seed}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        report = train_trial(arrays, make_split(arrays["labels"], split_seed), args, trial_dir)
        traditional_sigma = report["traditional"]["gaussian_fit"]["sigma_cm"]
        hybrid_sigma = report["hybrid"]["gaussian_fit"]["sigma_cm"]
        result = {
            "split_seed": split_seed,
            "traditional_sigma_cm": traditional_sigma,
            "hybrid_sigma_cm": hybrid_sigma,
            "improvement_cm": traditional_sigma - hybrid_sigma,
            "traditional_rmse_cm": report["traditional"]["rmse_cm"],
            "hybrid_rmse_cm": report["hybrid"]["rmse_cm"],
        }
        results.append(result)
        print(
            f"[TRIAL DONE] split={split_seed} | traditional={traditional_sigma:.3f} cm | "
            f"hybrid={hybrid_sigma:.3f} cm | improvement={result['improvement_cm']:.3f} cm", flush=True
        )
    improvements = np.asarray([row["improvement_cm"] for row in results])
    summary = {
        "protocol": "repeated stratified event-level splits; train-only traditional calibration",
        "cache": str(args.cache),
        "model_seed": args.model_seed,
        "trials": results,
        "improvement_mean_cm": float(improvements.mean()),
        "improvement_sample_std_cm": float(improvements.std(ddof=1)) if improvements.size > 1 else 0.0,
        "all_trials_hybrid_better": bool(np.all(improvements > 0.0)),
    }
    (args.output_root / "repeated_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("===== Repeated-split summary =====", flush=True)
    for row in results:
        print(
            f"split={row['split_seed']} | traditional={row['traditional_sigma_cm']:.3f} | "
            f"hybrid={row['hybrid_sigma_cm']:.3f} | improvement={row['improvement_cm']:.3f}", flush=True
        )
    print(
        f"mean improvement={summary['improvement_mean_cm']:.3f} +/- "
        f"{summary['improvement_sample_std_cm']:.3f} cm | "
        f"all hybrid better={summary['all_trials_hybrid_better']}", flush=True
    )
    print(f"saved: {args.output_root / 'repeated_split_summary.json'}", flush=True)
    print("OHBA REPEATED ALL-USABLE SPLIT BENCHMARK: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
