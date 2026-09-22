"""Train a waveform model against the quality-selected Ohba baseline.

The quality threshold, traditional calibration, and fusion weight are read
from the preceding leakage-free quality-selection study.  The model therefore
uses exactly the same selected train/validation/test populations as the
4.683 cm traditional reference, while learning only a waveform residual.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import Scintillator_Project
from Scintillator_Project.src.models.hybrid_residual import HybridResidualNet
from Scintillator_Project.src.scripts.reproduce_ohba_traditional import cfd_time_ns, fit_residual_sigma


PROJECT_ROOT = Path(Scintillator_Project.__file__).resolve().parent
CFD_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quality-results", type=Path,
        default=PROJECT_ROOT / "results/ohba_quality_selection/quality_selection_results.json",
    )
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "dataset/split")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_selected_split(
    path: Path, protocol: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return jointly normalized ROI waveforms, raw features, baseline, labels."""
    config = protocol["config"]
    selected = protocol["selected"]
    baseline_stop = int(config["baseline_stop"])
    roi_start = int(config["roi_start"])
    roi_stop = int(config["roi_stop"])
    cfd_fraction = float(config["cfd_fraction"])
    threshold = float(selected["snr_threshold"])
    calibration = selected["calibration"]
    weight = float(selected["charge_weight"])

    with path.open("r", encoding="utf-8") as handle:
        events = json.load(handle)["events"]

    waveforms: list[np.ndarray] = []
    features: list[np.ndarray] = []
    baselines: list[float] = []
    labels: list[float] = []
    for event in events:
        raw = np.asarray([event["CH0"], event["CH1"]], dtype=np.float64)
        baseline_level = np.median(raw[:, :baseline_stop], axis=1, keepdims=True)
        baseline_noise = np.std(raw[:, :baseline_stop], axis=1)
        pulse = np.maximum(baseline_level - raw, 0.0)
        peaks = pulse.max(axis=1)
        peak_indices = pulse.argmax(axis=1)
        snr = float(np.min(peaks / np.maximum(baseline_noise, 1e-12)))
        if (
            snr < threshold
            or not np.all((peak_indices >= roi_start) & (peak_indices < roi_stop))
        ):
            continue

        charge = pulse[:, roi_start:roi_stop].sum(axis=1)
        if np.any(charge <= 0.0):
            continue
        crossings = np.asarray(
            [[cfd_time_ns(pulse[channel], fraction) for fraction in CFD_FRACTIONS]
             for channel in range(2)],
            dtype=np.float64,
        )
        if not np.isfinite(crossings).all():
            continue
        charge_ratio = float(np.log(charge[1] / charge[0]))
        time_difference = float(
            cfd_time_ns(pulse[0], cfd_fraction) - cfd_time_ns(pulse[1], cfd_fraction)
        )
        charge_position = (
            charge_ratio - float(calibration["charge_intercept"])
        ) / float(calibration["charge_slope"])
        time_position = (
            time_difference - float(calibration["time_intercept"])
        ) / float(calibration["time_slope"])
        traditional_position = weight * charge_position + (1.0 - weight) * time_position

        roi = pulse[:, roi_start:roi_stop]
        scale = float(roi.max())
        waveform = roi / scale if scale > 1e-12 else roi
        total_charge = float(charge.sum())
        rise = crossings[:, -1] - crossings[:, 0]
        feature = np.asarray(
            [
                charge_ratio,
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
        baselines.append(float(traditional_position))
        labels.append(float(event["position_label"]))

    return (
        np.asarray(waveforms, dtype=np.float32),
        np.asarray(features, dtype=np.float32),
        np.asarray(baselines, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


class SelectedWaveformDataset(Dataset):
    def __init__(
        self,
        waveforms: np.ndarray,
        features_raw: np.ndarray,
        baseline: np.ndarray,
        labels: np.ndarray,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        self.waveforms = waveforms
        self.features = (features_raw - feature_mean) / feature_std
        self.baseline = baseline
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.waveforms[index]),
            torch.from_numpy(self.features[index]),
            torch.tensor(self.baseline[index]),
            torch.tensor(self.labels[index]),
        )


def make_loader(dataset: SelectedWaveformDataset, batch_size: int, workers: int, training: bool) -> DataLoader:
    sampler = None
    if training:
        positions, counts = np.unique(dataset.labels, return_counts=True)
        count_by_position = dict(zip(positions.tolist(), counts.tolist()))
        weights = np.asarray([1.0 / count_by_position[value] for value in dataset.labels])
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(dataset), replacement=True
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def train_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_events = 0
    for waveforms, features, baseline, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        baseline = baseline[:, None].to(device, non_blocking=True)
        labels = labels[:, None].to(device, non_blocking=True)
        prediction, log_variance = model(waveforms, features, baseline)
        error_scaled = (prediction - labels) / 10.0
        nll = 0.5 * (torch.exp(-log_variance) * error_scaled.square() + log_variance)
        loss = F.smooth_l1_loss(prediction, labels, beta=3.0) + 0.5 * nll.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * labels.shape[0]
        total_events += labels.shape[0]
    return total_loss / total_events


@torch.no_grad()
def predict(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, baselines, labels = [], [], []
    for waveforms, features, baseline, target in loader:
        prediction, _ = model(
            waveforms.to(device, non_blocking=True),
            features.to(device, non_blocking=True),
            baseline[:, None].to(device, non_blocking=True),
        )
        predictions.append(prediction[:, 0].cpu().numpy())
        baselines.append(baseline.numpy())
        labels.append(target.numpy())
    return tuple(map(np.concatenate, (predictions, baselines, labels)))


def gaussian_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    residual = predictions - labels
    fit = fit_residual_sigma(residual, -40.0, 40.0, 0.5)
    return {
        "events": int(labels.size),
        "rmse_cm": float(np.sqrt(np.mean(residual ** 2))),
        "mae_cm": float(np.mean(np.abs(residual))),
        "bias_cm": float(np.mean(residual)),
        "gaussian_fit": fit,
    }


def plot_result(path: Path, labels: np.ndarray, baseline: np.ndarray, prediction: np.ndarray) -> None:
    baseline_fit = fit_residual_sigma(baseline - labels, -40.0, 40.0, 0.5)
    model_fit = fit_residual_sigma(prediction - labels, -40.0, 40.0, 0.5)
    edges = np.asarray(model_fit["histogram_edges_cm"])
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.step(centers, baseline_fit["histogram_counts"], where="mid", label=(
        f"Traditional selected baseline, sigma={baseline_fit['sigma_cm']:.3f} cm"
    ))
    ax.step(centers, model_fit["histogram_counts"], where="mid", label=(
        f"Hybrid waveform model, sigma={model_fit['sigma_cm']:.3f} cm"
    ))
    ax.set_xlabel("Position residual [cm]")
    ax.set_ylabel("Events / 0.5 cm")
    ax.set_title("Quality-selected held-out test")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not args.quality_results.is_file():
        raise FileNotFoundError(f"quality-selection results not found: {args.quality_results}")
    protocol = json.loads(args.quality_results.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()

    raw_splits = {}
    for name in ("train", "val", "test"):
        path = args.split_dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"split file not found: {path}")
        print(f"[LOAD] {name}: {path}", flush=True)
        raw_splits[name] = load_selected_split(path, protocol)

    train_waveforms, train_features, train_baseline, train_labels = raw_splits["train"]
    feature_mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-6)
    datasets = {
        name: SelectedWaveformDataset(*raw_splits[name], feature_mean, feature_std)
        for name in raw_splits
    }
    loaders = {
        "train": make_loader(datasets["train"], args.batch_size, args.num_workers, True),
        "val": make_loader(datasets["val"], args.batch_size, args.num_workers, False),
        "test": make_loader(datasets["test"], args.batch_size, args.num_workers, False),
    }
    model = HybridResidualNet(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    checkpoint = args.output_dir / "best_model.pth"

    print("===== Quality-selected hybrid waveform training =====", flush=True)
    print(f"device: {device}", flush=True)
    print(
        "events train/val/test: "
        f"{len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}",
        flush=True,
    )
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
    initial_baseline = gaussian_metrics(train_labels, train_baseline)["gaussian_fit"]["sigma_cm"]
    print(f"train traditional baseline sigma={initial_baseline:.3f} cm", flush=True)

    best_rmse = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, device)
        val_prediction, _, val_labels = predict(model, loaders["val"], device)
        val_rmse = float(np.sqrt(np.mean((val_prediction - val_labels) ** 2)))
        scheduler.step(val_rmse)
        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            stale = 0
            torch.save(model.state_dict(), checkpoint)
            marker = " best"
        else:
            stale += 1
            marker = ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} train_loss={train_loss:.5f} "
            f"val_rmse={val_rmse:.4f} lr={optimizer.param_groups[0]['lr']:.2e}{marker}",
            flush=True,
        )
        if epoch >= 50 and stale >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_prediction, test_baseline, test_labels = predict(model, loaders["test"], device)
    report = {
        "protocol": "quality threshold and traditional calibration fixed before ML training",
        "seed": args.seed,
        "device": str(device),
        "quality_selection_source": str(args.quality_results),
        "snr_threshold": protocol["selected"]["snr_threshold"],
        "split_events": {name: len(dataset) for name, dataset in datasets.items()},
        "traditional_selected_baseline": gaussian_metrics(test_labels, test_baseline),
        "hybrid_waveform_model": gaussian_metrics(test_labels, test_prediction),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        labels=test_labels,
        traditional_baseline=test_baseline,
        hybrid_prediction=test_prediction,
    )
    plot_result(args.output_dir / "test_residual_comparison.png", test_labels, test_baseline, test_prediction)
    print("===== Held-out quality-selected test =====", flush=True)
    print(
        f"traditional sigma={report['traditional_selected_baseline']['gaussian_fit']['sigma_cm']:.3f} cm | "
        f"hybrid sigma={report['hybrid_waveform_model']['gaussian_fit']['sigma_cm']:.3f} cm | "
        f"hybrid RMSE={report['hybrid_waveform_model']['rmse_cm']:.3f} cm",
        flush=True,
    )
    print(f"saved: {args.output_dir / 'metrics.json'}", flush=True)
    print("OHBA QUALITY-SELECTED HYBRID TRAINING: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
