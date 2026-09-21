import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, WeightedRandomSampler

import Scintillator_Project
from Scintillator_Project.src.data_parse.hybrid_waveform_dataset import (
    HybridWaveformDataset,
    ensure_caches,
)
from Scintillator_Project.src.models.hybrid_residual import HybridResidualNet


PROJECT_ROOT = Path(Scintillator_Project.__file__).parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_physics_baseline(train_dataset):
    # First two inputs are charge log-ratio/asymmetry; CFD@30% is column 10.
    columns = [0, 1, 10]
    model = Ridge(alpha=1e-3)
    model.fit(train_dataset.features_raw[:, columns], train_dataset.labels)
    return model, columns


class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, baseline):
        self.dataset = dataset
        self.baseline = np.asarray(baseline, dtype=np.float32)
        self.labels = dataset.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, features, label = self.dataset[index]
        return waveform, features, torch.tensor(self.baseline[index]), label


def make_baseline_loader(dataset, batch_size, num_workers, training=False):
    sampler = None
    if training:
        positions, counts = np.unique(dataset.labels, return_counts=True)
        count_map = dict(zip(positions.tolist(), counts.tolist()))
        weights = np.asarray([1.0 / count_map[label] for label in dataset.labels])
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(dataset), replacement=True
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    count = 0
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
        total += float(loss) * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions, labels, uncertainties, baselines = [], [], [], []
    for waveforms, features, baseline, target in loader:
        prediction, log_variance = model(
            waveforms.to(device),
            features.to(device),
            baseline[:, None].to(device),
        )
        predictions.append(prediction[:, 0].cpu().numpy())
        labels.append(target.numpy())
        uncertainties.append((10.0 * torch.exp(0.5 * log_variance[:, 0])).cpu().numpy())
        baselines.append(baseline.numpy())
    return tuple(map(np.concatenate, (predictions, labels, uncertainties, baselines)))


def metrics(labels, predictions):
    residual = predictions - labels
    residual_std = float(np.std(residual))
    return {
        "events": int(len(labels)),
        "mae_cm": float(np.mean(np.abs(residual))),
        "rmse_cm": float(np.sqrt(np.mean(residual ** 2))),
        "bias_cm": float(np.mean(residual)),
        "residual_std_cm": residual_std,
        "normal_mle_sigma_cm": residual_std,
        "p95_absolute_error_cm": float(np.quantile(np.abs(residual), 0.95)),
    }


def coverage_metrics(val_uncertainty, test_uncertainty, labels, predictions):
    output = {}
    for coverage in (1.0, 0.95, 0.90, 0.80):
        threshold = np.inf if coverage == 1.0 else np.quantile(val_uncertainty, coverage)
        mask = test_uncertainty <= threshold
        item = metrics(labels[mask], predictions[mask])
        item["requested_coverage"] = coverage
        item["actual_test_coverage"] = float(mask.mean())
        item["uncertainty_threshold_cm"] = None if coverage == 1.0 else float(threshold)
        output[f"coverage_{coverage:.2f}"] = item
    return output


def plot_results(output_dir, labels, baseline, prediction, coverage):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    bins = np.linspace(-30, 30, 121)
    axes[0].hist(baseline - labels, bins=bins, histtype="step", label="CFD + charge baseline")
    axes[0].hist(prediction - labels, bins=bins, histtype="step", label="Hybrid residual model")
    axes[0].set_xlabel("Residual (cm)")
    axes[0].set_ylabel("Events")
    axes[0].legend()
    keys = ["coverage_1.00", "coverage_0.95", "coverage_0.90", "coverage_0.80"]
    x = [coverage[key]["actual_test_coverage"] * 100 for key in keys]
    y = [coverage[key]["residual_std_cm"] for key in keys]
    axes[1].plot(x, y, marker="o")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Retained test events (%)")
    axes[1].set_ylabel("Residual sigma (cm)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "hybrid_residual_evaluation.png")
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = PROJECT_ROOT / "dataset" / "split"
    cache_dir = args.cache_dir or PROJECT_ROOT / "dataset" / "cache"
    ensure_caches(split_dir, cache_dir)
    train_raw = HybridWaveformDataset(cache_dir / "train_hybrid_roi450_800.npz")
    val_raw = HybridWaveformDataset(
        cache_dir / "val_hybrid_roi450_800.npz",
        train_raw.feature_mean,
        train_raw.feature_std,
    )
    test_raw = HybridWaveformDataset(
        cache_dir / "test_hybrid_roi450_800.npz",
        train_raw.feature_mean,
        train_raw.feature_std,
    )

    baseline_model, columns = fit_physics_baseline(train_raw)
    baseline_train = baseline_model.predict(train_raw.features_raw[:, columns])
    baseline_val = baseline_model.predict(val_raw.features_raw[:, columns])
    baseline_test = baseline_model.predict(test_raw.features_raw[:, columns])
    train_data = BaselineDataset(train_raw, baseline_train)
    val_data = BaselineDataset(val_raw, baseline_val)
    test_data = BaselineDataset(test_raw, baseline_test)

    train_loader = make_baseline_loader(train_data, args.batch_size, args.num_workers, True)
    val_loader = make_baseline_loader(val_data, args.batch_size, args.num_workers)
    test_loader = make_baseline_loader(test_data, args.batch_size, args.num_workers)
    model = HybridResidualNet(train_raw.features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5)

    print(f"device: {device}", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"events: {len(train_data)}/{len(val_data)}/{len(test_data)}", flush=True)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    best_rmse = float("inf")
    stale = 0
    checkpoint = args.output_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_prediction, val_labels, _, _ = predict(model, val_loader, device)
        val_rmse = metrics(val_labels, val_prediction)["rmse_cm"]
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
        if epoch >= 30 and stale >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    val_prediction, val_labels, val_uncertainty, _ = predict(model, val_loader, device)
    test_prediction, test_labels, test_uncertainty, test_baseline = predict(model, test_loader, device)
    report = {
        "seed": args.seed,
        "device": str(device),
        "split_events": {
            "train": len(train_data), "val": len(val_data), "test": len(test_data)
        },
        "external_reference": {
            "ohba_unpublished_selected_gaussian_sigma_cm": 4.72,
            "note": "External reference only; selection criteria and retained fraction are unknown.",
        },
        "physics_baseline": metrics(test_labels, test_baseline),
        "hybrid_all_events": metrics(test_labels, test_prediction),
        "hybrid_by_validation_uncertainty": coverage_metrics(
            val_uncertainty, test_uncertainty, test_labels, test_prediction
        ),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        labels=test_labels,
        physics_baseline=test_baseline,
        hybrid_prediction=test_prediction,
        predicted_uncertainty=test_uncertainty,
    )
    plot_results(
        args.output_dir,
        test_labels,
        test_baseline,
        test_prediction,
        report["hybrid_by_validation_uncertainty"],
    )
    print(json.dumps(report, indent=2), flush=True)
    print(f"output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
