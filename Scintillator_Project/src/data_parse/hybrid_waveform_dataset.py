import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROI = (450, 800)
BASELINE_END = 400
CFD_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5)


def _crossing_time(pulse: np.ndarray, fraction: float) -> float:
    peak_index = int(np.argmax(pulse))
    peak = float(pulse[peak_index])
    if peak <= 1e-9:
        return float(peak_index)

    threshold = fraction * peak
    crossing = np.flatnonzero(pulse[: peak_index + 1] >= threshold)
    if crossing.size == 0:
        return float(peak_index)

    right = int(crossing[0])
    if right == 0:
        return 0.0
    left = right - 1
    y0 = float(pulse[left])
    y1 = float(pulse[right])
    if abs(y1 - y0) < 1e-12:
        return float(right)
    return left + (threshold - y0) / (y1 - y0)


def preprocess_event(ch0, ch1):
    raw = np.stack(
        [np.asarray(ch0, dtype=np.float32), np.asarray(ch1, dtype=np.float32)]
    )
    baseline = np.median(raw[:, :BASELINE_END], axis=1, keepdims=True)
    pulse = np.maximum(baseline - raw, 0.0)
    pulse = pulse[:, ROI[0] : ROI[1]]

    scale = float(np.max(pulse))
    normalized = pulse / scale if scale > 1e-9 else pulse.copy()

    charge = pulse.sum(axis=1) + 1e-9
    peak = pulse.max(axis=1) + 1e-9
    peak_index = pulse.argmax(axis=1).astype(np.float32)
    cfd = np.asarray(
        [[_crossing_time(pulse[ch], fraction) for fraction in CFD_FRACTIONS]
         for ch in range(2)],
        dtype=np.float32,
    )

    rise = cfd[:, 4] - cfd[:, 0]
    total_charge = float(charge.sum())
    features = np.asarray(
        [
            np.log(charge[1] / charge[0]),
            (charge[1] - charge[0]) / total_charge,
            np.log(total_charge),
            np.log(charge[0]),
            np.log(charge[1]),
            np.log(peak[1] / peak[0]),
            np.log(peak.sum()),
            peak_index[1] - peak_index[0],
            *(cfd[1] - cfd[0]),
            rise[0],
            rise[1],
            rise[1] - rise[0],
        ],
        dtype=np.float32,
    )
    return normalized.astype(np.float32), features


def build_split_cache(split_json: Path, cache_path: Path):
    with split_json.open("r", encoding="utf-8") as handle:
        events = json.load(handle)["events"]

    waveforms = []
    features = []
    labels = []
    event_ids = []
    for event in events:
        waveform, event_features = preprocess_event(event["CH0"], event["CH1"])
        waveforms.append(waveform)
        features.append(event_features)
        labels.append(float(event["position_label"]))
        event_ids.append(int(event.get("event_id", -1)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        waveforms=np.asarray(waveforms, dtype=np.float32),
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        event_ids=np.asarray(event_ids, dtype=np.int64),
    )


def ensure_caches(split_dir: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        source = split_dir / f"{split}.json"
        target = cache_dir / f"{split}_hybrid_roi450_800.npz"
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            print(f"[CACHE] building {target.name}", flush=True)
            build_split_cache(source, target)
        else:
            print(f"[CACHE] using {target.name}", flush=True)


class HybridWaveformDataset(Dataset):
    def __init__(self, cache_path: Path, feature_mean=None, feature_std=None):
        data = np.load(cache_path)
        self.waveforms = data["waveforms"]
        self.features_raw = data["features"]
        self.labels = data["labels"]
        self.event_ids = data["event_ids"]

        if feature_mean is None:
            feature_mean = self.features_raw.mean(axis=0)
        if feature_std is None:
            feature_std = self.features_raw.std(axis=0)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(feature_std, dtype=np.float32), 1e-6)
        self.features = (self.features_raw - self.feature_mean) / self.feature_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.waveforms[index]),
            torch.from_numpy(self.features[index]),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )
