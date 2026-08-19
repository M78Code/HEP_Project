"""Truth-free event topology summaries derived from TreeRec graph caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ALL_FEATURE_NAMES = (
    'log_spatial_eig_small',
    'log_spatial_eig_middle',
    'log_spatial_eig_large',
    'log_energy_weighted_spatial_rms',
    'log_time_span_ns',
    'log_time_iqr_ns',
    'energy_weighted_time_std_ns_log',
    'energy_top1_fraction',
    'energy_top3_fraction',
    'late_quartile_energy_fraction',
    'log_energy_mean',
    'log_energy_std',
    'log_energy_q10',
    'log_energy_q50',
    'log_energy_q90',
    'log_energy_q90_minus_q10',
    'tof_sili_centroid_distance_log',
    'tof_sili_time_gap_ns_log',
)

# Chosen only after the independent train/val/test smoke audit.  These features
# are observable from TreeRec hitseries and exclude TreeMc beta, track, and
# stopping information.
STABLE_FEATURE_NAMES = (
    'tof_sili_time_gap_ns_log',
    'log_time_iqr_ns',
    'log_spatial_eig_small',
    'energy_weighted_time_std_ns_log',
    'log_time_span_ns',
    'energy_top3_fraction',
)
STABLE_FEATURE_INDICES = tuple(ALL_FEATURE_NAMES.index(name) for name in STABLE_FEATURE_NAMES)


def load_node_normalizer(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the six continuous global-log node normalization constants."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('mode') != 'global_log':
        raise ValueError(f'{path} is not a global_log node normalizer')
    mean = np.asarray(payload['mean'], dtype=np.float64)
    std = np.asarray(payload['std'], dtype=np.float64)
    if mean.shape != (6,) or std.shape != (6,) or np.any(std <= 0.0):
        raise ValueError(f'{path} must contain finite positive six-dimensional mean/std')
    return mean, std


def topology_features(graph, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return all 18 deterministic event summaries from one global-log graph."""
    x = graph.x.detach().cpu().numpy().astype(np.float64, copy=False)
    pos = graph.pos.detach().cpu().numpy().astype(np.float64, copy=False)
    if x.ndim != 2 or x.shape[1] != 8 or pos.shape != (len(x), 3):
        raise ValueError(f'unexpected graph shapes: x={x.shape}, pos={pos.shape}')
    if len(x) == 0:
        raise ValueError('empty graph')

    log_energy = x[:, 3] * std[3] + mean[3]
    log_time = x[:, 4] * std[4] + mean[4]
    energy = np.expm1(np.clip(log_energy, 0.0, 50.0))
    time = np.expm1(np.clip(log_time, 0.0, 50.0))
    det_type = x[:, 6]

    weights = np.maximum(energy, 0.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0.0:
        weights = np.ones(len(energy), dtype=np.float64)
    weights /= weights.sum()

    center = np.sum(pos * weights[:, None], axis=0)
    delta = pos - center
    covariance = (delta * weights[:, None]).T @ delta
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    spatial_rms = float(np.sqrt(np.sum(weights * np.sum(delta * delta, axis=1))))

    time_span = float(time.max() - time.min()) if len(time) > 1 else 0.0
    time_iqr = float(np.quantile(time, 0.75) - np.quantile(time, 0.25))
    weighted_time_mean = float(np.sum(weights * time))
    weighted_time_std = float(np.sqrt(np.sum(weights * (time - weighted_time_mean) ** 2)))

    ranked_energy = np.sort(energy)[::-1]
    total_energy = max(float(energy.sum()), 1e-12)
    top1_fraction = float(ranked_energy[:1].sum() / total_energy)
    top3_fraction = float(ranked_energy[:3].sum() / total_energy)
    late_energy_fraction = float(
        energy[time >= np.quantile(time, 0.75)].sum() / total_energy)
    energy_q10, energy_q50, energy_q90 = np.quantile(energy, (0.10, 0.50, 0.90))

    tof_mask = det_type < 0.5
    sili_mask = ~tof_mask
    if tof_mask.any() and sili_mask.any():
        tof_weight = weights[tof_mask]
        sili_weight = weights[sili_mask]
        tof_weight /= tof_weight.sum()
        sili_weight /= sili_weight.sum()
        tof_center = np.sum(pos[tof_mask] * tof_weight[:, None], axis=0)
        sili_center = np.sum(pos[sili_mask] * sili_weight[:, None], axis=0)
        centroid_distance = float(np.linalg.norm(tof_center - sili_center))
        tof_time = float(np.sum(time[tof_mask] * tof_weight))
        sili_time = float(np.sum(time[sili_mask] * sili_weight))
        tof_sili_time_gap = abs(tof_time - sili_time)
    else:
        centroid_distance = 0.0
        tof_sili_time_gap = 0.0

    return np.asarray([
        np.log1p(eigenvalues[0]),
        np.log1p(eigenvalues[1]),
        np.log1p(eigenvalues[2]),
        np.log1p(spatial_rms),
        np.log1p(max(time_span, 0.0)),
        np.log1p(max(time_iqr, 0.0)),
        np.log1p(max(weighted_time_std, 0.0)),
        top1_fraction,
        top3_fraction,
        late_energy_fraction,
        np.log1p(max(float(np.mean(energy)), 0.0)),
        np.log1p(max(float(np.std(energy)), 0.0)),
        np.log1p(max(float(energy_q10), 0.0)),
        np.log1p(max(float(energy_q50), 0.0)),
        np.log1p(max(float(energy_q90), 0.0)),
        np.log1p(max(float(energy_q90 - energy_q10), 0.0)),
        np.log1p(centroid_distance),
        np.log1p(max(tof_sili_time_gap, 0.0)),
    ], dtype=np.float32)


def stable_topology_features(graph, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return the six stable features selected by the 1M smoke audit."""
    return topology_features(graph, mean, std)[list(STABLE_FEATURE_INDICES)]
