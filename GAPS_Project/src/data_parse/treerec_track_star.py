"""Truth-free track and annihilation-star candidates from TreeRec hit graphs.

The production TreeRec files do not contain usable persisted track or vertex
objects.  This module therefore derives deterministic *candidates* directly
from observed hit position, energy, time, and detector type.  It intentionally
does not read TreeMc metadata, labels, MC beta, or reconstructed ROOT fields.
"""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    'has_sili_track_candidate',
    'has_tof_sili_link',
    'log_sili_hit_count',
    'sili_linearity',
    'sili_planarity',
    'log_sili_line_residual_mm',
    'abs_sili_axis_time_correlation',
    'log_sili_axis_time_slope_ns_per_mm',
    'terminal_third_energy_fraction',
    'log_terminal_to_start_energy_ratio',
    'late_to_all_transverse_rms_ratio',
    'off_axis_energy_fraction_75mm',
    'log_sili_component_count_100mm',
    'largest_sili_component_energy_fraction_100mm',
    'terminal_cluster_energy_fraction_100mm',
    'tof_sili_axis_alignment_abs',
    'log_tof_to_sili_axis_distance_mm',
    'sili_transverse_to_longitudinal_rms_ratio',
    'sili_components_per_hit_100mm',
)

# The Si(Li) hit times in the current production cache are frequently
# degenerate.  These geometry-only quantities remain finite and are not simple
# copies of the existing total event hit count or energy summary.
STRUCTURAL_FEATURE_NAMES = (
    'sili_linearity',
    'sili_planarity',
    'sili_transverse_to_longitudinal_rms_ratio',
    'off_axis_energy_fraction_75mm',
)
STRUCTURAL_FEATURE_INDICES = tuple(
    FEATURE_NAMES.index(name) for name in STRUCTURAL_FEATURE_NAMES)

_OFF_AXIS_RADIUS_MM = 75.0
_COMPONENT_RADIUS_MM = 100.0


def _safe_weights(energy: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(energy, dtype=np.float64), 0.0)
    total = float(weights.sum())
    if not np.isfinite(weights).all() or total <= 0.0:
        return np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    return weights / total


def _weighted_line(
        points: np.ndarray, energy: np.ndarray, time: np.ndarray) -> dict:
    """Fit an energy-weighted PCA line and orient it along increasing time."""
    weights = _safe_weights(energy)
    center = np.sum(points * weights[:, None], axis=0)
    delta = points - center
    covariance = (delta * weights[:, None]).T @ delta
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    axis = eigenvectors[:, -1]
    longitudinal = delta @ axis

    time_center = float(np.sum(weights * time))
    time_delta = time - time_center
    long_var = float(np.sum(weights * longitudinal * longitudinal))
    time_var = float(np.sum(weights * time_delta * time_delta))
    covariance_st = float(np.sum(weights * longitudinal * time_delta))
    correlation = covariance_st / np.sqrt(max(long_var * time_var, 1e-24))
    if correlation < 0.0:
        axis = -axis
        longitudinal = -longitudinal
        correlation = -correlation
        covariance_st = -covariance_st

    slope = covariance_st / max(long_var, 1e-12)
    transverse_sq = np.maximum(
        np.sum(delta * delta, axis=1) - longitudinal * longitudinal, 0.0)
    transverse = np.sqrt(transverse_sq)
    residual = float(np.sqrt(np.sum(weights * transverse_sq)))
    largest = max(float(eigenvalues[-1]), 1e-12)

    return {
        'weights': weights,
        'center': center,
        'axis': axis,
        'longitudinal': longitudinal,
        'transverse': transverse,
        'linearity': float((eigenvalues[-1] - eigenvalues[-2]) / largest),
        'planarity': float((eigenvalues[-2] - eigenvalues[-3]) / largest),
        'residual': residual,
        'time_correlation': float(np.clip(correlation, 0.0, 1.0)),
        'time_slope': float(abs(slope)),
        'longitudinal_rms': float(np.sqrt(largest)),
    }


def _component_energy_fractions(
        points: np.ndarray, energy: np.ndarray) -> tuple[int, float]:
    """Return 3D connected-component count and largest energy fraction."""
    count = len(points)
    if count == 0:
        return 0, 0.0
    delta = points[:, None, :] - points[None, :, :]
    adjacent = np.sum(delta * delta, axis=-1) <= _COMPONENT_RADIUS_MM ** 2
    visited = np.zeros(count, dtype=bool)
    component_energies = []
    energy = np.maximum(energy, 0.0)
    for seed in range(count):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        members = []
        while stack:
            current = stack.pop()
            members.append(current)
            neighbors = np.flatnonzero(adjacent[current] & ~visited)
            visited[neighbors] = True
            stack.extend(neighbors.tolist())
        component_energies.append(float(energy[members].sum()))
    total = max(float(energy.sum()), 1e-12)
    return len(component_energies), max(component_energies) / total


def _terminal_cluster_fraction(
        points: np.ndarray, energy: np.ndarray, time: np.ndarray) -> float:
    """Energy around the energy-weighted late-hit center, within 100 mm."""
    late_mask = time >= np.quantile(time, 0.75)
    late_points = points[late_mask]
    late_energy = energy[late_mask]
    if len(late_points) == 0:
        return 0.0
    late_weights = _safe_weights(late_energy)
    center = np.sum(late_points * late_weights[:, None], axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    total = max(float(np.maximum(energy, 0.0).sum()), 1e-12)
    return float(np.maximum(energy[radius <= _COMPONENT_RADIUS_MM], 0.0).sum() / total)


def track_star_features(graph, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return track/terminal/star candidates from a global-log TreeRec graph.

    ``graph.pos`` retains positions in mm.  Energy and time are recovered from
    the train-global node normalizer only to calculate physical summaries.
    """
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
    sili_mask = det_type >= 0.5
    tof_mask = ~sili_mask

    sili_pos = pos[sili_mask]
    sili_energy = energy[sili_mask]
    sili_time = time[sili_mask]
    has_sili_track = len(sili_pos) >= 3
    has_link = bool(has_sili_track and tof_mask.any())

    values = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    values[0] = float(has_sili_track)
    values[1] = float(has_link)
    values[2] = np.log1p(len(sili_pos))
    if not has_sili_track:
        return values.astype(np.float32)

    line = _weighted_line(sili_pos, sili_energy, sili_time)
    longitudinal = line['longitudinal']
    energy_weights = np.maximum(sili_energy, 0.0)
    total_energy = max(float(energy_weights.sum()), 1e-12)
    q1, q2 = np.quantile(longitudinal, (1.0 / 3.0, 2.0 / 3.0))
    start_energy = float(energy_weights[longitudinal <= q1].sum())
    terminal_energy = float(energy_weights[longitudinal >= q2].sum())

    late_mask = sili_time >= np.quantile(sili_time, 0.75)
    late_weights = _safe_weights(sili_energy[late_mask])
    late_delta = sili_pos[late_mask] - line['center']
    late_transverse_sq = np.maximum(
        np.sum(late_delta * late_delta, axis=1)
        - (late_delta @ line['axis']) ** 2,
        0.0)
    late_transverse_rms = float(np.sqrt(np.sum(late_weights * late_transverse_sq)))
    component_count, largest_component = _component_energy_fractions(
        sili_pos, sili_energy)

    values[3] = line['linearity']
    values[4] = line['planarity']
    values[5] = np.log1p(line['residual'])
    values[6] = line['time_correlation']
    values[7] = np.log1p(line['time_slope'])
    values[8] = terminal_energy / total_energy
    values[9] = np.log((terminal_energy + 1e-8) / (start_energy + 1e-8))
    values[10] = late_transverse_rms / max(line['residual'], 1e-8)
    values[11] = float(energy_weights[line['transverse'] > _OFF_AXIS_RADIUS_MM].sum() / total_energy)
    values[12] = np.log1p(component_count)
    values[13] = largest_component
    values[14] = _terminal_cluster_fraction(sili_pos, sili_energy, sili_time)

    tof_distance = 0.0
    if has_link:
        tof_pos = pos[tof_mask]
        tof_weights = _safe_weights(energy[tof_mask])
        tof_center = np.sum(tof_pos * tof_weights[:, None], axis=0)
        tof_delta = tof_center - line['center']
        tof_distance = np.linalg.norm(
            tof_delta - np.dot(tof_delta, line['axis']) * line['axis'])
        direction = tof_center - line['center']
        direction_norm = float(np.linalg.norm(direction))
        alignment = (
            abs(float(np.dot(direction, line['axis']))) / direction_norm
            if direction_norm > 1e-12 else 0.0)
        values[15] = alignment
    values[16] = np.log1p(tof_distance)

    values[17] = line['residual'] / max(line['longitudinal_rms'], 1e-8)
    values[18] = component_count / max(len(sili_pos), 1)

    if not np.isfinite(values).all():
        raise ValueError('track/star feature calculation produced non-finite values')
    return values.astype(np.float32)


def structural_track_star_features(
        graph, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return four non-degenerate geometry candidates for strict A/B tests."""
    return track_star_features(graph, mean, std)[list(STRUCTURAL_FEATURE_INDICES)]
