"""Truth-free Si(Li) cluster and annihilation-vertex candidates for TreeRec.

The production TreeRec cache only provides observed hit-level quantities.  This
module identifies a local energy-density maximum as a candidate stopping or
annihilation vertex, then clusters hits outside that core into candidate
prongs.  It deliberately never reads TreeMc metadata, labels, beta, or ROOT
reconstruction objects.
"""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    'has_sili_vertex_candidate',
    'log_sili_cluster_count_75mm',
    'largest_sili_cluster_energy_fraction_75mm',
    'vertex_energy_fraction_75mm',
    'vertex_energy_fraction_125mm',
    'log_outer_prong_count_100mm',
    'largest_outer_prong_energy_fraction',
    'outer_prong_energy_entropy',
    'top2_prong_opening_angle_rad',
    'max_prong_opening_angle_rad',
    'energy_weighted_prong_opening_angle_rad',
)

_VERTEX_RADIUS_MM = 75.0
_VERTEX_ENERGY_RADIUS_MM = 125.0
_CLUSTER_LINK_RADIUS_MM = 75.0
_PRONG_LINK_RADIUS_MM = 100.0


def _safe_weights(energy: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(energy, dtype=np.float64), 0.0)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(weights).all():
        return np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    return weights / total


def _components(points: np.ndarray, energy: np.ndarray, radius_mm: float) -> list[dict]:
    """Return spatial connected components with energy-weighted centers."""
    count = len(points)
    if count == 0:
        return []
    delta = points[:, None, :] - points[None, :, :]
    adjacent = np.sum(delta * delta, axis=-1) <= radius_mm ** 2
    visited = np.zeros(count, dtype=bool)
    components = []
    clipped_energy = np.maximum(energy, 0.0)
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
        members = np.asarray(members, dtype=np.int64)
        member_energy = clipped_energy[members]
        weights = _safe_weights(member_energy)
        components.append({
            'energy': float(member_energy.sum()),
            'center': np.sum(points[members] * weights[:, None], axis=0),
        })
    return components


def _pair_angles(directions: np.ndarray) -> np.ndarray:
    if len(directions) < 2:
        return np.empty(0, dtype=np.float64)
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    upper = np.triu_indices(len(directions), k=1)
    return np.arccos(cosine[upper])


def cluster_vertex_features(graph, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Compute vertex-core and outer-prong candidates from observed Si(Li) hits."""
    x = graph.x.detach().cpu().numpy().astype(np.float64, copy=False)
    pos = graph.pos.detach().cpu().numpy().astype(np.float64, copy=False)
    if x.ndim != 2 or x.shape[1] != 8 or pos.shape != (len(x), 3):
        raise ValueError(f'unexpected graph shapes: x={x.shape}, pos={pos.shape}')

    values = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    if len(x) == 0:
        return values.astype(np.float32)
    log_energy = x[:, 3] * std[3] + mean[3]
    energy = np.expm1(np.clip(log_energy, 0.0, 50.0))
    sili_mask = x[:, 6] >= 0.5
    points = pos[sili_mask]
    energy = np.maximum(energy[sili_mask], 0.0)
    if len(points) < 3:
        return values.astype(np.float32)

    values[0] = 1.0
    total_energy = max(float(energy.sum()), 1e-12)
    delta = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))

    # A discrete energy-density maximum is stable under hit ordering and does
    # not require an unavailable fitted track or truth stopping point.
    local_energy = (distance <= _VERTEX_RADIUS_MM) @ energy
    vertex_index = int(np.argmax(local_energy))
    vertex = points[vertex_index]
    vertex_distance = distance[vertex_index]
    values[3] = float(energy[vertex_distance <= _VERTEX_RADIUS_MM].sum() / total_energy)
    values[4] = float(energy[vertex_distance <= _VERTEX_ENERGY_RADIUS_MM].sum() / total_energy)

    clusters = _components(points, energy, _CLUSTER_LINK_RADIUS_MM)
    cluster_energy = np.asarray([component['energy'] for component in clusters])
    values[1] = np.log1p(len(clusters))
    values[2] = float(cluster_energy.max() / total_energy)

    outer = vertex_distance > _VERTEX_RADIUS_MM
    prongs = _components(points[outer], energy[outer], _PRONG_LINK_RADIUS_MM)
    if not prongs:
        return values.astype(np.float32)
    prong_energy = np.asarray([prong['energy'] for prong in prongs])
    prong_total = max(float(prong_energy.sum()), 1e-12)
    fractions = prong_energy / prong_total
    values[5] = np.log1p(len(prongs))
    values[6] = float(fractions.max())
    nonzero = fractions[fractions > 0.0]
    values[7] = float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(prongs))) if len(prongs) > 1 else 0.0

    centers = np.stack([prong['center'] for prong in prongs])
    direction = centers - vertex
    norm = np.linalg.norm(direction, axis=1)
    valid = norm > 1e-8
    direction = direction[valid] / norm[valid, None]
    valid_energy = prong_energy[valid]
    angles = _pair_angles(direction)
    if len(angles):
        values[9] = float(angles.max())
        pair_weights = np.outer(valid_energy, valid_energy)
        values[10] = float(np.average(
            angles, weights=pair_weights[np.triu_indices(len(valid_energy), k=1)]))
    if len(direction) >= 2:
        top_two = np.argsort(valid_energy)[-2:]
        values[8] = float(np.arccos(np.clip(
            np.dot(direction[top_two[0]], direction[top_two[1]]), -1.0, 1.0)))

    if not np.isfinite(values).all():
        raise ValueError('cluster/vertex feature calculation produced non-finite values')
    return values.astype(np.float32)
