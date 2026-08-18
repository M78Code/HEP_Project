"""Derived event-level features for TreeRec graph models."""

from __future__ import annotations

import json
from pathlib import Path

import torch


C_MM_PER_NS = 299.792458
TOF_TIME_SCALE_NS = 50.0
TOF_POSITION_SCALE_MM = 1000.0

# graph_feat is assembled as
# [n_hits, total_energy, Si(Li) profile(16), TOF profile(16), TOF primary(11)].
# The first 38 fields are non-negative counts or energy deposits.  The remaining
# TOF flight time and two 3D positions can be signed, so they are only z-scored.
BASE_GRAPH_FEATURE_DIM = 45
LOG1P_GRAPH_FEATURE_END = 38


def build_base_graph_feat(batch) -> torch.Tensor:
    """Return the 45-D TreeRec event summary stored in every graph cache."""
    return torch.cat([
        batch.n_hits.view(-1, 1),
        batch.total_energy.view(-1, 1),
        batch.sili_profile.view(-1, 16),
        batch.tof_profile.view(-1, 16),
        batch.tof_feat.view(-1, 11),
    ], dim=1)


def transform_base_graph_feat(graph_feat: torch.Tensor) -> torch.Tensor:
    """Apply the fixed log transform used before global graph-feature scaling."""
    if graph_feat.ndim != 2 or graph_feat.size(1) != BASE_GRAPH_FEATURE_DIM:
        raise ValueError(
            f"expected graph_feat [batch, {BASE_GRAPH_FEATURE_DIM}], "
            f"got {tuple(graph_feat.shape)}")
    transformed = graph_feat.clone()
    transformed[:, :LOG1P_GRAPH_FEATURE_END] = torch.log1p(
        transformed[:, :LOG1P_GRAPH_FEATURE_END].clamp_min(0.0))
    return transformed


def load_graph_feature_normalizer(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a train-only 45-D graph-feature normalizer written by the audit tool."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    mean = torch.tensor(payload["mean"], dtype=torch.float32)
    std = torch.tensor(payload["std"], dtype=torch.float32)
    if mean.shape != (BASE_GRAPH_FEATURE_DIM,) or std.shape != mean.shape:
        raise ValueError(
            f"{path} must contain mean/std with shape "
            f"({BASE_GRAPH_FEATURE_DIM},)")
    if payload.get("transform") != "log1p_first_38_then_global_zscore":
        raise ValueError(f"unsupported graph-feature transform in {path}")
    return mean, std.clamp_min(1e-6)


def normalize_base_graph_feat(
        graph_feat: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor) -> torch.Tensor:
    """Apply the fitted train-only transform without changing feature order."""
    transformed = transform_base_graph_feat(graph_feat)
    return (transformed - mean) / std


def reconstruct_tof_beta(tof_feat: torch.Tensor) -> torch.Tensor:
    """Return ``[beta_tof, valid]`` from the 11-D TreeRec TOF features.

    ``tof_feat`` stores ``dt / 50 ns`` at index 4, the outer entry position at
    indices 5:8, and the inner entry position at indices 8:11. Positions are
    stored in units of 1000 mm.
    """
    if tof_feat.ndim != 2 or tof_feat.size(1) != 11:
        raise ValueError(
            f"expected tof_feat with shape [batch, 11], got {tuple(tof_feat.shape)}"
        )

    outer_hits = tof_feat[:, 2]
    inner_hits = tof_feat[:, 3]
    delta_t_ns = tof_feat[:, 4] * TOF_TIME_SCALE_NS
    path_mm = (
        torch.linalg.vector_norm(tof_feat[:, 8:11] - tof_feat[:, 5:8], dim=1)
        * TOF_POSITION_SCALE_MM
    )

    valid = (
        (outer_hits > 0)
        & (inner_hits > 0)
        & torch.isfinite(delta_t_ns)
        & torch.isfinite(path_mm)
        & (delta_t_ns > 0)
        & (path_mm > 0)
    )
    beta = path_mm / (C_MM_PER_NS * delta_t_ns.clamp_min(1e-6))
    beta = torch.nan_to_num(beta, nan=0.0, posinf=2.0, neginf=0.0)
    beta = beta.clamp(0.0, 2.0)
    beta = torch.where(valid, beta, torch.zeros_like(beta))

    return torch.stack([beta, valid.to(dtype=tof_feat.dtype)], dim=1)
