"""Derived event-level features for TreeRec graph models."""

import torch


C_MM_PER_NS = 299.792458
TOF_TIME_SCALE_NS = 50.0
TOF_POSITION_SCALE_MM = 1000.0


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
