"""Utilities for fixed-length TOF paddle energy features."""
import json
from pathlib import Path

import numpy as np


N_TOF_PADDLES = 172


def tof_paddle_id(volume_id: int) -> int:
    """Remove the final two component digits from a TOF volume ID."""
    return int(volume_id) // 100


def collect_tof_paddle_ids(events) -> list[int]:
    """Collect the sorted TOF paddle IDs present in training events."""
    paddle_ids = set()
    for event in events:
        volume_ids = np.asarray(event['volume_id'], dtype=np.int64)
        tof_ids = volume_ids[(volume_ids // 100_000_000) == 1]
        paddle_ids.update((tof_ids // 100).tolist())
    return sorted(int(pid) for pid in paddle_ids)


def save_paddle_ids(paddle_ids: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'n_paddles': len(paddle_ids),
                'paddle_ids': paddle_ids,
                'definition': 'volume_id // 100 for TOF hits',
            },
            f,
            indent=2,
        )


def load_paddle_ids(path: Path) -> list[int]:
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    paddle_ids = [int(pid) for pid in payload['paddle_ids']]
    if len(paddle_ids) != N_TOF_PADDLES:
        raise ValueError(
            f'expected {N_TOF_PADDLES} TOF paddles, got {len(paddle_ids)}')
    return paddle_ids


def make_paddle_index(paddle_ids: list[int]) -> dict[int, int]:
    if len(paddle_ids) != len(set(paddle_ids)):
        raise ValueError('paddle_ids contains duplicates')
    return {int(pid): idx for idx, pid in enumerate(paddle_ids)}


def build_tof_paddle_energy(energies, volume_ids, paddle_index) -> np.ndarray:
    """Aggregate reconstructed TOF hit energy into a fixed 172-D vector."""
    energies = np.asarray(energies, dtype=np.float32)
    volume_ids = np.asarray(volume_ids, dtype=np.int64)
    result = np.zeros(len(paddle_index), dtype=np.float32)

    is_tof = (volume_ids // 100_000_000) == 1
    for energy, volume_id in zip(energies[is_tof], volume_ids[is_tof]):
        paddle_id = tof_paddle_id(volume_id)
        index = paddle_index.get(paddle_id)
        if index is None:
            raise KeyError(f'unknown TOF paddle ID: {paddle_id}')
        result[index] += energy
    return result
