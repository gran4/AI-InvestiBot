"""
Utilities for deriving directional labels from price series.
"""

from __future__ import annotations

import numpy as np


def generate_directional_labels(
    values: np.ndarray,
    forward_horizon: int = 5,
    threshold: float = 0.0,
    balance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create directional labels using forward returns over a specified horizon.

    Args:
        values: Flattened array of price values.
        forward_horizon: Number of steps to look ahead when computing returns.
        threshold: Minimum absolute return to keep (before balancing).
        balance: Whether to cap positive/negative samples to the same count.
    """
    flat = values.reshape(-1)
    if flat.size <= forward_horizon:
        raise ValueError("Not enough values to compute forward returns")

    returns = flat[forward_horizon:] - flat[:-forward_horizon]
    indices = np.arange(returns.shape[0])

    if threshold > 0:
        mask = np.abs(returns) >= threshold
        if np.any(mask):
            returns = returns[mask]
            indices = indices[mask]

    labels = (returns >= 0).astype(float)

    if not balance:
        return indices, labels

    pos_idx = indices[labels == 1]
    neg_idx = indices[labels == 0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        take = max(1, returns.shape[0] // 4)
        order = np.argsort(np.abs(returns))[::-1][:take]
        return indices[order], (returns[order] >= 0).astype(float)

    def _even_sample(values: np.ndarray, target: int) -> np.ndarray:
        if values.size <= target:
            return values
        positions = np.linspace(0, values.size - 1, num=target, dtype=int)
        return values[positions]

    min_count = min(pos_idx.size, neg_idx.size)
    pos_idx = _even_sample(pos_idx, min_count)
    neg_idx = _even_sample(neg_idx, min_count)
    combined_idx = np.concatenate([pos_idx, neg_idx])
    combined_labels = np.concatenate(
        [np.ones_like(pos_idx, dtype=float), np.zeros_like(neg_idx, dtype=float)]
    )
    order = np.argsort(combined_idx)
    return combined_idx[order], combined_labels[order]
