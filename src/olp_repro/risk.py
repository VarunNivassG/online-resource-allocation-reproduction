"""Scarcity-aware price adjustments used by the optional Stage 3 algorithm."""
from __future__ import annotations
import numpy as np


def scarcity_premium(
    rewards_seen: np.ndarray,
    resources_seen: np.ndarray,
    used: np.ndarray,
    budget: np.ndarray,
    t: int,
    T: int,
    risk_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-resource price premium and normalized ahead-of-pace scarcity.

    The premium is a heuristic, not a theoretical dual-price guarantee. Its
    magnitude is scaled to observed reward per unit of resource consumption.
    """
    if risk_weight < 0:
        raise ValueError("risk_weight must be nonnegative")
    valid = budget > 1e-12
    progress = t / max(T, 1)
    remaining_horizon = max(1.0 - progress, 1.0 / max(T, 1))
    utilization = np.divide(used, budget, out=np.zeros_like(used, dtype=float), where=valid)
    scarcity = np.zeros_like(budget, dtype=float)
    scarcity[valid] = np.clip(np.maximum(0.0, utilization[valid] - progress) / remaining_horizon, 0.0, 1.0)

    if resources_seen.size == 0 or rewards_seen.size == 0:
        scale = np.ones_like(budget, dtype=float)
    else:
        reward_scale = max(float(np.median(np.abs(rewards_seen))), 1e-3)
        resource_scale = np.median(np.abs(resources_seen), axis=0)
        scale = reward_scale / np.maximum(resource_scale, 1e-3)
    premium = risk_weight * scarcity * scale
    return premium, scarcity
