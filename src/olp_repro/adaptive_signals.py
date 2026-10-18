"""Causal online signals used to decide when to refresh OLP dual prices.

All signals use only the current state and requests observed through time t.
No future request is inspected by the adaptive controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SignalSnapshot:
    dual_drift: float
    resource_pressure: float
    decision_instability: float
    score: float


class AdaptiveSignalMonitor:
    """Compute normalized drift, inventory-pressure, and decision-change signals."""

    def __init__(self, weights: tuple[float, float, float] = (0.40, 0.35, 0.25), window: int = 25):
        if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("weights must be three nonnegative values with positive sum")
        self.weights = np.asarray(weights, dtype=float) / sum(weights)
        self.window = max(1, int(window))

    @staticmethod
    def _relative_price_drift(current: np.ndarray, anchor: np.ndarray) -> float:
        numerator = float(np.linalg.norm(current - anchor))
        denominator = 1.0 + float(np.linalg.norm(anchor))
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    @staticmethod
    def _resource_pressure(used: np.ndarray, budget: np.ndarray, t: int, T: int) -> float:
        """How far usage is ahead of a uniform budget-consumption trajectory."""
        valid = budget > 1e-12
        if not np.any(valid):
            return 0.0
        progress = t / max(T, 1)
        utilization = np.divide(used[valid], budget[valid], out=np.zeros_like(used[valid]), where=budget[valid] > 1e-12)
        remaining_horizon = max(1.0 - progress, 1.0 / max(T, 1))
        ahead = np.maximum(0.0, utilization - progress) / remaining_horizon
        return float(np.clip(np.max(ahead, initial=0.0), 0.0, 1.0))

    def _decision_instability(
        self,
        rewards_seen: np.ndarray,
        resources_seen: np.ndarray,
        current_prices: np.ndarray,
        anchor_prices: np.ndarray,
    ) -> float:
        if rewards_seen.size == 0:
            return 0.0
        start = max(0, rewards_seen.shape[0] - self.window)
        r = rewards_seen[start:]
        A = resources_seen[start:]
        current_decisions = r >= (A @ current_prices)
        anchored_decisions = r >= (A @ anchor_prices)
        return float(np.mean(current_decisions != anchored_decisions))

    def evaluate(
        self,
        *,
        t: int,
        T: int,
        budget: np.ndarray,
        used: np.ndarray,
        current_prices: np.ndarray,
        anchor_prices: np.ndarray,
        rewards_seen: np.ndarray,
        resources_seen: np.ndarray,
    ) -> SignalSnapshot:
        drift = self._relative_price_drift(current_prices, anchor_prices)
        pressure = self._resource_pressure(used, budget, t, T)
        instability = self._decision_instability(rewards_seen, resources_seen, current_prices, anchor_prices)
        score = float(np.dot(self.weights, [drift, pressure, instability]))
        return SignalSnapshot(drift, pressure, instability, score)
