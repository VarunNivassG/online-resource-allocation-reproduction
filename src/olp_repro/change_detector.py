"""Windowed, causal change-point detector for rewards/resource-consumption streams."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ChangeDetectionSnapshot:
    score: float
    event: bool


class WindowedShiftDetector:
    """Two-window standardized mean-shift detector with hysteresis.

    The detector examines only completed observations. The event is a rising
    threshold crossing; the algorithm can hold it pending until a re-solve is
    permitted. It is intended as an empirical detector, not a formal test with
    guaranteed false-alarm probability.
    """
    def __init__(self, window: int = 10, threshold: float = 3.0, release_ratio: float = 0.5):
        if window < 3:
            raise ValueError("window must be >= 3")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 0.0 < release_ratio < 1.0:
            raise ValueError("release_ratio must be in (0, 1)")
        self.window = int(window)
        self.threshold = float(threshold)
        self.release_ratio = float(release_ratio)
        self.armed = True

    def update(self, rewards_seen: np.ndarray, resources_seen: np.ndarray) -> ChangeDetectionSnapshot:
        n = len(rewards_seen)
        w = self.window
        if n < 2 * w:
            return ChangeDetectionSnapshot(0.0, False)
        features = np.column_stack([np.asarray(rewards_seen, dtype=float), np.asarray(resources_seen, dtype=float)])
        # Compare a recent window with all earlier history. This retains a
        # persistent reference to the pre-change regime instead of forgetting
        # a shift as soon as two adjacent windows both enter the new regime.
        past = features[:-w]
        recent = features[-w:]
        diff = np.mean(recent, axis=0) - np.mean(past, axis=0)
        pooled = np.vstack([past, recent])
        std = np.std(pooled, axis=0, ddof=1)
        standard_error = np.maximum(std, 1e-8) * np.sqrt(1.0 / len(past) + 1.0 / len(recent))
        z = np.abs(diff) / standard_error
        # The maximum feature score reacts to a shift isolated to one resource.
        statistic = float(np.max(z, initial=0.0))
        score = float(np.clip(statistic / (2.0 * self.threshold), 0.0, 1.0))
        event = bool(self.armed and statistic >= self.threshold)
        if event:
            self.armed = False
        elif statistic < self.release_ratio * self.threshold:
            self.armed = True
        return ChangeDetectionSnapshot(score, event)
