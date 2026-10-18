"""LP-solve budget and trigger logic for adaptive re-solving."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveControllerConfig:
    max_lp_solves: int
    threshold: float = 0.42
    warmup: int = 5
    min_interval: int = 2
    max_interval: int | None = None
    force_final_refresh: bool = False

    def validate(self, T: int) -> None:
        if self.max_lp_solves < 0:
            raise ValueError("max_lp_solves must be nonnegative")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.warmup < 1 or self.min_interval < 1:
            raise ValueError("warmup and min_interval must be >= 1")
        if self.max_interval is not None and self.max_interval < self.min_interval:
            raise ValueError("max_interval must be >= min_interval")
        if self.max_lp_solves > T:
            raise ValueError("max_lp_solves cannot exceed T")


class AdaptiveResolveController:
    """Choose whether to re-solve after observing request t and its outcome."""

    def __init__(self, config: AdaptiveControllerConfig):
        self.config = config
        self.solves = 0
        self.last_solve_t = 0

    def should_resolve(self, t: int, T: int, score: float, change_detected: bool = False) -> tuple[bool, str]:
        """Return (trigger, reason); t is 1-indexed and refers to observed history."""
        cfg = self.config
        if self.solves >= cfg.max_lp_solves:
            return False, "budget_exhausted"
        if t >= T:
            return False, "terminal_step"
        if t < cfg.warmup:
            return False, "warmup"
        elapsed = t - self.last_solve_t
        if elapsed < cfg.min_interval:
            return False, "minimum_interval"
        if change_detected:
            return True, "distribution_change"
        if cfg.max_interval is not None and elapsed >= cfg.max_interval:
            return True, "maximum_interval"
        if score >= cfg.threshold:
            return True, "signal_threshold"
        return False, "below_threshold"

    def record_solve(self, t: int) -> None:
        if self.solves >= self.config.max_lp_solves:
            raise RuntimeError("adaptive controller exceeded its LP-solve budget")
        self.solves += 1
        self.last_solve_t = int(t)
