"""Controlled non-stationary request sequences for research stress tests."""
from __future__ import annotations
import numpy as np
from .problem import OLPInstance
from .distributions import generate_input

SUPPORTED_SCENARIOS = ("stationary", "abrupt_resource", "abrupt_reward", "abrupt_combined", "gradual")


def generate_nonstationary_input(
    T: int,
    m: int,
    kind: str = "uniform",
    *,
    scenario: str = "abrupt_combined",
    change_fraction: float = 0.5,
    shift_strength: float = 0.5,
    average_resource: float | np.ndarray | None = None,
    seed: int | None = None,
) -> OLPInstance:
    """Generate a reproducible instance with a controlled request-distribution shift.

    The original per-instance resource budget is preserved across stationary and
    shifted versions; this isolates the effect of the request sequence changing.
    For stationary inputs, exactly the base generator output is returned.
    """
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(f"scenario must be one of {SUPPORTED_SCENARIOS}")
    if not 0.0 < change_fraction < 1.0:
        raise ValueError("change_fraction must be between 0 and 1")
    if not 0.0 <= shift_strength <= 2.0:
        raise ValueError("shift_strength must be in [0, 2]")
    base = generate_input(T, m, kind, average_resource=average_resource, seed=seed)
    if scenario == "stationary":
        return base

    r = base.r.copy()
    A = base.A.copy()
    change = max(1, min(T - 1, int(round(T * change_fraction))))
    strength = float(shift_strength)

    if scenario in {"abrupt_resource", "abrupt_combined"}:
        # Resource pressure jumps while budget remains fixed, making the shift meaningful.
        A[change:] *= (1.0 + strength)
    if scenario in {"abrupt_reward", "abrupt_combined"}:
        # Reward distribution shifts upward or downward based on sign convention;
        # default positive strength increases late-horizon reward opportunities.
        reward_scale = 1.0 + 0.5 * strength
        r[change:] *= reward_scale
    if scenario == "gradual":
        ramp = np.linspace(0.0, 1.0, T - change, endpoint=True)
        A[change:] *= (1.0 + strength * ramp[:, None])
        r[change:] *= (1.0 + 0.5 * strength * ramp)

    shifted = OLPInstance(r=r, A=A, b=base.b.copy())
    shifted.validate()
    return shifted


def known_change_point(T: int, scenario: str, change_fraction: float = 0.5) -> int | None:
    if scenario == "stationary":
        return None
    return max(1, min(T - 1, int(round(T * change_fraction))))
