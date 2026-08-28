from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dual import acceptance_rule, projected_subgradient_update, solve_sample_dual
from .problem import OLPInstance


@dataclass
class RunResult:
    x: np.ndarray
    prices: np.ndarray
    lp_solves: int
    first_order_updates: int
    batch_ids: np.ndarray


def _safe_accept(r_t: float, a_t: np.ndarray, p_t: np.ndarray, remaining: np.ndarray) -> int:
    """Decision rule plus hard feasibility guard used by practical simulation."""
    x = acceptance_rule(r_t, a_t, p_t)
    if x and np.any(a_t > remaining + 1e-12):
        return 0
    return x


def run_first_order(instance: OLPInstance, alpha_mode: str = "sqrtT") -> RunResult:
    instance.validate()
    T, m = instance.T, instance.m
    d = instance.d
    p = np.zeros(m)
    x = np.zeros(T, dtype=float)
    prices = np.zeros((T, m))
    remaining = instance.b.copy()
    fo_updates = 0

    for t in range(T):
        prices[t] = p
        x[t] = _safe_accept(instance.r[t], instance.A[t], p, remaining)
        remaining -= instance.A[t] * x[t]
        if alpha_mode == "sqrtT":
            alpha = 1.0 / np.sqrt(T)
        elif alpha_mode == "t":
            # The paper's cited O(T^{1/3}) restart baseline uses a
            # two-stage learning-rate schedule around T^{2/3}.
            split = max(1, int(T ** (2.0 / 3.0)))
            alpha = T ** (-1.0 / 3.0) if t <= split else T ** (-2.0 / 3.0)
        else:
            raise ValueError("alpha_mode must be sqrtT or t")
        p = projected_subgradient_update(p, d, instance.A[t], int(x[t]), alpha)
        fo_updates += 1

    return RunResult(x, prices, 0, fo_updates, np.arange(T) // max(1, T))


def run_lp_based(instance: OLPInstance) -> RunResult:
    """Reference LP-based method: re-solve the updated sample dual every step."""
    instance.validate()
    T, m = instance.T, instance.m
    x = np.zeros(T, dtype=float)
    prices = np.zeros((T, m))
    remaining = instance.b.copy()
    p = np.zeros(m)
    solves = 0

    for t in range(T):
        prices[t] = p
        x[t] = _safe_accept(instance.r[t], instance.A[t], p, remaining)
        remaining -= instance.A[t] * x[t]
        rem_horizon = T - (t + 1)
        if rem_horizon > 0:
            d_t = remaining / rem_horizon
        else:
            d_t = np.zeros(m)
        p = solve_sample_dual(instance.r[: t + 1], instance.A[: t + 1], d_t)
        solves += 1

    return RunResult(x, prices, solves, 0, np.arange(T) // max(1, T))


def _lp_resolve(instance: OLPInstance, t: int, remaining: np.ndarray) -> np.ndarray:
    rem_horizon = instance.T - t
    if rem_horizon <= 0:
        return np.zeros(instance.m)
    d_t = remaining / rem_horizon
    return solve_sample_dual(instance.r[:t], instance.A[:t], d_t)


def run_algorithm1(instance: OLPInstance, f: int) -> RunResult:
    """Paper Algorithm 1, implemented literally at the decision/update level.

    - initial batch: first-order updates, alpha ~ f^-1/2
    - intermediate batches: hold the latest LP dual price
    - final batch: first-order updates, alpha ~ f^-2/3
    - LP is re-solved at t divisible by f if constraints are feasible.
    """
    instance.validate()
    T, m = instance.T, instance.m
    if not 1 <= f <= T:
        raise ValueError("f must be in [1,T]")
    k = T // f
    d = instance.d
    p = np.zeros(m)
    x = np.zeros(T, dtype=float)
    prices = np.zeros((T, m))
    batch_ids = np.arange(T) // f
    remaining = instance.b.copy()
    solves = 0
    fo_updates = 0

    for t0 in range(T):
        t = t0 + 1  # paper is 1-indexed
        prices[t0] = p
        x[t0] = _safe_accept(instance.r[t0], instance.A[t0], p, remaining)
        remaining -= instance.A[t0] * x[t0]

        # Paper: first-order in initial and final batches.
        if t <= f or t >= k * f:
            if t <= f:
                alpha = f ** (-0.5)
            else:
                alpha = f ** (-2.0 / 3.0)
            p = projected_subgradient_update(
                p, d, instance.A[t0], int(x[t0]), alpha
            )
            fo_updates += 1

        # Paper: re-solve at batch boundaries when constraints are not violated.
        if t % f == 0 and np.all(remaining >= -1e-12):
            p = _lp_resolve(instance, t, remaining)
            solves += 1

    return RunResult(x, prices, solves, fo_updates, batch_ids)


def run_algorithm2(instance: OLPInstance, f: int) -> RunResult:
    """Paper Algorithm 2: first-order fine-tuning between LP re-solves."""
    instance.validate()
    T, m = instance.T, instance.m
    if not 1 <= f <= T:
        raise ValueError("f must be in [1,T]")
    d = instance.d
    p = np.zeros(m)
    x = np.zeros(T, dtype=float)
    prices = np.zeros((T, m))
    batch_ids = np.arange(T) // f
    remaining = instance.b.copy()
    solves = 0
    fo_updates = 0

    for t0 in range(T):
        t = t0 + 1
        prices[t0] = p
        x[t0] = _safe_accept(instance.r[t0], instance.A[t0], p, remaining)
        remaining -= instance.A[t0] * x[t0]

        if t % f != 0:
            alpha = 1.0 / max(1, t)
            p = projected_subgradient_update(
                p, d, instance.A[t0], int(x[t0]), alpha
            )
            fo_updates += 1

        if t % f == 0 and np.all(remaining >= -1e-12):
            p = _lp_resolve(instance, t, remaining)
            solves += 1

    return RunResult(x, prices, solves, fo_updates, batch_ids)
