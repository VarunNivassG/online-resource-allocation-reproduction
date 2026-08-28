from __future__ import annotations

import math


def theoretical_objective(T: int, f: int) -> float:
    """Theorem 3.2 rate term log(T/f) + sqrt(f)."""
    if not 1 <= f <= T:
        raise ValueError("f must be in [1,T]")
    return math.log(T / f) + math.sqrt(f)


def choose_f_grid(T: int, lp_cost_per_solve: float, fo_cost_per_step: float, budget: float) -> int:
    """Discrete implementation of the paper's computational-budget tuning idea.

    Approximate cost: ceil(T/f) LP solves plus 2f first-order work, matching
    the structure c1(k)+2c2(f) <= R used in Proposition 3.6.
    """
    candidates = []
    for f in range(1, T + 1):
        k = math.ceil(T / f)
        cost = lp_cost_per_solve * k + 2.0 * fo_cost_per_step * f
        if cost <= budget:
            candidates.append((theoretical_objective(T, f), f))
    if not candidates:
        raise ValueError("No feasible f under the supplied budget")
    return min(candidates)[1]
