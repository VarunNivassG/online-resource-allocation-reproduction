from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, eye


def acceptance_rule(r_t: float, a_t: np.ndarray, p_t: np.ndarray) -> int:
    """Paper equation (4): x_t = I(r_t >= a_t^T p_t)."""
    return int(r_t >= float(a_t @ p_t))


def projected_subgradient_update(
    p_t: np.ndarray,
    d: np.ndarray,
    a_t: np.ndarray,
    x_t: int,
    alpha_t: float,
) -> np.ndarray:
    """Paper first-order update: p_{t+1}=[p_t-alpha_t(d-a_tx_t)]_+."""
    if alpha_t <= 0:
        raise ValueError("alpha_t must be positive")
    return np.maximum(p_t - alpha_t * (d - a_t * x_t), 0.0)


def solve_sample_dual(r: np.ndarray, A: np.ndarray, d_t: np.ndarray) -> np.ndarray:
    """Solve the sample dual LP used by the paper.

    min_{p>=0,y>=0} d_t^T p + (1/t) sum_j y_j
    s.t. A_j p + y_j >= r_j.

    The formulation is sparse and uses scipy HiGHS. This is mathematically
    faithful to the sample dual; it is intentionally not a surrogate solver.
    """
    r = np.asarray(r, dtype=float)
    A = np.asarray(A, dtype=float)
    d_t = np.asarray(d_t, dtype=float)
    t, m = A.shape
    if r.shape != (t,):
        raise ValueError("r shape mismatch")
    if d_t.shape != (m,):
        raise ValueError("d_t shape mismatch")
    if t == 0:
        return np.zeros(m)

    # -A p - I y <= -r
    Aub = hstack([
        -csr_matrix(A),
        -eye(t, format="csr"),
    ], format="csr")
    c = np.concatenate([d_t, np.full(t, 1.0 / t)])
    bounds = [(0.0, None)] * (m + t)
    res = linprog(c, A_ub=Aub, b_ub=-r, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"dual LP failed: {res.message}")
    return np.asarray(res.x[:m], dtype=float)


def solve_hindsight_primal(r: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the offline hindsight LP max r^T x s.t. A^T x <= b, 0<=x<=1."""
    r = np.asarray(r, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    T, m = A.shape
    res = linprog(-r, A_ub=A.T, b_ub=b, bounds=[(0.0, 1.0)] * T, method="highs")
    if not res.success:
        raise RuntimeError(f"hindsight LP failed: {res.message}")
    return np.asarray(res.x, dtype=float)
