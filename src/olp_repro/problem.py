from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class OLPInstance:
    """A stochastic OLP sample path.

    r[t] is the offered reward and A[t] is the resource vector a_t.
    The paper's primal LP uses 0 <= x_t <= 1 and A^T x <= b.
    """

    r: np.ndarray  # shape (T,)
    A: np.ndarray  # shape (T, m)
    b: np.ndarray  # shape (m,)

    @property
    def T(self) -> int:
        return int(self.r.shape[0])

    @property
    def m(self) -> int:
        return int(self.A.shape[1])

    @property
    def d(self) -> np.ndarray:
        return self.b / self.T

    def validate(self) -> None:
        if self.r.ndim != 1:
            raise ValueError("r must have shape (T,)")
        if self.A.ndim != 2 or self.A.shape[0] != self.T:
            raise ValueError("A must have shape (T, m)")
        if self.b.shape != (self.m,):
            raise ValueError("b must have shape (m,)")
        if not np.all(np.isfinite(self.r)) or not np.all(np.isfinite(self.A)):
            raise ValueError("r and A must be finite")
        if not np.all(np.isfinite(self.b)) or np.any(self.b < 0):
            raise ValueError("b must be finite and nonnegative")


def make_budget(A: np.ndarray, average_resource: np.ndarray | float) -> np.ndarray:
    T = A.shape[0]
    if np.isscalar(average_resource):
        d = np.full(A.shape[1], float(average_resource))
    else:
        d = np.asarray(average_resource, dtype=float)
    return d * T
