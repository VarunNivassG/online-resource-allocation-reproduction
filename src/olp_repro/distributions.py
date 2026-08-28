from __future__ import annotations

import numpy as np
from .problem import OLPInstance, make_budget


def generate_input(
    T: int,
    m: int,
    kind: str = "uniform",
    average_resource: float | np.ndarray | None = None,
    seed: int | None = None,
) -> OLPInstance:
    """Generate the paper's principal synthetic input families.

    Input I: a_it ~ Uniform[0,2], r_t ~ Uniform[0,10].
    Input II: a_it ~ N(0.5, 1), r_t ~ N(0.5*m, m).

    For Input II we interpret N(mu, sigma^2), i.e. the second parameter
    is variance, matching the conventional mathematical notation.
    """
    if T <= 0 or m <= 0:
        raise ValueError("T and m must be positive")
    rng = np.random.default_rng(seed)

    kind = kind.lower()
    if kind in {"uniform", "input_i", "i"}:
        A = rng.uniform(0.0, 2.0, size=(T, m))
        r = rng.uniform(0.0, 10.0, size=T)
    elif kind in {"normal", "input_ii", "ii"}:
        A = rng.normal(0.5, 1.0, size=(T, m))
        r = rng.normal(0.5 * m, np.sqrt(m), size=T)
    else:
        raise ValueError(f"unknown distribution: {kind}")

    if average_resource is None:
        d = rng.uniform(1.0 / 3.0, 2.0 / 3.0, size=m)
    else:
        d = np.asarray(average_resource, dtype=float)
        if d.ndim == 0:
            d = np.full(m, float(d))
    b = make_budget(A, d)
    instance = OLPInstance(r=r, A=A, b=b)
    instance.validate()
    return instance
