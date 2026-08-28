from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .problem import OLPInstance


def dataset_path(root: str | Path, distribution: str, T: int, m: int, seed: int) -> Path:
    """Canonical path for one immutable synthetic OLP instance."""
    return Path(root) / distribution / f"T{T}_m{m}_seed{seed}.npz"


def save_instance(instance: OLPInstance, path: str | Path, *, distribution: str, seed: int) -> Path:
    """Save an OLP instance as a compressed NPZ with reproducibility metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    instance.validate()
    metadata = {
        "distribution": distribution,
        "seed": int(seed),
        "T": instance.T,
        "m": instance.m,
        "format": "olp-repro-npz-v1",
    }
    np.savez_compressed(
        path,
        r=instance.r,
        A=instance.A,
        b=instance.b,
        d=instance.d,
        metadata=np.array(json.dumps(metadata)),
    )
    return path


def load_instance(path: str | Path) -> OLPInstance:
    """Load a previously saved synthetic OLP instance."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        instance = OLPInstance(
            r=np.asarray(data["r"], dtype=float),
            A=np.asarray(data["A"], dtype=float),
            b=np.asarray(data["b"], dtype=float),
        )
    instance.validate()
    return instance
