from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class Evaluation:
    reward: float
    offline_reward: float
    regret: float
    violation_l2: float
    resource_used: np.ndarray
    leftover: np.ndarray


def evaluate(r: np.ndarray, A: np.ndarray, b: np.ndarray, x: np.ndarray, x_star: np.ndarray) -> Evaluation:
    used = A.T @ x
    offline_reward = float(r @ x_star)
    reward = float(r @ x)
    regret = offline_reward - reward
    violation = np.maximum(used - b, 0.0)
    return Evaluation(
        reward=reward,
        offline_reward=offline_reward,
        regret=float(regret),
        violation_l2=float(np.linalg.norm(violation)),
        resource_used=used,
        leftover=b - used,
    )
