"""Targeted multi-seed studies for horizon scaling and risk-tail behavior.

These experiments complement the supplementary suite. They share generated
instances across methods/weights and export row-level measurements for auditing.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .algorithms import run_adaptive_resolving, run_algorithm2, run_first_order
from .distributions import generate_input
from .dual import solve_hindsight_primal
from .metrics import evaluate
from .nonstationary import generate_nonstationary_input
from .research import _controller_config, _fixed_interval


def _elapsed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _cap(T: int, fraction: float) -> int:
    return max(1, min(max(1, T - 1), int(np.ceil(T * fraction))))


def _summary_record(instance, x_star, result, runtime: float, *, seed: int,
                    scenario: str, algorithm: str, budget_cap: int,
                    risk_weight: float = 0.0, **extra: Any) -> dict[str, Any]:
    ev = evaluate(instance.r, instance.A, instance.b, result.x, x_star)
    scheduled_calls = int(result.lp_solves)
    interval = None
    if algorithm == "algorithm2_t13":
        interval = int(extra.get("algorithm2_interval", 0) or 0)
    elif algorithm == "fixed_frequency":
        interval = int(extra.get("fixed_interval", 0) or 0)
    actual_calls = scheduled_calls
    if interval is not None and interval > 0 and instance.T % interval == 0 and actual_calls > 0:
        # The final boundary only returns zero prices and does not call HiGHS.
        actual_calls -= 1
    positive = instance.b > 1e-12
    util = np.divide(ev.resource_used, instance.b, out=np.zeros_like(instance.b), where=positive)
    leftover = np.divide(ev.leftover, instance.b, out=np.ones_like(instance.b), where=positive)
    row = {
        "seed": int(seed), "T": int(instance.T), "m": int(instance.m),
        "scenario": str(scenario), "algorithm": str(algorithm),
        "risk_weight": float(risk_weight), "budget_cap": int(budget_cap),
        "reward": float(ev.reward), "offline_reward": float(ev.offline_reward),
        "regret": float(ev.regret),
        "regret_fraction": float(ev.regret / max(abs(ev.offline_reward), 1e-9)),
        "runtime_seconds": float(runtime), "lp_solves": int(actual_calls),
        "lp_solve_schedule_points": scheduled_calls,
        "first_order_updates": int(result.first_order_updates),
        "violation_l2": float(ev.violation_l2),
        "mean_resource_utilization": float(np.mean(util)) if util.size else 0.0,
        "max_resource_utilization": float(np.max(util, initial=0.0)),
        "minimum_leftover_fraction": float(np.min(leftover, initial=1.0)),
        "guard_rejections": int(result.guard_rejections),
    }
    row.update(extra)
    return row


def run_horizon_sweep(config: dict[str, Any]) -> pd.DataFrame:
    """Compare algorithms over multiple horizons using paired seeds."""
    horizons = [int(x) for x in config.get("horizons", [50, 100, 200, 500])]
    trials = int(config.get("trials", 10))
    m = int(config.get("m", 3))
    distribution = str(config.get("distribution", "uniform"))
    budget_fraction = float(config.get("budget_fraction", 0.08))
    adaptive = dict(config.get("adaptive", {}))
    rows: list[dict[str, Any]] = []

    for T in horizons:
        cap = _cap(T, budget_fraction)
        fixed_f = _fixed_interval(T, cap)
        paper_f = max(1, min(T, int(round(T ** (1.0 / 3.0)))))
        for seed in range(trials):
            inst = generate_input(T, m, distribution, seed=seed)
            x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
            cfg = _controller_config(T, cap, adaptive)
            methods = [
                ("first_order", lambda i=inst: run_first_order(i, "sqrtT"), -1),
                ("algorithm2_t13", lambda i=inst, f=paper_f: run_algorithm2(i, f), -1),
                ("fixed_frequency", lambda i=inst, f=fixed_f: run_algorithm2(i, f), cap),
                ("adaptive", lambda i=inst, c=cfg: run_adaptive_resolving(
                    i, c,
                    signal_weights=tuple(adaptive.get("weights", [0.40, 0.35, 0.25])),
                    signal_window=int(adaptive.get("signal_window", 25)),
                    change_detection=False,
                ), cap),
            ]
            for name, fn, reported_cap in methods:
                result, runtime = _elapsed(fn)
                rows.append(_summary_record(
                    inst, x_star, result, runtime, seed=seed, scenario="stationary",
                    algorithm=name, budget_cap=reported_cap,
                    budget_fraction=budget_fraction, fixed_interval=fixed_f,
                    algorithm2_interval=paper_f,
                ))
    return pd.DataFrame(rows)


def run_risk_tail_sweep(config: dict[str, Any]) -> pd.DataFrame:
    """Estimate mean and upper-tail regret for scarcity-aware pricing.

    This study does not claim safety from empirical results. The hard feasibility
    guard remains active for every risk weight; residual capacity and guard
    rejections are logged as descriptive resource-pressure indicators.
    """
    T = int(config.get("risk_T", 120))
    trials = int(config.get("risk_trials", 24))
    m = int(config.get("m", 3))
    distribution = str(config.get("distribution", "uniform"))
    budget_fraction = float(config.get("budget_fraction", 0.08))
    strengths = float(config.get("shift_strength", 0.5))
    change_fraction = float(config.get("change_fraction", 0.5))
    scenarios = list(config.get("risk_tail_scenarios", ["stationary", "abrupt_resource", "abrupt_combined", "gradual"]))
    weights = [float(x) for x in config.get("risk_tail_weights", [0.0, 0.1, 0.25, 0.5])]
    adaptive = dict(config.get("adaptive", {}))
    cap = _cap(T, budget_fraction)
    cfg = _controller_config(T, cap, adaptive)
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        for seed in range(trials):
            inst = generate_nonstationary_input(
                T, m, distribution, scenario=scenario,
                shift_strength=strengths, change_fraction=change_fraction, seed=seed,
            )
            x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
            for weight in weights:
                result, runtime = _elapsed(lambda i=inst, c=cfg, w=weight, sc=scenario: run_adaptive_resolving(
                    i, c,
                    signal_weights=tuple(adaptive.get("weights", [0.40, 0.35, 0.25])),
                    signal_window=int(adaptive.get("signal_window", 25)),
                    risk_weight=w,
                    change_detection=(sc != "stationary"),
                    change_window=adaptive.get("change_window"),
                    change_threshold=float(adaptive.get("change_threshold", 3.0)),
                ))
                rows.append(_summary_record(
                    inst, x_star, result, runtime, seed=seed, scenario=scenario,
                    algorithm="risk_aware_adaptive" if weight > 0 else "adaptive_no_risk_premium",
                    budget_cap=cap, risk_weight=weight,
                    shift_strength=strengths, change_fraction=change_fraction,
                ))
    return pd.DataFrame(rows)
