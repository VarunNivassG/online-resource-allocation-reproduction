"""Research extension suite: adaptive computation, non-stationarity, and risk-aware prices."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .adaptive_controller import AdaptiveControllerConfig
from .algorithms import (
    RunResult, run_adaptive_resolving, run_algorithm1, run_algorithm2,
    run_first_order, run_lp_based, run_risk_aware_adaptive,
)
from .distributions import generate_input
from .dual import acceptance_rule, solve_hindsight_primal
from .metrics import evaluate
from .nonstationary import generate_nonstationary_input, known_change_point


def _count_guard_rejections(instance, result: RunResult) -> int:
    remaining = instance.b.copy()
    count = 0
    for t in range(instance.T):
        p = result.prices[t]
        proposed = acceptance_rule(float(instance.r[t]), instance.A[t], p)
        feasible = not (proposed and np.any(instance.A[t] > remaining + 1e-12))
        if proposed and not feasible:
            count += 1
        remaining -= instance.A[t] * result.x[t]
    return count


def _effective_lp_calls(algorithm: str, T: int, budget_cap: int, recorded_calls: int) -> int:
    """Exclude terminal scheduled refreshes that do not invoke the LP solver.

    Legacy baseline functions retain their original boundary-counter semantics for
    compatibility. For research metrics, a refresh at t=T is not an LP call because
    `_lp_resolve` immediately returns zero prices when there are no future requests.
    """
    actual = int(recorded_calls)
    interval = None
    if algorithm == "lp_based":
        return max(0, actual - 1) if T > 0 else actual
    if algorithm in {"algorithm1", "algorithm2"}:
        interval = max(1, min(T, int(round(T ** (1.0 / 3.0)))))
    elif algorithm == "fixed_frequency" and budget_cap > 0:
        interval = _fixed_interval(T, budget_cap)
    if interval is not None and T % interval == 0 and actual > 0:
        actual -= 1
    return max(0, actual)


def _make_record(
    *, stage: str, scenario: str, distribution: str, seed: int,
    instance, x_star: np.ndarray, algorithm: str, result: RunResult,
    runtime: float, budget_cap: int, risk_weight: float = 0.0,
) -> dict[str, Any]:
    ev = evaluate(instance.r, instance.A, instance.b, result.x, x_star)
    scheduled_calls = int(result.lp_solves)
    actual_calls = _effective_lp_calls(algorithm, instance.T, budget_cap, scheduled_calls)
    budget_positive = instance.b > 1e-12
    util = np.divide(ev.resource_used, instance.b, out=np.zeros_like(instance.b), where=budget_positive)
    leftover_fraction = np.divide(ev.leftover, instance.b, out=np.ones_like(instance.b), where=budget_positive)
    return {
        "stage": stage, "scenario": scenario, "distribution": distribution,
        "seed": int(seed), "T": instance.T, "m": instance.m,
        "budget_cap": int(budget_cap), "algorithm": algorithm,
        "reward": ev.reward, "offline_reward": ev.offline_reward,
        "regret": ev.regret, "regret_fraction": ev.regret / max(abs(ev.offline_reward), 1e-9),
        "violation_l2": ev.violation_l2,
        "max_resource_utilization": float(np.max(util, initial=0.0)),
        "mean_resource_utilization": float(np.mean(util)) if util.size else 0.0,
        "minimum_leftover_fraction": float(np.min(leftover_fraction, initial=1.0)),
        "lp_solves": int(actual_calls),
        "lp_solve_schedule_points": scheduled_calls,
        "solve_budget_utilization": float(actual_calls / budget_cap) if budget_cap > 0 else np.nan,
        "first_order_updates": int(result.first_order_updates),
        "guard_rejections": int(result.guard_rejections if result.guard_rejections else _count_guard_rejections(instance, result)),
        "runtime_seconds": float(runtime),
        "risk_weight": float(risk_weight),
    }


def _timed_run(fn) -> tuple[RunResult, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _controller_config(T: int, budget: int, options: dict) -> AdaptiveControllerConfig:
    warmup = int(options.get("warmup", max(3, min(10, int(np.sqrt(T) / 2)))) )
    min_interval = int(options.get("min_interval", 2))
    configured_max_interval = options.get("max_interval")
    if configured_max_interval is None:
        # Safety net: avoid indefinite price staleness, but allow materially
        # longer intervals than the matched fixed-frequency baseline.
        configured_max_interval = min(T, max(min_interval, int(np.ceil(T / max(1, budget)) * 1.5)))
    return AdaptiveControllerConfig(
        max_lp_solves=min(max(0, int(budget)), T),
        threshold=float(options.get("threshold", 0.42)),
        warmup=min(max(1, warmup), max(1, T - 1)),
        min_interval=min_interval,
        max_interval=int(configured_max_interval),
    )


def _budget_caps(T: int, fractions: list[float]) -> list[int]:
    caps = sorted(set(max(1, min(max(1, T - 1), int(np.ceil(T * float(f))))) for f in fractions))
    return caps


def _fixed_interval(T: int, budget: int) -> int:
    # Largest interval that yields no more than `budget` nominal boundary solves.
    return max(1, min(T, int(np.ceil(T / max(1, budget)))))


def _store_trace(
    records: list[dict], *, stage: str, scenario: str, distribution: str,
    seed: int, instance, result: RunResult, algorithm: str,
    change_point: int | None,
    budget_cap: int = -1,
) -> None:
    used = np.zeros(instance.m, dtype=float)
    cumulative_reward = 0.0
    solve_cum = 0
    for t in range(instance.T):
        used += instance.A[t] * result.x[t]
        cumulative_reward += float(instance.r[t] * result.x[t])
        solved = bool(result.solve_steps[t]) if result.solve_steps is not None else False
        solve_cum += int(solved)
        row = {
            "stage": stage, "scenario": scenario, "distribution": distribution,
            "seed": int(seed), "T": instance.T, "algorithm": algorithm,
            "budget_cap": int(budget_cap), "t": t + 1, "reward_t": float(instance.r[t] * result.x[t]),
            "accepted": int(result.x[t]), "cumulative_reward": cumulative_reward,
            "lp_solve_event": int(solved), "lp_solves_cumulative": solve_cum,
            "trigger_score": float(result.trigger_scores[t]) if result.trigger_scores is not None else np.nan,
            "dual_drift": float(result.dual_drift[t]) if result.dual_drift is not None else np.nan,
            "resource_pressure": float(result.resource_pressure[t]) if result.resource_pressure is not None else np.nan,
            "decision_instability": float(result.decision_instability[t]) if result.decision_instability is not None else np.nan,
            "change_score": float(result.change_scores[t]) if result.change_scores is not None else np.nan,
            "change_detected": int(result.change_detected[t]) if result.change_detected is not None else 0,
            "trigger_reason": result.trigger_reasons[t] if result.trigger_reasons is not None else "baseline",
            "risk_premium_mean": float(np.mean(result.risk_premiums[t])) if result.risk_premiums is not None else 0.0,
            "change_point": change_point,
        }
        for j in range(instance.m):
            row[f"used_r{j}"] = float(used[j])
            row[f"budget_r{j}"] = float(instance.b[j])
            row[f"utilization_r{j}"] = float(used[j] / instance.b[j]) if instance.b[j] > 1e-12 else np.nan
        records.append(row)


def run_research_suite(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Stages 1–3 with paired instances and preserve representative telemetry."""
    options = config.get("adaptive", {})
    horizons = [int(x) for x in config.get("horizons", [50, 100])]
    trials = int(config.get("trials", 2))
    m = int(config.get("m", 2))
    distributions = list(config.get("stationary_distributions", ["uniform"]))
    fractions = [float(x) for x in config.get("solve_budget_fractions", [0.05, 0.10])]
    scenarios = list(config.get("nonstationary_scenarios", ["abrupt_resource", "gradual"]))
    change_fraction = float(config.get("change_fraction", 0.5))
    shift_strength = float(config.get("shift_strength", 0.5))
    risk_weight = float(config.get("risk_weight", 0.25))
    trace_horizon = min(horizons)
    max_lp_t = int(config.get("lp_based_max_T", 200))
    result_rows: list[dict] = []
    trace_rows: list[dict] = []

    def run_and_record(stage, scenario, distribution, seed, inst, x_star, algorithm, fn,
                       budget_cap, risk=0.0, save_trace=False):
        result, runtime = _timed_run(fn)
        record = _make_record(
            stage=stage, scenario=scenario, distribution=distribution, seed=seed,
            instance=inst, x_star=x_star, algorithm=algorithm, result=result,
            runtime=runtime, budget_cap=budget_cap, risk_weight=risk,
        )
        cp = known_change_point(inst.T, scenario, change_fraction)
        detected_times = (np.flatnonzero(result.change_detected) + 1) if result.change_detected is not None else np.array([], dtype=int)
        post_change_times = detected_times[detected_times >= cp] if cp is not None else np.array([], dtype=int)
        record["change_point"] = cp if cp is not None else np.nan
        record["change_detections"] = int(len(detected_times))
        record["first_change_detection_t"] = int(post_change_times[0]) if post_change_times.size else np.nan
        record["change_detection_delay"] = (float(post_change_times[0] - cp) if post_change_times.size and cp is not None else np.nan)
        record["false_change_alarms"] = int(np.sum(detected_times < cp)) if cp is not None else int(len(detected_times))
        result_rows.append(record)
        if save_trace:
            _store_trace(trace_rows, stage=stage, scenario=scenario, distribution=distribution,
                         seed=seed, instance=inst, result=result, algorithm=algorithm,
                         change_point=known_change_point(inst.T, scenario, change_fraction),
                         budget_cap=budget_cap)
        return result

    # Stage 1: stationary workloads — preserve and exercise the original baselines.
    for distribution in distributions:
        for T in horizons:
            for seed in range(trials):
                inst = generate_input(T, m, distribution, seed=seed)
                x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
                f_base = max(1, min(T, int(round(T ** (1.0 / 3.0)))))
                save_trace = (T == trace_horizon and seed == 0)
                base_methods = [
                    ("first_order", lambda i=inst: run_first_order(i, "sqrtT")),
                    ("first_order_t13", lambda i=inst: run_first_order(i, "t")),
                    ("algorithm1", lambda i=inst, f=f_base: run_algorithm1(i, f)),
                    ("algorithm2", lambda i=inst, f=f_base: run_algorithm2(i, f)),
                ]
                if T <= max_lp_t:
                    base_methods.append(("lp_based", lambda i=inst: run_lp_based(i)))
                for name, fn in base_methods:
                    run_and_record("stage1", "stationary", distribution, seed, inst, x_star,
                                   name, fn, budget_cap=-1,
                                   save_trace=save_trace and name in {"first_order", "algorithm2"})
                for budget in _budget_caps(T, fractions):
                    f = _fixed_interval(T, budget)
                    run_and_record("stage1", "stationary", distribution, seed, inst, x_star,
                                   "fixed_frequency", lambda i=inst, f=f: run_algorithm2(i, f),
                                   budget_cap=budget, save_trace=save_trace)
                    cc = _controller_config(T, budget, options)
                    run_and_record("stage1", "stationary", distribution, seed, inst, x_star,
                                   "adaptive", lambda i=inst, c=cc: run_adaptive_resolving(
                                       i, c,
                                       signal_weights=tuple(options.get("weights", [0.40, 0.35, 0.25])),
                                       signal_window=int(options.get("signal_window", 25))),
                                   budget_cap=budget, save_trace=save_trace)

    # Stage 2: distribution shifts, with the same fixed budgets and no future knowledge.
    for scenario in scenarios:
        for T in horizons:
            for seed in range(trials):
                inst = generate_nonstationary_input(
                    T, m, config.get("nonstationary_base_distribution", "uniform"),
                    scenario=scenario, change_fraction=change_fraction,
                    shift_strength=shift_strength, seed=seed,
                )
                x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
                f_base = max(1, min(T, int(round(T ** (1.0 / 3.0)))))
                save_trace = (T == trace_horizon and seed == 0)
                for name, fn in [
                    ("first_order", lambda i=inst: run_first_order(i, "sqrtT")),
                    ("algorithm2", lambda i=inst, f=f_base: run_algorithm2(i, f)),
                ]:
                    run_and_record("stage2", scenario, "uniform", seed, inst, x_star,
                                   name, fn, budget_cap=-1,
                                   save_trace=save_trace and name == "first_order")
                for budget in _budget_caps(T, fractions):
                    f = _fixed_interval(T, budget)
                    run_and_record("stage2", scenario, "uniform", seed, inst, x_star,
                                   "fixed_frequency", lambda i=inst, f=f: run_algorithm2(i, f),
                                   budget_cap=budget, save_trace=save_trace)
                    cc = _controller_config(T, budget, options)
                    run_and_record("stage2", scenario, "uniform", seed, inst, x_star,
                                   "adaptive", lambda i=inst, c=cc: run_adaptive_resolving(
                                       i, c,
                                       signal_weights=tuple(options.get("weights", [0.40, 0.35, 0.25])),
                                       signal_window=int(options.get("signal_window", 25)),
                                       change_detection=True,
                                       change_window=options.get("change_window"),
                                       change_threshold=float(options.get("change_threshold", 3.0))),
                                   budget_cap=budget, save_trace=save_trace)

    # Stage 3: isolate the value (and possible cost) of scarcity-aware prices.
    risk_scenarios = ["stationary"] + scenarios
    for scenario in risk_scenarios:
        for T in horizons:
            for seed in range(trials):
                if scenario == "stationary":
                    distribution = config.get("nonstationary_base_distribution", "uniform")
                    inst = generate_input(T, m, distribution, seed=seed)
                    stage_name = "stage3"
                else:
                    distribution = "uniform"
                    inst = generate_nonstationary_input(
                        T, m, distribution, scenario=scenario,
                        change_fraction=change_fraction, shift_strength=shift_strength, seed=seed,
                    )
                    stage_name = "stage3"
                x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
                save_trace = (T == trace_horizon and seed == 0)
                for budget in _budget_caps(T, fractions):
                    cc = _controller_config(T, budget, options)
                    run_and_record(stage_name, scenario, distribution, seed, inst, x_star,
                                   "adaptive", lambda i=inst, c=cc: run_adaptive_resolving(
                                       i, c,
                                       signal_weights=tuple(options.get("weights", [0.40, 0.35, 0.25])),
                                       signal_window=int(options.get("signal_window", 25)),
                                       change_detection=(scenario != "stationary"),
                                       change_window=options.get("change_window"),
                                       change_threshold=float(options.get("change_threshold", 3.0))),
                                   budget_cap=budget, save_trace=save_trace)
                    run_and_record(stage_name, scenario, distribution, seed, inst, x_star,
                                   "risk_aware_adaptive",
                                   lambda i=inst, c=cc, rw=risk_weight, cd=(scenario != "stationary"): run_risk_aware_adaptive(
                                       i, c, risk_weight=rw,
                                       signal_weights=tuple(options.get("weights", [0.40, 0.35, 0.25])),
                                       signal_window=int(options.get("signal_window", 25)),
                                       change_detection=cd,
                                       change_window=options.get("change_window"),
                                       change_threshold=float(options.get("change_threshold", 3.0))),
                                   budget_cap=budget, risk=risk_weight, save_trace=save_trace)

    return pd.DataFrame(result_rows), pd.DataFrame(trace_rows)


def bootstrap_summary(df: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 123) -> pd.DataFrame:
    """Group-level means, standard deviations, and percentile bootstrap 95% CIs."""
    if df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["stage", "scenario", "distribution", "T", "m", "budget_cap", "algorithm"] if c in df.columns]
    metrics = [c for c in ["reward", "regret", "regret_fraction", "lp_solves", "runtime_seconds",
                           "violation_l2", "max_resource_utilization", "guard_rejections",
                           "solve_budget_utilization", "lp_solve_schedule_points", "minimum_leftover_fraction", "change_detections",
                           "change_detection_delay", "false_change_alarms"] if c in df.columns]
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n"] = len(group)
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_ci95_low"] = np.nan
                row[f"{metric}_ci95_high"] = np.nan
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            if values.size == 1:
                lo = hi = float(values[0])
            else:
                indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
                means = values[indices].mean(axis=1)
                lo, hi = np.quantile(means, [0.025, 0.975])
            row[f"{metric}_ci95_low"] = float(lo)
            row[f"{metric}_ci95_high"] = float(hi)
        rows.append(row)
    return pd.DataFrame(rows)


def save_research_outputs(results: pd.DataFrame, traces: pd.DataFrame, outdir: str | Path) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(outdir / "results.csv", index=False)
    traces.to_csv(outdir / "traces.csv", index=False)
    summary = bootstrap_summary(results)
    summary.to_csv(outdir / "summary.csv", index=False)
    return summary
