"""Supplementary experiments for research-grade comparison and sensitivity figures.

The routines run paired experiments on identical request instances. Outputs are raw
per-seed CSV tables; figures are generated separately from those CSVs. These are
empirical diagnostics, not formal regret or safety guarantees.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .adaptive_controller import AdaptiveControllerConfig
from .algorithms import (
    RunResult,
    run_adaptive_resolving,
    run_algorithm2,
    run_first_order,
    run_risk_aware_adaptive,
)
from .distributions import generate_input
from .dual import solve_hindsight_primal
from .metrics import evaluate
from .nonstationary import generate_nonstationary_input, known_change_point
from .change_detector import WindowedShiftDetector
from .research import _controller_config, _fixed_interval


def _cap(T: int, fraction: float) -> int:
    return max(1, min(max(1, T - 1), int(np.ceil(T * fraction))))


def _instance(T: int, m: int, distribution: str, scenario: str, strength: float,
              change_fraction: float, seed: int):
    if scenario == "stationary":
        return generate_input(T, m, distribution, seed=seed)
    return generate_nonstationary_input(
        T, m, distribution, scenario=scenario, shift_strength=strength,
        change_fraction=change_fraction, seed=seed,
    )


def _record_result(
    *, group: str, scenario: str, strength: float, seed: int, instance,
    x_star: np.ndarray, algorithm: str, result: RunResult, budget_cap: int,
    runtime: float, actual_lp_solves: int | None = None, **extra: Any,
) -> dict[str, Any]:
    ev = evaluate(instance.r, instance.A, instance.b, result.x, x_star)
    positive = instance.b > 1e-12
    util = np.divide(ev.resource_used, instance.b, out=np.zeros_like(instance.b), where=positive)
    leftover = np.divide(ev.leftover, instance.b, out=np.ones_like(instance.b), where=positive)
    recorded_calls = int(result.lp_solves)
    metric_calls = recorded_calls if actual_lp_solves is None else int(actual_lp_solves)
    row = {
        "experiment_group": group,
        "scenario": scenario,
        "shift_strength": float(strength),
        "seed": int(seed),
        "T": int(instance.T),
        "m": int(instance.m),
        "algorithm": algorithm,
        "budget_cap": int(budget_cap),
        "reward": float(ev.reward),
        "offline_reward": float(ev.offline_reward),
        "regret": float(ev.regret),
        "regret_fraction": float(ev.regret / max(abs(ev.offline_reward), 1e-9)),
        "lp_solves": metric_calls,
        "lp_solve_schedule_points": recorded_calls,
        "first_order_updates": int(result.first_order_updates),
        "runtime_seconds": float(runtime),
        "violation_l2": float(ev.violation_l2),
        "max_resource_utilization": float(np.max(util, initial=0.0)),
        "mean_resource_utilization": float(np.mean(util)) if util.size else 0.0,
        "minimum_leftover_fraction": float(np.min(leftover, initial=1.0)),
        "guard_rejections": int(result.guard_rejections),
        "solve_budget_utilization": float(result.lp_solves / budget_cap) if budget_cap > 0 else np.nan,
    }
    row.update(extra)
    return row


def _run_timed(fn) -> tuple[RunResult, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _run_controller(instance, budget: int, adaptive: dict[str, Any], *,
                    threshold: float | None = None, weights: tuple[float, float, float] | None = None,
                    risk_weight: float = 0.0, change_detection: bool = False) -> tuple[RunResult, float]:
    opts = dict(adaptive)
    if threshold is not None:
        opts["threshold"] = float(threshold)
    cfg = _controller_config(instance.T, budget, opts)
    shared = dict(
        signal_weights=weights or tuple(opts.get("weights", [0.40, 0.35, 0.25])),
        signal_window=int(opts.get("signal_window", 25)),
        change_detection=change_detection,
        change_window=opts.get("change_window"),
        change_threshold=float(opts.get("change_threshold", 3.0)),
    )
    if risk_weight > 0:
        return _run_timed(lambda: run_risk_aware_adaptive(
            instance, cfg, risk_weight=risk_weight, **shared
        ))
    return _run_timed(lambda: run_adaptive_resolving(instance, cfg, **shared))


def run_detector_noise_sensitivity(config: dict[str, Any]) -> pd.DataFrame:
    """Run a fast detector-only noise stress test, independently of policy sweeps."""
    T = int(config.get("T", 100))
    trials = int(config.get("trials", 8))
    m = int(config.get("m", 3))
    distribution = str(config.get("distribution", "uniform"))
    change_fraction = float(config.get("change_fraction", 0.5))
    base_strength = float(config.get("base_shift_strength", 0.5))
    adaptive = dict(config.get("adaptive", {}))
    shift_scenarios = list(config.get("shift_scenarios", ["abrupt_resource", "abrupt_reward", "abrupt_combined", "gradual"]))
    noise_levels = [float(v) for v in config.get("detector_noise_levels", [0.0, 0.1, 0.25, 0.5, 1.0])]
    detector_scenarios = ["stationary"] + shift_scenarios
    default_change_window = max(3, min(25, max(8, T // 8)))
    detector_window = int(adaptive.get("change_window") or default_change_window)
    detector_window = min(max(3, detector_window), max(3, (T - 1) // 2))
    detector_threshold = float(adaptive.get("change_threshold", 3.0))
    rows: list[dict[str, Any]] = []
    for scenario in detector_scenarios:
        for seed in range(trials):
            inst = _instance(T, m, distribution, scenario, base_strength, change_fraction, seed)
            cp = known_change_point(T, scenario, change_fraction)
            pre_end = cp - 1 if cp is not None else max(3, T // 2)
            reward_scale = max(float(np.std(inst.r[:pre_end], ddof=1)) if pre_end > 1 else 0.0, 1e-8)
            resource_scale = np.maximum(np.std(inst.A[:pre_end], axis=0, ddof=1) if pre_end > 1 else np.zeros(m), 1e-8)
            # Reuse standardized random noise across levels for paired comparisons.
            rng = np.random.default_rng(120000 + seed * 97 + sum(ord(ch) for ch in scenario))
            reward_noise = rng.normal(size=T)
            resource_noise = rng.normal(size=(T, m))
            for noise_level in noise_levels:
                observed_r = inst.r + noise_level * reward_scale * reward_noise
                observed_A = np.maximum(0.0, inst.A + noise_level * resource_scale[None, :] * resource_noise)
                detector = WindowedShiftDetector(window=detector_window, threshold=detector_threshold)
                event_times: list[int] = []
                for t in range(1, T + 1):
                    snap = detector.update(observed_r[:t], observed_A[:t])
                    if snap.event:
                        event_times.append(t)
                events = np.asarray(event_times, dtype=int)
                if cp is not None:
                    post = events[events >= cp]
                    pre = events[events < cp]
                    success = float(post.size > 0)
                    delay = float(post[0] - cp) if post.size else np.nan
                    false_count = int(pre.size)
                    false_rate = float(pre.size / max(cp - 1, 1) * 100.0)
                else:
                    success = np.nan
                    delay = np.nan
                    false_count = int(events.size)
                    false_rate = float(events.size / max(T, 1) * 100.0)
                rows.append({
                    "scenario": scenario, "shift_strength": 0.0 if scenario == "stationary" else base_strength,
                    "seed": seed, "T": T, "noise_level": noise_level,
                    "noise_scale_definition": "fraction of pre-change standard deviation per feature",
                    "change_point": cp if cp is not None else np.nan,
                    "detection_success": success, "detection_delay": delay,
                    "false_alarm_count": false_count,
                    "false_alarm_per_100_prechange": false_rate,
                    "detector_window": detector_window, "detector_threshold": detector_threshold,
                })

    return pd.DataFrame(rows)


def run_supplementary_suite(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Run trajectory, shift-strength, ablation, threshold, risk and budget sweeps."""
    T = int(config.get("T", 100))
    trials = int(config.get("trials", 8))
    m = int(config.get("m", 3))
    distribution = str(config.get("distribution", "uniform"))
    change_fraction = float(config.get("change_fraction", 0.5))
    base_strength = float(config.get("base_shift_strength", 0.5))
    base_budget_fraction = float(config.get("budget_fraction", 0.08))
    adaptive = dict(config.get("adaptive", {}))
    shift_scenarios = list(config.get("shift_scenarios", ["abrupt_resource", "abrupt_reward", "abrupt_combined", "gradual"]))
    sensitivity_scenarios = list(config.get("sensitivity_scenarios", ["stationary", "abrupt_resource", "abrupt_combined", "gradual"]))
    budget = _cap(T, base_budget_fraction)
    fixed_f = _fixed_interval(T, budget)
    default_f = max(1, min(T, int(round(T ** (1 / 3)))))

    trajectories: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []

    # 1) Multi-seed time-resolved diagnostics: regret to a prefix-hindsight oracle
    # under the same resource budget, plus actual utilization over time.
    trajectory_scenarios = list(config.get("trajectory_scenarios", shift_scenarios))
    for scenario in trajectory_scenarios:
        for seed in range(trials):
            inst = _instance(T, m, distribution, scenario, base_strength, change_fraction, seed)
            # Prefix oracle solves are shared by all policies on this identical instance.
            prefix_opt = np.zeros(T, dtype=float)
            for end in range(1, T + 1):
                x_pref = solve_hindsight_primal(inst.r[:end], inst.A[:end], inst.b)
                prefix_opt[end - 1] = float(inst.r[:end] @ x_pref)
            methods: list[tuple[str, Any, int]] = [
                ("first_order", lambda i=inst: run_first_order(i, "sqrtT"), -1),
                ("algorithm2", lambda i=inst, f=default_f: run_algorithm2(i, f), -1),
                ("fixed_frequency", lambda i=inst, f=fixed_f: run_algorithm2(i, f), budget),
            ]
            cfg = _controller_config(T, budget, adaptive)
            methods.append(("adaptive", lambda i=inst, c=cfg, sc=scenario: run_adaptive_resolving(
                i, c,
                signal_weights=tuple(adaptive.get("weights", [0.40, 0.35, 0.25])),
                signal_window=int(adaptive.get("signal_window", 25)),
                change_detection=True,
                change_window=adaptive.get("change_window"),
                change_threshold=float(adaptive.get("change_threshold", 3.0)),
            ), budget))
            for name, fn, cap in methods:
                result, _ = _run_timed(fn)
                reward_step = inst.r * result.x
                cumulative_reward = np.cumsum(reward_step)
                used = np.cumsum(inst.A * result.x[:, None], axis=0)
                used_frac = np.divide(used, inst.b[None, :], out=np.zeros_like(used),
                                      where=inst.b[None, :] > 1e-12)
                cp = known_change_point(T, scenario, change_fraction)
                for idx in range(T):
                    trajectories.append({
                        "scenario": scenario, "seed": seed, "T": T, "m": m,
                        "shift_strength": base_strength, "change_point": cp,
                        "algorithm": name, "budget_cap": cap, "t": idx + 1,
                        "cumulative_reward": float(cumulative_reward[idx]),
                        "prefix_optimal_reward": float(prefix_opt[idx]),
                        "prefix_regret": float(max(0.0, prefix_opt[idx] - cumulative_reward[idx])),
                        "mean_resource_utilization": float(np.mean(used_frac[idx])),
                        "max_resource_utilization": float(np.max(used_frac[idx], initial=0.0)),
                        "lp_solves_cumulative": int(np.sum(result.solve_steps[:idx + 1])) if result.solve_steps is not None else np.nan,
                    })

    # 2) Shift magnitude and detector reliability. All policies share each instance.
    shift_strengths = [float(v) for v in config.get("shift_strengths", [0.25, 0.5, 0.75, 1.0])]
    for scenario in shift_scenarios:
        for strength in shift_strengths:
            for seed in range(trials):
                inst = _instance(T, m, distribution, scenario, strength, change_fraction, seed)
                x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
                cp = known_change_point(T, scenario, change_fraction)
                # Baselines plus both adaptive modes under the same request stream.
                methods: list[tuple[str, Any]] = [
                    ("first_order", lambda i=inst: run_first_order(i, "sqrtT")),
                    ("algorithm2", lambda i=inst, f=default_f: run_algorithm2(i, f)),
                    ("fixed_frequency", lambda i=inst, f=fixed_f: run_algorithm2(i, f)),
                ]
                methods.append(("adaptive", lambda i=inst: _run_controller(
                    i, budget, adaptive, change_detection=True
                )[0]))
                for name, fn in methods:
                    result, runtime = _run_timed(fn)
                    detected_times = (np.flatnonzero(result.change_detected) + 1) if result.change_detected is not None else np.array([], dtype=int)
                    if cp is not None:
                        post = detected_times[detected_times >= cp]
                        pre = detected_times[detected_times < cp]
                        success = float(post.size > 0) if name == "adaptive" else np.nan
                        delay = float(post[0] - cp) if name == "adaptive" and post.size else np.nan
                        false_alarms = int(pre.size) if name == "adaptive" else np.nan
                        false_rate = float(pre.size / max(cp - 1, 1) * 100.0) if name == "adaptive" else np.nan
                    else:
                        success, delay = np.nan, np.nan
                        false_alarms = int(detected_times.size) if name == "adaptive" else np.nan
                        false_rate = float(detected_times.size / T * 100.0) if name == "adaptive" else np.nan
                    detection_rows.append(_record_result(
                        group="shift_sensitivity", scenario=scenario, strength=strength,
                        seed=seed, instance=inst, x_star=x_star, algorithm=name,
                        result=result, budget_cap=budget if name in {"fixed_frequency", "adaptive"} else -1,
                        runtime=runtime,
                        actual_lp_solves=(max(0, result.lp_solves - int(T % (default_f if name == "algorithm2" else fixed_f) == 0))
                                          if name in {"algorithm2", "fixed_frequency"} else result.lp_solves),
                        change_point=cp if cp is not None else np.nan,
                        detection_success=success, detection_delay=delay,
                        false_alarm_count=false_alarms, false_alarm_per_100_prechange=false_rate,
                    ))

    # Stationary control is needed to quantify false alarms where no change exists.
    for seed in range(trials):
        inst = generate_input(T, m, distribution, seed=seed)
        x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
        result, runtime = _run_controller(inst, budget, adaptive, change_detection=True)
        events = np.flatnonzero(result.change_detected) + 1 if result.change_detected is not None else np.array([], dtype=int)
        detection_rows.append(_record_result(
            group="shift_sensitivity", scenario="stationary", strength=0.0,
            seed=seed, instance=inst, x_star=x_star, algorithm="adaptive",
            result=result, budget_cap=budget, runtime=runtime,
            change_point=np.nan, detection_success=np.nan, detection_delay=np.nan,
            false_alarm_count=int(events.size),
            false_alarm_per_100_prechange=float(events.size / T * 100.0),
        ))

    # 3) Detector-only measurement-noise sensitivity is independent of the
    # policy sweeps and can also be run separately for quick diagnostics.
    noise_table = run_detector_noise_sensitivity(config)

    # 4) Signal ablation. Change detection is disabled so only the weighted signals vary.
    weight_sets: dict[str, tuple[float, float, float]] = {
        "full": tuple(adaptive.get("weights", [0.40, 0.35, 0.25])),
        "no_dual_drift": (0.0, 0.35, 0.25),
        "no_resource_pressure": (0.40, 0.0, 0.25),
        "no_decision_instability": (0.40, 0.35, 0.0),
        "dual_drift_only": (1.0, 0.0, 0.0),
        "resource_pressure_only": (0.0, 1.0, 0.0),
        "decision_instability_only": (0.0, 0.0, 1.0),
    }
    ablation_scenarios = list(config.get("ablation_scenarios", ["stationary", "abrupt_resource", "abrupt_combined"]))
    for scenario in ablation_scenarios:
        for seed in range(trials):
            inst = _instance(T, m, distribution, scenario, base_strength, change_fraction, seed)
            x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
            for variant, weights in weight_sets.items():
                result, runtime = _run_controller(inst, budget, adaptive, weights=weights, change_detection=False)
                ablation_rows.append(_record_result(
                    group="signal_ablation", scenario=scenario, strength=base_strength,
                    seed=seed, instance=inst, x_star=x_star, algorithm=variant,
                    result=result, budget_cap=budget, runtime=runtime,
                ))

    # 5) Threshold sensitivity at a matched resource request stream; shift detector off.
    thresholds = [float(v) for v in config.get("thresholds", [0.15, 0.25, 0.35, 0.42, 0.55, 0.70])]
    for scenario in sensitivity_scenarios:
        for seed in range(trials):
            inst = _instance(T, m, distribution, scenario, base_strength, change_fraction, seed)
            x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
            for threshold in thresholds:
                result, runtime = _run_controller(
                    inst, budget, adaptive, threshold=threshold, change_detection=False
                )
                threshold_rows.append(_record_result(
                    group="threshold_sensitivity", scenario=scenario, strength=base_strength,
                    seed=seed, instance=inst, x_star=x_star,
                    algorithm=f"threshold_{threshold:.2f}", result=result,
                    budget_cap=budget, runtime=runtime, threshold=threshold,
                ))

    # 6) Risk-premium sweep, keeping the same controller and instances.
    risk_weights = [float(v) for v in config.get("risk_weights", [0.0, 0.05, 0.10, 0.25, 0.50, 1.0])]
    risk_scenarios = list(config.get("risk_scenarios", sensitivity_scenarios))
    for scenario in risk_scenarios:
        for seed in range(trials):
            inst = _instance(T, m, distribution, scenario, base_strength, change_fraction, seed)
            x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
            for rw in risk_weights:
                result, runtime = _run_controller(
                    inst, budget, adaptive, risk_weight=rw,
                    change_detection=(scenario != "stationary"),
                )
                risk_rows.append(_record_result(
                    group="risk_weight_sensitivity", scenario=scenario, strength=base_strength,
                    seed=seed, instance=inst, x_star=x_star,
                    algorithm=f"risk_{rw:g}", result=result,
                    budget_cap=budget, runtime=runtime, risk_weight=rw,
                ))

    # 7) Compute-budget frontier across several matched LP-solve budgets.
    budget_fractions = [float(v) for v in config.get("budget_fractions", [0.02, 0.05, 0.08, 0.12, 0.18])]
    budget_scenarios = list(config.get("budget_scenarios", ["stationary", "abrupt_resource", "abrupt_combined"]))
    for scenario in budget_scenarios:
        for seed in range(trials):
            inst = _instance(T, m, distribution, scenario, base_strength, change_fraction, seed)
            x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
            for fraction in budget_fractions:
                cap = _cap(T, fraction)
                interval = _fixed_interval(T, cap)
                fixed, fixed_runtime = _run_timed(lambda i=inst, f=interval: run_algorithm2(i, f))
                budget_rows.append(_record_result(
                    group="budget_sweep", scenario=scenario, strength=base_strength,
                    seed=seed, instance=inst, x_star=x_star, algorithm="fixed_frequency",
                    result=fixed, budget_cap=cap, runtime=fixed_runtime,
                    actual_lp_solves=max(0, fixed.lp_solves - int(T % interval == 0)),
                    budget_fraction=fraction,
                ))
                adapt, adapt_runtime = _run_controller(
                    inst, cap, adaptive, change_detection=(scenario != "stationary")
                )
                budget_rows.append(_record_result(
                    group="budget_sweep", scenario=scenario, strength=base_strength,
                    seed=seed, instance=inst, x_star=x_star, algorithm="adaptive",
                    result=adapt, budget_cap=cap, runtime=adapt_runtime,
                    budget_fraction=fraction,
                ))

    return {
        "trajectories": pd.DataFrame(trajectories),
        "shift_sensitivity": pd.DataFrame(detection_rows),
        "ablation": pd.DataFrame(ablation_rows),
        "threshold_sensitivity": pd.DataFrame(threshold_rows),
        "risk_sensitivity": pd.DataFrame(risk_rows),
        "budget_sweep": pd.DataFrame(budget_rows),
        "noise_sensitivity": noise_table,
    }


def save_supplementary_outputs(tables: dict[str, pd.DataFrame], outdir: str) -> None:
    from pathlib import Path
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(path / f"{name}.csv", index=False)
    inventory = "\n".join(f"- `{name}.csv`: {len(frame):,} rows" for name, frame in tables.items())
    readme = f"""# Supplementary research experiments

This directory contains raw per-seed outputs for the supplementary study.

## Tables

{inventory}

## Figures

Figures `13`–`24` are in `plots/`: prefix regret trajectories, resource-utilization trajectories, change detection versus shift magnitude, detection delay and false alarms, signal ablation (regret and LP calls), trigger-threshold sensitivity, risk-weight sensitivity, regret versus shift strength, multi-budget Pareto frontier, solve-budget utilization, and detector-noise robustness.

The additional script `scripts/run_remaining_research_plots.py` adds `horizon_sweep.csv` and `risk_tail_sweep.csv`, and Figures `25`–`26`: multi-seed regret scaling versus horizon, and mean/tail risk-aware outcomes. These use 10 seeds per horizon and 24 seeds per risk condition, respectively.

## Experimental setup

The saved `config.json` is the exact configuration used to produce these tables. The current output uses 8 paired seeds, horizon T=100, and 3 resource types. This is an expanded experimental diagnostic, but 8 seeds are still not enough to establish robust statistical superiority; use a larger held-out multi-seed run before paper claims. Bootstrap intervals are pointwise for time-series curves and descriptive for sensitivity sweeps.

## Definitions and caveats

- Prefix regret is the optimum of an offline LP restricted to the first t observed requests, using the same total resource budget, minus the policy reward accumulated through t.
- Detector-noise experiments inject noise only into detector telemetry, not the policy's decisions; they isolate the detector's robustness.
- The practical allocation simulation retains a hard feasibility guard. Zero violations therefore do not prove that risk-aware pricing is superior or unnecessary; compare reward, regret, utilization, and guard rejections too.
- The change detector and risk premium are empirical heuristics, not formal statistical tests or safety guarantees.
- Runtime depends on the machine, solver version, and system load.
"""
    (path / "README.md").write_text(readme, encoding="utf-8")
