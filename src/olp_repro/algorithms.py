from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dual import acceptance_rule, projected_subgradient_update, solve_sample_dual
from .problem import OLPInstance
from .adaptive_controller import AdaptiveControllerConfig, AdaptiveResolveController
from .adaptive_signals import AdaptiveSignalMonitor
from .risk import scarcity_premium
from .change_detector import WindowedShiftDetector


@dataclass
class RunResult:
    x: np.ndarray
    prices: np.ndarray
    lp_solves: int
    first_order_updates: int
    batch_ids: np.ndarray
    # Optional research telemetry. The original baselines leave these fields unset.
    trigger_scores: np.ndarray | None = None
    dual_drift: np.ndarray | None = None
    resource_pressure: np.ndarray | None = None
    decision_instability: np.ndarray | None = None
    solve_steps: np.ndarray | None = None
    trigger_reasons: list[str] | None = None
    dual_prices: np.ndarray | None = None
    risk_premiums: np.ndarray | None = None
    scarcity: np.ndarray | None = None
    guard_rejections: int = 0
    change_scores: np.ndarray | None = None
    change_detected: np.ndarray | None = None


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


def run_adaptive_resolving(
    instance: OLPInstance,
    controller_config: AdaptiveControllerConfig,
    *,
    signal_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    signal_window: int = 25,
    risk_weight: float = 0.0,
    change_detection: bool = False,
    change_window: int | None = None,
    change_threshold: float = 3.0,
) -> RunResult:
    """Online adaptive hybrid: first-order updates plus budgeted LP refreshes.

    ``risk_weight=0`` is Stage 1/2. A positive value enables the optional Stage 3
    scarcity-aware effective-price premium. The original paper baselines above
    are intentionally unchanged. Decisions and triggers use no future requests.
    """
    instance.validate()
    T, m = instance.T, instance.m
    controller_config.validate(T)
    if risk_weight < 0:
        raise ValueError("risk_weight must be nonnegative")

    controller = AdaptiveResolveController(controller_config)
    monitor = AdaptiveSignalMonitor(signal_weights, signal_window)
    default_change_window = max(3, min(25, max(8, T // 8)))
    detector_window = min(change_window or default_change_window, max(3, (T - 1) // 2))
    detector = WindowedShiftDetector(
        window=detector_window, threshold=change_threshold
    ) if change_detection else None
    pending_change = False
    p = np.zeros(m, dtype=float)
    anchor_p = p.copy()
    remaining = instance.b.copy()
    used = np.zeros(m, dtype=float)
    x = np.zeros(T, dtype=float)
    prices = np.zeros((T, m), dtype=float)
    dual_prices = np.zeros((T, m), dtype=float)
    premiums = np.zeros((T, m), dtype=float)
    scarcity_trace = np.zeros((T, m), dtype=float)
    trigger_scores = np.zeros(T, dtype=float)
    dual_drifts = np.zeros(T, dtype=float)
    pressure_trace = np.zeros(T, dtype=float)
    instability_trace = np.zeros(T, dtype=float)
    change_scores = np.zeros(T, dtype=float)
    change_detected_trace = np.zeros(T, dtype=bool)
    solve_steps = np.zeros(T, dtype=bool)
    reasons: list[str] = ["not_evaluated"] * T
    fo_updates = 0
    guard_rejections = 0
    batch_ids = np.zeros(T, dtype=int)

    for t0 in range(T):
        t = t0 + 1  # online observed-history length; controller is 1-indexed
        # Risk premium is computed from requests strictly before the current one.
        premium, scarcity = scarcity_premium(
            instance.r[:t0], instance.A[:t0], used, instance.b, t0, T, risk_weight
        ) if risk_weight > 0 else (np.zeros(m), np.zeros(m))
        effective_p = p + premium
        dual_prices[t0] = p
        prices[t0] = effective_p
        premiums[t0] = premium
        scarcity_trace[t0] = scarcity
        batch_ids[t0] = controller.solves

        proposed = acceptance_rule(instance.r[t0], instance.A[t0], effective_p)
        feasible = not (proposed and np.any(instance.A[t0] > remaining + 1e-12))
        if proposed and not feasible:
            guard_rejections += 1
        x[t0] = int(proposed and feasible)
        used += instance.A[t0] * x[t0]
        remaining -= instance.A[t0] * x[t0]

        # Preserve a cheap first-order update on every observed request.
        alpha = 1.0 / max(1, t)
        p = projected_subgradient_update(p, instance.d, instance.A[t0], int(x[t0]), alpha)
        fo_updates += 1

        snapshot = monitor.evaluate(
            t=t, T=T, budget=instance.b, used=used, current_prices=p,
            anchor_prices=anchor_p, rewards_seen=instance.r[:t],
            resources_seen=instance.A[:t],
        )
        trigger_scores[t0] = snapshot.score
        dual_drifts[t0] = snapshot.dual_drift
        pressure_trace[t0] = snapshot.resource_pressure
        instability_trace[t0] = snapshot.decision_instability
        if detector is not None:
            change_snapshot = detector.update(instance.r[:t], instance.A[:t])
            change_scores[t0] = change_snapshot.score
            change_detected_trace[t0] = change_snapshot.event
            pending_change = pending_change or change_snapshot.event
        should_solve, reason = controller.should_resolve(
            t, T, snapshot.score, change_detected=pending_change
        )
        reasons[t0] = reason
        if should_solve:
            # _lp_resolve uses only r[:t] and A[:t]; it is an online causal refresh.
            p = _lp_resolve(instance, t, remaining)
            anchor_p = p.copy()
            controller.record_solve(t)
            solve_steps[t0] = True
            reasons[t0] = reason
            pending_change = False

    if controller.solves > controller_config.max_lp_solves:
        raise AssertionError("LP-solve budget exceeded")
    return RunResult(
        x=x, prices=prices, lp_solves=controller.solves,
        first_order_updates=fo_updates, batch_ids=batch_ids,
        trigger_scores=trigger_scores, dual_drift=dual_drifts,
        resource_pressure=pressure_trace, decision_instability=instability_trace,
        solve_steps=solve_steps, trigger_reasons=reasons, dual_prices=dual_prices,
        risk_premiums=premiums if risk_weight > 0 else None,
        scarcity=scarcity_trace if risk_weight > 0 else None,
        guard_rejections=guard_rejections,
        change_scores=change_scores if detector is not None else None,
        change_detected=change_detected_trace if detector is not None else None,
    )


def run_risk_aware_adaptive(
    instance: OLPInstance, controller_config: AdaptiveControllerConfig,
    *, risk_weight: float = 0.25,
    signal_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
    signal_window: int = 25,
    change_detection: bool = False,
    change_window: int | None = None,
    change_threshold: float = 3.0,
) -> RunResult:
    """Stage 3 variant; scarcity-aware premiums influence acceptance prices."""
    if risk_weight <= 0:
        raise ValueError("risk_weight must be positive for risk-aware variant")
    return run_adaptive_resolving(
        instance, controller_config, signal_weights=signal_weights,
        signal_window=signal_window, risk_weight=risk_weight,
        change_detection=change_detection, change_window=change_window,
        change_threshold=change_threshold,
    )
