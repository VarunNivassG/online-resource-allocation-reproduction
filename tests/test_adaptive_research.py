import numpy as np

from olp_repro.adaptive_controller import AdaptiveControllerConfig, AdaptiveResolveController
from olp_repro.algorithms import run_adaptive_resolving, run_algorithm1, run_algorithm2, run_first_order, run_lp_based, run_risk_aware_adaptive
from olp_repro.distributions import generate_input
from olp_repro.dual import solve_hindsight_primal
from olp_repro.metrics import evaluate
from olp_repro.nonstationary import generate_nonstationary_input


def test_controller_never_exceeds_budget_and_is_causal():
    T = 60
    instance = generate_input(T, 2, "uniform", seed=41)
    cfg = AdaptiveControllerConfig(max_lp_solves=4, threshold=0.0, warmup=3, min_interval=2)
    result = run_adaptive_resolving(instance, cfg)
    assert result.lp_solves <= 4
    assert int(np.sum(result.solve_steps)) == result.lp_solves
    assert np.all(result.x >= 0) and np.all(result.x <= 1)
    assert result.trigger_scores.shape == (T,)
    assert np.all(np.isfinite(result.trigger_scores))
    assert np.all(instance.A.T @ result.x <= instance.b + 1e-8)


def test_risk_aware_outputs_price_premiums_and_respects_budget():
    instance = generate_input(50, 2, "uniform", seed=7)
    cfg = AdaptiveControllerConfig(max_lp_solves=3, threshold=0.2, warmup=3, min_interval=1)
    result = run_risk_aware_adaptive(instance, cfg, risk_weight=0.2)
    assert result.lp_solves <= 3
    assert result.risk_premiums.shape == (instance.T, instance.m)
    assert np.all(result.risk_premiums >= 0)
    assert np.all(instance.A.T @ result.x <= instance.b + 1e-8)


def test_nonstationary_generator_preserves_budget_and_changes_requests():
    base = generate_input(100, 2, "uniform", seed=5)
    shifted = generate_nonstationary_input(100, 2, "uniform", scenario="abrupt_resource", seed=5)
    assert np.array_equal(base.b, shifted.b)
    assert np.array_equal(base.r, shifted.r)
    assert np.allclose(shifted.A[:50], base.A[:50])
    assert np.allclose(shifted.A[50:], base.A[50:] * 1.5)


def test_original_baselines_remain_callable_and_feasible():
    instance = generate_input(30, 2, "uniform", seed=14)
    x_star = solve_hindsight_primal(instance.r, instance.A, instance.b)
    results = [
        run_first_order(instance),
        run_algorithm1(instance, 5),
        run_algorithm2(instance, 5),
        run_lp_based(instance),
    ]
    assert [r.lp_solves for r in results] == [0, 6, 6, 30]
    for result in results:
        evaluation = evaluate(instance.r, instance.A, instance.b, result.x, x_star)
        assert evaluation.violation_l2 <= 1e-8
        assert result.trigger_scores is None


def test_change_detector_responds_to_shift_using_observed_history():
    from olp_repro.change_detector import WindowedShiftDetector

    instance = generate_nonstationary_input(100, 2, "uniform", scenario="abrupt_resource", seed=0)
    detector = WindowedShiftDetector(window=10, threshold=3.0)
    events = []
    for t in range(instance.T):
        snapshot = detector.update(instance.r[:t + 1], instance.A[:t + 1])
        if snapshot.event:
            events.append(t + 1)
    assert events
    assert events[0] >= 50


def test_adaptive_controller_can_trigger_on_change():
    instance = generate_nonstationary_input(100, 2, "uniform", scenario="abrupt_resource", seed=0)
    cfg = AdaptiveControllerConfig(max_lp_solves=5, threshold=1.0, warmup=3, min_interval=1)
    result = run_adaptive_resolving(instance, cfg, change_detection=True, change_window=10, change_threshold=3.0)
    assert result.change_detected is not None
    assert np.any(result.change_detected)
    assert any(reason == "distribution_change" for reason in result.trigger_reasons)
    assert result.lp_solves <= 5


def test_detector_noise_sensitivity_includes_no_shift_control():
    from olp_repro.supplementary import run_detector_noise_sensitivity

    config = {
        "T": 30,
        "trials": 2,
        "m": 2,
        "distribution": "uniform",
        "change_fraction": 0.5,
        "base_shift_strength": 0.5,
        "shift_scenarios": ["abrupt_resource"],
        "detector_noise_levels": [0.0, 0.5],
        "adaptive": {"change_threshold": 3.0, "change_window": 5},
    }
    frame = run_detector_noise_sensitivity(config)
    assert len(frame) == 8  # two scenarios x two noise levels x two seeds
    assert set(frame["scenario"]) == {"stationary", "abrupt_resource"}
    assert frame.loc[frame.scenario == "stationary", "detection_success"].isna().all()
    assert frame.loc[frame.scenario == "abrupt_resource", "detection_success"].notna().all()
