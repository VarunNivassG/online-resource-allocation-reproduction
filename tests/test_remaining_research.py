import json
from pathlib import Path

import pandas as pd

from olp_repro.remaining_research import run_horizon_sweep, run_risk_tail_sweep
from olp_repro.remaining_research_plots import plot_horizon_sweep, plot_risk_tail_sweep


def _small_config():
    return {
        "horizons": [20, 30], "trials": 2, "m": 2,
        "distribution": "uniform", "budget_fraction": 0.10,
        "risk_T": 30, "risk_trials": 2,
        "risk_tail_scenarios": ["stationary", "abrupt_resource"],
        "risk_tail_weights": [0.0, 0.1],
        "shift_strength": 0.5, "change_fraction": 0.5,
        "adaptive": {
            "threshold": 0.42, "warmup": 4, "min_interval": 2,
            "max_interval": None, "signal_window": 8,
            "change_window": None, "change_threshold": 3.0,
            "weights": [0.40, 0.35, 0.25],
        },
    }


def test_horizon_sweep_pairs_methods_per_horizon():
    df = run_horizon_sweep(_small_config())
    assert len(df) == 2 * 2 * 4
    assert set(df.algorithm) == {"first_order", "algorithm2_t13", "fixed_frequency", "adaptive"}
    assert set(df["T"]) == {20, 30}
    assert df.groupby(["T", "seed"]).algorithm.nunique().eq(4).all()
    assert (df.loc[df.algorithm == "adaptive", "lp_solves"] <= df.loc[df.algorithm == "adaptive", "budget_cap"]).all()


def test_risk_tail_sweep_uses_paired_instances_and_feasibility_guard():
    df = run_risk_tail_sweep(_small_config())
    assert len(df) == 2 * 2 * 2
    assert set(df.risk_weight) == {0.0, 0.1}
    assert set(df.scenario) == {"stationary", "abrupt_resource"}
    assert (df.violation_l2 <= 1e-9).all()
    assert df.groupby(["scenario", "seed"]).risk_weight.nunique().eq(2).all()


def test_new_research_plots_render(tmp_path: Path):
    cfg = _small_config()
    horizon = run_horizon_sweep(cfg)
    risk = run_risk_tail_sweep(cfg)
    p1 = plot_horizon_sweep(horizon, tmp_path / "horizon.png")
    p2 = plot_risk_tail_sweep(risk, tmp_path / "risk.png")
    assert Path(p1).stat().st_size > 1000
    assert Path(p2).stat().st_size > 1000
