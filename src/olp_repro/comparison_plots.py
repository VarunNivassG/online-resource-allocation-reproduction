"""Direct algorithm-comparison figures for the OLP research suite.

These figures complement the existing trajectory/diagnostic plots.  They use
summary CIs for aggregate metrics and paired seed-level points for method-vs-
method comparisons so the experimental unit remains visible.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ALGORITHM_ORDER = [
    "first_order", "first_order_t13", "algorithm1", "algorithm2",
    "fixed_frequency", "adaptive", "lp_based", "risk_aware_adaptive",
]
ALGORITHM_LABELS = {
    "first_order": "First-order (sqrt T)",
    "first_order_t13": "First-order (T¹ᐟ³)",
    "algorithm1": "Algorithm 1",
    "algorithm2": "Algorithm 2",
    "fixed_frequency": "Fixed-frequency",
    "adaptive": "Adaptive re-solving",
    "lp_based": "LP every step",
    "risk_aware_adaptive": "Risk-aware adaptive",
}


def _errors(mean: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return np.vstack([np.maximum(0.0, mean - lo), np.maximum(0.0, hi - mean)])


def _stage1_selection(summary: pd.DataFrame) -> pd.DataFrame:
    """One comparable row per algorithm at the longest stationary horizon.

    Fixed-frequency/adaptive methods are evaluated at the same largest configured
    solve cap; non-budgeted baselines keep their original settings.
    """
    d = summary[(summary["stage"] == "stage1") &
                (summary["scenario"] == "stationary") &
                (summary["distribution"] == "uniform")].copy()
    if d.empty:
        return d
    horizon = int(d["T"].max())
    d = d[d["T"] == horizon].copy()
    picked: list[pd.DataFrame] = []
    for algorithm, group in d.groupby("algorithm", sort=False):
        if algorithm in {"adaptive", "fixed_frequency"}:
            eligible = group[group["budget_cap"] > 0]
            if not eligible.empty:
                chosen_cap = eligible["budget_cap"].max()
                picked.append(eligible[eligible["budget_cap"] == chosen_cap].head(1))
        else:
            eligible = group[group["budget_cap"] == -1]
            if not eligible.empty:
                picked.append(eligible.head(1))
    if not picked:
        return d.iloc[0:0]
    out = pd.concat(picked, ignore_index=True)
    rank = {name: i for i, name in enumerate(ALGORITHM_ORDER)}
    out["_rank"] = out["algorithm"].map(rank).fillna(999)
    return out.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)


def _nice_labels(values) -> list[str]:
    return [ALGORITHM_LABELS.get(str(x), str(x).replace("_", " ")) for x in values]


def _paired_rows(results: pd.DataFrame, stage: str, algorithm_a: str, algorithm_b: str,
                 metric_cols: list[str], select_largest_budget: bool = True) -> pd.DataFrame:
    """Return A/B values paired by scenario, seed, horizon and budget.

    algorithm_a is the reference; algorithm_b is the compared method.
    """
    d = results[results["stage"] == stage].copy()
    if d.empty:
        return pd.DataFrame()
    horizon = int(d["T"].max())
    d = d[d["T"] == horizon]
    d = d[d["algorithm"].isin([algorithm_a, algorithm_b])]
    if select_largest_budget:
        budgeted = d[d["budget_cap"] > 0]
        if budgeted.empty:
            return pd.DataFrame()
        # A common cap must exist for both methods within each scenario.
        keys = ["scenario", "T"]
        caps = (budgeted.groupby(keys)["budget_cap"].max().rename("selected_cap").reset_index())
        d = budgeted.merge(caps, on=keys, how="inner")
        d = d[d["budget_cap"] == d["selected_cap"]].drop(columns="selected_cap")
    else:
        d = d[d["budget_cap"] == -1]
    join_keys = [c for c in ["stage", "scenario", "distribution", "seed", "T", "m", "budget_cap"] if c in d.columns]
    a = d[d["algorithm"] == algorithm_a][join_keys + metric_cols].copy()
    b = d[d["algorithm"] == algorithm_b][join_keys + metric_cols].copy()
    merged = a.merge(b, on=join_keys, how="inner", suffixes=("_a", "_b"))
    merged["algorithm_a"] = algorithm_a
    merged["algorithm_b"] = algorithm_b
    return merged


def _paired_comparison_plot(results: pd.DataFrame, outdir: Path, written: list[str]) -> None:
    # Adaptive vs fixed frequency: positive values mean adaptive has lower regret.
    paired = _paired_rows(results, "stage2", "fixed_frequency", "adaptive", ["regret"])
    if not paired.empty:
        paired["benefit_pct"] = 100.0 * (paired["regret_a"] - paired["regret_b"]) / np.maximum(np.abs(paired["regret_a"]), 1e-9)
        scenarios = sorted(paired["scenario"].unique())
        fig, ax = plt.subplots(figsize=(9.5, max(4.6, 0.8 * len(scenarios))))
        ybase = np.arange(len(scenarios))
        rng = np.random.default_rng(20261018)
        for yi, scenario in zip(ybase, scenarios):
            vals = paired.loc[paired["scenario"] == scenario, "benefit_pct"].to_numpy(float)
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(vals, yi + jitter, s=42, alpha=0.8, label="paired seed" if yi == 0 else None)
            ax.plot([float(np.mean(vals))], [yi], marker="D", markersize=7, color="black",
                    linestyle="None", label="mean" if yi == 0 else None)
            for val, dy in zip(vals, jitter):
                ax.annotate("", xy=(val, yi + dy), xytext=(val, yi),
                            arrowprops={"arrowstyle": "-", "color": "0.65", "lw": 0.7})
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(ybase, scenarios)
        ax.invert_yaxis()
        ax.set_xlabel("Regret reduction vs fixed-frequency (%)  ·  positive = adaptive better")
        ax.set_title(f"Paired comparison under demand shifts (T={int(paired['T'].max())})")
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(loc="best")
        fig.text(0.01, 0.01, f"Each dot is one paired seed; n={paired['seed'].nunique()} seeds in smoke config. Descriptive only.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = outdir / "10_adaptive_vs_fixed_shift_comparison.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # Risk-aware adaptive vs plain adaptive: show paired reward/regret/resource effects.
    paired = _paired_rows(results, "stage3", "adaptive", "risk_aware_adaptive",
                          ["regret", "reward", "max_resource_utilization", "guard_rejections", "runtime_seconds"])
    if not paired.empty:
        paired["regret_benefit_pct"] = 100.0 * (paired["regret_a"] - paired["regret_b"]) / np.maximum(np.abs(paired["regret_a"]), 1e-9)
        paired["reward_change_pct"] = 100.0 * (paired["reward_b"] - paired["reward_a"]) / np.maximum(np.abs(paired["reward_a"]), 1e-9)
        paired["peak_util_reduction_pp"] = 100.0 * (paired["max_resource_utilization_a"] - paired["max_resource_utilization_b"])
        paired["guard_rejections_reduced"] = paired["guard_rejections_a"] - paired["guard_rejections_b"]
        specs = [
            ("regret_benefit_pct", "Regret reduction (%)", "positive = lower regret"),
            ("reward_change_pct", "Reward change (%)", "positive = higher reward"),
            ("peak_util_reduction_pp", "Peak utilization reduction (pp)", "positive = lower peak use"),
            ("guard_rejections_reduced", "Guard rejections reduced (count)", "positive = fewer guard blocks"),
        ]
        scenarios = sorted(paired["scenario"].unique())
        fig, axes = plt.subplots(1, len(specs), figsize=(16, max(4.8, 0.8 * len(scenarios))), sharey=True)
        ybase = np.arange(len(scenarios))
        for ax, (metric, xlabel, subtitle) in zip(axes, specs):
            for yi, scenario in zip(ybase, scenarios):
                vals = paired.loc[paired["scenario"] == scenario, metric].to_numpy(float)
                jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.zeros(1)
                ax.scatter(vals, yi + jitter, s=34, alpha=0.8)
                ax.plot([float(np.mean(vals))], [yi], marker="D", markersize=6, color="black", linestyle="None")
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel(xlabel)
            ax.set_title(subtitle, fontsize=8)
            ax.grid(True, axis="x", alpha=0.22)
        axes[0].set_yticks(ybase, scenarios)
        axes[0].invert_yaxis()
        fig.suptitle(f"Paired risk-aware vs adaptive comparison (T={int(paired['T'].max())}, largest solve budget)", y=1.02)
        fig.text(0.01, 0.01, f"Each dot is one matched seed; n={paired['seed'].nunique()} seeds in smoke config. Positive is not universally better for utilization: interpret alongside reward and regret.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        path = outdir / "11_risk_aware_paired_comparison.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))


def plot_comparison_bundle(results: pd.DataFrame, summary: pd.DataFrame, outdir: str | Path) -> list[str]:
    """Generate direct cross-algorithm, paired-method and resource comparisons."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 7) Direct regret comparison across every core stationary algorithm.
    d = _stage1_selection(summary)
    if not d.empty:
        labels = _nice_labels(d["algorithm"])
        x = np.arange(len(d))
        mean = d["regret_mean"].to_numpy(float)
        yerr = _errors(mean, d["regret_ci95_low"].to_numpy(float), d["regret_ci95_high"].to_numpy(float))
        fig, ax = plt.subplots(figsize=(11, 5.8))
        bars = ax.bar(x, mean, yerr=yerr, capsize=3, alpha=0.85)
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.set_ylabel("Cumulative regret (lower is better)")
        ax.set_title(f"Stationary workload: direct algorithm comparison (T={int(d['T'].max())})")
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, mean):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        fig.text(0.01, 0.01, f"Means with percentile-bootstrap 95% CIs; smoke config has n={int(d['n'].min())} seeds per cell.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = outdir / "07_algorithm_regret_comparison.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # 8) Wall-clock runtime, deliberately shown on a log axis due to scale differences.
    if not d.empty:
        d2 = d.sort_values("runtime_seconds_mean", ascending=True).copy()
        labels = _nice_labels(d2["algorithm"])
        mean = d2["runtime_seconds_mean"].to_numpy(float)
        low = d2["runtime_seconds_ci95_low"].to_numpy(float)
        high = d2["runtime_seconds_ci95_high"].to_numpy(float)
        xerr = _errors(mean, low, high)
        fig, ax = plt.subplots(figsize=(10, 5.6))
        y = np.arange(len(d2))
        ax.barh(y, mean, alpha=0.82)
        ax.errorbar(mean, y, xerr=xerr, fmt="none", capsize=3, color="black", linewidth=1)
        ax.set_yticks(y, labels)
        ax.set_xscale("log")
        ax.set_xlabel("Measured wall-clock runtime in seconds (log scale)")
        ax.set_title(f"Runtime comparison at T={int(d2['T'].max())}")
        ax.grid(True, axis="x", alpha=0.25, which="both")
        fig.text(0.01, 0.01, "Runtime is environment-dependent; use as a secondary metric alongside LP-solve counts.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = outdir / "08_algorithm_runtime_comparison.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # 9) Actual LP-solve count comparison.
    if not d.empty:
        d3 = d.sort_values("lp_solves_mean", ascending=True).copy()
        labels = _nice_labels(d3["algorithm"])
        mean = d3["lp_solves_mean"].to_numpy(float)
        yerr = _errors(mean, d3["lp_solves_ci95_low"].to_numpy(float), d3["lp_solves_ci95_high"].to_numpy(float))
        fig, ax = plt.subplots(figsize=(10, 5.6))
        y = np.arange(len(d3))
        ax.barh(y, mean, xerr=yerr, capsize=3, alpha=0.85)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Mean number of LP solves")
        ax.set_title(f"LP-solver workload comparison at T={int(d3['T'].max())}")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path = outdir / "09_algorithm_lp_solve_comparison.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # 10 and 11) Paired method comparisons across identical seeds/instances.
    _paired_comparison_plot(results, outdir, written)

    # 12) Resource use diagnostics across stationary algorithms (not a feasibility score).
    if not d.empty:
        d4 = d.sort_values("max_resource_utilization_mean", ascending=True).copy()
        labels = _nice_labels(d4["algorithm"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
        yy = np.arange(len(d4))
        mu = d4["max_resource_utilization_mean"].to_numpy(float)
        muerr = _errors(mu, d4["max_resource_utilization_ci95_low"].to_numpy(float),
                        d4["max_resource_utilization_ci95_high"].to_numpy(float))
        axes[0].barh(yy, mu, xerr=muerr, capsize=3, alpha=0.85)
        axes[0].set_xlabel("Peak resource utilization (fraction of budget)")
        axes[0].set_xlim(left=0)
        lf = d4["minimum_leftover_fraction_mean"].to_numpy(float)
        lferr = _errors(lf, d4["minimum_leftover_fraction_ci95_low"].to_numpy(float),
                        d4["minimum_leftover_fraction_ci95_high"].to_numpy(float))
        axes[1].barh(yy, lf, xerr=lferr, capsize=3, alpha=0.85)
        axes[1].set_xlabel("Minimum leftover fraction")
        axes[1].set_xlim(left=0)
        axes[0].set_yticks(yy, labels)
        for ax in axes:
            ax.grid(True, axis="x", alpha=0.22)
        fig.suptitle(f"Resource-use comparison at T={int(d4['T'].max())}")
        fig.text(0.01, 0.01, "These are descriptive utilization metrics; higher utilization is not always preferable.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        path = outdir / "12_resource_utilization_comparison.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    return written
