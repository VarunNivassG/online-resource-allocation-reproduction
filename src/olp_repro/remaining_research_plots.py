"""Research figures covering multi-horizon scaling and tail outcomes."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = {
    "first_order": "First-order",
    "algorithm2_t13": r"Algorithm 2 ($f\approx T^{1/3}$)",
    "fixed_frequency": "Fixed-frequency (matched cap)",
    "adaptive": "Adaptive re-solving",
    "adaptive_no_risk_premium": "Adaptive (risk weight 0)",
    "risk_aware_adaptive": "Risk-aware adaptive",
}


def _mean_ci(values, seed: int, n_boot: int = 3000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if x.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    # One vectorized bootstrap call per point keeps rendering fast.
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = x[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return mean, float(lo), float(hi)


def plot_horizon_sweep(df: pd.DataFrame, path: str | Path) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    methods = ["first_order", "algorithm2_t13", "fixed_frequency", "adaptive"]
    for ax, metric, ylabel in [
        (axes[0], "regret", "Final regret (lower is better)"),
        (axes[1], "lp_solves", "Actual LP-solver calls"),
    ]:
        for mi, alg in enumerate(methods):
            g = df[df.algorithm == alg]
            xs, means, lows, highs = [], [], [], []
            for T, gt in g.groupby("T", sort=True):
                mean, lo, hi = _mean_ci(gt[metric], seed=12000 + mi * 71 + int(T))
                xs.append(T); means.append(mean); lows.append(lo); highs.append(hi)
            if not xs:
                continue
            ax.plot(xs, means, marker="o", linewidth=2, label=LABELS.get(alg, alg))
            ax.fill_between(xs, lows, highs, alpha=0.14)
        ax.set_xlabel("Horizon T")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["T"].unique()))
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=8)
    n = int(df.seed.nunique())
    fig.suptitle("Multi-seed horizon scaling: regret and computational work")
    fig.text(0.01, 0.01, f"Uniform stationary requests; m={int(df.m.iloc[0])}; n={n} paired seeds per horizon. Shading: percentile-bootstrap 95% CI. Fixed-frequency and adaptive methods share a nominal solve cap of about 8% of T; measured solve counts are shown separately.", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
    return str(path)


def _cvar_upper(values: np.ndarray, alpha: float = 0.75) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return float("nan")
    cutoff = np.quantile(x, alpha)
    tail = x[x >= cutoff]
    return float(np.mean(tail)) if tail.size else float(cutoff)


def plot_risk_tail_sweep(df: pd.DataFrame, path: str | Path) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = [s for s in ["stationary", "abrupt_resource", "abrupt_combined", "gradual"] if s in set(df.scenario)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [
        (axes[0, 0], "regret", "Mean regret (lower is better)", "mean"),
        (axes[0, 1], "regret", "Upper-quartile mean regret / CVaR75 (lower is better)", "cvar"),
        (axes[1, 0], "guard_rejections", "Feasibility-guard rejections (mean)", "mean"),
        (axes[1, 1], "minimum_leftover_fraction", "Minimum residual capacity fraction (higher = more reserve)", "mean"),
    ]
    for ax, metric, ylabel, stat_kind in specs:
        for si, scenario in enumerate(scenarios):
            g = df[df.scenario == scenario]
            rows = []
            for weight, gw in g.groupby("risk_weight", sort=True):
                if stat_kind == "cvar":
                    # Bootstrap the upper-quartile mean across paired seed rows.
                    vals = gw[metric].to_numpy(float)
                    stat = _cvar_upper(vals, 0.75)
                    if len(vals) > 1:
                        rng = np.random.default_rng(15000 + si * 101 + int(weight * 1000))
                        idx = rng.integers(0, len(vals), size=(2500, len(vals)))
                        boot = np.asarray([_cvar_upper(vals[ix], 0.75) for ix in idx])
                        lo, hi = np.quantile(boot, [0.025, 0.975])
                    else:
                        lo = hi = stat
                else:
                    stat, lo, hi = _mean_ci(gw[metric], seed=16000 + si * 101 + int(weight * 1000))
                rows.append((float(weight), stat, float(lo), float(hi)))
            if not rows:
                continue
            xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
            errs = np.asarray([[max(0.0, y-lo) for _, y, lo, hi in rows], [max(0.0, hi-y) for _, y, lo, hi in rows]])
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, linewidth=1.7, label=scenario.replace("_", " "))
        ax.set_xlabel("Scarcity-risk weight")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=7)
    n = int(df.seed.nunique())
    fig.suptitle("Risk-aware extension: mean and tail outcomes across matched seeds")
    fig.text(0.01, 0.005, f"T={int(df["T"].iloc[0])}; {n} seeds per scenario/weight. Error bars are percentile-bootstrap 95% CIs. CVaR75 is the mean of the worst quarter of regret outcomes; it is descriptive, not a formal risk or safety guarantee. Hard feasibility guard remains enabled.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
    return str(path)
