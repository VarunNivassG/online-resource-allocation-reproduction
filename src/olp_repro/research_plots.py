"""Publication-oriented plots for the research extension suite."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .comparison_plots import plot_comparison_bundle


def _errorbar(ax, x, mean, lo, hi, **kwargs):
    mean = np.asarray(mean, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    lower = np.maximum(0.0, mean - lo)
    upper = np.maximum(0.0, hi - mean)
    ax.errorbar(x, mean, yerr=np.vstack([lower, upper]), capsize=3, **kwargs)


def plot_research_bundle(results: pd.DataFrame, summary: pd.DataFrame, traces: pd.DataFrame, outdir: str | Path) -> list[str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1) Main contribution: regret vs actual LP solves under the same horizon.
    d = summary[(summary.stage == "stage1") & (summary.scenario == "stationary") &
                (summary.distribution == "uniform")].copy()
    if not d.empty:
        T = int(d["T"].max())
        d = d[d["T"] == T]
        fig, ax = plt.subplots(figsize=(9, 6))
        for alg, g in d.groupby("algorithm"):
            g = g.sort_values("lp_solves_mean")
            if alg in {"adaptive", "fixed_frequency"}:
                g = g[g.budget_cap > 0]
            if g.empty:
                continue
            _errorbar(ax, g.lp_solves_mean, g.regret_mean, g.regret_ci95_low, g.regret_ci95_high,
                      marker="o", linewidth=2, label=alg)
        ax.set_xlabel("Mean actual LP solves")
        ax.set_ylabel("Cumulative regret (mean ± 95% bootstrap CI)")
        ax.set_title(f"Computation–regret trade-off (uniform, T={T})")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = outdir / "01_computation_regret_frontier.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # 2) Stationary scalability with baseline references and highest solve budget.
    d = summary[(summary.stage == "stage1") & (summary.scenario == "stationary") &
                (summary.distribution == "uniform")].copy()
    if not d.empty:
        selected = []
        for alg, g in d.groupby("algorithm"):
            if alg in {"adaptive", "fixed_frequency"}:
                g = g[g.budget_cap == g.groupby("T").budget_cap.transform("max")]
            else:
                g = g[g.budget_cap == -1]
            selected.append(g)
        d = pd.concat(selected, ignore_index=True) if selected else d.iloc[0:0]
        fig, ax = plt.subplots(figsize=(9, 6))
        for alg, g in d.groupby("algorithm"):
            g = g.sort_values("T")
            _errorbar(ax, g["T"], g.regret_mean, g.regret_ci95_low, g.regret_ci95_high,
                      marker="o", linewidth=1.8, label=alg)
        ax.set_xlabel("Horizon T"); ax.set_ylabel("Cumulative regret (mean ± 95% bootstrap CI)")
        ax.set_title("Stationary workload: regret vs horizon")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout()
        path = outdir / "02_stationary_regret_horizon.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # 3) Shift robustness at the highest tested budget.
    d = summary[(summary.stage == "stage2") & (summary["T"] == summary[summary.stage == "stage2"]["T"].max())].copy()
    if not d.empty:
        references = d[d.budget_cap == -1]
        budgeted = d[d.budget_cap > 0]
        budgeted = budgeted[budgeted.budget_cap == budgeted.groupby(["scenario", "T"]).budget_cap.transform("max")]
        d = pd.concat([references, budgeted], ignore_index=True)
        scenarios = list(dict.fromkeys(d.scenario.tolist()))
        algorithms = [a for a in ["first_order", "algorithm2", "fixed_frequency", "adaptive"] if a in set(d.algorithm)]
        x = np.arange(len(scenarios)); width = 0.8 / max(1, len(algorithms))
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, alg in enumerate(algorithms):
            g = d[d.algorithm == alg].set_index("scenario").reindex(scenarios)
            xpos = x + (i - (len(algorithms)-1)/2) * width
            means = g.regret_mean.to_numpy(dtype=float)
            lower = np.maximum(0, means - g.regret_ci95_low.to_numpy(dtype=float))
            upper = np.maximum(0, g.regret_ci95_high.to_numpy(dtype=float) - means)
            ax.bar(xpos, means, width=width, label=alg, yerr=np.vstack([lower, upper]), capsize=2)
        ax.set_xticks(x, scenarios, rotation=15, ha="right")
        ax.set_ylabel("Cumulative regret (mean ± 95% bootstrap CI)")
        ax.set_title("Robustness to non-stationary request distributions")
        ax.grid(True, axis="y", alpha=0.25); ax.legend()
        fig.tight_layout()
        path = outdir / "03_nonstationary_regret.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # 4) Risk extension diagnostics: trade-off between reward and guard rejections.
    d = summary[(summary.stage == "stage3") & (summary.budget_cap > 0)].copy()
    if not d.empty:
        # Use each scenario's largest budget and selected horizon.
        d = d[d["T"] == d["T"].max()]
        d = d[d.budget_cap == d.groupby(["scenario", "T"]).budget_cap.transform("max")]
        fig, ax = plt.subplots(figsize=(9, 6))
        for alg, g in d.groupby("algorithm"):
            ax.scatter(g.guard_rejections_mean, g.reward_mean, s=55, label=alg, alpha=0.85)
            for _, row in g.iterrows():
                ax.annotate(str(row.scenario), (row.guard_rejections_mean, row.reward_mean),
                            xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Potential accepts blocked by hard feasibility guard (mean)")
        ax.set_ylabel("Online reward (mean)")
        ax.set_title("Stage 3: reward vs scarcity-related blocking")
        ax.grid(True, alpha=0.25); ax.legend()
        fig.tight_layout()
        path = outdir / "04_risk_reward_guard_tradeoff.png"
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # 5) Time-series diagnostics and resource utilization for representative runs.
    if not traces.empty:
        scenarios = list(dict.fromkeys(traces.scenario.tolist()))
        for scenario in scenarios:
            g = traces[(traces.scenario == scenario) & (traces.algorithm.isin(["adaptive", "risk_aware_adaptive"]))]
            if g.empty:
                continue
            # Avoid overplotting multiple budgets: retain the largest solve cap per run/algorithm.
            run_keys = [c for c in ["stage", "scenario", "distribution", "seed", "T", "algorithm"] if c in g.columns]
            if "budget_cap" in g.columns:
                max_cap = g.groupby(run_keys, dropna=False)["budget_cap"].transform("max")
                g = g[g["budget_cap"] == max_cap]
            if g.empty:
                continue
            g = g.sort_values("t")
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            for alg, h in g.groupby("algorithm"):
                h = h.sort_values("t")
                axes[0].plot(h.t, h.trigger_score, label=f"{alg}: score", linewidth=1.7)
                axes[0].plot(h.t, h.resource_pressure, linestyle="--", alpha=0.75, label=f"{alg}: pressure")
                axes[1].plot(h.t, h.cumulative_reward, label=alg, linewidth=1.8)
                util_cols = [c for c in h.columns if c.startswith("utilization_r")]
                if util_cols:
                    axes[2].plot(h.t, h[util_cols[0]], label=f"{alg}: resource 0", linewidth=1.8)
                solve_t = h.loc[h.lp_solve_event == 1, "t"]
                if len(solve_t):
                    for st in solve_t:
                        axes[0].axvline(st, color="grey", alpha=0.12, linewidth=0.8)
            cp_vals = g.change_point.dropna().unique()
            if cp_vals.size:
                cp = int(cp_vals[0])
                for ax in axes:
                    ax.axvline(cp, color="black", linestyle=":", linewidth=1.4, label="distribution shift")
            axes[0].set_ylabel("Trigger / pressure")
            axes[1].set_ylabel("Cumulative reward")
            axes[2].set_ylabel("Resource use / budget")
            axes[2].set_xlabel("Request time t")
            axes[0].set_title(f"Online controller diagnostics — {scenario}")
            for ax in axes:
                ax.grid(True, alpha=0.22)
                ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            slug = "".join(ch if ch.isalnum() else "_" for ch in scenario.lower())
            path = outdir / f"05_trace_diagnostics_{slug}.png"
            fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

        # Separate trigger events against the composite score for a readable example.
        g = traces[(traces.stage == "stage2") & (traces.algorithm == "adaptive")]
        if not g.empty:
            chosen = g[g.seed == g.seed.min()]
            scenario = chosen.scenario.iloc[0]
            chosen = chosen[chosen.scenario == scenario].sort_values("t")
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(chosen.t, chosen.trigger_score, label="composite trigger score", linewidth=2)
            ax.axhline(0.42, color="tab:blue", linestyle="--", alpha=0.35, label="default score threshold")
            ax.plot(chosen.t, chosen.dual_drift, label="dual-price drift", alpha=0.8)
            ax.plot(chosen.t, chosen.resource_pressure, label="resource pressure", alpha=0.8)
            if "change_score" in chosen and chosen.change_score.notna().any():
                ax.plot(chosen.t, chosen.change_score, label="distribution-change score", alpha=0.8)
                ax.axhline(0.5, color="tab:red", linestyle="--", alpha=0.35, label="change score threshold")
            events = chosen[chosen.lp_solve_event == 1]
            ax.scatter(events.t, events.trigger_score, marker="x", s=55, label="LP re-solve")
            if chosen.change_point.notna().any():
                ax.axvline(int(chosen.change_point.dropna().iloc[0]), color="black", linestyle=":", label="known shift point")
            ax.set_xlabel("Request time t"); ax.set_ylabel("Normalized signal")
            ax.set_title(f"Adaptive trigger timeline — {scenario}")
            ax.grid(True, alpha=0.25); ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            path = outdir / "06_trigger_timeline.png"
            fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(path))

    # Direct comparison plots supplement the controller-specific diagnostics.
    written.extend(plot_comparison_bundle(results, summary, outdir))
    return written
