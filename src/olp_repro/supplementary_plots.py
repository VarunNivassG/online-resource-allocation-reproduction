"""Additional paper-oriented plots missing from the initial research suite."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LABELS = {
    "first_order": "First-order",
    "algorithm2": "Algorithm 2",
    "fixed_frequency": "Fixed-frequency",
    "adaptive": "Adaptive re-solving",
    "full": "Full controller",
    "no_dual_drift": "Without dual drift",
    "no_resource_pressure": "Without resource pressure",
    "no_decision_instability": "Without decision instability",
    "dual_drift_only": "Dual drift only",
    "resource_pressure_only": "Resource pressure only",
    "decision_instability_only": "Decision instability only",
}


def _mean_ci(values, *, seed: int = 123, n_boot: int = 800) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if x.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = x[ix].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return mean, float(low), float(high)


def _err(mean: float, low: float, high: float) -> list[float]:
    return [max(0.0, mean - low), max(0.0, high - mean)]


def _scenario_order(values) -> list[str]:
    preferred = ["abrupt_resource", "abrupt_reward", "abrupt_combined", "gradual", "stationary"]
    present = set(map(str, values))
    return [s for s in preferred if s in present] + sorted(present - set(preferred))


def plot_supplementary_bundle(
    tables: dict[str, pd.DataFrame], outdir: str | Path, *, only: set[str] | None = None
) -> list[str]:
    """Render supplementary figures. ``only`` may contain figure numbers ("13".."23").

    This permits rerendering only missing/failed figures from saved CSVs without
    rerunning the experiments or spending time regenerating existing trajectory plots.
    """
    selected = set(only) if only is not None else {str(i) for i in range(13, 25)}
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    trajectories = tables.get("trajectories", pd.DataFrame())
    shifts = tables.get("shift_sensitivity", pd.DataFrame())
    ablation = tables.get("ablation", pd.DataFrame())
    threshold = tables.get("threshold_sensitivity", pd.DataFrame())
    risk = tables.get("risk_sensitivity", pd.DataFrame())
    budget = tables.get("budget_sweep", pd.DataFrame())
    noise = tables.get("noise_sensitivity", pd.DataFrame())

    # 13) Prefix regret trajectories with pointwise seed-bootstrap confidence bands.
    if "13" in selected and not trajectories.empty:
        scenarios = _scenario_order(trajectories.scenario.unique())
        ncols = 2
        nrows = int(np.ceil(len(scenarios) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows), squeeze=False)
        algorithms = [a for a in ["first_order", "algorithm2", "fixed_frequency", "adaptive"] if a in set(trajectories.algorithm)]
        for ax, scenario in zip(axes.flat, scenarios):
            d = trajectories[trajectories.scenario == scenario]
            for ai, alg in enumerate(algorithms):
                g = d[d.algorithm == alg]
                if g.empty:
                    continue
                xs, means, lows, highs = [], [], [], []
                for t, gt in g.groupby("t", sort=True):
                    mean, lo, hi = _mean_ci(gt.prefix_regret, seed=1000 + int(t) + ai * 37)
                    xs.append(t); means.append(mean); lows.append(lo); highs.append(hi)
                ax.plot(xs, means, linewidth=1.8, label=LABELS.get(alg, alg))
                ax.fill_between(xs, lows, highs, alpha=0.14)
            cpv = d.change_point.dropna().unique()
            if cpv.size:
                ax.axvline(int(cpv[0]), color="black", linestyle="--", linewidth=1.0, label="Known shift")
            ax.set_title(scenario.replace("_", " ").title())
            ax.set_xlabel("Observed requests (t)")
            ax.set_ylabel("Prefix hindsight regret")
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=7)
        for ax in axes.flat[len(scenarios):]:
            ax.set_visible(False)
        nseed = int(trajectories.seed.nunique())
        fig.suptitle("Time-resolved prefix regret after controlled demand shifts", y=1.01)
        fig.text(0.01, 0.005, f"Shading: pointwise percentile-bootstrap 95% CI across {nseed} paired seeds. Prefix oracle uses the same resource budget on requests observed up to t.", fontsize=8)
        fig.tight_layout(rect=(0, 0.025, 1, 0.98))
        p = out / "13_prefix_regret_trajectories.png"
        fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    # 14) Multi-seed utilization trajectories: descriptive, not a risk guarantee.
    if "14" in selected and not trajectories.empty:
        scenarios = _scenario_order(trajectories.scenario.unique())
        ncols = 2
        nrows = int(np.ceil(len(scenarios) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.1 * nrows), squeeze=False)
        algorithms = [a for a in ["first_order", "algorithm2", "fixed_frequency", "adaptive"] if a in set(trajectories.algorithm)]
        for ax, scenario in zip(axes.flat, scenarios):
            d = trajectories[trajectories.scenario == scenario]
            for ai, alg in enumerate(algorithms):
                g = d[d.algorithm == alg]
                if g.empty:
                    continue
                xs, means, lows, highs = [], [], [], []
                for t, gt in g.groupby("t", sort=True):
                    mean, lo, hi = _mean_ci(gt.mean_resource_utilization, seed=2000 + int(t) + ai * 41)
                    xs.append(t); means.append(mean); lows.append(lo); highs.append(hi)
                ax.plot(xs, means, linewidth=1.8, label=LABELS.get(alg, alg))
                ax.fill_between(xs, lows, highs, alpha=0.14)
            cpv = d.change_point.dropna().unique()
            if cpv.size:
                ax.axvline(int(cpv[0]), color="black", linestyle="--", linewidth=1.0, label="Known shift")
            ax.set_title(scenario.replace("_", " ").title())
            ax.set_xlabel("Observed requests (t)")
            ax.set_ylabel("Mean resource utilization (fraction of budget)")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=7)
        for ax in axes.flat[len(scenarios):]:
            ax.set_visible(False)
        fig.suptitle("Resource-utilization trajectories across matched seeds", y=1.01)
        fig.text(0.01, 0.005, f"Shading: pointwise percentile-bootstrap 95% CI; n={int(trajectories.seed.nunique())} seeds. Higher utilization is not necessarily better.", fontsize=8)
        fig.tight_layout(rect=(0, 0.025, 1, 0.98))
        p = out / "14_resource_utilization_trajectories.png"
        fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    # 15) Detection probability versus shift magnitude.
    if "15" in selected and not shifts.empty:
        d = shifts[(shifts.algorithm == "adaptive") & shifts.detection_success.notna()].copy()
        if not d.empty:
            scenarios = _scenario_order(d.scenario.unique())
            fig, ax = plt.subplots(figsize=(9.5, 5.8))
            for si, scenario in enumerate(scenarios):
                g = d[d.scenario == scenario]
                points = []
                for strength, gs in g.groupby("shift_strength", sort=True):
                    mean, lo, hi = _mean_ci(gs.detection_success, seed=3000 + int(strength * 1000) + si)
                    points.append((strength, mean, lo, hi))
                if not points:
                    continue
                xs = [x[0] for x in points]; ys = [x[1] for x in points]
                yerr = np.asarray([_err(y, lo, hi) for _, y, lo, hi in points]).T
                ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, linewidth=1.8, label=scenario.replace("_", " "))
            ax.set_xlabel("Shift strength")
            ax.set_ylabel("Detection probability")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("Change-detector sensitivity to shift magnitude")
            ax.grid(True, alpha=0.25); ax.legend()
            fig.text(0.01, 0.01, f"Pointwise percentile-bootstrap 95% CI across seeds; n={int(d.seed.nunique())} seeds per strength. A detected event is one at or after the known shift point.", fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            p = out / "15_detection_probability_vs_shift_strength.png"
            fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    # 16) Detection delay among successful detections and pre-change false alarm rate.
    if "16" in selected and not shifts.empty:
        d = shifts[(shifts.algorithm == "adaptive") & (shifts.scenario != "stationary")].copy()
        if not d.empty:
            scenarios = _scenario_order(d.scenario.unique())
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
            for si, scenario in enumerate(scenarios):
                g = d[d.scenario == scenario]
                delays, dl, dh, fars, fl, fh, strengths = [], [], [], [], [], [], []
                for strength, gs in g.groupby("shift_strength", sort=True):
                    detected = gs.loc[gs.detection_success > 0, "detection_delay"].dropna().to_numpy(float)
                    if detected.size:
                        dm, dlo, dhi = _mean_ci(detected, seed=4100 + int(strength * 1000) + si)
                    else:
                        dm = dlo = dhi = np.nan
                    fm, flo, fhi = _mean_ci(gs.false_alarm_per_100_prechange, seed=4200 + int(strength * 1000) + si)
                    strengths.append(strength); delays.append(dm); dl.append(dlo); dh.append(dhi)
                    fars.append(fm); fl.append(flo); fh.append(fhi)
                axes[0].errorbar(strengths, delays,
                    yerr=np.asarray([[max(0, m-l) if np.isfinite(m) and np.isfinite(l) else 0 for m,l in zip(delays,dl)], [max(0, h-m) if np.isfinite(m) and np.isfinite(h) else 0 for m,h in zip(delays,dh)]]),
                    marker="o", capsize=3, label=scenario.replace("_", " "))
                axes[1].errorbar(strengths, fars,
                    yerr=np.asarray([[max(0, m-l) for m,l in zip(fars,fl)], [max(0, h-m) for m,h in zip(fars,fh)]]),
                    marker="o", capsize=3, label=scenario.replace("_", " "))
            axes[0].set_xlabel("Shift strength"); axes[0].set_ylabel("Mean detection delay (requests)")
            axes[0].set_title("Delay among detected shifts")
            axes[1].set_xlabel("Shift strength"); axes[1].set_ylabel("False alarms per 100 pre-shift requests")
            axes[1].set_title("Pre-change false alarms")
            for ax in axes:
                ax.grid(True, alpha=0.23); ax.legend(fontsize=7)
            fig.suptitle("Change-detector delay and false-alarm analysis")
            fig.text(0.01, 0.01, "Detection delay excludes missed shifts; interpret it together with Figure 15 detection probability. Error bars are bootstrap 95% CIs.", fontsize=8)
            fig.tight_layout(rect=(0, 0.04, 1, 0.94))
            p = out / "16_detection_delay_false_alarms.png"
            fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    # 17) Paired regret impact of removing each signal on the same seed and instance.
    if (selected & {"17", "18"}) and not ablation.empty and "full" in set(ablation.algorithm):
        scenarios = _scenario_order(ablation.scenario.unique())
        variants = [v for v in ["no_dual_drift", "no_resource_pressure", "no_decision_instability", "dual_drift_only", "resource_pressure_only", "decision_instability_only"] if v in set(ablation.algorithm)]
        ncols = 1
        fig, axes = plt.subplots(len(scenarios), ncols, figsize=(11, max(4.0, 3.6 * len(scenarios))), squeeze=False)
        for ax, scenario in zip(axes.flat, scenarios):
            d = ablation[ablation.scenario == scenario]
            pivot = d.pivot(index="seed", columns="algorithm", values="regret")
            rows = []
            for variant in variants:
                if variant not in pivot or "full" not in pivot:
                    continue
                delta = (pivot[variant] - pivot["full"]).dropna().to_numpy(float)
                mean, lo, hi = _mean_ci(delta, seed=5100 + len(rows))
                rows.append((variant, delta, mean, lo, hi))
            yy = np.arange(len(rows))
            for y, (variant, values, mean, lo, hi) in zip(yy, rows):
                jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(1)
                ax.scatter(values, y + jitter, s=23, alpha=0.65)
                ax.errorbar(mean, y, xerr=np.asarray(_err(mean, lo, hi)).reshape(2, 1), fmt="D", color="black", capsize=3)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
            ax.set_yticks(yy, [LABELS.get(r[0], r[0]) for r in rows])
            ax.invert_yaxis()
            ax.set_title(scenario.replace("_", " ").title())
            ax.set_xlabel("Paired regret difference vs full controller (positive = ablation worse)")
            ax.grid(True, axis="x", alpha=0.22)
        fig.suptitle("Signal ablation: which trigger components contribute?", y=1.01)
        fig.text(0.01, 0.005, f"Dots are paired seeds; black diamonds with bootstrap 95% CIs show mean change from the full controller. n={int(ablation.seed.nunique())} seeds per scenario.", fontsize=8)
        fig.tight_layout(rect=(0, 0.025, 1, 0.98))
        p = out / "17_signal_ablation_paired_regret.png"
        fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

        # 18) Ablation computational cost side-by-side.
        if "18" in selected:
            agg_rows = []
            for (scenario, variant), g in ablation.groupby(["scenario", "algorithm"]):
                mean, lo, hi = _mean_ci(g.lp_solves, seed=6100 + len(agg_rows))
                agg_rows.append({"scenario": scenario, "algorithm": variant, "mean": mean, "lo": lo, "hi": hi})
            agg = pd.DataFrame(agg_rows)
            fig, axes = plt.subplots(1, len(scenarios), figsize=(5.0 * len(scenarios), 5.4), squeeze=False)
            for ax, scenario in zip(axes.flat, scenarios):
                d = agg[agg.scenario == scenario].set_index("algorithm").reindex(["full"] + variants).dropna(subset=["mean"]).reset_index()
                y = np.arange(len(d)); labels = [LABELS.get(v, v) for v in d.algorithm]
                xerr = np.asarray([_err(m, lo, hi) for m, lo, hi in zip(d["mean"], d["lo"], d["hi"])]).T
                ax.barh(y, d["mean"], xerr=xerr, capsize=2, alpha=0.82)
                ax.set_yticks(y, labels, fontsize=7); ax.invert_yaxis()
                ax.set_xlabel("Actual LP solves"); ax.set_title(scenario.replace("_", " "))
                ax.grid(True, axis="x", alpha=0.2)
            fig.suptitle("Signal ablation: computational cost")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            p = out / "18_signal_ablation_lp_solves.png"
            fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))


    # 19) Threshold sensitivity: regret and solver work on identical seeds.
    if "19" in selected and not threshold.empty:
        d = threshold.copy()
        d["threshold"] = pd.to_numeric(d["threshold"])
        scenarios = _scenario_order(d.scenario.unique())
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        metrics = [("regret", "Cumulative regret"), ("lp_solves", "Actual LP-solve count")]
        for ax, (metric, ylabel) in zip(axes, metrics):
            for si, scenario in enumerate(scenarios):
                g = d[d.scenario == scenario]
                xs, ms, ls, hs = [], [], [], []
                for val, gv in g.groupby("threshold", sort=True):
                    mean, lo, hi = _mean_ci(gv[metric], seed=7100 + si + int(val * 1000))
                    xs.append(val); ms.append(mean); ls.append(lo); hs.append(hi)
                ax.plot(xs, ms, marker="o", linewidth=1.8, label=scenario.replace("_", " "))
                ax.fill_between(xs, ls, hs, alpha=0.12)
            ax.set_xlabel("Trigger threshold")
            ax.set_ylabel(ylabel + " (mean with bootstrap CI)")
            ax.grid(True, alpha=0.23); ax.legend(fontsize=7)
        fig.suptitle("Adaptive re-solving: trigger-threshold sensitivity")
        fig.text(0.01, 0.01, "Change detection is disabled in this sweep to isolate the composite trigger threshold. Bands are pointwise 95% bootstrap CIs.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        p = out / "19_trigger_threshold_sensitivity.png"
        fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    # 20) Risk-weight sensitivity across regret, reward, utilization, and guard events.
    if "20" in selected and not risk.empty:
        d = risk.copy()
        d["risk_weight"] = pd.to_numeric(d["risk_weight"])
        scenarios = _scenario_order(d.scenario.unique())
        specs = [
            ("regret", "Cumulative regret"),
            ("reward", "Online reward"),
            ("max_resource_utilization", "Peak resource utilization"),
            ("guard_rejections", "Feasibility-guard rejections"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (metric, ylabel) in zip(axes.flat, specs):
            for si, scenario in enumerate(scenarios):
                g = d[d.scenario == scenario]
                xs, ms, ls, hs = [], [], [], []
                for val, gv in g.groupby("risk_weight", sort=True):
                    mean, lo, hi = _mean_ci(gv[metric], seed=8100 + si + int(val * 1000))
                    xs.append(val); ms.append(mean); ls.append(lo); hs.append(hi)
                ax.plot(xs, ms, marker="o", linewidth=1.7, label=scenario.replace("_", " "))
                ax.fill_between(xs, ls, hs, alpha=0.12)
            ax.set_xlabel("Scarcity-risk weight")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.23); ax.legend(fontsize=7)
        fig.suptitle("Risk-aware dual-price extension: risk-weight sensitivity", y=1.01)
        fig.text(0.01, 0.005, "The hard feasibility guard remains active; zero violations are therefore not evidence that risk pricing is unnecessary. Shading is pointwise 95% bootstrap CI.", fontsize=8)
        fig.tight_layout(rect=(0, 0.025, 1, 0.98))
        p = out / "20_risk_weight_sensitivity.png"
        fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    # 21) Regret as a function of shift magnitude; methods share random seeds/instances.
    if "21" in selected and not shifts.empty:
        d = shifts[shifts.scenario != "stationary"].copy()
        scenarios = _scenario_order(d.scenario.unique())
        algs = [a for a in ["first_order", "algorithm2", "fixed_frequency", "adaptive"] if a in set(d.algorithm)]
        ncols = 2; nrows = int(np.ceil(len(scenarios) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.4 * nrows), squeeze=False)
        for ax, scenario in zip(axes.flat, scenarios):
            ds = d[d.scenario == scenario]
            for ai, alg in enumerate(algs):
                g = ds[ds.algorithm == alg]
                xs, ms, ls, hs = [], [], [], []
                for strength, gs in g.groupby("shift_strength", sort=True):
                    mean, lo, hi = _mean_ci(gs.regret, seed=9100 + ai + int(strength * 1000))
                    xs.append(strength); ms.append(mean); ls.append(lo); hs.append(hi)
                if xs:
                    ax.plot(xs, ms, marker="o", linewidth=1.7, label=LABELS.get(alg, alg))
                    ax.fill_between(xs, ls, hs, alpha=0.12)
            ax.set_title(scenario.replace("_", " ").title())
            ax.set_xlabel("Shift strength"); ax.set_ylabel("Final regret")
            ax.grid(True, alpha=0.22); ax.legend(fontsize=7)
        for ax in axes.flat[len(scenarios):]: ax.set_visible(False)
        fig.suptitle("Algorithm robustness as distribution-shift strength increases", y=1.01)
        fig.text(0.01, 0.005, f"Methods share identical generated instances per seed. Shading is pointwise 95% bootstrap CI; n={int(d.seed.nunique())} seeds per strength.", fontsize=8)
        fig.tight_layout(rect=(0, 0.025, 1, 0.98))
        p = out / "21_regret_vs_shift_strength.png"
        fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    # 22) Robust compute-performance Pareto frontiers over several budget caps.
    if "22" in selected and not budget.empty:
        scenarios = _scenario_order(budget.scenario.unique())
        ncols = 2; nrows = int(np.ceil(len(scenarios) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.6 * nrows), squeeze=False)
        for ax, scenario in zip(axes.flat, scenarios):
            ds = budget[budget.scenario == scenario]
            for alg in ["fixed_frequency", "adaptive"]:
                g = ds[ds.algorithm == alg]
                stats = []
                for cap, gc in g.groupby("budget_cap", sort=True):
                    xm, xl, xh = _mean_ci(gc.lp_solves, seed=10100 + int(cap))
                    ym, yl, yh = _mean_ci(gc.regret, seed=10200 + int(cap))
                    stats.append((xm, xl, xh, ym, yl, yh, cap))
                if not stats: continue
                stats.sort(key=lambda z: z[0])
                xs = [z[0] for z in stats]; ys = [z[3] for z in stats]
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=LABELS.get(alg, alg))
                yerr = np.asarray([_err(z[3], z[4], z[5]) for z in stats]).T
                ax.errorbar(xs, ys, yerr=yerr, fmt="none", capsize=2, linewidth=1)
                for x, y, *_, cap in stats:
                    ax.annotate(f"cap={cap}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
            ax.set_title(scenario.replace("_", " ").title())
            ax.set_xlabel("Mean actual LP-solve count")
            ax.set_ylabel("Final regret (lower is better)")
            ax.grid(True, alpha=0.23); ax.legend(fontsize=8)
        for ax in axes.flat[len(scenarios):]: ax.set_visible(False)
        fig.suptitle("Matched compute-budget frontier: adaptive versus fixed schedule", y=1.01)
        fig.text(0.01, 0.005, f"Each point is a requested solve cap; error bars are 95% bootstrap CIs across {int(budget.seed.nunique())} paired seeds. Runtime is recorded separately in the CSV.", fontsize=8)
        fig.tight_layout(rect=(0, 0.025, 1, 0.98))
        p = out / "22_budget_sweep_pareto_frontier.png"
        fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    # 23) Requested budget utilization / budget saturation.
    if "23" in selected and not budget.empty:
        scenarios = _scenario_order(budget.scenario.unique())
        ncols = 2; nrows = int(np.ceil(len(scenarios) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.3 * nrows), squeeze=False)
        for ax, scenario in zip(axes.flat, scenarios):
            ds = budget[budget.scenario == scenario]
            for alg in ["fixed_frequency", "adaptive"]:
                g = ds[ds.algorithm == alg]
                xs, ms, ls, hs = [], [], [], []
                for cap, gc in g.groupby("budget_cap", sort=True):
                    mean, lo, hi = _mean_ci(gc.solve_budget_utilization, seed=11100 + int(cap))
                    xs.append(cap); ms.append(mean); ls.append(lo); hs.append(hi)
                if xs:
                    ax.plot(xs, ms, marker="o", linewidth=1.7, label=LABELS.get(alg, alg))
                    ax.fill_between(xs, ls, hs, alpha=0.12)
            ax.set_title(scenario.replace("_", " ").title())
            ax.set_xlabel("Configured LP-solve cap")
            ax.set_ylabel("Actual solves / cap")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.22); ax.legend(fontsize=8)
        for ax in axes.flat[len(scenarios):]: ax.set_visible(False)
        fig.suptitle("Solve-budget utilization: does the controller spend its budget?", y=1.01)
        fig.text(0.01, 0.005, "Research metrics exclude terminal schedule boundaries that return zero prices without invoking the LP solver; the raw legacy counter is retained as lp_solve_schedule_points in each CSV.", fontsize=8)
        fig.tight_layout(rect=(0, 0.025, 1, 0.98))
        p = out / "23_solve_budget_utilization.png"
        fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    # 24) Robustness of detector-only telemetry to increasing measurement noise.
    if "24" in selected and not noise.empty:
        scenarios = _scenario_order(noise.scenario.unique())
        shifted_scenarios = [s for s in scenarios if s != "stationary"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
        for si, scenario in enumerate(shifted_scenarios):
            g = noise[noise.scenario == scenario]
            xs, means, lows, highs = [], [], [], []
            for level, gl in g.groupby("noise_level", sort=True):
                mean, lo, hi = _mean_ci(gl.detection_success, seed=12100 + si + int(level * 1000))
                xs.append(level); means.append(mean); lows.append(lo); highs.append(hi)
            if xs:
                axes[0].errorbar(xs, means, yerr=np.asarray([
                    [max(0.0, m - lo) for m, lo in zip(means, lows)],
                    [max(0.0, hi - m) for m, hi in zip(means, highs)],
                ]), marker="o", capsize=3, linewidth=1.7, label=scenario.replace("_", " "))
        axes[0].set_xlabel("Detector measurement-noise level (fraction of pre-shift feature SD)")
        axes[0].set_ylabel("Detection probability")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_title("Detection probability under noisy telemetry")

        for si, scenario in enumerate(scenarios):
            g = noise[noise.scenario == scenario]
            xs, means, lows, highs = [], [], [], []
            for level, gl in g.groupby("noise_level", sort=True):
                mean, lo, hi = _mean_ci(gl.false_alarm_per_100_prechange, seed=12200 + si + int(level * 1000))
                xs.append(level); means.append(mean); lows.append(lo); highs.append(hi)
            if xs:
                axes[1].errorbar(xs, means, yerr=np.asarray([
                    [max(0.0, m - lo) for m, lo in zip(means, lows)],
                    [max(0.0, hi - m) for m, hi in zip(means, highs)],
                ]), marker="o", capsize=3, linewidth=1.6, label=scenario.replace("_", " "))
        axes[1].set_xlabel("Detector measurement-noise level (fraction of pre-shift feature SD)")
        axes[1].set_ylabel("False alarms per 100 pre-change observations")
        axes[1].set_title("False-alarm sensitivity (includes stationary control)")
        for ax in axes:
            ax.grid(True, alpha=0.23); ax.legend(fontsize=7)
        fig.suptitle("Change-detector robustness to noisy observed telemetry")
        fig.text(0.01, 0.01, "This is a detector-only stress test: noise is injected into the detector's observed reward/resource stream; policy decisions still use the original request sequence. Bands are pointwise bootstrap 95% CIs.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        p = out / "24_detector_noise_robustness.png"
        fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig); written.append(str(p))

    return written
