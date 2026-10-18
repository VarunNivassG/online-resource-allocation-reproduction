# Research Plot Guide

This project now has a consolidated research gallery with **29 image files** (numbered Figures 01–26 plus four separate trace-diagnostic views grouped as Figure 05). The index/contact sheet is at `results_research_complete/`.

## Figure map

| Figure(s) | Question answered | Source data / run |
|---|---|---|
| 01 | Regret–computation trade-off | `results_research_smoke/summary.csv` |
| 02 | Initial stationary regret vs horizon | `results_research_smoke/results.csv` |
| 03–04 | Initial non-stationary and risk/reward comparisons | `results_research_smoke/results.csv` |
| 05 (four scenarios) | Trigger, price, utilization, and solve timeline diagnostics | `results_research_smoke/traces.csv` |
| 06 | Adaptive trigger signals on an example instance | `results_research_smoke/traces.csv` |
| 07–12 | Algorithm, runtime, LP calls, paired shift effects, risk extension, resource use | `results_research_smoke/results.csv` and related trace/result tables |
| 13 | Prefix-regret trajectories around controlled shifts | `results_research_supplementary/trajectories.csv` |
| 14 | Resource-utilization trajectories | `results_research_supplementary/trajectories.csv` |
| 15–16 | Detection probability, detection delay, and false alarms | `results_research_supplementary/shift_sensitivity.csv` |
| 17–18 | Signal ablations: paired regret effect and LP-solve cost | `results_research_supplementary/ablation.csv` |
| 19 | Trigger-threshold sensitivity | `results_research_supplementary/threshold_sensitivity.csv` |
| 20 | Risk-weight sensitivity | `results_research_supplementary/risk_sensitivity.csv` |
| 21 | Regret vs distribution-shift magnitude | `results_research_supplementary/shift_sensitivity.csv` |
| 22–23 | Matched compute-budget frontier and budget utilization | `results_research_supplementary/budget_sweep.csv` |
| 24 | Detector-only robustness to noisy telemetry | `results_research_supplementary/noise_sensitivity.csv` |
| 25 | Multi-seed regret scaling vs horizon and LP calls | `results_research_supplementary/horizon_sweep.csv` |
| 26 | Mean and upper-tail regret under risk-aware price weights | `results_research_supplementary/risk_tail_sweep.csv` |

## Regenerating the additional plots

The supplementary tables for Figures 13–24 are produced by:

```bash
python scripts/run_research_supplementary.py
```

That sweep can be computationally intensive because it constructs prefix hindsight oracles and evaluates several sensitivity grids. If the CSV tables already exist and only some figures were not rendered, render Figures 17–24 without rerunning the algorithms:

```bash
python scripts/render_missing_research_plots.py
```

The additional multi-seed horizon and risk-tail studies (Figures 25–26) are run with:

```bash
python scripts/run_remaining_research_plots.py
```

Use `--force` to rerun both additional experiments instead of using config-matched cached CSVs.

The script caches `horizon_sweep.csv` and `risk_tail_sweep.csv`, so it can be rerun to regenerate the figures without repeating a completed sweep.

## Statistical interpretation and limits

- Figures 01–12 are from the **two-seed smoke configuration**. They are useful for pipeline checks and qualitative debugging, not strong statistical claims.
- Figures 13–24 are from the supplementary configuration with **eight seeds per scenario/condition**. Confidence bands are descriptive and the sample remains small for subtle effects.
- Figure 25 uses **10 independent seeds per horizon**. Figure 26 uses **24 independent seeds per scenario and risk weight**. These provide better empirical evidence but are still pilot-scale, not final publication-grade statistical power for every metric.
- Bootstrap bands are percentile intervals over seeds. For highly paired comparisons, use paired seed-level differences in the final statistical analysis rather than inferring significance from overlapping marginal confidence bands.
- The operational feasibility guard rejects requests that would exceed remaining resources. Hence zero observed violations is a property of the guard, not evidence that risk-aware pricing has a formal safety guarantee. Figure 26 reports resource pressure and the upper-quartile mean regret (`CVaR75`) descriptively.
- Thresholds, signal weights, change-detection settings, and risk weights should be calibrated on validation instances and evaluated on held-out seeds before results are presented as confirmatory.
