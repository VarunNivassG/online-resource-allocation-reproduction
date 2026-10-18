# Supplementary Research Results

This folder contains the supplementary experiment tables and Figures 13–26. CSV files preserve per-seed observations rather than only aggregated means.

- `trajectories.csv`: per-step prefix regret, cumulative reward, resource utilization, and LP-solve counts.
- `shift_sensitivity.csv`: algorithm outcomes, detection probability, delay, and pre-shift false-alarm counts across shift strengths.
- `ablation.csv`: paired signal ablation results.
- `threshold_sensitivity.csv`: controller trigger-threshold sweep.
- `risk_sensitivity.csv`: risk-weight sweep.
- `budget_sweep.csv`: matched requested LP-solve cap sweep.
- `noise_sensitivity.csv`: detector-only noise robustness stress test.
- `horizon_sweep.csv`: 10-seed stationary scaling study at T=50, 100, 200, and 500.
- `risk_tail_sweep.csv`: 24-seed risk-weight study under stationary and shifted demand.

All plots are in `plots/`. For the figure-by-figure interpretation and limitations, see `docs/research_plot_guide.md`.

## Important interpretation note

The supplementary sweep uses eight seeds per scenario for Figures 13–24; Figure 25 uses ten seeds per horizon and Figure 26 uses 24 seeds per scenario/risk weight. These are stronger than a two-seed smoke run but remain pilot-scale. Confidence intervals are descriptive. The implementation's hard feasibility guard prevents resource-budget overrun, so zero violation is not by itself evidence of a risk-aware algorithmic improvement.
