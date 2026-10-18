# Research Extension: Adaptive, Non-Stationary, and Risk-Aware OLP

## Scope and research questions

This extension adds three experimentally testable questions to the baseline reproduction:

1. **Adaptive computation:** Can a causal trigger schedule sample-dual LP solves more effectively than a fixed interval under a hard maximum solve budget?
2. **Non-stationarity:** Can reward/resource-distribution shifts be detected from completed requests and use that signal to refresh dual prices?
3. **Resource risk:** Does a scarcity-aware price premium change reward and potential feasibility-guard rejection behavior compared with the same adaptive controller without the premium?

These are research hypotheses, not guaranteed positive results or claims of established novelty. The included risk and change-detection mechanisms are heuristic baselines for evaluation, not formal guarantees.

## Architecture

- `adaptive_signals.py`: causal normalized dual-price drift, ahead-of-pace resource use, and recent decision instability.
- `adaptive_controller.py`: hard maximum number of LP solves, warm-up, minimum re-solve gap, score threshold, and a maximum-gap fallback.
- `change_detector.py`: two-window/expanding-history standardized mean-shift detector on observed reward and resource vectors. It uses only completed requests. The detector is empirical; its threshold should be calibrated on validation seeds and false-alarm rates reported.
- `risk.py`: estimates resource scarcity relative to a linear consumption trajectory and adds a per-resource price premium scaled by observed reward/resource magnitudes.
- `algorithms.py`: adds `run_adaptive_resolving` and `run_risk_aware_adaptive`. The original first-order, LP-based, Algorithm 1, and Algorithm 2 routines remain intact.
- `nonstationary.py`: controlled shift generators with a fixed pre-generated resource budget, isolating request changes from budget changes.
- `research.py`: paired experiments, metrics, representative per-step traces, and bootstrap summary statistics.
- `research_plots.py`: computation/regret frontier, horizon scaling, non-stationary stress comparison, reward/guard-rejection view, trigger and resource traces.
- `comparison_plots.py`: direct cross-algorithm regret/runtime/LP-call comparison, paired adaptive-vs-fixed shift comparison, paired risk-aware-vs-adaptive effects, and resource-use comparisons.

## Adaptive signal and controller

At the end of request `t`, the monitor computes

`score_t = w_D * drift_t + w_P * pressure_t + w_I * instability_t`,

where the three signals are normalized to `[0, 1]`. If the score crosses the configured threshold, a change alarm is pending, or the maximum interval has elapsed, the controller re-solves the sample dual LP using `r[:t]` and `A[:t]` and the remaining budget scaled to the unobserved horizon. The updated prices are used for future requests only. The maximum number of actual LP calls is enforced independently of the trigger.

The computation cap in experiment configs is an upper bound, not a target number of solves. The adaptive method can use fewer calls; the matched fixed-frequency comparator uses a batch interval selected from the same cap. Both actual calls and their measured wall-clock runtime are reported.

## Non-stationary scenario definitions

`generate_nonstationary_input` creates a base instance from a fixed seed. It keeps `b` equal to the base instance budget, then changes request data:

- `abrupt_resource`: resource requirements increase after `change_fraction` of the horizon.
- `abrupt_reward`: late-horizon rewards are scaled upward.
- `abrupt_combined`: both reward and resource sequences change.
- `gradual`: resource and reward multipliers ramp after the designated point.

This is a controlled stress-test suite. It is not a real-world dataset and should not be described as such. For an actual paper, report the shift strength, change point, detector window/threshold, and calibration procedure.

## Risk-aware price heuristic

The optional variant computes an ahead-of-trajectory scarcity score for each resource and adds a positive premium to the baseline dual price when consumption is running ahead of a linear pacing target. The premium is scaled by observed reward/resource magnitude. It only affects the acceptance threshold; the normal projected update and adaptive LP controller remain in use.

The simulation already has a hard feasibility guard. Thus `violation_l2` may remain zero for all methods and cannot alone show risk improvement. The suite additionally reports potential requests blocked by the guard, reward/regret, remaining budget fractions, and resource utilization. The `risk_weight` parameter must be selected on validation seeds; do not tune it on held-out final-test seeds.

## Outputs and how to interpret them

`results.csv` has one row per algorithm/instance/budget; `summary.csv` contains means, standard deviations, and percentile bootstrap 95% confidence intervals by experimental group; `traces.csv` contains selected per-step trajectories.

Figures generated under `plots/`:

1. `01_computation_regret_frontier.png`: mean regret against actual LP solves with bootstrap CIs.
2. `02_stationary_regret_horizon.png`: stationary regret scaling against horizon.
3. `03_nonstationary_regret.png`: regret comparison across shift scenarios.
4. `04_risk_reward_guard_tradeoff.png`: reward against potential hard-guard blocks.
5. `05_trace_diagnostics_*.png`: trigger signals, cumulative reward, and resource utilization over time.
6. `06_trigger_timeline.png`: a representative change/trigger timeline, including the known synthetic shift point.
7. `07_algorithm_regret_comparison.png`: stationary regret across all core baselines and the adaptive method.
8. `08_algorithm_runtime_comparison.png`: measured runtime comparison on a log x-axis.
9. `09_algorithm_lp_solve_comparison.png`: actual solver-call comparison.
10. `10_adaptive_vs_fixed_shift_comparison.png`: paired seed-level percentage regret change for adaptive against fixed-frequency under non-stationarity.
11. `11_risk_aware_paired_comparison.png`: paired effects of risk-aware pricing on regret, reward, peak utilization, and guard rejections.
12. `12_resource_utilization_comparison.png`: peak utilization and minimum leftover resource fraction across baselines.

The paired comparison plots show per-seed paired points instead of implying that the smoke sample supports reliable interval estimates. Aggregate bars use the summary CSV's bootstrap intervals, but those intervals are not dependable with only two seeds.

The smoke config uses only two seeds and is intended to verify pipeline integrity, not support strong statistical claims. Use `configs/research_scale.json` for a larger evaluation; adjust the trial count for available compute. Report mean/CI alongside raw seed-level results, and avoid declaring statistical significance solely because bootstrap intervals look separated.

## Reproducibility and caveats

- Same generator seed, scenario, and budget are paired across algorithms.
- The hindsight LP is recomputed for each changed request sequence with its fixed budget.
- Existing baseline source logic is retained; baseline tests include expected LP solve counts.
- `runtime_seconds` is measured in-process and depends on hardware, SciPy/HiGHS version, and system load. It is supplementary to actual LP-call counts.
- The change detector is not a calibrated statistical test with a formal false-alarm guarantee. Tune/calibrate the threshold on separate seeds and report detection delay and false alarms.
- Risk-aware prices have no theoretical guarantee in this implementation. Any theoretical claim needs a proof under clearly stated assumptions.


## Supplementary multi-seed research plots (Figures 13–24)

Run `python scripts/run_research_supplementary.py` to generate paired sensitivity experiments and Figures 13–24 under `results_research_supplementary/`. The run saves the exact config and raw per-seed tables, so figures can be rendered again with `python scripts/render_missing_research_plots.py` without rerunning optimization.

- `13_prefix_regret_trajectories.png`: per-time prefix regret against the hindsight LP optimum for the same observed prefix and fixed resource budget, with pointwise seed-bootstrap intervals and the known shift point.
- `14_resource_utilization_trajectories.png`: cumulative utilization across resources and time, comparing the same policies on matched instances.
- `15_detection_probability_vs_shift_strength.png`: empirical change detection rate as synthetic shift magnitude changes.
- `16_detection_delay_false_alarms.png`: delay among detected shifts and pre-shift false alarms; interpret delay jointly with Figure 15 because missed shifts have no delay value.
- `17_signal_ablation_paired_regret.png`: paired change in regret after removing or isolating controller signals. Positive regret difference means the ablated policy performed worse than the full controller on that seed.
- `18_signal_ablation_lp_solves.png`: solver-call cost of each signal set.
- `19_trigger_threshold_sensitivity.png`: regret and LP-solve count as the composite threshold changes, with change detection disabled to isolate the trigger threshold.
- `20_risk_weight_sensitivity.png`: regret, reward, resource utilization and guard rejections against the scarcity-price premium weight.
- `21_regret_vs_shift_strength.png`: final regret by shift type and magnitude for first-order, Algorithm 2, fixed-frequency and adaptive methods on paired sequences.
- `22_budget_sweep_pareto_frontier.png`: regret against mean LP-solve count under multiple requested solver budgets.
- `23_solve_budget_utilization.png`: recorded solve-budget usage across caps; solver-call fields should be interpreted alongside the baseline implementation's legacy terminal-boundary counter.
- `24_detector_noise_robustness.png`: a detector-only stress test in which measurement noise is injected into observed reward/resource telemetry. Policy allocation decisions use the clean generated instance in this specific plot; it isolates detector robustness and does not claim full policy robustness under noisy inputs.

The supplementary config currently uses 8 paired seeds, T=100, and m=3. Bootstrap intervals are pointwise in trajectory plots and descriptive with this sample size. Repeat with more independent seeds and held-out validation/test data before drawing inferential conclusions. The change detector and scarcity premium remain empirical heuristics, not formal statistical tests or theoretical safety guarantees.


## Additional horizon and risk-tail studies (Figures 25–26)

`python scripts/run_remaining_research_plots.py` produces two additional figures from paired runs:

- `25_multi_seed_regret_vs_horizon.png`: compares regret and actual solver calls across T=50, 100, 200, 500 with 10 independent seeds per horizon.
- `26_risk_tail_outcomes.png`: compares mean regret, upper-quartile mean regret (`CVaR75`), guard rejections, and minimum residual-capacity fraction for 24 seeds per scenario and risk weight.

`horizon_sweep.csv` and `risk_tail_sweep.csv` retain per-seed records. `CVaR75` is descriptive; it is not a formal safety guarantee. The hard feasibility guard remains active, and any publication claim requires held-out validation and sufficient statistical power.
