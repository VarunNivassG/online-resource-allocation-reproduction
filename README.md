# Wait-Less OLP Reproduction — A Clean Research Implementation

This project is our own, from-scratch implementation of the mathematical and experimental ideas in:

**Jingruo Sun, Wenzhi Gao, Ellen Vitercik, Yinyu Ye — “Wait-Less Offline Tuning and Re-solving for Online Decision Making,” ICML 2025 / arXiv:2412.09594v3.**

The implementation is deliberately organized so that the paper can be reproduced, inspected, modified, and extended rather than treated as a black-box script.

## What is implemented

### Core mathematics

1. Online Linear Programming primal problem.
2. Sample dual formulation.
3. Dual-price acceptance rule.
4. Projected first-order/subgradient update.
5. Hindsight offline LP benchmark.
6. Regret and resource-violation metrics.

### Algorithms

- `first_order`: classic first-order baseline.
- `lp_based`: LP re-solving every step baseline.
- `algorithm1`: paper's parallel multi-phase hybrid.
- `algorithm2`: enhanced multi-start hybrid.

### Paper-style experiments

- Input I: \(a_{it}\sim U[0,2]\), \(r_t\sim U[0,10]\).
- Input II: \(a_{it}\sim N(0.5,1)\), \(r_t\sim N(0.5m,m)\), interpreting the second parameter as variance.
- Frequency settings \(f\in\{T^{1/3},T^{1/2},T^{2/3}\}\).
- Baseline comparison with \(m=5\).
- CSV outputs and log-log plots.

## Project structure

```text
olp_repro/
├── configs/
│   ├── smoke.json
│   └── paper_scale.json
├── datasets/                 # generated/reproducible synthetic instances
│   ├── uniform/
│   └── normal/
├── docs/
│   └── mathematical_map.md
├── results/
├── scripts/
│   ├── run_smoke.py
│   └── run_paper_scale.py
├── src/olp_repro/
│   ├── __init__.py
│   ├── algorithms.py
│   ├── cli.py
│   ├── distributions.py
│   ├── dual.py
│   ├── experiment.py
│   ├── metrics.py
│   ├── plotting.py
│   └── problem.py
├── tests/
│   ├── test_core.py
│   └── test_distributions.py
├── pyproject.toml
└── README.md
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -e .[dev]
```

## Run tests

```bash
pytest -q
```

## Run a fast smoke reproduction

```bash
python scripts/run_smoke.py
```

This produces:

```text
results/frequency.csv
results/frequency.png
results/comparison.csv
results/comparison.png
```

and saves every generated synthetic OLP instance under `datasets/` as a compressed
`.npz` file. The same `(T, m, distribution, seed)` instance is reused across
all algorithms and all re-solving frequencies, so comparisons are made on the
exact same request sequence.

Each saved instance contains:

- `r`: rewards
- `A`: resource-consumption matrix
- `b`: total resource budget
- `d`: average resource budget
- JSON metadata containing distribution, seed, `T`, and `m`

## Run the paper-scale configuration

```bash
python scripts/run_paper_scale.py
```

**Important:** the paper-scale configuration intentionally contains horizons up to \(10^3\) and 10 trials. The LP-based reference requires repeated large LP solves and can be extremely expensive in memory and wall-clock time. Start with the smoke configuration, then scale one dimension at a time.

## Important reproduction detail: feasibility guard

The paper's decision rule is

\[
x_t=I(r_t\ge a_t^Tp_t).
\]

For a practical simulator we also prevent an accepted order from exceeding remaining inventory. This makes the simulation operationally feasible and avoids artificial negative inventory. The theoretical analysis in the paper separately handles resource violation through its performance metric.

## Main theorem implemented as the experiment target

For re-solving interval \(f\), the paper proves the spectrum bound

\[
\Delta_T=O\left(\log\frac{T}{f}+\sqrt f\right).
\]

Extreme cases:

- \(f=1\): LP-like \(O(\log T)\) behavior.
- \(f=T\): first-order-like \(O(\sqrt T)\) behavior.

## Research extensions we can build on top

Saved datasets can be loaded with `olp_repro.data_io.load_instance()` and are
portable between runs. Delete `datasets/` if you intentionally want to regenerate
all stochastic instances.

This repository is intentionally modular. Natural next experiments include:

1. adaptive \(f_t\) instead of fixed \(f\);
2. non-stationary request distributions;
3. unknown horizon;
4. noisy or delayed LP solutions;
5. larger numbers of resource types;
6. alternative first-order methods such as mirror descent;
7. empirical computation-budget optimization;
8. confidence intervals and statistical significance tests;
9. real-world resource-allocation datasets.

## Scope note

This is a research reproduction, not a claim of bit-for-bit reproduction of the authors' original implementation. Random seeds, solver versions, hardware, and implementation details can change numerical results. The mathematical definitions and principal experimental settings are based on the supplied paper.
