# Implementation notes

## Fidelity choices

This repository follows the supplied paper's equations and principal experiment settings, but it is not the authors' original codebase.

### Hindsight benchmark

The paper defines the offline optimal solution using the relaxed LP

`max r^T x` subject to `A^T x <= b`, `0 <= x <= 1`.

We solve this LP with SciPy HiGHS.

### Sample dual

At a re-solving time t, we solve the finite-sample dual in LP form using auxiliary variables y:

`min d_t^T p + (1/t) sum_j y_j`

subject to

`a_j^T p + y_j >= r_j`, `p >= 0`, `y >= 0`.

### Feasibility

The theoretical metric permits constraint violation and explicitly measures it. For an operational simulator, we additionally prevent accepting an order that would exceed remaining inventory. This is a simulation guard, not a change to the paper's theorem.

### Runtime

Runtime is measured around the algorithm only, excluding hindsight-benchmark solving and data generation, so comparisons focus on decision/learning computation. Exact wall-clock numbers will differ from the paper because solver versions, CPUs, Python overhead, and implementation details differ.

### Normal distribution

The paper writes `N(0.5,1)` and `N(0.5m,m)`. This implementation interprets the second parameter as variance, so NumPy receives the corresponding standard deviation.
