# Mathematical map of the reproduction

## 1. Primal OLP

\[
\max_{0\le x\le1}\; r^T x
\quad\text{s.t.}\quad A^T x\le b.
\]

At time t, request t has reward r_t and resource vector a_t.

## 2. Sample dual

With d_t=b_t/(T-t), the sample dual used for re-solving is

\[
\min_{p\ge0}\; d_t^Tp + \frac1t\sum_{j=1}^t(r_j-a_j^Tp)_+.
\]

Equivalent LP form:

\[
\min_{p\ge0,y\ge0}\; d_t^Tp+\frac1t\mathbf1^Ty
\]

subject to

\[
a_j^Tp+y_j\ge r_j.
\]

## 3. Decision rule

\[
x_t=\mathbf 1(r_t\ge a_t^Tp_t).
\]

The simulator additionally applies a hard feasibility guard so an accepted request cannot exceed remaining inventory.

## 4. First-order update

\[
p_{t+1}=[p_t-\alpha_t(d-a_tx_t)]_+.
\]

Therefore:

- if \(a_tx_t<d\), then \(p\) tends to decrease;
- if \(a_tx_t>d\), then \(p\) tends to increase.

## 5. Algorithm 1

- first batch \([1,f]\): first-order, \(\alpha_t=O(f^{-1/2})\);
- intermediate batches: latest LP dual is held between re-solves;
- every \(f\) steps: solve the sample dual and refresh the price;
- final batch: first-order, \(\alpha_t=O(f^{-2/3})\).

The paper proves

\[
\Delta_T=O\left(\log(T/f)+\sqrt f\right).
\]

## 6. Algorithm 2

Between LP re-solves, fine-tune from the latest LP solution using

\[
\alpha_t=O(1/t).
\]

## 7. Evaluation

Offline benchmark:

\[
x^*=\arg\max_{0\le x\le1}\{r^Tx:A^Tx\le b\}.
\]

Regret:

\[
r(x)=r^Tx^*-r^Tx.
\]

Constraint violation:

\[
v(x)=\|(Ax-b)_+\|.
\]

Combined performance:

\[
\Delta_T=\sup_{P\in\Xi}\mathbb E[r(x)+v(x)].
\]
