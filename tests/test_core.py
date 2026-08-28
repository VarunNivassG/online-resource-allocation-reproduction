import numpy as np

from olp_repro.distributions import generate_input
from olp_repro.dual import acceptance_rule, projected_subgradient_update, solve_hindsight_primal
from olp_repro.algorithms import run_algorithm1, run_algorithm2
from olp_repro.metrics import evaluate


def test_acceptance_rule():
    assert acceptance_rule(10.0, np.array([2.0]), np.array([4.0])) == 1
    assert acceptance_rule(7.0, np.array([2.0]), np.array([4.0])) == 0


def test_update_sign():
    p = np.array([5.0])
    d = np.array([10.0])
    a = np.array([13.0])
    p2 = projected_subgradient_update(p, d, a, 1, 0.1)
    # d-a*x=-3, hence p'=5-0.1*(-3)=5.3
    assert np.allclose(p2, [5.3])


def test_update_underuse_decreases_price():
    p = np.array([5.0])
    d = np.array([10.0])
    a = np.array([7.0])
    p2 = projected_subgradient_update(p, d, a, 1, 0.1)
    # d-a*x=+3, hence p'=5-0.1*3=4.7
    assert np.allclose(p2, [4.7])


def test_hindsight_and_algorithms():
    inst = generate_input(30, 2, "uniform", seed=3)
    x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
    r1 = run_algorithm1(inst, 5)
    r2 = run_algorithm2(inst, 5)
    e1 = evaluate(inst.r, inst.A, inst.b, r1.x, x_star)
    e2 = evaluate(inst.r, inst.A, inst.b, r2.x, x_star)
    assert e1.reward >= 0
    assert e2.reward >= 0
    assert r1.lp_solves == 6
    assert r2.lp_solves == 6
