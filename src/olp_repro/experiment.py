from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .algorithms import run_algorithm1, run_algorithm2, run_first_order, run_lp_based
from .distributions import generate_input
from .dual import solve_hindsight_primal
from .data_io import dataset_path, load_instance, save_instance
from .metrics import evaluate


def run_single(
    T: int,
    m: int,
    distribution: str,
    f: int,
    seed: int,
    algorithms: list[str],
    dataset_dir: str | Path = "datasets",
) -> pd.DataFrame:
    path = dataset_path(dataset_dir, distribution, T, m, seed)
    if path.exists():
        # Reuse the exact saved instance across every f and algorithm.
        inst = load_instance(path)
    else:
        inst = generate_input(T, m, distribution, seed=seed)
        save_instance(inst, path, distribution=distribution, seed=seed)
    x_star = solve_hindsight_primal(inst.r, inst.A, inst.b)
    rows = []
    for name in algorithms:
        start = time.perf_counter()
        if name == "algorithm1":
            result = run_algorithm1(inst, f)
        elif name == "algorithm2":
            result = run_algorithm2(inst, f)
        elif name == "first_order":
            result = run_first_order(inst, "sqrtT")
        elif name == "first_order_t13":
            result = run_first_order(inst, "t")
        elif name == "lp_based":
            result = run_lp_based(inst)
        else:
            raise ValueError(f"unknown algorithm: {name}")
        runtime = time.perf_counter() - start
        ev = evaluate(inst.r, inst.A, inst.b, result.x, x_star)
        rows.append({
            "seed": seed,
            "T": T,
            "m": m,
            "distribution": distribution,
            "f": f,
            "algorithm": name,
            "reward": ev.reward,
            "offline_reward": ev.offline_reward,
            "regret": ev.regret,
            "violation_l2": ev.violation_l2,
            "lp_solves": result.lp_solves,
            "first_order_updates": result.first_order_updates,
            "runtime_seconds": runtime,
        })
    return pd.DataFrame(rows)


def run_frequency_sweep(config: dict) -> pd.DataFrame:
    rows = []

    trials = int(config.get("trials", 10))
    Ts = config["horizons"]
    f_exponents = config.get(
        "f_exponents",
        [1 / 3, 1 / 2, 2 / 3]
    )
    m = int(config.get("m", 1))
    algorithms = config.get(
        "algorithms",
        ["algorithm1", "algorithm2", "first_order"]
    )
    distributions = config.get(
        "distributions",
        ["uniform", "normal"]
    )
    dataset_dir = config.get("dataset_dir", "datasets")

    total = (
        len(distributions)
        * len(Ts)
        * len(f_exponents)
        * trials
    )

    completed = 0
    start_total = time.perf_counter()

    print(f"\nTotal frequency configurations: {total}", flush=True)

    for distribution in distributions:
        for T in Ts:
            for exponent in f_exponents:

                f = max(
                    1,
                    min(
                        T,
                        int(round(T ** float(exponent)))
                    )
                )

                for seed in range(trials):

                    completed += 1

                    elapsed = time.perf_counter() - start_total

                    if completed > 1:
                        eta = (
                            elapsed / completed
                            * (total - completed)
                        )
                    else:
                        eta = 0

                    print(
                        f"[{completed}/{total}] "
                        f"{completed / total * 100:6.2f}% | "
                        f"dist={distribution:<7} "
                        f"T={T:<8} "
                        f"f={f:<6} "
                        f"seed={seed:<4} | "
                        f"elapsed={elapsed / 60:.2f}m "
                        f"ETA={eta / 60:.2f}m",
                        flush=True,
                    )

                    rows.append(
                        run_single(
                            T,
                            m,
                            distribution,
                            f,
                            seed,
                            algorithms,
                            dataset_dir,
                        )
                    )

    return pd.concat(rows, ignore_index=True)

def run_baseline_comparison(config: dict) -> pd.DataFrame:
    rows = []

    trials = int(config.get("trials", 10))
    Ts = config["horizons"]
    m = int(config.get("m", 5))
    exponent = float(
        config.get("f_exponent", 1 / 3)
    )

    algorithms = config.get(
        "algorithms",
        [
            "first_order",
            "first_order_t13",
            "lp_based",
            "algorithm1",
            "algorithm2",
        ],
    )

    distribution = config.get(
        "distribution",
        "uniform"
    )

    dataset_dir = config.get(
        "dataset_dir",
        "datasets"
    )

    total = len(Ts) * trials
    completed = 0
    start_total = time.perf_counter()

    print(
        f"\nTotal comparison configurations: {total}",
        flush=True
    )

    for T in Ts:

        f = max(
            1,
            min(
                T,
                int(round(T ** exponent))
            )
        )

        for seed in range(trials):

            completed += 1

            elapsed = time.perf_counter() - start_total

            if completed > 1:
                eta = (
                    elapsed / completed
                    * (total - completed)
                )
            else:
                eta = 0

            print(
                f"[{completed}/{total}] "
                f"{completed / total * 100:6.2f}% | "
                f"T={T:<8} "
                f"f={f:<6} "
                f"seed={seed:<4} | "
                f"elapsed={elapsed / 60:.2f}m "
                f"ETA={eta / 60:.2f}m",
                flush=True,
            )

            rows.append(
                run_single(
                    T,
                    m,
                    distribution,
                    f,
                    seed,
                    algorithms,
                    dataset_dir,
                )
            )

    return pd.concat(rows, ignore_index=True)

def save_results(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
