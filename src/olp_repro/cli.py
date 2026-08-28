from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_baseline_comparison, run_frequency_sweep, save_results
from .plotting import plot_frequency, plot_regret


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the wait-less OLP hybrid experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    freq = sub.add_parser("frequency", help="run a re-solving-frequency sweep")
    freq.add_argument("--config", default="configs/smoke.json")
    freq.add_argument("--out", default="results/frequency.csv")
    freq.add_argument("--plot", default="results/frequency.png")

    comp = sub.add_parser("comparison", help="compare hybrid and baseline algorithms")
    comp.add_argument("--config", default="configs/smoke.json")
    comp.add_argument("--out", default="results/comparison.csv")
    comp.add_argument("--plot", default="results/comparison.png")

    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())

    if args.cmd == "frequency":
        df = run_frequency_sweep(config["frequency"])
        save_results(df, args.out)
        plot_frequency(df, args.plot)
    else:
        df = run_baseline_comparison(config["comparison"])
        save_results(df, args.out)
        plot_regret(df, args.plot)

    print(f"saved results to {args.out}")
    print(f"saved plot to {args.plot}")


if __name__ == "__main__":
    main()
