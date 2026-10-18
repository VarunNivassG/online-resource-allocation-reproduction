from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_baseline_comparison, run_frequency_sweep, save_results
from .plotting import plot_frequency, plot_regret
from .research import run_research_suite, save_research_outputs
from .research_plots import plot_research_bundle
from .supplementary import run_supplementary_suite, save_supplementary_outputs
from .supplementary_plots import plot_supplementary_bundle


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

    research = sub.add_parser("research", help="run adaptive, non-stationary, and risk-aware research stages")
    research.add_argument("--config", default="configs/research_smoke.json")
    research.add_argument("--outdir", default="results/research")

    supplementary = sub.add_parser("supplementary", help="run multi-seed trajectory, ablation, sensitivity, shift-strength, and compute-budget experiments")
    supplementary.add_argument("--config", default="configs/research_supplementary.json")
    supplementary.add_argument("--outdir", default="results/research_supplementary")

    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())

    if args.cmd == "frequency":
        df = run_frequency_sweep(config["frequency"])
        save_results(df, args.out)
        plot_frequency(df, args.plot)
        print(f"saved results to {args.out}")
        print(f"saved plot to {args.plot}")
    elif args.cmd == "comparison":
        df = run_baseline_comparison(config["comparison"])
        save_results(df, args.out)
        plot_regret(df, args.plot)
        print(f"saved results to {args.out}")
        print(f"saved plot to {args.plot}")
    elif args.cmd == "research":
        results, traces = run_research_suite(config)
        outdir = Path(args.outdir)
        summary = save_research_outputs(results, traces, outdir)
        plots = plot_research_bundle(results, summary, traces, outdir / "plots")
        print(f"saved {len(results)} experiment rows to {outdir / 'results.csv'}")
        print(f"saved {len(summary)} summary groups to {outdir / 'summary.csv'}")
        print(f"saved {len(traces)} trace rows to {outdir / 'traces.csv'}")
        print(f"saved {len(plots)} research figures to {outdir / 'plots'}")
    else:
        tables = run_supplementary_suite(config)
        outdir = Path(args.outdir)
        save_supplementary_outputs(tables, outdir)
        (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        plots = plot_supplementary_bundle(tables, outdir / "plots")
        counts = {name: len(frame) for name, frame in tables.items()}
        print(f"saved supplementary rows: {counts}")
        print(f"saved {len(plots)} supplementary research figures to {outdir / 'plots'}")


if __name__ == "__main__":
    main()
