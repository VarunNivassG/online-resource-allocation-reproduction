from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
config = root / "configs" / "paper_scale.json"

frequency_out = root / "results_full" / "paper_frequency.csv"
frequency_plot = root / "results_full" / "paper_frequency"

comparison_out = root / "results_full" / "paper_comparison.csv"
comparison_plot = root / "results_full" / "paper_comparison.png"


def run_stage(name, command):
    print("\n" + "=" * 70, flush=True)
    print(f"STARTING: {name}", flush=True)
    print("=" * 70, flush=True)

    start = time.perf_counter()

    result = subprocess.run(command)

    elapsed = time.perf_counter() - start

    print("\n" + "-" * 70, flush=True)
    print(f"FINISHED: {name}", flush=True)
    print(f"TIME: {elapsed / 60:.2f} minutes", flush=True)
    print("-" * 70, flush=True)

    if result.returncode != 0:
        raise SystemExit(
            f"{name} failed with exit code {result.returncode}"
        )


run_stage(
    "FREQUENCY EXPERIMENT",
    [
        sys.executable,
        "-m",
        "olp_repro.cli",
        "frequency",
        "--config",
        str(config),
        "--out",
        str(frequency_out),
        "--plot",
        str(frequency_plot),
    ],
)

run_stage(
    "BASELINE COMPARISON",
    [
        sys.executable,
        "-m",
        "olp_repro.cli",
        "comparison",
        "--config",
        str(config),
        "--out",
        str(comparison_out),
        "--plot",
        str(comparison_plot),
    ],
)

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 70)