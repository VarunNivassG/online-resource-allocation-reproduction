"""Run the missing multi-seed horizon and risk-tail studies and render figures 25-26."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from olp_repro.remaining_research import run_horizon_sweep, run_risk_tail_sweep
from olp_repro.remaining_research_plots import plot_horizon_sweep, plot_risk_tail_sweep

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true", help="re-run both experiments even when compatible cached CSVs exist")
args = parser.parse_args()
config_path = ROOT / "configs/research_remaining.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
out = ROOT / "results_research_supplementary"
out.mkdir(parents=True, exist_ok=True)
metadata_path = out / "remaining_research_config.json"
old_config = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
cache_compatible = old_config == config and not args.force
if not cache_compatible:
    # Avoid silently mixing cached data generated with a different configuration.
    args.force = True
metadata_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

horizon_path = out / "horizon_sweep.csv"
if horizon_path.exists() and not args.force: 
    horizon = pd.read_csv(horizon_path)
else:
    print("Running multi-seed horizon sweep...", flush=True)
    horizon = run_horizon_sweep(config)
    horizon.to_csv(horizon_path, index=False)
plot_horizon_sweep(horizon, out / "plots" / "25_multi_seed_regret_vs_horizon.png")
print(f"Horizon sweep: {len(horizon)} rows, {horizon.seed.nunique()} seeds per horizon", flush=True)

risk_path = out / "risk_tail_sweep.csv"
if risk_path.exists() and not args.force:
    risk_tail = pd.read_csv(risk_path)
else:
    print("Running paired risk-tail sweep...", flush=True)
    risk_tail = run_risk_tail_sweep(config)
    risk_tail.to_csv(risk_path, index=False)
plot_risk_tail_sweep(risk_tail, out / "plots" / "26_risk_tail_outcomes.png")
print(f"Risk-tail sweep: {len(risk_tail)} rows, {risk_tail.seed.nunique()} seeds per scenario/weight", flush=True)
print("Saved figures 25-26.", flush=True)
subprocess.run([sys.executable, str(ROOT / "scripts/build_research_plot_gallery.py")], cwd=ROOT, check=True)
