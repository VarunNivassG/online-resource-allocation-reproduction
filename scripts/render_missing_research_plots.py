"""Render supplementary figures from already computed experiment tables.

This avoids rerunning optimization experiments when only plot formatting changes.
"""
from pathlib import Path
import sys
import subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from olp_repro.supplementary_plots import plot_supplementary_bundle
from olp_repro.remaining_research_plots import plot_horizon_sweep, plot_risk_tail_sweep

DATA = ROOT / "results_research_supplementary"
OUT = DATA / "plots"
required = {
    "trajectories": "trajectories.csv",
    "shift_sensitivity": "shift_sensitivity.csv",
    "ablation": "ablation.csv",
    "threshold_sensitivity": "threshold_sensitivity.csv",
    "risk_sensitivity": "risk_sensitivity.csv",
    "budget_sweep": "budget_sweep.csv",
    "noise_sensitivity": "noise_sensitivity.csv",
    "horizon_sweep": "horizon_sweep.csv",
    "risk_tail_sweep": "risk_tail_sweep.csv",
}
missing = [name for name, filename in required.items() if not (DATA / filename).exists()]
if missing:
    raise FileNotFoundError(f"Missing supplementary CSV tables: {missing}. Run scripts/run_research_supplementary.py first.")
tables = {name: pd.read_csv(DATA / filename) for name, filename in required.items()}
extra = {name: tables.pop(name) for name in ["horizon_sweep", "risk_tail_sweep"]}
written = plot_supplementary_bundle(tables, OUT)
written.append(plot_horizon_sweep(extra["horizon_sweep"], OUT / "25_multi_seed_regret_vs_horizon.png"))
written.append(plot_risk_tail_sweep(extra["risk_tail_sweep"], OUT / "26_risk_tail_outcomes.png"))
print(f"Rendered {len(written)} supplementary research figures:")
for path in written:
    print(path)
subprocess.run([sys.executable, str(ROOT / "scripts/build_research_plot_gallery.py")], cwd=ROOT, check=True)
