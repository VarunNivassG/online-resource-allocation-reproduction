"""Run supplementary multi-seed plots and sensitivity experiments."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
subprocess.run([
    sys.executable, "-m", "olp_repro.cli", "supplementary",
    "--config", str(ROOT / "configs/research_supplementary.json"),
    "--outdir", str(ROOT / "results_research_supplementary"),
], cwd=ROOT, env=env, check=True)
# Adds multi-horizon scaling and a higher-seed risk-tail/CVaR evaluation.
subprocess.run([sys.executable, str(ROOT / "scripts/run_remaining_research_plots.py")],
               cwd=ROOT, env=env, check=True)
