"""Run the original baseline smoke experiments without requiring an editable install."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
subprocess.run([sys.executable, "-m", "olp_repro.cli", "frequency", "--config", str(root / "configs/smoke.json"), "--out", str(root / "results_smoke/frequency.csv"), "--plot", str(root / "results_smoke/frequency")], cwd=root, env=env, check=True)
subprocess.run([sys.executable, "-m", "olp_repro.cli", "comparison", "--config", str(root / "configs/smoke.json"), "--out", str(root / "results_smoke/comparison.csv"), "--plot", str(root / "results_smoke/comparison.png")], cwd=root, env=env, check=True)
