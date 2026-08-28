from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, "-m", "olp_repro.cli", "frequency", "--config", str(root / "configs/smoke.json"), "--out", str(root / "results_smoke/frequency.csv"), "--plot", str(root / "results_smoke/frequency")], check=True)
subprocess.run([sys.executable, "-m", "olp_repro.cli", "comparison", "--config", str(root / "configs/smoke.json"), "--out", str(root / "results_smoke/comparison.csv"), "--plot", str(root / "results_smoke/comparison.png")], check=True)
