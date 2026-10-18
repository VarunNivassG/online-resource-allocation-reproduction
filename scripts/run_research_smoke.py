"""Run all three research stages on a small deterministic smoke configuration."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
subprocess.run([
    sys.executable, "-m", "olp_repro.cli", "research",
    "--config", str(ROOT / "configs/research_smoke.json"),
    "--outdir", str(ROOT / "results_research_smoke"),
], cwd=ROOT, env=env, check=True)
