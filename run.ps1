# Windows PowerShell quick start
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
pytest -q
python scripts\run_smoke.py
