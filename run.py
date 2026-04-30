#!/usr/bin/env python3
"""
run.py — Shield-Fi one-shot setup script.
Generates data and trains the model in one command:

    python run.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"

def run(script: str, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    result = subprocess.run([sys.executable, str(SRC / script)], check=True)
    return result

if __name__ == "__main__":
    run("generate_data.py", "Step 1/2 — Generating synthetic transaction data")
    run("train.py",         "Step 2/2 — Training XGBoost fraud detection model")
    print("\n" + "="*60)
    print("  ✅  Shield-Fi setup complete!")
    print("  👉  Start the API: uvicorn src.api:app --reload --port 8000")
    print("  👉  API docs:      http://localhost:8000/docs")
    print("  👉  Run tests:     pytest tests/ -v")
    print("="*60)
