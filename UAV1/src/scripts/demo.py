"""
Quick demo script to showcase the project.
Runs a short training, evaluation, and generates plots.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main():
    print("=" * 60)
    print("Dual UAV-ISAC Project Demo")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("runs/demo", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    print("\n1. Training SAC (short run: 50k steps)...")
    cmd = [
        sys.executable, "-m", "src.training.train_sac",
        "--total_timesteps", "50000",
        "--alpha", "0.5",
        "--n_users", "8",
        "--n_targets", "1",
        "--save_dir", "runs/demo/sac",
        "--seed", "42",
    ]
    subprocess.run(cmd, check=True)
    
    print("\n2. Evaluating trained model...")
    cmd = [
        sys.executable, "-m", "src.eval.eval_rollout",
        "--model_path", "runs/demo/sac/model.zip",
        "--alpha", "0.5",
        "--n_episodes", "10",
    ]
    subprocess.run(cmd, check=True)
    
    print("\n3. Running baselines...")
    for baseline in ["circle", "greedy_comms", "greedy_sense"]:
        cmd = [
            sys.executable, "-m", "src.eval.eval_rollout",
            "--baseline", baseline,
            "--alpha", "0.5",
            "--n_episodes", "10",
            "--out", f"runs/demo/{baseline}",
        ]
        subprocess.run(cmd, check=True)
    
    print("\n4. Generating plots...")
    cmd = [
        sys.executable, "-m", "src.eval.plots",
        "--inputs", "runs/demo/*/results.jsonl",
        "--out", "reports/figures",
    ]
    subprocess.run(cmd, check=True)
    
    print("\n" + "=" * 60)
    print("Demo complete! Check reports/figures/ for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()

