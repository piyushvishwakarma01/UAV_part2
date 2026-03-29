from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from src.training.train_sac import Args as SACArgs, main as train_sac_main
from src.eval.eval_rollout import run_model


def main() -> None:
    alphas: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
    out_dir = "runs/pareto"
    os.makedirs(out_dir, exist_ok=True)

    for a in alphas:
        run_dir = os.path.join(out_dir, f"sac_alpha_{a}")
        args = SACArgs(total_timesteps=300_000, seed=42, alpha=a, n_users=8, n_targets=1, save_dir=run_dir)
        print(f"Training SAC alpha={a}...")
        train_sac_main(args)

        print("Evaluating...")
        results = run_model(os.path.join(run_dir, "model.zip"), a, 8, 1, n_episodes=10)
        # Cache aggregate
        with open(os.path.join(run_dir, "pareto_agg.json"), "w", encoding="utf-8") as f:
            json.dump(results["aggregate"], f, indent=2)

    print("Pareto sweep complete.")


if __name__ == "__main__":
    main()


