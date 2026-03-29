from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import tyro
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.envs.dual_isac_env import make_env


@dataclass
class Args:
    total_timesteps: int = 300_000
    seed: int = 42
    alpha: float = 0.5
    n_users: int = 8
    n_targets: int = 1
    save_dir: str = "runs/sac_run"


def main(args: Args) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    def make() :
        env = make_env(n_users=args.n_users, n_targets=args.n_targets, alpha=args.alpha, seed=args.seed)
        return Monitor(env)

    vec_env = DummyVecEnv([make])
    vec_env = VecMonitor(vec_env)

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tb"),
        seed=args.seed,
        ent_coef="auto",
        gamma=0.99,
        buffer_size=200_000,
        batch_size=256,
        learning_rate=3e-4,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path=args.save_dir, name_prefix="sac_checkpoint"
    )

    model.learn(total_timesteps=args.total_timesteps, callback=[checkpoint_callback])
    model.save(os.path.join(args.save_dir, "model"))

    # Save a small config
    cfg = {
        "algo": "SAC",
        "total_timesteps": args.total_timesteps,
        "alpha": args.alpha,
        "n_users": args.n_users,
        "n_targets": args.n_targets,
        "seed": args.seed,
    }
    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main(tyro.cli(Args))


