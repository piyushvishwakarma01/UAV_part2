from __future__ import annotations

import json
import os
from dataclasses import dataclass

import tyro
from stable_baselines3 import TD3
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
    save_dir: str = "runs/td3_run"


def main(args: Args) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    def make() :
        env = make_env(n_users=args.n_users, n_targets=args.n_targets, alpha=args.alpha, seed=args.seed)
        return Monitor(env)

    vec_env = DummyVecEnv([make])
    vec_env = VecMonitor(vec_env)

    model = TD3(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tb"),
        seed=args.seed,
        gamma=0.99,
        policy_delay=2,
        target_policy_noise=0.2,
        tau=0.005,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path=args.save_dir, name_prefix="td3_checkpoint"
    )

    model.learn(total_timesteps=args.total_timesteps, callback=[checkpoint_callback])
    model.save(os.path.join(args.save_dir, "model"))

    cfg = {
        "algo": "TD3",
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


