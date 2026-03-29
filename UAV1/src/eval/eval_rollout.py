from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.dual_isac_env import make_env
from src.baselines.heuristics import circle_strategy, greedy_strategy


def run_model(model_path: str, alpha: float, n_users: int, n_targets: int, n_episodes: int, seed: int = 123) -> Dict:
    def mk():
        return Monitor(make_env(n_users=n_users, n_targets=n_targets, alpha=alpha, seed=seed))

    env = DummyVecEnv([mk])
    # Try loading SAC then TD3
    model = None
    try:
        model = SAC.load(model_path, env=env)
    except Exception:
        model = TD3.load(model_path, env=env)

    ep_metrics = []
    for _ in tqdm(range(n_episodes), desc="Eval Episodes"):
        obs = env.reset()
        done = False
        ep = {"reward": 0.0, "sum_rate": 0.0, "sensing": 0.0, "leakage": 0.0, "energy": 0.0, "safety": 0.0}
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            info = infos[0]
            ep["reward"] += float(rewards[0])
            for k in ["sum_rate", "sensing", "leakage", "energy", "safety"]:
                ep[k] += float(info.get(k, 0.0))
            steps += 1
        for k in ep.keys():
            ep[k] /= max(1, steps)
        ep_metrics.append(ep)
    agg = {k: float(np.mean([m[k] for m in ep_metrics])) for k in ep_metrics[0].keys()}
    return {"episodes": ep_metrics, "aggregate": agg}


def run_baseline(name: str, alpha: float, n_users: int, n_targets: int, n_episodes: int, seed: int = 123) -> Dict:
    env = make_env(n_users=n_users, n_targets=n_targets, alpha=alpha, seed=seed)
    if name == "circle":
        return circle_strategy(env, n_episodes=n_episodes)
    elif name == "greedy_comms":
        return greedy_strategy(env, mode="comms", n_episodes=n_episodes)
    elif name == "greedy_sense":
        return greedy_strategy(env, mode="sense", n_episodes=n_episodes)
    else:
        raise ValueError(f"Unknown baseline: {name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--baseline", type=str, default=None, choices=[None, "circle", "greedy_comms", "greedy_sense"], nargs="?")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n_users", type=int, default=8)
    p.add_argument("--n_targets", type=int, default=1)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.model_path is None and args.baseline is None:
        raise SystemExit("Provide --model_path or --baseline")

    results = (
        run_model(args.model_path, args.alpha, args.n_users, args.n_targets, args.n_episodes)
        if args.model_path is not None
        else run_baseline(args.baseline, args.alpha, args.n_users, args.n_targets, args.n_episodes)
    )

    # Decide output path
    out_dir = args.out or os.path.dirname(args.model_path or "runs/baselines")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.jsonl")
    # Append episodes and an aggregate record
    with open(out_path, "a", encoding="utf-8") as f:
        for ep in results["episodes"]:
            f.write(json.dumps({"type": "episode", **ep}) + "\n")
        f.write(json.dumps({"type": "aggregate", **results["aggregate"]}) + "\n")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


