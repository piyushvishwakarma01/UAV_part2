from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

import gymnasium as gym


def rollout(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray, dict], np.ndarray],
    n_episodes: int = 10,
    seed: int | None = 123,
) -> Dict:
    rng = np.random.default_rng(seed)
    metrics: List[Dict] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep = {"reward": 0.0, "sum_rate": 0.0, "sensing": 0.0, "leakage": 0.0, "energy": 0.0, "safety": 0.0}
        steps = 0
        while not done:
            act = policy_fn(obs, {})
            obs, rew, term, trunc, info = env.step(act)
            done = term or trunc
            ep["reward"] += float(rew)
            for k in ["sum_rate", "sensing", "leakage", "energy", "safety"]:
                ep[k] += float(info.get(k, 0.0))
            steps += 1
        for k in ep.keys():
            ep[k] /= max(1, steps)
        metrics.append(ep)

    # Aggregate
    agg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0].keys()}
    return {"episodes": metrics, "aggregate": agg}


def circle_strategy(env: gym.Env, radius: float = 400.0, n_episodes: int = 10) -> Dict:
    center1 = np.array([0.33 * env.cfg.area_size, 0.33 * env.cfg.area_size], dtype=np.float32)
    center2 = np.array([0.66 * env.cfg.area_size, 0.66 * env.cfg.area_size], dtype=np.float32)

    def policy_fn(obs: np.ndarray, _: dict) -> np.ndarray:
        # Ignore obs, produce two UAV actions circling opposing centers
        # [Δheading, speed_frac, Δz_norm, time_split_sense, power_split_AN]
        dhead = 0.2  # slow turn
        speed = 0.6
        dz = 0.0
        time_sense = 0.5
        an = 0.2
        a1 = np.array([dhead, speed, dz, 2 * time_sense - 1, 2 * an - 1], dtype=np.float32)
        a2 = np.array([-dhead, speed, dz, 2 * time_sense - 1, 2 * an - 1], dtype=np.float32)
        return np.concatenate([a1, a2])

    return rollout(env, policy_fn, n_episodes=n_episodes)


def greedy_strategy(env: gym.Env, mode: str = "comms", n_episodes: int = 10) -> Dict:
    users = env.users_xy
    targets = env.targets_xy
    user_centroid = np.mean(users, axis=0)
    tgt_centroid = np.mean(targets, axis=0)
    rng = np.random.default_rng(42)  # Seeded for reproducibility

    def policy_fn(obs: np.ndarray, _: dict) -> np.ndarray:
        # Use current headings to steer towards centroid
        head_gain = 0.5
        speed = 0.8
        dz = 0.0
        if mode == "comms":
            time_sense, an = 0.2, 0.2
            goal = user_centroid
        else:
            time_sense, an = 0.8, 0.2
            goal = tgt_centroid

        # Approx: choose heading delta sign randomly since we don't know position directly from obs in baseline
        dhead1 = head_gain * np.sign(rng.standard_normal())
        dhead2 = -head_gain * np.sign(rng.standard_normal())
        a1 = np.array([dhead1, speed, dz, 2 * time_sense - 1, 2 * an - 1], dtype=np.float32)
        a2 = np.array([dhead2, speed, dz, 2 * time_sense - 1, 2 * an - 1], dtype=np.float32)
        return np.concatenate([a1, a2])

    return rollout(env, policy_fn, n_episodes=n_episodes)


