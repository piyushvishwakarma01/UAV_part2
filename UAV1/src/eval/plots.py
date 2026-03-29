from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(paths: List[str]) -> List[Dict]:
    records = []
    for pattern in paths:
        for fp in glob.glob(pattern):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    return records


def plot_reward_curve(records: List[Dict], out_dir: str) -> None:
    episodes = [r for r in records if r.get("type") == "episode"]
    if not episodes:
        return
    rewards = np.array([e["reward"] for e in episodes])
    window = max(1, len(rewards) // 20)
    kernel = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode="valid")
    plt.figure(figsize=(6, 4))
    plt.plot(smoothed)
    plt.title("Reward (smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_curve.png"), dpi=200)
    plt.close()


def plot_secrecy_bar(groups: Dict[str, List[Dict]], out_dir: str) -> None:
    labels = []
    means = []
    stds = []
    for name, recs in groups.items():
        eps = [r for r in recs if r.get("type") == "episode"]
        if not eps:
            continue
        vals = np.array([e["leakage"] for e in eps])
        labels.append(name)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    if not labels:
        return
    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Secrecy proxy (lower is better)")
    plt.title("Secrecy Proxy: SAC vs TD3 vs Baselines")
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "secrecy_bar.png"), dpi=200)
    plt.close()


def plot_cdf_per_user_rates(records: List[Dict], out_dir: str) -> None:
    # Placeholder: use sum_rate as proxy for per-user distribution (no per-user detail logged)
    eps = [r for r in records if r.get("type") == "episode"]
    if not eps:
        return
    vals = np.array([e["sum_rate"] for e in eps])
    xs = np.sort(vals)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.xlabel("Rate proxy")
    plt.ylabel("CDF")
    plt.title("CDF of rate proxy")
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cdf_rates.png"), dpi=200)
    plt.close()


def plot_pareto(records: List[Dict], out_dir: str) -> None:
    aggs = [r for r in records if r.get("type") == "aggregate"]
    if not aggs:
        return
    xs = [a.get("sum_rate", 0.0) for a in aggs]
    ys = [a.get("sensing", 0.0) for a in aggs]
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, c=np.linspace(0, 1, len(xs)), cmap="viridis")
    plt.xlabel("Comms Utility (proxy)")
    plt.ylabel("Sensing Utility (proxy)")
    plt.title("Pareto: Comms vs Sensing")
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pareto.png"), dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="results.jsonl glob(s)")
    p.add_argument("--out", required=True, help="output dir for figures")
    args = p.parse_args()

    recs = load_results(args.inputs)
    # Group by parent dir name for bar chart
    groups: Dict[str, List[Dict]] = {}
    for pattern in args.inputs:
        for fp in glob.glob(pattern):
            name = os.path.basename(os.path.dirname(fp)) or os.path.basename(fp)
            with open(fp, "r", encoding="utf-8") as f:
                groups[name] = [json.loads(line) for line in f]

    plot_reward_curve(recs, args.out)
    plot_secrecy_bar(groups, args.out)
    plot_cdf_per_user_rates(recs, args.out)
    plot_pareto(recs, args.out)
    print(f"Saved plots to {args.out}")


if __name__ == "__main__":
    main()


