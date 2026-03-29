"""Show results from the quick run."""
import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from src.envs.dual_isac_env import DualISACEnv, EnvConfig
from src.baselines.heuristics import circle_strategy, greedy_strategy

# Load trained model
model = SAC.load("runs/quick_run/model.zip")

# Evaluate SAC
cfg = EnvConfig(
    n_users=8, n_targets=1, alpha=0.5, seed=99,
    channel_model="rician", sensing_model="crb",
    energy_model="aerodynamic", secrecy_model="capacity",
    qos_model="queue"
)
env = DualISACEnv(cfg=cfg)

print("=" * 65)
print(" SAC EVALUATION (5 episodes, deterministic policy)")
print("=" * 65)

sac_metrics = {"reward": [], "sum_rate": [], "sensing": [], "leakage": [], "energy": []}
for ep in range(5):
    obs, _ = env.reset()
    done = False
    ep_r = 0
    steps = 0
    info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, t, tr, info = env.step(a)
        done = t or tr
        ep_r += r
        steps += 1
    sac_metrics["reward"].append(ep_r)
    sac_metrics["sum_rate"].append(info["sum_rate"])
    sac_metrics["sensing"].append(info["sensing"])
    sac_metrics["leakage"].append(info["leakage"])
    sac_metrics["energy"].append(info["energy"])
    print(f"  Ep {ep+1}: reward={ep_r:.3f} | steps={steps} | rate={info['sum_rate']:.2f} | sense={info['sensing']:.3f} | leak={info['leakage']:.3f}")

import numpy as np
avg_sac = {k: float(np.mean(v)) for k, v in sac_metrics.items()}
print(f"\n  Average: reward={avg_sac['reward']:.3f} rate={avg_sac['sum_rate']:.2f} sense={avg_sac['sensing']:.3f} leak={avg_sac['leakage']:.3f}")

# Baselines
print("\n" + "=" * 65)
print(" BASELINE COMPARISON")
print("=" * 65)

env1 = DualISACEnv(cfg=EnvConfig(n_users=8, n_targets=1, alpha=0.5, seed=42,
    channel_model="rician", sensing_model="crb", energy_model="aerodynamic",
    secrecy_model="capacity", qos_model="queue"))
c = circle_strategy(env1, n_episodes=5)

env2 = DualISACEnv(cfg=EnvConfig(n_users=8, n_targets=1, alpha=0.5, seed=42,
    channel_model="rician", sensing_model="crb", energy_model="aerodynamic",
    secrecy_model="capacity", qos_model="queue"))
g = greedy_strategy(env2, mode="comms", n_episodes=5)

header = f"  {'Method':<14} {'Reward':>10} {'SumRate':>10} {'Sensing':>10} {'Leakage':>10}"
sep = "  " + "-" * 56
print(header)
print(sep)
ca = c["aggregate"]
ga = g["aggregate"]
print(f"  {'Circle':<14} {ca['reward']:>10.4f} {ca['sum_rate']:>10.2f} {ca['sensing']:>10.4f} {ca['leakage']:>10.4f}")
print(f"  {'Greedy':<14} {ga['reward']:>10.4f} {ga['sum_rate']:>10.2f} {ga['sensing']:>10.4f} {ga['leakage']:>10.4f}")
print(f"  {'SAC (10k)':<14} {avg_sac['reward']:>10.4f} {avg_sac['sum_rate']:>10.2f} {avg_sac['sensing']:>10.4f} {avg_sac['leakage']:>10.4f}")
print(sep)
print("=" * 65)
print(" Project ran successfully!")
print("=" * 65)
