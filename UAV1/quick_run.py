"""
Quick Project Demo — runs a short SAC training (10k steps) and evaluates.
Designed to complete in ~5 minutes on CPU.
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.envs.dual_isac_env import DualISACEnv, EnvConfig
from src.baselines.heuristics import circle_strategy, greedy_strategy

# ==========================================================
print("=" * 60)
print(" Dual UAV-ISAC Project — Quick Run")
print("=" * 60)

# --- Step 1: Quick Environment Smoke Test ---
print("\n[1/4] Environment smoke test...")
cfg = EnvConfig(
    n_users=8, n_targets=1, alpha=0.5, seed=42,
    channel_model="rician", sensing_model="crb",
    energy_model="aerodynamic", secrecy_model="capacity",
    qos_model="queue"
)
env = DualISACEnv(cfg=cfg)
obs, info = env.reset()
print(f"  ✓ Observation shape: {obs.shape}")
print(f"  ✓ Action space: {env.action_space.shape}")

total_reward = 0
for step in range(50):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    if term or trunc:
        break

print(f"  ✓ Ran 50 random steps | Total reward: {total_reward:.3f}")
print(f"  ✓ Last step info: sum_rate={info['sum_rate']:.2f}, sensing={info['sensing']:.3f}, "
      f"leakage={info['leakage']:.3f}, energy={info['energy']:.4f}")

# --- Step 2: SAC Training (10k steps — fast) ---
print("\n[2/4] Training SAC (10,000 steps on CPU)...")
save_dir = "runs/quick_run"
os.makedirs(save_dir, exist_ok=True)

def make_env():
    return Monitor(DualISACEnv(cfg=EnvConfig(
        n_users=8, n_targets=1, alpha=0.5, seed=42,
        channel_model="rician", sensing_model="crb",
        energy_model="aerodynamic", secrecy_model="capacity",
        qos_model="queue"
    )))

vec_env = VecMonitor(DummyVecEnv([make_env]))

start = time.time()
model = SAC(
    policy="MlpPolicy",
    env=vec_env,
    verbose=0,
    seed=42,
    ent_coef="auto",
    gamma=0.99,
    buffer_size=50_000,
    batch_size=128,
    learning_rate=3e-4,
)
model.learn(total_timesteps=10_000)
model.save(os.path.join(save_dir, "model"))
elapsed = time.time() - start
print(f"  ✓ Training completed in {elapsed:.1f}s")
print(f"  ✓ Model saved to {save_dir}/model.zip")

# --- Step 3: Evaluate Trained Model ---
print("\n[3/4] Evaluating trained model (5 episodes)...")
eval_env = DualISACEnv(cfg=EnvConfig(
    n_users=8, n_targets=1, alpha=0.5, seed=99,
    channel_model="rician", sensing_model="crb",
    energy_model="aerodynamic", secrecy_model="capacity",
    qos_model="queue"
))

ep_rewards = []
for ep in range(5):
    obs, _ = eval_env.reset()
    done = False
    ep_reward = 0
    step_count = 0
    ep_info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = eval_env.step(action)
        done = term or trunc
        ep_reward += reward
        ep_info = info
        step_count += 1
    ep_rewards.append(ep_reward)
    print(f"  Episode {ep+1}: reward={ep_reward:.3f} | steps={step_count} | "
          f"rate={ep_info['sum_rate']:.2f} | sense={ep_info['sensing']:.3f} | "
          f"leak={ep_info['leakage']:.3f}")

avg_reward = sum(ep_rewards) / len(ep_rewards)
print(f"  ✓ Average reward: {avg_reward:.3f}")

# --- Step 4: Compare with Baselines ---
print("\n[4/4] Running baselines for comparison...")
baseline_env = DualISACEnv(cfg=EnvConfig(
    n_users=8, n_targets=1, alpha=0.5, seed=42,
    channel_model="rician", sensing_model="crb",
    energy_model="aerodynamic", secrecy_model="capacity",
    qos_model="queue"
))

circle_results = circle_strategy(baseline_env, n_episodes=5)
print(f"  Circle:  avg_reward={circle_results['aggregate']['reward']:.3f}")

baseline_env2 = DualISACEnv(cfg=EnvConfig(
    n_users=8, n_targets=1, alpha=0.5, seed=42,
    channel_model="rician", sensing_model="crb",
    energy_model="aerodynamic", secrecy_model="capacity",
    qos_model="queue"
))
greedy_results = greedy_strategy(baseline_env2, mode="comms", n_episodes=5)
print(f"  Greedy:  avg_reward={greedy_results['aggregate']['reward']:.3f}")
print(f"  SAC:     avg_reward={avg_reward:.3f}")

# Summary
print("\n" + "=" * 60)
print(" RESULTS SUMMARY")
print("=" * 60)
print(f"  {'Method':<12} {'Avg Reward':>12} {'Sum Rate':>10} {'Sensing':>10} {'Leakage':>10}")
print(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
for name, res in [("Circle", circle_results), ("Greedy", greedy_results)]:
    a = res['aggregate']
    print(f"  {name:<12} {a['reward']:>12.4f} {a['sum_rate']:>10.2f} {a['sensing']:>10.4f} {a['leakage']:>10.4f}")
print(f"  {'SAC (10k)':<12} {avg_reward:>12.4f} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
print("=" * 60)
print(" Done! ✓")
