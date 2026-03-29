"""
Full Experimental Suite for IEEE Paper
========================================
This script runs ALL experiments needed to regenerate the paper figures and tables:
1. SAC-Simple (baseline with simple models)
2. SAC-Advanced (our IEEE-worthy models)
3. TD3-Simple (comparison algorithm)
4. Heuristics (circle, greedy)
5. Pareto frontier sweep (α ∈ {0.1, 0.3, 0.5, 0.7, 0.9})
6. Ablation study (each advanced model individually)

Run this on a GPU machine. Total time: ~4-6 hours.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tyro
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.envs.dual_isac_env import EnvConfig, DualISACEnv
from src.baselines.heuristics import circle_strategy, greedy_strategy
from src.eval.eval_rollout import run_model


@dataclass
class ExperimentConfig:
    """Configuration for the full experimental suite"""
    base_dir: str = "runs/ieee_experiments"
    seeds: List[int] = None  # Will default to [42, 123, 456, 789, 1011]
    total_timesteps: int = 300_000
    n_users: int = 8
    n_targets: int = 1
    alpha: float = 0.5  # Default for main experiments
    eval_episodes: int = 20
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1011]


# EXPERIMENT 1: Baseline vs Advanced (Main Comparison)


def exp1_baseline_vs_advanced(cfg: ExperimentConfig) -> None:
    """Compare SAC with simple vs advanced models (5 seeds each)"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Baseline (Simple) vs Advanced Models")
    print("="*70)
    
    configs = {
        "SAC-Simple": EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=cfg.alpha,
            channel_model="simple",
            sensing_model="geometric",
            energy_model="simple",
            secrecy_model="proxy",
            qos_model="deficit"
        ),
        "SAC-Advanced": EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=cfg.alpha,
            channel_model="rician",
            sensing_model="crb",
            energy_model="aerodynamic",
            secrecy_model="capacity",
            qos_model="queue"
        )
    }
    
    for name, env_cfg in configs.items():
        for seed in cfg.seeds:
            run_dir = os.path.join(cfg.base_dir, "exp1_main", name, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            
            print(f"\n>>> Training {name} with seed={seed}...")
            start = time.time()
            
            # Create environment
            def make_env():
                env = DualISACEnv(cfg=env_cfg)
                return Monitor(env)
            
            vec_env = VecMonitor(DummyVecEnv([make_env]))
            
            # Train SAC
            model = SAC(
                policy="MlpPolicy",
                env=vec_env,
                verbose=1,
                tensorboard_log=os.path.join(run_dir, "tb"),
                seed=seed,
                ent_coef="auto",
                gamma=0.99,
                buffer_size=200_000,
                batch_size=256,
                learning_rate=3e-4,
            )
            
            model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)
            model.save(os.path.join(run_dir, "model"))
            
            elapsed = time.time() - start
            print(f"    ✓ Completed in {elapsed/60:.1f} minutes")
            
            # Evaluate
            print(f"    Evaluating {name} seed={seed}...")
            results = run_model(
                os.path.join(run_dir, "model.zip"),
                cfg.alpha,
                cfg.n_users,
                cfg.n_targets,
                n_episodes=cfg.eval_episodes
            )
            
            # Save results
            with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"    ✓ Avg Reward: {results['aggregate']['reward']:.3f}")
            print(f"    ✓ Sum Rate: {results['aggregate']['sum_rate']:.2f} Mbps")
            print(f"    ✓ Sensing: {results['aggregate']['sensing']:.3f}")
            print(f"    ✓ Secrecy: {1 - results['aggregate']['leakage']:.3f}")


# EXPERIMENT 2: Algorithm Comparison (SAC vs TD3)


def exp2_sac_vs_td3(cfg: ExperimentConfig) -> None:
    """Compare SAC vs TD3 with simple models (3 seeds)"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: SAC vs TD3 (Simple Models)")
    print("="*70)
    
    env_cfg = EnvConfig(
        channel_model="simple",
        sensing_model="geometric",
        energy_model="linear",
        secrecy_model="proxy",
        qos_model="deficit"
    )
    
    seeds = cfg.seeds[:3]  # Use 3 seeds for computational efficiency
    
    for algo_name, AlgoClass in [("SAC", SAC), ("TD3", TD3)]:
        for seed in seeds:
            run_dir = os.path.join(cfg.base_dir, "exp2_algos", algo_name, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            
            print(f"\n>>> Training {algo_name} with seed={seed}...")
            
            def make_env():
                env = DualISACEnv(cfg=env_cfg)
                return Monitor(env)
            
            vec_env = VecMonitor(DummyVecEnv([make_env]))
            
            if algo_name == "SAC":
                model = SAC(
                    policy="MlpPolicy",
                    env=vec_env,
                    verbose=1,
                    seed=seed,
                    gamma=0.99,
                    buffer_size=200_000,
                    batch_size=256,
                    learning_rate=3e-4,
                )
            else:  # TD3
                model = TD3(
                    policy="MlpPolicy",
                    env=vec_env,
                    verbose=1,
                    seed=seed,
                    gamma=0.99,
                    buffer_size=200_000,
                    batch_size=256,
                    learning_rate=3e-4,
                )
            
            model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)
            model.save(os.path.join(run_dir, "model"))
            
            # Evaluate
            results = run_model(
                os.path.join(run_dir, "model.zip"),
                cfg.alpha,
                cfg.n_users,
                cfg.n_targets,
                n_episodes=cfg.eval_episodes
            )
            
            with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"    ✓ {algo_name} Reward: {results['aggregate']['reward']:.3f}")


# ============================================================================
# EXPERIMENT 3: Heuristic Baselines
# ============================================================================

def exp3_heuristics(cfg: ExperimentConfig) -> None:
    """Evaluate heuristic strategies (Circle, Greedy)"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Heuristic Baselines")
    print("="*70)
    
    # Use advanced environment for fair comparison
    env_cfg = EnvConfig(
        channel_model="rician",
        sensing_model="crb",
        energy_model="aerodynamic",
        secrecy_model="capacity",
        qos_model="queue"
    )
    
    strategies = {
        "circle": circle_strategy,
        "greedy": greedy_strategy
    }
    
    for name, strategy_fn in strategies.items():
        run_dir = os.path.join(cfg.base_dir, "exp3_heuristics", name)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n>>> Evaluating {name} strategy...")
        
        # Create environment and run strategy
        env_cfg_copy = EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=cfg.alpha,
            channel_model="rician",
            sensing_model="crb",
            energy_model="aerodynamic",
            secrecy_model="capacity",
            qos_model="queue"
        )
        env = DualISACEnv(cfg=env_cfg_copy)
        
        # Use the heuristic rollout function
        results = strategy_fn(env, n_episodes=cfg.eval_episodes)
        
        with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"    ✓ {name.capitalize()} Avg Reward: {results['aggregate']['reward']:.3f}")


# ============================================================================
# EXPERIMENT 4: Pareto Frontier (α sweep)
# ============================================================================

def exp4_pareto_sweep(cfg: ExperimentConfig) -> None:
    """Generate Pareto frontier by sweeping α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Pareto Frontier (α sweep)")
    print("="*70)
    
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    env_cfg = EnvConfig(
        channel_model="rician",
        sensing_model="crb",
        energy_model="aerodynamic",
        secrecy_model="capacity",
        qos_model="queue"
    )
    
    for alpha in alphas:
        run_dir = os.path.join(cfg.base_dir, "exp4_pareto", f"alpha_{alpha}")
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n>>> Training SAC-Advanced with α={alpha}...")
        
        alpha_env_cfg = EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=alpha,
            channel_model="rician",
            sensing_model="crb",
            energy_model="aerodynamic",
            secrecy_model="capacity",
            qos_model="queue"
        )
        
        def make_env():
            env = DualISACEnv(cfg=alpha_env_cfg)
            return Monitor(env)
        
        vec_env = VecMonitor(DummyVecEnv([make_env]))
        
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=cfg.seeds[0],
            gamma=0.99,
            buffer_size=200_000,
            batch_size=256,
            learning_rate=3e-4,
        )
        
        model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)
        model.save(os.path.join(run_dir, "model"))
        
        # Evaluate
        results = run_model(
            os.path.join(run_dir, "model.zip"),
            alpha,
            cfg.n_users,
            cfg.n_targets,
            n_episodes=cfg.eval_episodes
        )
        
        with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"    ✓ α={alpha}: Rate={results['aggregate']['sum_rate']:.2f}, "
              f"Sense={results['aggregate']['sensing']:.3f}")


# ============================================================================
# EXPERIMENT 5: Ablation Study
# ============================================================================

def exp5_ablation(cfg: ExperimentConfig) -> None:
    """Ablation: test each advanced model individually"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Ablation Study (Individual Advanced Models)")
    print("="*70)
    
    ablation_configs = {
        "Baseline": EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=cfg.alpha,
            channel_model="simple",
            sensing_model="geometric",
            energy_model="simple",
            secrecy_model="proxy",
            qos_model="deficit"
        ),
        "ChannelOnly": EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=cfg.alpha,
            channel_model="rician",  # Advanced
            sensing_model="geometric",
            energy_model="simple",
            secrecy_model="proxy",
            qos_model="deficit"
        ),
        "SensingOnly": EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=cfg.alpha,
            channel_model="simple",
            sensing_model="crb",  # Advanced
            energy_model="simple",
            secrecy_model="proxy",
            qos_model="deficit"
        ),
        "SecrecyOnly": EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=cfg.alpha,
            channel_model="simple",
            sensing_model="geometric",
            energy_model="simple",
            secrecy_model="capacity",  # Advanced
            qos_model="deficit"
        ),
        "EnergyOnly": EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=cfg.alpha,
            channel_model="simple",
            sensing_model="geometric",
            energy_model="aerodynamic",  # Advanced
            secrecy_model="proxy",
            qos_model="deficit"
        ),
        "AllCombined": EnvConfig(
            n_users=cfg.n_users,
            n_targets=cfg.n_targets,
            alpha=cfg.alpha,
            channel_model="rician",
            sensing_model="crb",
            energy_model="aerodynamic",
            secrecy_model="capacity",
            qos_model="queue"
        )
    }
    
    seed = cfg.seeds[0]  # Use single seed for ablation
    
    for name, ablation_cfg in ablation_configs.items():
        run_dir = os.path.join(cfg.base_dir, "exp5_ablation", name)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n>>> Training {name}...")
        
        def make_env(local_cfg=ablation_cfg):
            env = DualISACEnv(cfg=local_cfg)
            return Monitor(env)
        
        vec_env = VecMonitor(DummyVecEnv([make_env]))
        
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=seed,
            gamma=0.99,
            buffer_size=200_000,
            batch_size=256,
            learning_rate=3e-4,
        )
        
        model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True)
        model.save(os.path.join(run_dir, "model"))
        
        # Evaluate
        results = run_model(
            os.path.join(run_dir, "model.zip"),
            cfg.alpha,
            cfg.n_users,
            cfg.n_targets,
            n_episodes=cfg.eval_episodes
        )
        
        with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"    ✓ {name}: Reward={results['aggregate']['reward']:.3f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(
    base_dir: str = "runs/ieee_experiments",
    quick_test: bool = False
) -> None:
    """
    Run the full experimental suite.
    
    Args:
        base_dir: Output directory for all experiments
        quick_test: If True, run with reduced timesteps for validation (30k instead of 300k)
    """
    
    cfg = ExperimentConfig(base_dir=base_dir)
    
    if quick_test:
        print("\n" + "!"*70)
        print("QUICK TEST MODE: Using 30k timesteps for validation only!")
        print("!"*70)
        cfg.total_timesteps = 30_000
        cfg.seeds = [42]  # Single seed for quick test
        cfg.eval_episodes = 5
    
    print("\n" + "="*70)
    print("IEEE PAPER - FULL EXPERIMENTAL SUITE")
    print("="*70)
    print(f"Base directory: {cfg.base_dir}")
    print(f"Seeds: {cfg.seeds}")
    print(f"Timesteps per run: {cfg.total_timesteps:,}")
    print(f"Evaluation episodes: {cfg.eval_episodes}")
    print("="*70)
    
    overall_start = time.time()
    
    # Run all experiments
    exp1_baseline_vs_advanced(cfg)
    exp2_sac_vs_td3(cfg)
    exp3_heuristics(cfg)
    exp4_pareto_sweep(cfg)
    exp5_ablation(cfg)
    
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"Total time: {overall_elapsed/3600:.2f} hours")
    print(f"Results saved to: {cfg.base_dir}")
    print("\nNext steps:")
    print("  1. Run: python generate_paper_figures.py")
    print("  2. Run: python generate_paper_tables.py")
    print("  3. Compile LaTeX: cd latex && pdflatex main_enhanced.tex")
    print("="*70)


if __name__ == "__main__":
    tyro.cli(main)
