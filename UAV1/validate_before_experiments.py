"""
Pre-Flight Validation Script
=============================
This script performs a comprehensive check of the codebase before running
expensive experiments on a GPU machine. It verifies:
  1. All dependencies are installed
  2. Advanced models work correctly
  3. Environment integration is functional
  4. Training can start without errors
  5. Evaluation pipeline works

Run this on BOTH your local machine AND the GPU machine before starting
the full experiment suite.
"""

from __future__ import annotations

import sys
import time
from typing import List, Tuple

import numpy as np


def print_section(title: str) -> None:
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def check_pass(name: str) -> None:
    """Print a passing check"""
    print(f"  ✓ {name}")


def check_fail(name: str, error: str) -> None:
    """Print a failing check"""
    print(f"  ✗ {name}")
    print(f"    Error: {error}")


# ============================================================================
# CHECK 1: Dependencies
# ============================================================================

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required packages are installed"""
    print_section("CHECK 1: Dependencies")
    
    required_packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("torch", "PyTorch"),
        ("tyro", "Tyro"),
        ("yaml", "PyYAML"),
    ]
    
    all_ok = True
    errors = []
    
    for module_name, display_name in required_packages:
        try:
            __import__(module_name)
            check_pass(f"{display_name:20s} installed")
        except ImportError as e:
            check_fail(f"{display_name:20s} missing", str(e))
            errors.append(f"Install: pip install {module_name}")
            all_ok = False
    
    # Check Python version
    if sys.version_info >= (3, 11):
        check_pass(f"Python {sys.version_info.major}.{sys.version_info.minor} (>= 3.11)")
    else:
        check_fail("Python version", f"Found {sys.version_info.major}.{sys.version_info.minor}, need >= 3.11")
        errors.append("Upgrade Python to 3.11+")
        all_ok = False
    
    return all_ok, errors


# ============================================================================
# CHECK 2: Advanced Models
# ============================================================================

def check_advanced_models() -> Tuple[bool, List[str]]:
    """Test all 5 advanced models"""
    print_section("CHECK 2: Advanced Models")
    
    try:
        from src.envs.advanced_models import (
            RicianChannelModel,
            CRBSensingModel,
            SecrecyCapacityModel,
            UAVEnergyModel,
            QoSQueueModel
        )
    except ImportError as e:
        check_fail("Import advanced_models", str(e))
        return False, ["Check src/envs/advanced_models.py exists"]
    
    errors = []
    all_ok = True
    
    # Test 1: Rician Channel
    try:
        channel = RicianChannelModel(fc_ghz=2.4, environment="suburban")
        
        # Test LoS probability
        los_prob = channel.los_probability(45.0)
        assert 0 <= los_prob <= 1, f"LoS prob out of range: {los_prob}"
        
        # Test path loss
        pl_db = channel.path_loss_db(100.0, 45.0, True, np.random.default_rng(42))
        assert 30 < pl_db < 100, f"Path loss out of range: {pl_db}"
        
        check_pass("Rician Channel Model")
    except Exception as e:
        check_fail("Rician Channel Model", str(e))
        errors.append("Debug RicianChannelModel methods")
        all_ok = False
    
    # Test 2: CRB Sensing
    try:
        sensing = CRBSensingModel(carrier_freq_ghz=24.0, bandwidth_mhz=100.0, target_rcs_dbsm=10.0)
        
        # Test the full CRB computation (NO time_fraction parameter!)
        crb_error, sensing_score = sensing.compute_crb_position_error(
            uav1_pos=np.array([1000, 1000, 100]),
            uav2_pos=np.array([1100, 1000, 100]),
            target_pos=np.array([1050, 1050]),
            tx_power_w=1.0
        )
        assert 0 < crb_error < 100, f"CRB error out of range: {crb_error}"
        assert 0 <= sensing_score <= 1, f"Sensing score out of range: {sensing_score}"
        
        check_pass("CRB Sensing Model")
    except Exception as e:
        check_fail("CRB Sensing Model", str(e))
        errors.append("Debug CRBSensingModel methods")
        all_ok = False
    
    # Test 3: Secrecy Capacity
    try:
        secrecy = SecrecyCapacityModel(noise_power_dbm=-90.0, an_power_fraction=0.3)
        
        # Test compute_secrecy_rate with correct parameters
        rate, utility = secrecy.compute_secrecy_rate(
            snr_user=20.0,
            snr_eve=15.0,
            an_power_allocated=0.3,  # NOT p_an!
            jammer_active=False,
            jammer_to_eve_gain=0.5
        )
        assert rate >= 0, f"Secrecy capacity negative: {rate}"
        assert 0 <= utility <= 1, f"Utility out of range: {utility}"
        
        check_pass("Secrecy Capacity Model")
    except Exception as e:
        check_fail("Secrecy Capacity Model", str(e))
        errors.append("Debug SecrecyCapacityModel methods")
        all_ok = False
    
    # Test 4: Aerodynamic Energy
    try:
        energy = UAVEnergyModel(mass_kg=2.0, battery_capacity_wh=100.0)
        
        # Test power_consumption with correct parameters
        power = energy.power_consumption(
            speed_mps=10.0,  # NOT v_horizontal!
            altitude_rate_mps=0.0,  # NOT vz!
            tx_power_w=1.0
        )
        assert 100 < power < 1000, f"Power out of range: {power}"
        
        check_pass("Aerodynamic Energy Model")
    except Exception as e:
        check_fail("Aerodynamic Energy Model", str(e))
        errors.append("Debug UAVEnergyModel methods")
        all_ok = False
    
    # Test 5: QoS Queue
    try:
        qos = QoSQueueModel(arrival_rate_mbps=1.0, max_queue_size_mb=10.0)
        
        # Test update_queue method
        new_queue, score = qos.update_queue(
            current_queue_mb=5.0,
            service_rate_mbps=2.0,
            dt=1.0
        )
        assert 0 <= new_queue <= 10, f"Queue out of range: {new_queue}"
        assert 0 <= score <= 1, f"QoS score out of range: {score}"
        
        check_pass("QoS Queue Model")
    except Exception as e:
        check_fail("QoS Queue Model", str(e))
        errors.append("Debug QoSQueueModel methods")
        all_ok = False
    
    return all_ok, errors


# ============================================================================
# CHECK 3: Environment Integration
# ============================================================================

def check_environment_integration() -> Tuple[bool, List[str]]:
    """Test that the environment works with advanced models"""
    print_section("CHECK 3: Environment Integration")
    
    try:
        from src.envs.dual_isac_env import DualISACEnv, EnvConfig
    except ImportError as e:
        check_fail("Import DualISACEnv", str(e))
        return False, ["Check src/envs/dual_isac_env.py"]
    
    errors = []
    all_ok = True
    
    # Test 1: Simple models (backward compatibility)
    try:
        config_simple = EnvConfig(
            n_users=8,
            n_targets=1,
            alpha=0.5,
            channel_model="simple",
            sensing_model="geometric",
            energy_model="simple",
            secrecy_model="proxy",
            qos_model="deficit"
        )
        env_simple = DualISACEnv(cfg=config_simple)
        obs, info = env_simple.reset(seed=42)
        action = env_simple.action_space.sample()
        obs, reward, terminated, truncated, info = env_simple.step(action)
        assert isinstance(reward, (int, float)), "Reward is not a number"
        check_pass("Simple models (backward compatibility)")
    except Exception as e:
        check_fail("Simple models", str(e))
        errors.append("Debug DualISACEnv with simple config")
        all_ok = False
    
    # Test 2: Advanced models
    try:
        config_advanced = EnvConfig(
            n_users=8,
            n_targets=1,
            alpha=0.5,
            channel_model="rician",
            sensing_model="crb",
            energy_model="aerodynamic",
            secrecy_model="capacity",
            qos_model="queue"
        )
        env_advanced = DualISACEnv(cfg=config_advanced)
        obs, info = env_advanced.reset(seed=42)
        
        # Run 10 steps
        for _ in range(10):
            action = env_advanced.action_space.sample()
            obs, reward, terminated, truncated, info = env_advanced.step(action)
            
            if terminated or truncated:
                break
        
        # Check info dict
        required_keys = ["sum_rate", "sensing", "energy", "leakage"]
        for key in required_keys:
            assert key in info, f"Missing key '{key}' in info dict"
        
        check_pass("Advanced models integration")
    except Exception as e:
        check_fail("Advanced models", str(e))
        errors.append("Debug DualISACEnv with advanced config")
        all_ok = False
    
    return all_ok, errors


# ============================================================================
# CHECK 4: Training Pipeline
# ============================================================================

def check_training_pipeline() -> Tuple[bool, List[str]]:
    """Test that SAC training can start"""
    print_section("CHECK 4: Training Pipeline")
    
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        from src.envs.dual_isac_env import DualISACEnv, EnvConfig
    except ImportError as e:
        check_fail("Import training modules", str(e))
        return False, ["Check imports"]
    
    errors = []
    all_ok = True
    
    try:
        print("  Testing SAC training (100 steps)...")
        
        # Create environment
        config = EnvConfig(
            n_users=8,
            n_targets=1,
            alpha=0.5,
            channel_model="rician",
            sensing_model="crb",
            energy_model="aerodynamic",
            secrecy_model="capacity",
            qos_model="queue"
        )
        
        def make_env():
            env = DualISACEnv(cfg=config)
            return Monitor(env)
        
        vec_env = VecMonitor(DummyVecEnv([make_env]))
        
        # Create SAC model
        model = SAC(
            policy="MlpPolicy",
            env=vec_env,
            verbose=0,
            seed=42,
            gamma=0.99,
            buffer_size=1000,  # Small for testing
            batch_size=32,
            learning_rate=3e-4,
        )
        
        # Train for 100 steps
        start = time.time()
        model.learn(total_timesteps=100, progress_bar=False)
        elapsed = time.time() - start
        
        check_pass(f"SAC training (100 steps in {elapsed:.2f}s)")
        
        # Estimate full training time
        steps_per_sec = 100 / elapsed
        full_time_hours = 300_000 / steps_per_sec / 3600
        print(f"     → Estimated time for 300k steps: {full_time_hours:.2f} hours")
        
    except Exception as e:
        check_fail("SAC training", str(e))
        errors.append("Debug SAC training setup")
        all_ok = False
    
    return all_ok, errors


# ============================================================================
# CHECK 5: Evaluation Pipeline
# ============================================================================

def check_evaluation_pipeline() -> Tuple[bool, List[str]]:
    """Test that evaluation and plotting work"""
    print_section("CHECK 5: Evaluation Pipeline")
    
    errors = []
    all_ok = True
    
    # Test eval_rollout
    try:
        from src.eval.eval_rollout import run_model
        check_pass("Import eval_rollout")
    except ImportError as e:
        check_fail("Import eval_rollout", str(e))
        errors.append("Check src/eval/eval_rollout.py")
        all_ok = False
    
    # Test plotting modules
    try:
        import matplotlib.pyplot as plt
        
        # Create a dummy plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        plt.close(fig)
        
        check_pass("Matplotlib plotting")
    except Exception as e:
        check_fail("Matplotlib", str(e))
        errors.append("Check matplotlib installation")
        all_ok = False
    
    return all_ok, errors


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run all validation checks"""
    
    print("\n" + "█"*70)
    print("  PRE-FLIGHT VALIDATION FOR IEEE EXPERIMENTS")
    print("█"*70)
    
    all_checks = []
    all_errors = []
    
    # Run all checks
    checks = [
        check_dependencies,
        check_advanced_models,
        check_environment_integration,
        check_training_pipeline,
        check_evaluation_pipeline,
    ]
    
    for check_func in checks:
        success, errors = check_func()
        all_checks.append(success)
        all_errors.extend(errors)
    
    # Summary
    print_section("VALIDATION SUMMARY")
    
    passed = sum(all_checks)
    total = len(all_checks)
    
    if all(all_checks):
        print("\n  ✅ ALL CHECKS PASSED!")
        print("\n  You are ready to run the full experimental suite.")
        print("\n  Next steps:")
        print("    1. Copy this codebase to the GPU machine")
        print("    2. Run this validation script again on that machine")
        print("    3. Execute: python run_full_experiments.py")
        print("    4. Expected time: ~4-6 hours for full suite")
    else:
        print(f"\n  ⚠️  {passed}/{total} checks passed")
        print("\n  Please fix the following issues:")
        for i, error in enumerate(all_errors, 1):
            print(f"    {i}. {error}")
        print("\n  After fixing, run this script again.")
        sys.exit(1)
    
    print("\n" + "█"*70)


if __name__ == "__main__":
    main()
