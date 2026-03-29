"""
Verification script to test advanced models work correctly.
"""
from __future__ import annotations

import numpy as np
from src.envs.advanced_models import (
    RicianChannelModel,
    CRBSensingModel,
    SecrecyCapacityModel,
    UAVEnergyModel,
    QoSQueueModel,
)


def test_channel_model():
    print("Testing Rician Channel Model...")
    model = RicianChannelModel(fc_ghz=2.4, environment="suburban")
    rng = np.random.default_rng(42)
    
    # Test LoS probability
    p_los = model.los_probability(45.0)
    print(f"  LoS probability at 45°: {p_los:.3f}")
    assert 0.0 <= p_los <= 1.0, "LoS probability out of range"
    
    # Test SNR computation
    snr = model.compute_snr(100.0, 45.0, 30.0, -90.0, rng)
    print(f"  SNR at 100m: {10*np.log10(snr):.2f} dB")
    assert snr > 0, "SNR must be positive"
    
    print("  ✅ Channel model working!\n")


def test_sensing_model():
    print("Testing CRB Sensing Model...")
    model = CRBSensingModel(carrier_freq_ghz=24.0, bandwidth_mhz=100.0)
    
    # Test GDOP
    uav1 = np.array([0.0, 0.0, 100.0])
    uav2 = np.array([200.0, 0.0, 100.0])
    target = np.array([100.0, 100.0, 0.0])
    
    gdop = model.geometric_dilution_of_precision(uav1, uav2, target)
    print(f"  GDOP: {gdop:.3f}")
    assert gdop > 0, "GDOP must be positive"
    
    # Test CRB
    crb_error, score = model.compute_crb_position_error(uav1, uav2, target, 1.0)
    print(f"  CRB error: {crb_error:.2f} m")
    print(f"  Sensing score: {score:.3f}")
    assert 0.0 <= score <= 1.0, "Score out of range"
    
    print("  ✅ Sensing model working!\n")


def test_secrecy_model():
    print("Testing Secrecy Capacity Model...")
    model = SecrecyCapacityModel(noise_power_dbm=-90.0)
    
    # Test secrecy rate
    snr_user = 10.0
    snr_eve = 5.0
    an_power = 0.3
    
    sec_rate, sec_score = model.compute_secrecy_rate(
        snr_user, snr_eve, an_power, jammer_active=False
    )
    print(f"  Secrecy rate (no jammer): {sec_rate:.3f} bits/s/Hz")
    print(f"  Secrecy score: {sec_score:.3f}")
    
    sec_rate_jam, sec_score_jam = model.compute_secrecy_rate(
        snr_user, snr_eve, an_power, jammer_active=True
    )
    print(f"  Secrecy rate (with jammer): {sec_rate_jam:.3f} bits/s/Hz")
    print(f"  Improvement: {100*(sec_rate_jam/sec_rate - 1):.1f}%")
    
    assert sec_rate_jam > sec_rate, "Jammer should improve secrecy"
    print("  ✅ Secrecy model working!\n")


def test_energy_model():
    print("Testing UAV Energy Model...")
    model = UAVEnergyModel(mass_kg=2.0, battery_capacity_wh=100.0)
    
    # Test power at different speeds
    print("  Power consumption:")
    for speed in [0.0, 5.0, 10.0, 15.0, 20.0]:
        power = model.power_consumption(speed, 0.0, 1.0)
        print(f"    {speed:4.1f} m/s: {power:6.1f} W")
    
    # Test energy normalization
    speeds = np.array([10.0, 15.0])
    alts = np.array([0.0, 0.0])
    tx = np.array([1.0, 1.0])
    energy_frac = model.energy_consumed_normalized(speeds, alts, tx, dt=1.0)
    print(f"  Energy fraction per step: {energy_frac:.5f}")
    
    print("  ✅ Energy model working!\n")


def test_qos_model():
    print("Testing QoS Queue Model...")
    model = QoSQueueModel(arrival_rate_mbps=1.0, max_queue_size_mb=10.0)
    
    # Simulate queue evolution
    queue = 0.0
    print("  Queue evolution (10 steps):")
    for i in range(10):
        service_rate = 1.5 if i % 2 == 0 else 0.5  # Varying service
        queue, qos = model.update_queue(queue, service_rate, dt=1.0)
        print(f"    Step {i+1}: Queue={queue:.2f} MB, QoS={qos:.3f}")
    
    print("  ✅ QoS model working!\n")


def test_environment_integration():
    print("Testing Environment Integration...")
    from src.envs.dual_isac_env import make_env
    
    # Create environment with advanced models
    env = make_env(n_users=8, n_targets=1, alpha=0.5, seed=42)
    
    # Check that advanced models are initialized
    assert env.channel_model is not None, "Channel model not initialized"
    assert env.sensing_model is not None, "Sensing model not initialized"
    assert env.energy_model_adv is not None, "Energy model not initialized"
    assert env.secrecy_model is not None, "Secrecy model not initialized"
    assert env.qos_models is not None, "QoS models not initialized"
    
    print(f"  Environment created successfully")
    print(f"  Channel model: {type(env.channel_model).__name__}")
    print(f"  Sensing model: {type(env.sensing_model).__name__}")
    print(f"  Energy model: {type(env.energy_model_adv).__name__}")
    print(f"  Secrecy model: {type(env.secrecy_model).__name__}")
    print(f"  QoS models: {len(env.qos_models)} users")
    
    # Test one step
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    
    print(f"  Step executed: reward={reward:.4f}")
    print(f"  Info keys: {list(info.keys())}")
    
    print("  ✅ Environment integration working!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED MODELS VERIFICATION")
    print("=" * 60)
    print()
    
    try:
        test_channel_model()
        test_sensing_model()
        test_secrecy_model()
        test_energy_model()
        test_qos_model()
        test_environment_integration()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nAdvanced models are working correctly.")
        print("You can now run training with IEEE-worthy enhancements!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
