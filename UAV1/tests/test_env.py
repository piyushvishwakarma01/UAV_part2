from __future__ import annotations

import numpy as np

from src.envs.dual_isac_env import make_env, EnvConfig, DualISACEnv


def test_reset_shapes():
    env = make_env(n_users=8, n_targets=1, alpha=0.5, seed=0)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))


def test_step_zero_action():
    env = make_env(n_users=8, n_targets=1, alpha=0.5, seed=0)
    obs, _ = env.reset()
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, rew, term, trunc, info = env.step(act)
    assert np.isfinite(rew)
    assert (rew >= -1.0) and (rew <= 1.0)
    assert env.t == 1


def test_safety_penalty_close_uavs():
    env: DualISACEnv = make_env(n_users=8, n_targets=1, alpha=0.5, seed=0)
    env.reset()
    # Force UAVs too close
    env.uav_pos[0] = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    env.uav_pos[1] = np.array([100.0, 120.0, 100.0], dtype=np.float32)  # 20 m apart
    obs, rew, term, trunc, info = env.step(np.zeros(10, dtype=np.float32))
    assert info["safety"] >= env.cfg.safety_mu
    assert -1.0 <= rew <= 1.0


