from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np

from .utils import (
    AreaSpec,
    baseline_geometry_score,
    clamp_position,
    dist3d,
    energy_model,
    no_fly_violations,
    normalize_obs,
    secrecy_leakage_proxy,
    seed_everything,
    wrap_angle,
)

from .advanced_models import (
    RicianChannelModel,
    CRBSensingModel,
    SecrecyCapacityModel,
    UAVEnergyModel,
    QoSQueueModel,
)


@dataclass      
class EnvConfig:
    area_size: float = 2000.0
    zmin: float = 80.0
    zmax: float = 180.0
    dt: float = 1.0
    vmax: float = 25.0
    kc: float = 1.0
    horizon: int = 400
    alpha: float = 0.5
    energy_lambda: float = 0.05
    safety_mu: float = 0.2
    leakage_eta: float = 0.1
    min_separation: float = 50.0
    no_fly_enabled: bool = False
    jammer_threshold: float = 0.7
    n_users: int = 8
    n_targets: int = 1
    seed: Optional[int] = None
    
    # Advanced model selection
    channel_model: str = "rician"  # "simple" or "rician"
    sensing_model: str = "crb"  # "geometric" or "crb"
    energy_model: str = "aerodynamic"  # "simple" or "aerodynamic"
    secrecy_model: str = "capacity"  # "proxy" or "capacity"
    qos_model: str = "queue"  # "deficit" or "queue"
    
    # Channel parameters
    carrier_freq_ghz: float = 2.4
    tx_power_dbm: float = 30.0
    noise_power_dbm: float = -90.0
    environment: str = "suburban"
    
    # Sensing parameters
    sensing_carrier_freq_ghz: float = 24.0
    sensing_bandwidth_mhz: float = 100.0
    target_rcs_dbsm: float = 10.0
    
    # Energy parameters
    uav_mass_kg: float = 2.0
    battery_capacity_wh: float = 100.0
    
    # Secrecy parameters
    an_power_fraction: float = 0.3
    
    # QoS parameters
    arrival_rate_mbps: float = 1.0
    max_queue_size_mb: float = 10.0


class DualISACEnv(gym.Env):
    metadata = {"render_modes": ["none"], "render_fps": 10}

    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = seed_everything(cfg.seed)
        self.area = AreaSpec(width=cfg.area_size, height=cfg.area_size, zmin=cfg.zmin, zmax=cfg.zmax)

        # Initialize advanced models based on configuration
        if cfg.channel_model == "rician":
            self.channel_model = RicianChannelModel(
                fc_ghz=cfg.carrier_freq_ghz,
                environment=cfg.environment,
            )
        else:
            self.channel_model = None
        
        if cfg.sensing_model == "crb":
            self.sensing_model = CRBSensingModel(
                carrier_freq_ghz=cfg.sensing_carrier_freq_ghz,
                bandwidth_mhz=cfg.sensing_bandwidth_mhz,
                target_rcs_dbsm=cfg.target_rcs_dbsm,
            )
        else:
            self.sensing_model = None
        
        if cfg.energy_model == "aerodynamic":
            self.energy_model_adv = UAVEnergyModel(
                mass_kg=cfg.uav_mass_kg,
                battery_capacity_wh=cfg.battery_capacity_wh,
            )
        else:
            self.energy_model_adv = None
        
        if cfg.secrecy_model == "capacity":
            self.secrecy_model = SecrecyCapacityModel(
                noise_power_dbm=cfg.noise_power_dbm,
                an_power_fraction=cfg.an_power_fraction,
            )
        else:
            self.secrecy_model = None
        
        if cfg.qos_model == "queue":
            self.qos_models = [
                QoSQueueModel(
                    arrival_rate_mbps=cfg.arrival_rate_mbps,
                    max_queue_size_mb=cfg.max_queue_size_mb,
                )
                for _ in range(cfg.n_users)
            ]
            self.user_queues = np.zeros(cfg.n_users, dtype=np.float32)
        else:
            self.qos_models = None
            self.user_queues = None

        # Observations: build fixed-size vector per n_users/n_targets
        self.n_users = cfg.n_users
        self.n_targets = cfg.n_targets

        # UAV state (2 x 5: x,y,z,heading,battery)
        uav_dim = 2 * 5
        users_dim = self.n_users * 3  # (x,y,deficit/queue)
        targets_dim = self.n_targets * 3  # (x,y,confidence)
        time_dim = 1
        nofly_dim = 100 if cfg.no_fly_enabled else 0
        self.obs_dim = uav_dim + users_dim + targets_dim + time_dim + nofly_dim

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # Action per UAV: 5 dims, total 10 dims
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        # Static users and targets
        self.users_xy = self.rng.uniform(0.1 * self.cfg.area_size, 0.9 * self.cfg.area_size, size=(self.n_users, 2))
        self.user_deficits = np.ones(self.n_users, dtype=np.float32)  # start with high deficit
        self.targets_xy = self.rng.uniform(0.2 * self.cfg.area_size, 0.8 * self.cfg.area_size, size=(self.n_targets, 2))
        self.target_conf = np.zeros(self.n_targets, dtype=np.float32)

        # Eavesdropper sector (angle ±spread)
        self.eaves_angle = float(self.rng.uniform(-np.pi, np.pi))
        self.eaves_spread = float(np.deg2rad(30.0))
        self.eaves_pos = self.rng.uniform(0.1 * self.cfg.area_size, 0.9 * self.cfg.area_size, size=(2,))

        # No-fly mask
        if self.cfg.no_fly_enabled:
            mask = self.rng.choice([0, 1], size=(100,), p=[0.9, 0.1])
        else:
            mask = np.zeros(100, dtype=np.int32)
        self.nofly_mask = mask

        # State holders
        self.uav_pos = np.zeros((2, 3), dtype=np.float32)
        self.uav_heading = np.zeros(2, dtype=np.float32)
        self.uav_battery = np.ones(2, dtype=np.float32)
        self.t = 0

    def _spawn_uavs(self) -> None:
        # Place UAVs at opposite corners at mid altitude
        z0 = 0.5 * (self.cfg.zmin + self.cfg.zmax)
        self.uav_pos[0] = np.array([0.25 * self.cfg.area_size, 0.25 * self.cfg.area_size, z0], dtype=np.float32)
        self.uav_pos[1] = np.array([0.75 * self.cfg.area_size, 0.75 * self.cfg.area_size, z0], dtype=np.float32)
        self.uav_heading[:] = 0.0
        self.uav_battery[:] = 1.0

    def _obs(self) -> np.ndarray:
        # Build raw obs vector
        uav_raw = np.concatenate([
            self.uav_pos.reshape(-1),
            self.uav_heading,
            self.uav_battery,
        ])  # length 10
        users_raw = np.concatenate([self.users_xy.reshape(-1), self.user_deficits])  # 2N + N
        targets_raw = np.concatenate([self.targets_xy.reshape(-1), self.target_conf])  # 2T + T
        time_left = np.array([1.0 - self.t / max(1, self.cfg.horizon)], dtype=np.float32)
        if self.cfg.no_fly_enabled:
            raw = np.concatenate([uav_raw, users_raw, targets_raw, time_left, self.nofly_mask])
        else:
            raw = np.concatenate([uav_raw, users_raw, targets_raw, time_left])

        # Construct normalization bounds
        lows = []
        highs = []
        # uav: x,y ∈ [0,area], z ∈ [zmin,zmax], heading ∈ [-pi,pi], battery ∈ [0,1]
        for _ in range(2):
            lows += [0.0, 0.0, self.cfg.zmin]
            highs += [self.cfg.area_size, self.cfg.area_size, self.cfg.zmax]
        lows += [-np.pi, -np.pi]
        highs += [np.pi, np.pi]
        lows += [0.0, 0.0]
        highs += [1.0, 1.0]
        # users: x,y,deficit
        for _ in range(self.n_users):
            lows += [0.0, 0.0, 0.0]
            highs += [self.cfg.area_size, self.cfg.area_size, 1.0]
        # targets: x,y,confidence
        for _ in range(self.n_targets):
            lows += [0.0, 0.0, 0.0]
            highs += [self.cfg.area_size, self.cfg.area_size, 1.0]
        lows += [0.0]
        highs += [1.0]
        if self.cfg.no_fly_enabled:
            lows += [0.0] * 100
            highs += [1.0] * 100

        low = np.asarray(lows, dtype=np.float32)
        high = np.asarray(highs, dtype=np.float32)
        obs = normalize_obs(raw.astype(np.float32), low, high)
        assert obs.shape == (self.obs_dim,), f"Obs shape mismatch: {obs.shape}"
        return obs

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        if seed is not None:
            self.rng = seed_everything(seed)
        self.t = 0
        self._spawn_uavs()
        # Re-randomize users/targets per episode for diversity
        self.users_xy = self.rng.uniform(0.1 * self.cfg.area_size, 0.9 * self.cfg.area_size, size=(self.n_users, 2))
        self.user_deficits[:] = 1.0
        if self.user_queues is not None:
            self.user_queues[:] = 0.0
        self.targets_xy = self.rng.uniform(0.2 * self.cfg.area_size, 0.8 * self.cfg.area_size, size=(self.n_targets, 2))
        self.target_conf[:] = 0.0
        self.eaves_angle = float(self.rng.uniform(-np.pi, np.pi))
        self.eaves_pos = self.rng.uniform(0.1 * self.cfg.area_size, 0.9 * self.cfg.area_size, size=(2,))
        obs = self._obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (10,):
            raise ValueError(f"Action shape must be (10,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        a1 = action[:5]
        a2 = action[5:]

        # Extract per-UAV controls
        def apply(uav_idx: int, a: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
            d_heading = float(a[0]) * np.deg2rad(30.0)  # at most 30 deg/s turn
            speed = float(np.clip((a[1] + 1.0) * 0.5, 0.0, 1.0)) * self.cfg.vmax
            dz = float(a[2]) * 5.0  # ±5 m/s vertical
            time_sense = float(np.clip((a[3] + 1.0) * 0.5, 0.0, 1.0))
            power_an = float(np.clip((a[4] + 1.0) * 0.5, 0.0, 1.0))

            # Update heading
            self.uav_heading[uav_idx] = float(wrap_angle(self.uav_heading[uav_idx] + d_heading))

            # Position update in xy
            dx = speed * np.cos(self.uav_heading[uav_idx]) * self.cfg.dt
            dy = speed * np.sin(self.uav_heading[uav_idx]) * self.cfg.dt
            dz_dt = dz * self.cfg.dt
            new_pos = self.uav_pos[uav_idx] + np.array([dx, dy, dz_dt], dtype=np.float32)
            new_pos = clamp_position(new_pos, self.area)
            self.uav_pos[uav_idx] = new_pos

            return new_pos, speed, dz_dt, time_sense, power_an

        p1, s1, dz1, t1, an1 = apply(0, a1)
        p2, s2, dz2, t2, an2 = apply(1, a2)

        avg_time_sense = 0.5 * (t1 + t2)
        avg_power_an = 0.5 * (an1 + an2)

        # ========== COMMUNICATION UTILITY ==========
        # Assign users to nearest UAV and compute SNR
        d1 = np.linalg.norm(self.users_xy - p1[:2], axis=1)
        d2_ = np.linalg.norm(self.users_xy - p2[:2], axis=1)
        nearest = (d1 <= d2_).astype(np.float32)
        d3d1 = np.sqrt(d1**2 + (p1[2])**2)
        d3d2 = np.sqrt(d2_**2 + (p2[2])**2)
        
        # Use advanced channel model if enabled
        if self.channel_model is not None:
            # Rician fading channel with realistic propagation
            snr1 = np.zeros(self.n_users)
            snr2 = np.zeros(self.n_users)
            tx_power_w = 10.0 ** ((self.cfg.tx_power_dbm - 30.0) / 10.0)  # dBm to Watts
            
            for i in range(self.n_users):
                # Elevation angles
                elev1 = np.rad2deg(np.arctan(p1[2] / (d1[i] + 1e-6)))
                elev2 = np.rad2deg(np.arctan(p2[2] / (d2_[i] + 1e-6)))
                
                # Compute SNR with fading
                snr1[i] = self.channel_model.compute_snr(
                    d3d1[i], elev1, 
                    self.cfg.tx_power_dbm * (1.0 - an1),  # Reduce power by AN
                    self.cfg.noise_power_dbm, 
                    self.rng
                )
                snr2[i] = self.channel_model.compute_snr(
                    d3d2[i], elev2,
                    self.cfg.tx_power_dbm * (1.0 - an2),
                    self.cfg.noise_power_dbm,
                    self.rng
                )
        else:
            # Simple free-space path loss model (baseline)
            snr1 = self.cfg.kc / (d3d1**2 + 1.0)
            snr2 = self.cfg.kc / (d3d2**2 + 1.0)
        
        # Select SNR based on nearest UAV
        snr = nearest * snr1 + (1 - nearest) * snr2
        sum_rate = float(np.sum(np.log2(1.0 + snr)))
        
        # Update QoS: queue model or simple deficit
        if self.qos_models is not None:
            # Queue-theoretic model
            qos_total = 0.0
            for i in range(self.n_users):
                service_rate = np.log2(1.0 + snr[i])  # Shannon capacity
                new_queue, qos_score = self.qos_models[i].update_queue(
                    self.user_queues[i], service_rate, self.cfg.dt
                )
                self.user_queues[i] = new_queue
                qos_total += qos_score
            qos_metric = qos_total / max(1, self.n_users)
        else:
            # Simple deficit model (original)
            served = (snr > 0.5).astype(np.float32)
            self.user_deficits = np.clip(self.user_deficits - 0.01 * served, 0.0, 1.0)
            qos_metric = 1.0 - np.mean(self.user_deficits)
        
        # Communication utility scaled by time and AN allocation
        comms_util = sum_rate * (1.0 - avg_time_sense) * (1.0 - avg_power_an) * 0.1  # Scale factor

        # ========== SENSING UTILITY ==========
        sense_total = 0.0
        if self.sensing_model is not None:
            # CRB-based sensing model (IEEE-worthy)
            for k in range(self.n_targets):
                tgt = np.array([self.targets_xy[k, 0], self.targets_xy[k, 1], 0.0], dtype=np.float32)
                tx_power_w = max(0.01, avg_time_sense) * 1.0  # Allocate power based on time split, avoid 0
                crb_error, sensing_score = self.sensing_model.compute_crb_position_error(
                    self.uav_pos[0], self.uav_pos[1], tgt, tx_power_w
                )
                # sensing_score directly from CRBSensingModel might be low magnitude, ensure it scales reasonably
                sense_total += sensing_score * 10.0
        else:
            # Geometric heuristic (original baseline)
            for k in range(self.n_targets):
                tgt = np.array([self.targets_xy[k, 0], self.targets_xy[k, 1], 0.0], dtype=np.float32)
                prox = np.tanh(400.0 / (1e-6 + 0.5 * (np.linalg.norm(p1[:2] - tgt[:2]) + np.linalg.norm(p2[:2] - tgt[:2]))))
                base = baseline_geometry_score(self.uav_pos, tgt)
                sense_total += 0.5 * prox + 0.5 * base
        
        sensing_util = float(sense_total / max(1, self.n_targets)) * max(0.01, avg_time_sense)
        
        # Update target confidence
        self.target_conf = np.clip(self.target_conf + 0.01 * avg_time_sense, 0.0, 1.0)

        # ========== ENERGY MODEL ==========
        if self.energy_model_adv is not None:
            # Aerodynamic UAV energy model (IEEE-worthy)
            tx_powers = np.array([1.0 - an1, 1.0 - an2])  # Power allocated to transmission
            energy = self.energy_model_adv.energy_consumed_normalized(
                np.array([s1, s2]), np.array([dz1, dz2]), tx_powers, self.cfg.dt
            )
        else:
            # Simple linear energy model (baseline)
            energy = energy_model(np.array([s1, s2]), np.array([dz1, dz2]), np.array([1.0 - an1, 1.0 - an2]))

        # Safety violations
        sep = dist3d(self.uav_pos[0], self.uav_pos[1])
        sep_violation = 1.0 if sep < self.cfg.min_separation else 0.0
        bounds_violation = float(
            np.any(self.uav_pos[:, 0] <= 0.0)
            or np.any(self.uav_pos[:, 0] >= self.cfg.area_size)
            or np.any(self.uav_pos[:, 1] <= 0.0)
            or np.any(self.uav_pos[:, 1] >= self.cfg.area_size)
        )
        nofly_violation = 0.0
        if self.cfg.no_fly_enabled:
            nofly_violation = float(no_fly_violations(self.uav_pos[:, :2], self.nofly_mask, self.area) > 0)
        safety_pen = self.cfg.safety_mu * (sep_violation + bounds_violation + nofly_violation)

        # ========== SECRECY MODEL ==========
        jammer_active = (an2 >= self.cfg.jammer_threshold)
        
        if self.secrecy_model is not None:
            # Information-theoretic secrecy capacity (IEEE-worthy)
            # Compute eavesdropper's SNR
            eaves_3d = np.array([self.eaves_pos[0], self.eaves_pos[1], 0.0], dtype=np.float32)
            d_eve_1 = np.linalg.norm(p1[:2] - eaves_3d[:2])
            d_eve_2 = np.linalg.norm(p2[:2] - eaves_3d[:2])
            d3d_eve = np.sqrt(min(d_eve_1, d_eve_2)**2 + ((p1[2] + p2[2])/2)**2)
            
            if self.channel_model is not None:
                elev_eve = np.rad2deg(np.arctan(((p1[2] + p2[2])/2) / (d3d_eve + 1e-6)))
                snr_eve = self.channel_model.compute_snr(
                    d3d_eve, elev_eve, self.cfg.tx_power_dbm, self.cfg.noise_power_dbm, self.rng
                )
            else:
                snr_eve = self.cfg.kc / (d3d_eve**2 + 1.0)
            
            # Average user SNR
            snr_user_avg = float(np.mean(snr))
            
            # Compute secrecy rate
            jammer_to_eve_gain = 0.8 if jammer_active else 0.0
            secrecy_rate, secrecy_score = self.secrecy_model.compute_secrecy_rate(
                snr_user_avg, snr_eve, avg_power_an, jammer_active, jammer_to_eve_gain
            )
            leakage = 1.0 - secrecy_score  # Convert to leakage (lower is better)
        else:
            # Simple proxy model (baseline)
            leakage = secrecy_leakage_proxy(
                self.uav_heading, np.array([an1, an2]), self.eaves_angle, self.eaves_spread, jammer_active
            )

        # ========== REWARD COMPUTATION ==========
        reward = (
            self.cfg.alpha * comms_util
            + (1.0 - self.cfg.alpha) * sensing_util
            - self.cfg.energy_lambda * energy
            - self.cfg.leakage_eta * leakage
            - safety_pen
        )
        reward = float(np.clip(reward, -1.0, 1.0))

        # Battery drain
        self.uav_battery = np.clip(self.uav_battery - 0.001 - 0.0005 * np.array([s1, s2]) / self.cfg.vmax, 0.0, 1.0)

        self.t += 1
        terminated = self.t >= self.cfg.horizon or np.any(self.uav_battery <= 0.05)
        truncated = False
        info: Dict = {
            "sum_rate": sum_rate,
            "sensing": sensing_util,
            "energy": energy,
            "leakage": leakage,
            "safety": safety_pen,
            "jammer_duty_cycle": 1.0 if jammer_active else 0.0,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return None


def make_env(n_users: int = 8, n_targets: int = 1, alpha: float = 0.5, seed: int | None = None) -> DualISACEnv:
    cfg = EnvConfig(n_users=n_users, n_targets=n_targets, alpha=alpha, seed=seed)
    return DualISACEnv(cfg)


