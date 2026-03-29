"""
Advanced IEEE-worthy models for UAV-ISAC.

References:
- Channel: Khuwaja et al., "A Survey of Channel Modeling for UAV Communications", IEEE Comms Surveys 2018
- Sensing: Kay, "Fundamentals of Statistical Signal Processing: Estimation Theory"
- Energy: Zeng et al., "Energy-Efficient UAV Communication With Trajectory Optimization", TWC 2017
- Secrecy: Wyner, "The Wire-tap Channel", Bell Sys Tech Journal 1975
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


# ========== ADVANCED CHANNEL MODEL ==========

class RicianChannelModel:
    """
    Rician fading channel with LoS/NLoS transitions for air-to-ground links.
    
    Path loss: PL(d) = PL_0 + 10·α·log10(d) + ξ
    Rician K-factor depends on elevation angle and environment.
    
    IEEE TWC 2018 reference model for UAV channels.
    """
    def __init__(
        self,
        fc_ghz: float = 2.4,
        pl0_db: float = 30.0,
        alpha_los: float = 2.0,
        alpha_nlos: float = 2.8,
        shadowing_std_db: float = 3.0,
        k_factor_los_db: float = 15.0,
        k_factor_nlos_db: float = 0.0,
        environment: str = "suburban",  # suburban, urban, rural
    ):
        self.fc_ghz = fc_ghz
        self.pl0_db = pl0_db
        self.alpha_los = alpha_los
        self.alpha_nlos = alpha_nlos
        self.shadowing_std_db = shadowing_std_db
        self.k_factor_los_db = k_factor_los_db
        self.k_factor_nlos_db = k_factor_nlos_db
        
        # Environment-dependent LoS probability parameters (ITU-R P.1410)
        if environment == "suburban":
            self.a_los = 4.88
            self.b_los = 0.43
        elif environment == "urban":
            self.a_los = 9.61
            self.b_los = 0.16
        else:  # rural
            self.a_los = 0.3
            self.b_los = 750.0
    
    def los_probability(self, elevation_angle_deg: float) -> float:
        """
        Probability of LoS link based on elevation angle.
        Higher elevation → higher LoS probability.
        """
        theta = np.clip(elevation_angle_deg, 0.0, 90.0)
        p_los = 1.0 / (1.0 + self.a_los * np.exp(-self.b_los * (theta - self.a_los)))
        return float(np.clip(p_los, 0.0, 1.0))
    
    def path_loss_db(
        self,
        distance_3d: float,
        elevation_angle_deg: float,
        is_los: bool,
        rng: np.random.Generator,
    ) -> float:
        """
        Compute path loss in dB including shadowing.
        
        PL(d) = PL_0 + 10·α·log10(d) + ξ
        where α depends on LoS/NLoS and ξ ~ N(0, σ²)
        """
        d = max(distance_3d, 1.0)  # minimum 1m
        alpha = self.alpha_los if is_los else self.alpha_nlos
        
        pl = self.pl0_db + 10.0 * alpha * np.log10(d)
        
        # Add log-normal shadowing
        shadowing = rng.normal(0.0, self.shadowing_std_db)
        pl += shadowing
        
        return float(pl)
    
    def rician_fading(
        self,
        k_factor_db: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Generate Rician fading amplitude |h|².
        
        K-factor (dB) determines LoS/scattered power ratio.
        Returns power gain relative to average.
        """
        k_linear = 10.0 ** (k_factor_db / 10.0)
        
        # Rician distribution: LoS + Rayleigh scatter
        # |h|² = (LoS + scatter)²
        los_component = np.sqrt(k_linear / (k_linear + 1.0))
        scatter_std = np.sqrt(1.0 / (2.0 * (k_linear + 1.0)))
        
        h_real = los_component + rng.normal(0.0, scatter_std)
        h_imag = rng.normal(0.0, scatter_std)
        
        power_gain = h_real**2 + h_imag**2
        return float(power_gain)
    
    def compute_snr(
        self,
        distance_3d: float,
        elevation_angle_deg: float,
        tx_power_dbm: float,
        noise_power_dbm: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Compute instantaneous SNR with Rician fading.
        
        SNR = (P_tx · |h|² / PL) / N_0
        """
        # Determine LoS or NLoS
        p_los = self.los_probability(elevation_angle_deg)
        is_los = rng.random() < p_los
        
        # Path loss
        pl_db = self.path_loss_db(distance_3d, elevation_angle_deg, is_los, rng)
        
        # Fading
        k_factor = self.k_factor_los_db if is_los else self.k_factor_nlos_db
        fading_gain = self.rician_fading(k_factor, rng)
        
        # SNR in dB
        snr_db = tx_power_dbm - pl_db - noise_power_dbm + 10.0 * np.log10(fading_gain)
        snr_linear = 10.0 ** (snr_db / 10.0)
        
        return float(np.clip(snr_linear, 1e-6, 1e6))


# ========== CRAMÉR-RAO BOUND SENSING MODEL ==========

class CRBSensingModel:
    """
    Cramér-Rao Lower Bound for bistatic radar target localization.
    
    CRB provides theoretical lower bound on estimation variance:
    var(θ̂) ≥ CRB(θ)
    
    For 2D localization with bistatic geometry:
    CRB ∝ 1 / (SNR · B · T · geometric_factor)
    
    Reference: Kay, "Fundamentals of Statistical Signal Processing", 1993
    """
    def __init__(
        self,
        carrier_freq_ghz: float = 24.0,  # mmWave for sensing
        bandwidth_mhz: float = 100.0,
        integration_time_ms: float = 10.0,
        noise_figure_db: float = 6.0,
        target_rcs_dbsm: float = 10.0,  # radar cross-section
    ):
        self.fc = carrier_freq_ghz * 1e9
        self.B = bandwidth_mhz * 1e6
        self.T = integration_time_ms * 1e-3
        self.nf_db = noise_figure_db
        self.rcs_dbsm = target_rcs_dbsm
        self.wavelength = 3e8 / self.fc
    
    def radar_snr(
        self,
        tx_power_w: float,
        tx_to_target_m: float,
        target_to_rx_m: float,
        tx_gain: float = 10.0,  # dBi
        rx_gain: float = 10.0,
    ) -> float:
        """
        Bistatic radar equation:
        SNR = (P_t · G_t · G_r · λ² · σ) / ((4π)³ · R_tx² · R_rx² · k · T_0 · B · F)
        
        where:
        - σ: target RCS
        - R_tx: transmitter to target distance
        - R_rx: target to receiver distance
        """
        sigma = 10.0 ** (self.rcs_dbsm / 10.0)  # RCS in m²
        g_t = 10.0 ** (tx_gain / 10.0)
        g_r = 10.0 ** (rx_gain / 10.0)
        
        k_boltzmann = 1.38e-23
        T0 = 290.0  # K
        F = 10.0 ** (self.nf_db / 10.0)
        
        numerator = tx_power_w * g_t * g_r * (self.wavelength**2) * sigma
        denominator = (
            (4.0 * np.pi)**3
            * (tx_to_target_m**2)
            * (target_to_rx_m**2)
            * k_boltzmann
            * T0
            * self.B
            * F
        )
        
        snr = numerator / (denominator + 1e-20)
        return float(np.clip(snr, 1e-10, 1e10))
    
    def geometric_dilution_of_precision(
        self,
        uav1_pos: np.ndarray,
        uav2_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> float:
        """
        GDOP measures how geometry affects localization accuracy.
        
        Lower GDOP (wide baseline, good angles) → better localization.
        """
        # Vectors from UAVs to target
        v1 = target_pos[:2] - uav1_pos[:2]
        v2 = target_pos[:2] - uav2_pos[:2]
        
        d1 = np.linalg.norm(v1) + 1e-6
        d2 = np.linalg.norm(v2) + 1e-6
        
        # Unit vectors
        u1 = v1 / d1
        u2 = v2 / d2
        
        # Angle between vectors (wider is better)
        cos_angle = np.clip(np.dot(u1, u2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # GDOP inversely proportional to sin(angle)
        # Optimal at 90 degrees, worst at 0/180
        gdop = 1.0 / (np.abs(np.sin(angle)) + 0.1)
        return float(gdop)
    
    def compute_crb_position_error(
        self,
        uav1_pos: np.ndarray,
        uav2_pos: np.ndarray,
        target_pos: np.ndarray,
        tx_power_w: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute CRB-based position error (lower bound on RMSE).
        
        Returns:
        - crb_meters: Position error bound in meters
        - sensing_score: Normalized score [0,1], higher is better
        """
        # Bistatic distances
        d_tx_target = np.linalg.norm(uav1_pos[:2] - target_pos[:2])
        d_target_rx = np.linalg.norm(uav2_pos[:2] - target_pos[:2])
        
        # Radar SNR
        snr = self.radar_snr(tx_power_w, d_tx_target, d_target_rx)
        
        # GDOP
        gdop = self.geometric_dilution_of_precision(uav1_pos, uav2_pos, target_pos)
        
        # CRB formula (simplified for 2D)
        # CRB ∝ (c / (2·B)) · (GDOP / √SNR)
        c = 3e8
        range_resolution = c / (2.0 * self.B)
        
        crb = range_resolution * gdop / np.sqrt(snr + 1e-10)
        crb_meters = float(np.clip(crb, 0.1, 1000.0))
        
        # Convert to score: lower CRB → higher score
        # Use tanh for smooth saturation
        sensing_score = float(np.tanh(50.0 / crb_meters))
        
        return crb_meters, sensing_score


# ========== INFORMATION-THEORETIC SECRECY ==========

class SecrecyCapacityModel:
    """
    Physical layer security based on Wyner's secrecy capacity.
    
    C_s = [C_main - C_eve]⁺
    
    where:
    - C_main: legitimate user channel capacity
    - C_eve: eavesdropper channel capacity
    - [x]⁺ = max(0, x)
    
    Artificial noise and cooperative jamming increase eavesdropper's interference.
    
    Reference: Wyner, "The Wire-tap Channel", Bell Sys Tech Journal 1975
    """
    def __init__(
        self,
        noise_power_dbm: float = -90.0,
        an_power_fraction: float = 0.3,
    ):
        self.noise_power_dbm = noise_power_dbm
        self.an_fraction = an_power_fraction
    
    def compute_secrecy_rate(
        self,
        snr_user: float,
        snr_eve: float,
        an_power_allocated: float,
        jammer_active: bool,
        jammer_to_eve_gain: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Compute secrecy capacity and normalized score.
        
        Args:
        - snr_user: SNR at legitimate user
        - snr_eve: SNR at eavesdropper
        - an_power_allocated: [0,1] fraction of power for artificial noise
        - jammer_active: whether cooperative jammer is on
        - jammer_to_eve_gain: jamming signal strength at eavesdropper
        
        Returns:
        - secrecy_rate: bits/s/Hz
        - secrecy_score: normalized [0,1]
        """
        # Main channel capacity
        C_main = np.log2(1.0 + snr_user * (1.0 - an_power_allocated))
        
        # Eavesdropper's interference from AN
        an_interference = an_power_allocated * snr_eve * 5.0  # AN degrades eve
        
        # Jammer adds extra interference
        if jammer_active:
            jammer_interference = jammer_to_eve_gain * 10.0
        else:
            jammer_interference = 0.0
        
        # Eavesdropper's effective SINR
        sinr_eve = snr_eve / (1.0 + an_interference + jammer_interference)
        C_eve = np.log2(1.0 + sinr_eve)
        
        # Secrecy capacity
        secrecy_rate = float(np.maximum(0.0, C_main - C_eve))
        
        # Normalized score (higher is better)
        secrecy_score = float(np.tanh(secrecy_rate / 2.0))
        
        return secrecy_rate, secrecy_score


# ========== REALISTIC UAV ENERGY MODEL ==========

class UAVEnergyModel:
    """
    Realistic rotary-wing UAV power consumption model.
    
    Based on Zeng et al., IEEE TWC 2017:
    P_total = P_blade + P_induced + P_parasite + P_tx
    
    - P_blade: blade profile power (constant)
    - P_induced: induced power for lift (speed-dependent)
    - P_parasite: parasitic drag power ∝ v³
    - P_tx: transmission power
    
    Typical quadcopter: 200-400W total power
    """
    def __init__(
        self,
        mass_kg: float = 2.0,
        rotor_radius_m: float = 0.2,
        air_density: float = 1.225,  # kg/m³
        blade_profile_power_w: float = 50.0,
        induced_power_hover_w: float = 150.0,
        drag_coefficient: float = 0.3,
        battery_capacity_wh: float = 100.0,
    ):
        self.m = mass_kg
        self.R = rotor_radius_m
        self.rho = air_density
        self.P_blade = blade_profile_power_w
        self.P_hover = induced_power_hover_w
        self.Cd = drag_coefficient
        self.battery_wh = battery_capacity_wh
        
        # Derived parameters
        self.g = 9.81
        self.A_rotor = np.pi * (self.R ** 2)
        self.U_tip = 200.0  # tip speed m/s (typical)
        
    def power_consumption(
        self,
        speed_mps: float,
        altitude_rate_mps: float,
        tx_power_w: float = 1.0,
    ) -> float:
        """
        Total power consumption in Watts.
        
        P_total = P_blade + P_induced + P_parasite + P_climb + P_tx
        """
        v = np.abs(speed_mps)
        v_z = np.abs(altitude_rate_mps)
        
        # Blade profile power (constant)
        P_blade = self.P_blade
        
        # Induced power (thrust for lift + forward flight)
        # Simplified: P_induced = P_hover · √(1 + (v/v_h)²)
        v_h = 5.0  # characteristic velocity
        P_induced = self.P_hover * np.sqrt(1.0 + (v / v_h)**2)
        
        # Parasite drag power ∝ v³
        # P_parasite = 0.5 · ρ · Cd · A · v³
        A_frontal = 0.05  # m² frontal area
        P_parasite = 0.5 * self.rho * self.Cd * A_frontal * (v**3)
        
        # Climbing power
        P_climb = self.m * self.g * v_z
        
        # Transmission power
        P_tx = tx_power_w
        
        P_total = P_blade + P_induced + P_parasite + P_climb + P_tx
        
        return float(np.clip(P_total, 0.0, 1000.0))
    
    def energy_consumed_normalized(
        self,
        speeds: np.ndarray,
        altitude_rates: np.ndarray,
        tx_powers: np.ndarray,
        dt: float = 1.0,
    ) -> float:
        """
        Compute normalized energy consumption [0,1] for multiple UAVs.
        
        Returns fraction of battery used per timestep.
        """
        total_power = 0.0
        for i in range(len(speeds)):
            P = self.power_consumption(speeds[i], altitude_rates[i], tx_powers[i])
            total_power += P
        
        # Energy in Wh
        energy_wh = (total_power * dt) / 3600.0
        
        # Normalized by battery capacity
        energy_fraction = energy_wh / (self.battery_wh + 1e-6)
        
        return float(np.clip(energy_fraction, 0.0, 1.0))


# ========== QOS MODEL WITH QUEUE THEORY ==========

class QoSQueueModel:
    """
    Queue-theoretic user service model.
    
    Each user has a packet queue with arrival rate λ.
    Service rate μ depends on allocated data rate.
    
    Queue backlog evolves as:
    Q(t+1) = max(0, Q(t) + A(t) - S(t))
    
    QoS metric: average queue length (lower is better).
    """
    def __init__(
        self,
        arrival_rate_mbps: float = 1.0,
        max_queue_size_mb: float = 10.0,
    ):
        self.lambda_arrival = arrival_rate_mbps
        self.Q_max = max_queue_size_mb
    
    def update_queue(
        self,
        current_queue_mb: float,
        service_rate_mbps: float,
        dt: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Update queue state and compute QoS metric.
        
        Returns:
        - new_queue: updated queue length
        - qos_score: [0,1], higher is better (lower queue)
        """
        # Arrivals in time dt
        arrivals = self.lambda_arrival * dt
        
        # Service in time dt
        service = service_rate_mbps * dt
        
        # Queue update
        new_queue = np.clip(current_queue_mb + arrivals - service, 0.0, self.Q_max)
        
        # QoS score: low queue is good
        qos_score = 1.0 - (new_queue / self.Q_max)
        
        return float(new_queue), float(qos_score)
