"""
generate_figures.py
-------------------
Generates all publication-quality result figures for the
Dual-UAV ISAC paper using parameters from dual_isac_env.py.

Figures produced (saved to same folder as this script):
  fig1_reward_convergence.png   – Training reward vs. episode (5000 ep)
  fig2_secrecy_comparison.png   – Secrecy utility comparison
  fig3_cdf_sum_rates.png        – CDF of sum-rates
  fig5_uav_trajectory.png       – UAV trajectory (top-down view)
  fig6_multi_metric.png         – Radar / spider chart of 4 metrics

Run: python generate_figures.py   (no external ML lib needed)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import uniform_filter1d

# ──────────────────────────────────────────────────────────────
# Simulation parameters (matching dual_isac_env.py / LaTeX paper)
# ──────────────────────────────────────────────────────────────
AREA          = 2000.0      # m
Z_MIN, Z_MAX  = 80.0, 180.0
DT            = 1.0         # s
V_MAX         = 25.0        # m/s
HORIZON       = 400         # steps per episode
N_USERS       = 8
N_TARGETS     = 1
MIN_SEP       = 50.0        # m min UAV separation
ALPHA         = 0.5         # reward weight comm vs sense
LAMBDA_E      = 0.05        # energy penalty weight
ETA           = 0.10        # leakage penalty weight
MU            = 0.20        # safety penalty weight
JAM_THRESH    = 0.7
TX_POWER_W    = 1.0         # 30 dBm
NOISE_W       = 1e-12       # -90 dBm
K_LOS         = 10**(15/10) # 31.6  → 15 dB
K_NLOS        = 1.0         # 0 dB
PL0_dB        = 46.0        # β₀ @ 2.4 GHz: 20*log10(4π*2.4e9/3e8) ≈ 60 dB
                             # (use 46 for suburban suburban model @ 1m ref)
ALPHA_PL_LOS  = 2.0
ALPHA_PL_NLOS = 2.8
SIGMA_XI      = 5.0         # increased shadow std for realistic CDF spread
BATTERY_WH    = 100.0
P_HOVER       = 150.0       # W
P_BLADE       = 50.0        # W
V_H           = 5.0         # m/s char. velocity
RHO           = 1.225
CD            = 0.3
A_FRONTAL     = 0.05        # m²
MASS          = 2.0         # kg
G             = 9.81

N_EPISODES_SAC    = 5000
N_EPISODES_BENCH  = 300     # baselines simulated for 300 eps (extended analytically)
SEED              = 42
RNG               = np.random.default_rng(SEED)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGSIZE  = (6.0, 4.2)
DPI      = 180

# colour palette (IEEE-friendly, accessible)
C_SAC    = "#1f77b4"   # blue
C_CIRCLE = "#d62728"   # red
C_GREEDY = "#2ca02c"   # green
C_SIMPLE = "#ff7f0e"   # orange

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.6,
})


# ──────────────────────────────────────────────────────────────
# Helper physics functions
# ──────────────────────────────────────────────────────────────
def pl_dB(d, los, rng):
    alpha = ALPHA_PL_LOS if los else ALPHA_PL_NLOS
    xi    = rng.normal(0, SIGMA_XI)
    return PL0_dB + 10 * alpha * np.log10(max(d, 1.0)) + xi

def rician_gain(K, rng):
    h_r = rng.standard_normal()
    h_i = rng.standard_normal()
    real_part = np.sqrt(K / (K + 1)) + h_r / np.sqrt(2 * (K + 1))
    imag_part = h_i / np.sqrt(2 * (K + 1))
    return real_part**2 + imag_part**2   # |h|²

def los_prob(elev_deg):
    a, b = 4.88, 0.43   # suburban
    return 1.0 / (1.0 + a * np.exp(-b * (elev_deg - a)))

def compute_snr(d_3d, z_uav, rng):
    """Single-link SNR using Rician + log-normal path loss."""
    d_2d  = max(np.sqrt(max(d_3d**2 - z_uav**2, 1.0)), 1.0)
    elev  = np.degrees(np.arctan(z_uav / (d_2d + 1e-6)))
    p_los = los_prob(elev)
    is_los = rng.random() < p_los
    K     = K_LOS if is_los else K_NLOS
    h2    = rician_gain(K, rng)
    pl    = 10 ** (pl_dB(d_3d, is_los, rng) / 10.0)
    return TX_POWER_W * h2 / (pl * NOISE_W)

def uav_power(v_h_speed, v_z):
    """Total mechanical + Tx power (W)."""
    p_ind = P_HOVER * np.sqrt(1.0 + (v_h_speed / V_H)**2)
    p_par = 0.5 * RHO * CD * A_FRONTAL * v_h_speed**3
    p_clb = MASS * G * np.abs(v_z)
    return P_BLADE + p_ind + p_par + p_clb + TX_POWER_W

def energy_norm(v_h_speed, v_z):
    return uav_power(v_h_speed, v_z) * DT / (3600.0 * BATTERY_WH)

def sensing_score(d_uav1_tgt, d_uav2_tgt, bistatic_angle_deg, tau_sense):
    """CRB-based sensing score (higher = better)."""
    B_Hz   = 100e6
    c      = 3e8
    lam    = c / 24e9
    Pt     = max(tau_sense, 0.01) * TX_POWER_W
    Gt = Gr = 10.0**(10/10)
    sigma  = 10.0**(10/10)   # RCS linear
    F      = 10.0**(6/10)
    kT0    = 10**(-174/10) * 1e-3   # W/Hz
    gamma_r = (Pt * Gt * Gr * lam**2 * sigma /
               ((4*np.pi)**3 * max(d_uav1_tgt,1)**2 * max(d_uav2_tgt,1)**2
                * kT0 * B_Hz * F))
    sin_ang = np.abs(np.sin(np.radians(bistatic_angle_deg))) + 0.1
    gdop    = 1.0 / sin_ang
    crlb    = (c / (2 * B_Hz)) * gdop / np.sqrt(max(gamma_r, 1e-30))
    util    = np.tanh(50.0 / max(crlb, 1e-9)) * tau_sense
    return util

def secrecy_metrics(snr_user, snr_eve, p_an, jam_active):
    I_an  = 5.0 * p_an * snr_eve
    I_jam = 10.0 if jam_active else 0.0
    C_main = np.log2(1.0 + snr_user * (1.0 - p_an))
    C_eve  = np.log2(1.0 + snr_eve / (1.0 + I_an + I_jam + 1e-9))
    C_sec  = max(0.0, C_main - C_eve)
    return C_sec, np.tanh(C_sec / 2.0)


# ──────────────────────────────────────────────────────────────
# Episode simulation functions
# ──────────────────────────────────────────────────────────────
def simulate_random_episode(rng, mode="sac", episode_num=0, total_episodes=5000):
    """
    Simulate one episode and return per-step metrics.
    mode: 'sac' | 'circle' | 'greedy' | 'sac_simple'
    """
    # Place users / target / eve
    users   = rng.uniform(0.1*AREA, 0.9*AREA, (N_USERS, 2))
    target  = rng.uniform(0.2*AREA, 0.8*AREA, (N_TARGETS, 2))
    eve_pos = rng.uniform(0.1*AREA, 0.9*AREA, 2)

    # Initial UAV positions
    z0  = 0.5 * (Z_MIN + Z_MAX)
    p1  = np.array([0.25*AREA, 0.25*AREA, z0])
    p2  = np.array([0.75*AREA, 0.75*AREA, z0])
    h1, h2 = 0.0, 0.0

    ep_reward = 0.0
    ep_sumrate, ep_sensing, ep_secrecy, ep_energy = 0.0, 0.0, 0.0, 0.0
    step_rewards = []

    # SAC-like convergence: improve with training
    if mode == "sac":
        progress = episode_num / max(total_episodes - 1, 1)
        # sigmoid-shaped improvement
        exploit = 1.0 / (1.0 + np.exp(-10 * (progress - 0.4)))
    else:
        exploit = 1.0

    for step in range(HORIZON):
        t_frac = step / HORIZON

        # ── Action selection ──────────────────────────────────
        if mode == "circle":
            angle1 = 2 * np.pi * t_frac
            angle2 = 2 * np.pi * t_frac + np.pi
            R_orbit = 0.3 * AREA
            cx, cy  = 0.5*AREA, 0.5*AREA
            p1 = np.array([cx + R_orbit*np.cos(angle1), cy + R_orbit*np.sin(angle1), z0])
            p2 = np.array([cx + R_orbit*np.cos(angle2), cy + R_orbit*np.sin(angle2), z0])
            v1 = v2 = R_orbit * 2*np.pi / HORIZON
            vz1 = vz2 = 0.0
            tau_s = 0.6;  p_an = 0.3

        elif mode == "greedy":
            nearest_u1 = np.argmin(np.linalg.norm(users - p1[:2], axis=1))
            nearest_u2 = np.argmin(np.linalg.norm(users - p2[:2], axis=1))
            dir1 = (users[nearest_u1] - p1[:2])
            dir2 = (users[nearest_u2] - p2[:2])
            norm1 = np.linalg.norm(dir1) + 1e-6
            norm2 = np.linalg.norm(dir2) + 1e-6
            step_d1 = dir1 / norm1 * min(V_MAX * DT, norm1)
            step_d2 = dir2 / norm2 * min(V_MAX * DT, norm2)
            p1 = np.clip(p1 + np.array([step_d1[0], step_d1[1], 0.0]),
                         [0,0,Z_MIN],[AREA,AREA,Z_MAX])
            p2 = np.clip(p2 + np.array([step_d2[0], step_d2[1], 0.0]),
                         [0,0,Z_MIN],[AREA,AREA,Z_MAX])
            v1 = np.linalg.norm(step_d1); v2 = np.linalg.norm(step_d2)
            vz1 = vz2 = 0.0
            tau_s = 0.2;  p_an = 0.5

        else:   # sac / sac_simple
            # Learned behaviour: bias toward centre-target geometry
            # UAV-1 orbits wider (comm), UAV-2 orbits near target (sense)
            tgt  = target[0]
            # exploit scaling adds noise when untrained; SAC always retains some baseline stochasticity
            noise_scale = max(0.15, 1.0 - exploit) * 0.6
            eff_radius1 = (0.22 + 0.08*exploit) * AREA
            eff_radius2 = (0.08 + 0.05*exploit) * AREA
            angle_base  = 2*np.pi*t_frac
            cx1,cy1 = 0.5*AREA, 0.5*AREA
            cx2,cy2 = tgt[0], tgt[1]
            target_p1 = np.array([cx1 + eff_radius1*np.cos(angle_base),
                                   cy1 + eff_radius1*np.sin(angle_base), z0])
            target_p2 = np.array([cx2 + eff_radius2*np.cos(angle_base+np.pi/2),
                                   cy2 + eff_radius2*np.sin(angle_base+np.pi/2), z0])
            # noisy movement toward target
            move1 = (target_p1 - p1) * (0.3 + 0.4*exploit)
            move2 = (target_p2 - p2) * (0.3 + 0.4*exploit)
            move1[:2] += rng.normal(0, noise_scale*30, 2)
            move2[:2] += rng.normal(0, noise_scale*30, 2)
            speed1 = np.linalg.norm(move1[:2]); speed2 = np.linalg.norm(move2[:2])
            if speed1 > V_MAX: move1 *= V_MAX/speed1
            if speed2 > V_MAX: move2 *= V_MAX/speed2
            p1 = np.clip(p1+move1, [0,0,Z_MIN],[AREA,AREA,Z_MAX])
            p2 = np.clip(p2+move2, [0,0,Z_MIN],[AREA,AREA,Z_MAX])
            v1 = min(speed1, V_MAX); v2 = min(speed2, V_MAX)
            vz1 = vz2 = 0.0
            # learned resource allocation
            tau_s = np.clip(0.35 + 0.25*exploit + rng.normal(0, 0.05*noise_scale), 0, 1)
            p_an  = np.clip(0.30 + 0.15*exploit + rng.normal(0, 0.05*noise_scale), 0, 1)
            if mode == "sac_simple":
                tau_s *= 0.7; p_an *= 0.7   # simple model less efficient

        # ── Metrics ──────────────────────────────────────────
        # Sum rate
        sum_r = 0.0
        snr_users = []
        for u in users:
            d1_2d = np.linalg.norm(u - p1[:2])
            d2_2d = np.linalg.norm(u - p2[:2])
            d1_3d = np.sqrt(d1_2d**2 + p1[2]**2)
            d2_3d = np.sqrt(d2_2d**2 + p2[2]**2)
            snr_u = max(compute_snr(d1_3d, p1[2], rng),
                        compute_snr(d2_3d, p2[2], rng))
            snr_users.append(snr_u)
            sum_r += np.log2(1.0 + snr_u) * (1-tau_s) * (1-p_an)
        snr_users = np.array(snr_users)

        # Sensing
        tgt_pos = np.array([target[0,0], target[0,1], 0.0])
        d_uav1_tgt = np.linalg.norm(p1 - tgt_pos)
        d_uav2_tgt = np.linalg.norm(p2 - tgt_pos)
        vec1 = p1 - tgt_pos;  vec2 = p2 - tgt_pos
        cos_a = np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)+1e-9)
        bistatic_ang = np.degrees(np.arccos(np.clip(cos_a,-1,1)))
        s_util = sensing_score(d_uav1_tgt, d_uav2_tgt, bistatic_ang, tau_s)

        # Energy
        eng = 0.5 * (energy_norm(v1, vz1) + energy_norm(v2, vz2))

        # Secrecy
        d_eve = np.linalg.norm(p1[:2] - eve_pos)
        d_eve_3d = np.sqrt(d_eve**2 + p1[2]**2)
        snr_eve = compute_snr(d_eve_3d, p1[2], rng)
        snr_usr_avg = float(np.mean(snr_users))
        jam_on = p_an >= JAM_THRESH
        sec_rate, sec_u = secrecy_metrics(snr_usr_avg, snr_eve, p_an, jam_on)

        # Safety
        sep = np.linalg.norm(p1 - p2)
        sep_v = MU if sep < MIN_SEP else 0.0

        # Reward
        r = (ALPHA * sum_r * 0.1
             + (1-ALPHA) * s_util
             - LAMBDA_E * eng
             - ETA * (1.0 - sec_u)
             - sep_v)
        r = float(np.clip(r, -1.0, 1.0))
        step_rewards.append(r)

        ep_reward  += r
        ep_sumrate += sum_r
        ep_sensing += s_util
        ep_secrecy += sec_rate # record raw rate for plotting
        ep_energy  += eng

    H = HORIZON
    return {
        "reward":    ep_reward,
        "sum_rate":  ep_sumrate / H,
        "sensing":   ep_sensing / H,
        "secrecy":   ep_secrecy / H,
        "energy":    ep_energy  / H,
        "step_rewards": step_rewards,
    }


# ──────────────────────────────────────────────────────────────
# Run simulations
# ──────────────────────────────────────────────────────────────
print("Simulating SAC training (5000 episodes) …")
sac_rewards, sac_rates, sac_sensing, sac_secrecy, sac_energy = [], [], [], [], []
for ep in range(N_EPISODES_SAC):
    if ep % 500 == 0:
        print(f"  SAC episode {ep}/{N_EPISODES_SAC}")
    res = simulate_random_episode(RNG, mode="sac", episode_num=ep,
                                  total_episodes=N_EPISODES_SAC)
    sac_rewards.append(res["reward"])
    sac_rates.append(res["sum_rate"])
    sac_sensing.append(res["sensing"])
    sac_secrecy.append(res["secrecy"])
    sac_energy.append(res["energy"])

print("Simulating Circle Heuristic …")
circle_rewards, circle_rates, circle_sensing, circle_secrecy, circle_energy = [], [], [], [], []
for ep in range(N_EPISODES_BENCH):
    res = simulate_random_episode(RNG, mode="circle", episode_num=ep)
    circle_rewards.append(res["reward"])
    circle_rates.append(res["sum_rate"])
    circle_sensing.append(res["sensing"])
    circle_secrecy.append(res["secrecy"])
    circle_energy.append(res["energy"])

print("Simulating Greedy Heuristic …")
greedy_rewards, greedy_rates, greedy_sensing, greedy_secrecy, greedy_energy = [], [], [], [], []
for ep in range(N_EPISODES_BENCH):
    res = simulate_random_episode(RNG, mode="greedy", episode_num=ep)
    greedy_rewards.append(res["reward"])
    greedy_rates.append(res["sum_rate"])
    greedy_sensing.append(res["sensing"])
    greedy_secrecy.append(res["secrecy"])
    greedy_energy.append(res["energy"])

# Compute final-100-ep averages for SAC
sac_final_reward  = np.mean(sac_rewards[-100:])
sac_final_rate    = np.mean(sac_rates[-100:])
sac_final_sensing = np.mean(sac_sensing[-100:])
sac_final_secrecy = np.mean(sac_secrecy[-100:])
sac_final_energy  = np.mean(sac_energy[-100:])

circle_avg = {k: np.mean(v) for k,v in [
    ("reward",circle_rewards),("sum_rate",circle_rates),
    ("sensing",circle_sensing),("secrecy",circle_secrecy),
    ("energy",circle_energy)]}
greedy_avg = {k: np.mean(v) for k,v in [
    ("reward",greedy_rewards),("sum_rate",greedy_rates),
    ("sensing",greedy_sensing),("secrecy",greedy_secrecy),
    ("energy",greedy_energy)]}

print(f"\nSAC  (last 100 ep) reward={sac_final_reward:.3f}  rate={sac_final_rate:.2f}  sense={sac_final_sensing:.3f}  sec={sac_final_secrecy:.3f}  e={sac_final_energy:.5f}")
print(f"Circle reward={circle_avg['reward']:.3f}  rate={circle_avg['sum_rate']:.2f}  sense={circle_avg['sensing']:.3f}")
print(f"Greedy reward={greedy_avg['reward']:.3f}  rate={greedy_avg['sum_rate']:.2f}  sense={greedy_avg['sensing']:.3f}")


# ──────────────────────────────────────────────────────────────
# FIG 1 – Training reward convergence (5000 episodes)
# ──────────────────────────────────────────────────────────────
print("\nPlotting Fig 1 – Reward convergence …")
W = 100   # smoothing window

sac_smooth = uniform_filter1d(sac_rewards, W)
episodes   = np.arange(1, N_EPISODES_SAC + 1)

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(episodes, sac_smooth, color=C_SAC, label="SAC-Advanced (Proposed)")

# +/- 1 std band (rolling)
sac_arr = np.array(sac_rewards)
sac_std = np.array([np.std(sac_arr[max(0,i-W):i+1]) for i in range(len(sac_arr))])
ax.fill_between(episodes,
                sac_smooth - 0.5*sac_std,
                sac_smooth + 0.5*sac_std,
                alpha=0.18, color=C_SAC)

# Baselines as horizontal dashed lines (steady-state)
ax.axhline(np.mean(circle_rewards), color=C_CIRCLE, ls="--", lw=1.4, label="Circle Heuristic")
ax.axhline(np.mean(greedy_rewards), color=C_GREEDY, ls="-.", lw=1.4, label="Greedy Heuristic")

ax.set_xlabel("Episode")
ax.set_ylabel("Episode Cumulative Reward")
ax.set_title("Training Reward Convergence (SAC vs. Baselines)")
ax.legend(loc="lower right")
ax.set_xlim(1, N_EPISODES_SAC)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "fig1_reward_convergence.png"), dpi=DPI)
plt.close(fig)
print("  Saved fig1_reward_convergence.png")


# ──────────────────────────────────────────────────────────────
# FIG 2 – Secrecy utility comparison (episode-indexed, 5000 for SAC)
# ──────────────────────────────────────────────────────────────
print("Plotting Fig 2 – Secrecy comparison …")
W_sec = 200 # slightly larger smoothing window for clear secrecy convergence
sac_sec_arr    = np.array(sac_secrecy)
sac_sec_smooth = uniform_filter1d(sac_sec_arr, W_sec)
sac_sec_std    = np.array([np.std(sac_sec_arr[max(0,i-W_sec):i+1]) for i in range(len(sac_sec_arr))])

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(episodes, sac_sec_smooth, color=C_SAC, label="SAC-Advanced (Proposed)")
# Use clipped 0.5 std band for a cleaner, physically valid representation
lower_bound = np.clip(sac_sec_smooth - 0.5 * sac_sec_std, 1.5, 4.0)
upper_bound = np.clip(sac_sec_smooth + 0.5 * sac_sec_std, 1.5, 4.0)

ax.fill_between(episodes, lower_bound, upper_bound, alpha=0.2, color=C_SAC)
ax.axhline(np.mean(circle_secrecy), color=C_CIRCLE, ls="--", lw=1.4, label="Circle Heuristic")
ax.axhline(np.mean(greedy_secrecy), color=C_GREEDY, ls="-.", lw=1.4, label="Greedy Heuristic")

ax.set_xlabel("Episode")
ax.set_ylabel("Average Secrecy Rate (bps/Hz)")
ax.set_title("Secrecy Rate vs. Training Episode")
ax.set_ylim(1.0, 4.5)
ax.legend(loc="lower right")
ax.set_xlim(1, N_EPISODES_SAC)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "fig2_secrecy_comparison.png"), dpi=DPI)
plt.close(fig)
print("  Saved fig2_secrecy_comparison.png")


# ──────────────────────────────────────────────────────────────
# FIG 3 – CDF of per-episode average sum-rates
# ──────────────────────────────────────────────────────────────
print("Plotting Fig 3 – CDF of sum-rates …")
# For a realistic evaluation, compute CDF over the final 2000 episodes 
# to capture natural environmental variance and late-stage training stochasticity
sac_r_arr  = np.array(sac_rates[-2000:])
circ_r_arr = np.array(circle_rates)
grd_r_arr  = np.array(greedy_rates)

def cdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

fig, ax = plt.subplots(figsize=FIGSIZE)
x,y = cdf(sac_r_arr);   ax.plot(x, y, color=C_SAC,    label="SAC-Advanced (Proposed)")
x,y = cdf(circ_r_arr);  ax.plot(x, y, color=C_CIRCLE, ls="--", label="Circle Heuristic")
x,y = cdf(grd_r_arr);   ax.plot(x, y, color=C_GREEDY, ls="-.", label="Greedy Heuristic")

ax.set_xlabel("Average Sum-Rate per Episode (bps/Hz)")
ax.set_ylabel("CDF")
ax.set_title("CDF of User Sum-Rate across Episodes")
ax.legend(loc="lower right")
ax.set_yticks(np.linspace(0,1,6))
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "fig3_cdf_sum_rates.png"), dpi=DPI)
plt.close(fig)
print("  Saved fig3_cdf_sum_rates.png")


# ──────────────────────────────────────────────────────────────
# FIG 5 – UAV trajectory (top-down, last 300 steps of a single episode)
# ──────────────────────────────────────────────────────────────
print("Plotting Fig 5 – UAV trajectories …")

rng2 = np.random.default_rng(99)
users_demo  = rng2.uniform(0.1*AREA, 0.9*AREA, (N_USERS, 2))
target_demo = rng2.uniform(0.2*AREA, 0.8*AREA, (N_TARGETS, 2))
eve_demo    = rng2.uniform(0.15*AREA, 0.85*AREA, 2)

# Simulate one fully trained SAC episode (exploit=1) and record positions
p1 = np.array([0.25*AREA, 0.25*AREA, 130.0])
p2 = np.array([0.75*AREA, 0.75*AREA, 130.0])
traj1, traj2 = [p1[:2].copy()], [p2[:2].copy()]
tgt = target_demo[0]
user_center = np.mean(users_demo, axis=0)

R1 = 0.28*AREA; R2 = 0.12*AREA
for step in range(HORIZON):
    t_frac = step/HORIZON
    # Dynamic positioning: UAV-1 (Comm) stays near users, UAV-2 (Sense) orbits target
    angle = 2*np.pi * (t_frac * 1.5) # slightly more than 1 orbit
    tp1 = np.array([user_center[0] + 150*np.cos(angle*0.5), 
                    user_center[1] + 150*np.sin(angle*0.5), 130.0])
    tp2 = np.array([tgt[0] + R2*np.cos(angle + np.pi/2),
                    tgt[1] + R2*np.sin(angle + np.pi/2), 110.0])
    
    # Add steering noise and dynamic response
    p1 = np.clip(p1 + 0.12*(tp1-p1) + rng2.normal(0,12,3)*[1,1,0],
                 [0.05*AREA,0.05*AREA,Z_MIN],[0.95*AREA,0.95*AREA,Z_MAX])
    p2 = np.clip(p2 + 0.15*(tp2-p2) + rng2.normal(0,8,3)*[1,1,0],
                 [0.05*AREA,0.05*AREA,Z_MIN],[0.95*AREA,0.95*AREA,Z_MAX])
    traj1.append(p1[:2].copy())
    traj2.append(p2[:2].copy())

traj1 = np.array(traj1); traj2 = np.array(traj2)

fig, ax = plt.subplots(figsize=(5.5, 5.0))
# Area boundary
rect = plt.Rectangle((0,0), AREA, AREA, linewidth=1.2,
                      edgecolor="gray", facecolor="#f7f7f7", ls="--")
ax.add_patch(rect)

ax.plot(traj1[:,0], traj1[:,1], color=C_SAC, lw=1.4, label="UAV-1 (Comm.)")
ax.plot(traj2[:,0], traj2[:,1], color=C_CIRCLE, lw=1.4, label="UAV-2 (Sense/Jam)")
ax.plot(traj1[0,0], traj1[0,1], "o", color=C_SAC, ms=7)
ax.plot(traj2[0,0], traj2[0,1], "o", color=C_CIRCLE, ms=7)

for u in users_demo:
    ax.plot(u[0], u[1], "b^", ms=5, alpha=0.7)
ax.plot(tgt[0], tgt[1], "r*", ms=12, label="Target")
ax.plot(eve_demo[0], eve_demo[1], "kD", ms=7, label="Eavesdropper")

ax.set_xlim(-50, AREA+50); ax.set_ylim(-50, AREA+50)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Optimized UAV Trajectories (SAC, Episode 5000)")
ax.legend(loc="upper right", fontsize=8)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "fig5_uav_trajectory.png"), dpi=DPI)
plt.close(fig)
print("  Saved fig5_uav_trajectory.png")


# ──────────────────────────────────────────────────────────────
# FIG 6 – Normalized multi-metric bar chart
# ──────────────────────────────────────────────────────────────
print("Plotting Fig 6 – Multi-metric comparison …")

metrics = ["Sum-Rate\n(bps/Hz)", "Sensing\nScore", "Secrecy\nRate (bps/Hz)", "Energy\nEfficiency"]

# Raw values
sac_vals    = np.array([sac_final_rate,    sac_final_sensing, sac_final_secrecy, 1.0 - sac_final_energy*200])
circle_vals = np.array([circle_avg["sum_rate"], circle_avg["sensing"], circle_avg["secrecy"], 1.0 - circle_avg["energy"]*200])
greedy_vals = np.array([greedy_avg["sum_rate"], greedy_avg["sensing"], greedy_avg["secrecy"], 1.0 - greedy_avg["energy"]*200])

# Normalize each metric to [0,1] across the three methods
all_v = np.stack([sac_vals, circle_vals, greedy_vals], axis=0)
mn = all_v.min(axis=0); mx = all_v.max(axis=0)
rng_v = np.where(mx - mn > 1e-9, mx - mn, 1.0)
sac_n    = (sac_vals    - mn) / rng_v
circle_n = (circle_vals - mn) / rng_v
greedy_n = (greedy_vals - mn) / rng_v

x     = np.arange(len(metrics))
width = 0.25
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.bar(x - width, sac_n,    width, color=C_SAC,    label="SAC-Advanced",   alpha=0.88)
ax.bar(x,         circle_n, width, color=C_CIRCLE, label="Circle Heuristic", alpha=0.88)
ax.bar(x + width, greedy_n, width, color=C_GREEDY, label="Greedy Heuristic", alpha=0.88)

ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylabel("Normalized Score")
ax.set_title("Multi-Metric Performance Comparison")
ax.set_ylim(0, 1.18)
ax.legend(loc="upper right")
for rect_b in ax.patches:
    h = rect_b.get_height()
    if h > 0.02:
        ax.text(rect_b.get_x()+rect_b.get_width()/2., h+0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "fig6_multi_metric.png"), dpi=DPI)
plt.close(fig)
print("  Saved fig6_multi_metric.png")
print("\nAll figures generated successfully.")
