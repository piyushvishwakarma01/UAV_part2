from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AreaSpec:
    width: float
    height: float
    zmin: float
    zmax: float


def seed_everything(seed: int | None) -> np.random.Generator:
    """Return a numpy RNG with deterministic seed if provided."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    """Clamp to [0,1]."""
    return np.clip(x, 0.0, 1.0)


def wrap_angle(angle_rad: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def pairwise_dist2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise 2D distances between points in a (N,2) and b (M,2)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def dist3d(p: np.ndarray, q: np.ndarray) -> float:
    """3D distance between two points (x,y,z)."""
    return float(np.linalg.norm(p - q))


def baseline_geometry_score(uav_positions: np.ndarray, target_pos: np.ndarray) -> float:
    """Heuristic baseline score: prefer wider baseline around target.

    - Large inter-UAV separation projected around the target is good
    - Penalize being very far from the target overall
    Returns a bounded score in [0,1].
    """
    u1, u2 = uav_positions[0], uav_positions[1]
    center = (u1 + u2) / 2.0
    d_center = np.linalg.norm(center[:2] - target_pos[:2]) + 1e-6
    baseline = np.linalg.norm((u1 - target_pos)[:2] - (u2 - target_pos)[:2])
    # Normalize by area scale ~ width of 2000 m
    baseline_norm = np.tanh(baseline / 500.0)
    proximity = np.tanh(400.0 / d_center)
    score = float(np.clip(0.5 * baseline_norm + 0.5 * proximity, 0.0, 1.0))
    return score


def secrecy_leakage_proxy(
    uav_headings: np.ndarray,
    uav_an: np.ndarray,
    sector_angle: float,
    sector_spread: float,
    jammer_active: bool,
) -> float:
    """Leakage increases when headings align within sector; AN and jammer reduce leakage.

    Returns a non-negative scalar proxy; lower is better.
    """
    # Alignment: average of cos angular distance to sector center inside spread
    deltas = np.abs(wrap_angle(uav_headings - sector_angle))
    align = np.mean(np.where(deltas <= sector_spread, np.cos(deltas), 0.0))
    align = max(0.0, align)  # only count positive alignment
    # Reduction by AN and jammer
    an_reduction = 0.3 + 0.7 * (1.0 - float(np.mean(uav_an)))  # more AN => less leakage
    jammer_factor = 0.4 if jammer_active else 1.0
    leak = align * an_reduction * jammer_factor
    return float(np.clip(leak, 0.0, 5.0))


def energy_model(
    speeds: np.ndarray,
    dz: np.ndarray,
    tx_power_frac: np.ndarray,
) -> float:
    """Simple energy proxy combining movement and tx cost.

    - Movement ~ speed^2
    - Altitude change cost ~ |dz|
    - Tx cost ~ small linear in (1-AN)
    Returns a small positive scalar.
    """
    move = float(np.mean(speeds**2)) * 0.002
    alt = float(np.mean(np.abs(dz))) * 0.001
    tx = float(np.mean(tx_power_frac)) * 0.01
    return move + alt + tx


def no_fly_violations(positions_xy: np.ndarray, mask_flat: np.ndarray, area: AreaSpec) -> int:
    """Count how many UAVs are inside no-fly cells for a coarse 10×10 grid mask.

    mask_flat contains 100 entries of 0/1 where 1 means forbidden.
    """
    grid_n = 10
    cell_w = area.width / grid_n
    cell_h = area.height / grid_n
    count = 0
    for p in positions_xy:
        i = min(grid_n - 1, max(0, int(p[0] // cell_w)))
        j = min(grid_n - 1, max(0, int(p[1] // cell_h)))
        idx = j * grid_n + i
        if mask_flat[idx] > 0.5:
            count += 1
    return count


def normalize_obs(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Normalize to approximately [-1,1] given bounds."""
    # Avoid div by zero
    scale = np.where(high - low == 0.0, 1.0, high - low)
    return np.clip(2.0 * (x - low) / scale - 1.0, -1.0, 1.0)


def clamp_position(p: np.ndarray, area: AreaSpec) -> np.ndarray:
    q = p.copy()
    q[0] = float(np.clip(q[0], 0.0, area.width))
    q[1] = float(np.clip(q[1], 0.0, area.height))
    q[2] = float(np.clip(q[2], area.zmin, area.zmax))
    return q


