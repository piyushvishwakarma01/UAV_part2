"""Animate UAV trajectories in the dual-UAV ISAC environment."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from src.envs.dual_isac_env import make_env


def main():
    # Create environment
    env = make_env(n_users=8, n_targets=1, alpha=0.5, seed=42)
    obs, info = env.reset()

    # Store trajectory history
    uav_history = [[], []]
    area_size = env.cfg.area_size

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Dual UAV-ISAC Environment Animation")
    ax.grid(True, alpha=0.3)

    # Plot users and targets (static)
    user_positions = env.users_xy
    target_positions = env.targets_xy
    ax.plot(user_positions[:, 0], user_positions[:, 1], "g.", markersize=10, label="Users", alpha=0.7)
    ax.plot(target_positions[:, 0], target_positions[:, 1], "rx", markersize=15, label="Targets", markeredgewidth=2)

    # UAV trajectories (animated)
    uav1_line, = ax.plot([], [], "b-", alpha=0.3, linewidth=2, label="UAV1 trajectory")
    uav2_line, = ax.plot([], [], "r-", alpha=0.3, linewidth=2, label="UAV2 trajectory")
    uav1_dot, = ax.plot([], [], "bo", markersize=12, label="UAV1")
    uav2_dot, = ax.plot([], [], "ro", markersize=12, label="UAV2")

    ax.legend(loc="upper right")

    def init():
        uav1_line.set_data([], [])
        uav2_line.set_data([], [])
        uav1_dot.set_data([], [])
        uav2_dot.set_data([], [])
        return uav1_line, uav2_line, uav1_dot, uav2_dot

    def update(frame):
        if frame == 0:
            # Reset environment
            obs, info = env.reset()
            uav_history[0] = []
            uav_history[1] = []

        # Sample random action
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        # Get UAV positions (x, y only for 2D plot)
        uav1_pos = env.uav_pos[0, :2]
        uav2_pos = env.uav_pos[1, :2]

        # Update history
        uav_history[0].append(uav1_pos)
        uav_history[1].append(uav2_pos)

        # Update plot
        if len(uav_history[0]) > 1:
            hist1 = np.array(uav_history[0])
            hist2 = np.array(uav_history[1])
            uav1_line.set_data(hist1[:, 0], hist1[:, 1])
            uav2_line.set_data(hist2[:, 0], hist2[:, 1])

        uav1_dot.set_data([uav1_pos[0]], [uav1_pos[1]])
        uav2_dot.set_data([uav2_pos[0]], [uav2_pos[1]])

        # Reset if episode done
        if term or trunc:
            obs, info = env.reset()
            uav_history[0] = []
            uav_history[1] = []

        return uav1_line, uav2_line, uav1_dot, uav2_dot

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=400, init_func=init, interval=50, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
