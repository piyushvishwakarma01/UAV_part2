"""Setup script for dual-uav-isac project."""
from setuptools import find_packages, setup

setup(
    name="dual-uav-isac",
    version="0.1.0",
    description="Dual UAV-assisted ISAC with Deep Reinforcement Learning",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.11",
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.26",
        "scipy>=1.11",
        "torch>=2.1",
        "stable-baselines3==2.3.2",
        "matplotlib>=3.8",
        "tyro>=0.7.3",
        "tqdm>=4.66",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "ruff>=0.6",
            "tensorboard>=2.15",
        ],
    },
)


