"""Centralised configuration for the RL MLOps demo.

Every tunable knob lives here — device selection, model coordinates,
environment parameters, monitoring thresholds, and path conventions.
Import ``cfg`` from this module to access the singleton.

MLOps concepts demonstrated:
    - Hardware abstraction (MPS → CPU fallback)
    - Quality-gate thresholds (reward, latency)
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

# ── Ensure MPS fallback is set before any torch op ──────────────
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _default_n_envs() -> int:
    return min(os.cpu_count() or 4, 16)


def _detect_device() -> str:
    """Return device string for SB3.

    Priority: RL_DEVICE env var > cuda > auto (SB3 handles the rest).
    For CarRacing's small CNN, CPU is often faster than MPS on Apple Silicon,
    but CUDA provides real speedups on NVIDIA hardware.
    """
    override = os.environ.get("RL_DEVICE", "").strip().lower()
    if override in ("cpu", "cuda", "mps", "auto"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    # For Apple Silicon: CPU is typically faster for small RL CNNs
    # due to MPS data-transfer overhead. Let user opt-in via RL_DEVICE=mps.
    return "cpu"


# ── Paths ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs" / "runs"
VIDEOS_DIR = PROJECT_ROOT / "videos"


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and configuration for PPO training."""

    n_envs: int = field(default_factory=_default_n_envs)
    seed: int = 42
    total_timesteps: int = 4_000_000


@dataclass(frozen=True)
class EnvSpec:
    """Input/output contract for the Gymnasium environment.

    Documenting the contract explicitly prevents silent breakage
    when the environment is upgraded (the phantom-braking lesson).
    """

    env_id: str = "CarRacing-v3"
    obs_shape: tuple[int, int, int] = (96, 96, 3)
    action_dim: int = 3  # [steer, gas, brake]


@dataclass(frozen=True)
class Thresholds:
    """Quality-gate and alerting thresholds.

    In production these live in a config service; here they're
    constants that students can tweak to see the effect.
    """

    min_deploy_reward: float = 800.0
    max_latency_ms: float = 50.0
    max_failure_rate: float = 0.05


@dataclass
class Config:
    """Top-level, singleton-style configuration object."""

    # ── Hardware ─────────────────────────────────────────────
    device: str = "auto"

    # ── Training ─────────────────────────────────────────────
    train: TrainConfig = field(default_factory=TrainConfig)

    # ── Environment ──────────────────────────────────────────
    env: EnvSpec = field(default_factory=EnvSpec)

    # ── Inference ────────────────────────────────────────────
    deterministic: bool = True
    max_episode_steps: int = 1000
    num_eval_episodes: int = 5
    primary_model: str = "provided-fallback/fallback-model.zip"

    # ── Monitoring ───────────────────────────────────────────
    thresholds: Thresholds = field(default_factory=Thresholds)

    # ── Derived / system info ────────────────────────────────
    @property
    def system_info(self) -> dict[str, str]:
        return {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": f"{platform.system()}-{platform.machine()}",
            "device": str(self.device),
        }

    def ensure_dirs(self) -> None:
        """Create output directories if missing."""
        for d in (MODELS_DIR, LOGS_DIR, VIDEOS_DIR):
            d.mkdir(parents=True, exist_ok=True)


# ── Module-level singleton ───────────────────────────────────────
cfg = Config()
cfg.ensure_dirs()
