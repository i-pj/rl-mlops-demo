"""Inference engine — run episodes and collect metrics.

Provides both visual (pygame window) and headless (rgb_array) modes.
The core ``run_episode`` function is mode-agnostic; the environment's
``render_mode`` controls whether a window appears.

MLOps concepts demonstrated:
    - Consistent inference preprocessing (matching training)
    - Auto-detecting and loading VecNormalize running statistics
    - Per-step latency measurement
    - Graceful episode termination
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)

from rl_mlops_demo.config import MODELS_DIR, cfg
from rl_mlops_demo.metrics import (
    EpisodeRecord,
    MetricsCollector,
    RunSummary,
    generate_run_id,
    summarise_run,
    write_records,
)
from rl_mlops_demo.wrappers import make_car_env

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecNormalize

console = Console()


def find_latest_model() -> Path:
    """Find the classroom-stable primary model, then fall back to latest checkpoints."""
    override = os.environ.get("RL_MODEL_PATH", "").strip()
    preferred = Path(override) if override else MODELS_DIR / cfg.primary_model
    if preferred.exists():
        return preferred.resolve()

    models = list(MODELS_DIR.glob("**/best_model.zip"))
    if not models:
        # Also check for final_model.zip as fallback
        models = list(MODELS_DIR.glob("**/final_model.zip"))
        if not models:
            raise FileNotFoundError(
                "No trained models found in models/ directory. Run `uv run rl-demo train` first."
            )

    # Sort by modification time
    models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return models[0]


def _find_vec_normalize_path(model_path: Path) -> Path:
    """Return normalization statistics paired with a checkpoint when available."""
    step_marker = "rl_model_"
    if model_path.name.startswith(step_marker) and model_path.name.endswith("_steps.zip"):
        step = model_path.name.removeprefix(step_marker).removesuffix("_steps.zip")
        checkpoint_stats = model_path.parent / f"vecnorm_{step}_steps.pkl"
        if checkpoint_stats.exists():
            return checkpoint_stats
    return model_path.parent / "vec_normalize.pkl"


def load_model(path: Path | None = None) -> PPO:
    """Load a PPO checkpoint and place on configured device."""
    if path is None:
        path = find_latest_model()

    console.print(f"  📦 Loading [bold]{path.name}[/bold] -> {cfg.device}")

    # Fix SB3 PyTorch loading issues across environments
    custom_objects: dict[str, object] = {
        "map_location": cfg.device,
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    if "hf-vukpetar-ppo-carracing-v0-v3" in str(path):
        import gymnasium as gym
        import numpy as np

        # The checkpoint was produced with Gym 0.21 / SB3 1.5. Replacing the
        # serialized spaces allows explicit compatibility evaluation on v3.
        custom_objects.update(
            {
                "observation_space": gym.spaces.Box(0, 255, shape=(3, 96, 96), dtype=np.uint8),
                "action_space": gym.spaces.Box(
                    np.array([-1, 0, 0], dtype=np.float32),
                    np.array([1, 1, 1], dtype=np.float32),
                    dtype=np.float32,
                ),
                "clip_range": lambda _: 0.2,
            }
        )

    model = PPO.load(str(path), device=cfg.device, custom_objects=custom_objects)
    console.print(f"  [green]✓[/green] Model ready on [bold]{cfg.device}[/bold]")
    return model


def build_inference_env(model: PPO, *, render_mode: RenderMode = None):
    """Build the environment contract required by the loaded policy."""
    from rl_mlops_demo.wrappers import make_raw_car_env

    policy_shape = tuple(model.observation_space.shape)
    if policy_shape == (3, 96, 96):
        return VecTransposeImage(DummyVecEnv([make_raw_car_env(seed=42, render_mode=render_mode)]))
    if policy_shape == (2, 64, 64):
        vec_env = DummyVecEnv([make_car_env(seed=42, frame_skip=2, render_mode=render_mode)])
        vec_env = VecFrameStack(vec_env, n_stack=2)
        return VecTransposeImage(vec_env)
    raise ValueError(
        f"unsupported policy observation contract {policy_shape}; "
        "select a package compatible with raw RGB or stacked 64x64 grayscale"
    )


# ─── Single Episode ───────────────────────────────────────────────


def run_episode(
    model: PPO,
    vec_env: VecNormalize,
    collector: MetricsCollector,
    episode_num: int,
    *,
    deterministic: bool | None = None,
) -> EpisodeRecord:
    """Run one complete episode and return a metrics record."""
    det = deterministic if deterministic is not None else cfg.deterministic
    collector.start_episode(episode_num)

    obs = vec_env.reset()

    total_reward = 0.0
    step_count = 0
    done = False

    while not done and step_count < cfg.max_episode_steps:
        # ── Timed inference ──────────────────────────────────
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=det)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        collector.record_step(latency_ms)

        # ── Step environment ─────────────────────────────────
        obs, rewards, dones, _infos = vec_env.step(action)
        done = dones[0]

        total_reward += float(rewards[0])
        step_count += 1

    return collector.finish_episode(total_reward, step_count)


# ─── Multi-Episode Evaluation Run ─────────────────────────────────

RenderMode = Literal["human", "rgb_array", None]


def run_evaluation(
    model: PPO,
    model_path: Path,
    *,
    num_episodes: int | None = None,
    render_mode: RenderMode = None,
    deterministic: bool | None = None,
) -> tuple[list[EpisodeRecord], RunSummary, str]:
    """Run a full evaluation: N episodes -> records + summary."""
    n = num_episodes or cfg.num_eval_episodes

    run_id = generate_run_id()
    collector = MetricsCollector(run_id)

    # ── Replicate Training Preprocessing ──
    vec_env = build_inference_env(model, render_mode=render_mode)

    # NOTE: No VecNormalize needed at inference.
    # Training used norm_obs=False (raw pixels go directly to CNN) and
    # norm_reward=True (only for gradient stability during training).
    # Loading the saved VecNormalize stats here would corrupt the env.

    console.print(f"\n🏎️  Starting evaluation run [bold]{run_id[:16]}[/bold]")
    console.print(f"   Episodes: {n} | Deterministic: {deterministic or cfg.deterministic}")
    console.print(f"   Render: {render_mode or 'off'} | Device: {cfg.device}\n")

    records: list[EpisodeRecord] = []

    try:
        for ep in range(n):
            record = run_episode(model, vec_env, collector, ep, deterministic=deterministic)
            records.append(record)

            status_icon = "🟢" if record.total_reward >= cfg.thresholds.min_deploy_reward else "🔴"
            console.print(
                f"  {status_icon} Episode {ep + 1}/{n}  │  "
                f"Reward: {record.total_reward:>7.1f}  │  "
                f"Steps: {record.episode_length:>4d}  │  "
                f"Latency: {record.mean_latency_ms:.1f}ms"
            )
    finally:
        vec_env.close()

    csv_path = write_records(records, run_id)
    summary = summarise_run(records)

    console.print(summary.format_report())
    console.print(f"\n  📝 Logs saved to [link=file://{csv_path}]{csv_path.name}[/link]")
    try:
        from rl_mlops_demo.mlflow_tracker import log_evaluation_run

        mlflow_run_id = log_evaluation_run(
            model_path=model_path,
            records=records,
            summary=summary,
            csv_path=csv_path,
            deterministic=deterministic if deterministic is not None else cfg.deterministic,
            run_type="visual-demo" if render_mode == "human" else "evaluation",
        )
        console.print(f"  [green]✓[/green] MLflow run logged: [bold]{mlflow_run_id[:12]}[/bold]")
    except Exception as exc:
        console.print(f"  [yellow]MLflow logging skipped:[/yellow] {exc}")

    return records, summary, run_id


# ─── Quick Visual Demo ────────────────────────


def run_visual_demo(
    model: PPO,
    model_path: Path,
    *,
    num_episodes: int = 1,
) -> tuple[list[EpisodeRecord], RunSummary, str]:
    """Run with pygame window."""
    return run_evaluation(
        model,
        model_path,
        num_episodes=num_episodes,
        render_mode="human",
        deterministic=True,
    )


def record_evidence_video(
    model: PPO,
    output_path: Path,
    *,
    max_steps: int = 600,
    fps: int = 30,
) -> Path:
    """Record one deterministic episode through the policy's real contract."""
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".avi")
    vec_env = build_inference_env(model, render_mode="rgb_array")
    obs = vec_env.reset()
    frame = vec_env.get_images()[0]
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        vec_env.close()
        raise RuntimeError("OpenCV could not initialize the temporary video writer")

    done = False
    steps = 0
    try:
        while not done and steps < max_steps:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            action, _ = model.predict(obs, deterministic=True)
            obs, _rewards, dones, _infos = vec_env.step(action)
            done = bool(dones[0])
            frame = vec_env.get_images()[0]
            steps += 1
    finally:
        writer.release()
        vec_env.close()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temp_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    temp_path.unlink(missing_ok=True)
    console.print(f"  [green]✓[/green] Evidence video saved: [bold]{output_path}[/bold]")
    return output_path
