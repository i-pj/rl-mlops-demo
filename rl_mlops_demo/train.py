"""Training script for CarRacing PPO with MLflow integration.

This module sets up the environment pipeline and trains a PPO agent using
hyperparameters validated by the RL Baselines3 Zoo for CarRacing-v3.

MLOps concepts demonstrated:
    - Real-time experiment tracking (MLflow)
    - Hyperparameter and reproducibility logging
    - Checkpoint saving with normalization stats
    - Hardware acceleration abstraction
"""

import json
import os
import platform
import subprocess

import gymnasium as gym
import stable_baselines3
import torch
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)

from rl_mlops_demo.callbacks import (
    MLflowCallback,
    NormalizeCheckpointCallback,
    RichProgressCallback,
)
from rl_mlops_demo.config import LOGS_DIR, MODELS_DIR, cfg
from rl_mlops_demo.wrappers import make_car_env

console = Console()


def get_git_sha() -> str:
    """Safely get the current git commit SHA."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def linear_schedule(initial_value: float):
    """Linear learning rate decay schedule."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def build_train_env(n_envs: int, seed: int, frame_skip: int = 2):
    """Build the full training environment stack."""
    if n_envs > 1:
        env = SubprocVecEnv(
            [make_car_env(seed=seed + i, frame_skip=frame_skip) for i in range(n_envs)]
        )
    else:
        env = DummyVecEnv([make_car_env(seed=seed, frame_skip=frame_skip)])

    env = VecFrameStack(env, n_stack=2)  # RL Zoo: 2 (NOT 4)
    env = VecTransposeImage(env)  # [H,W,C] -> [C,H,W] for PyTorch
    env = VecNormalize(
        env,
        norm_obs=False,  # RL Zoo: DO NOT normalize pixel observations
        norm_reward=True,  # RL Zoo: DO normalize rewards
    )
    return env


def build_eval_env(seed: int, frame_skip: int = 2, vec_normalize_path: str | None = None):
    """Build the evaluation environment with frozen normalization stats."""
    env = DummyVecEnv([make_car_env(seed=seed + 100, frame_skip=frame_skip)])
    env = VecFrameStack(env, n_stack=2)
    env = VecTransposeImage(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)

    env.training = False  # CRITICAL: freeze running stats during eval
    env.norm_reward = False  # Don't normalize rewards during eval reporting
    return env


def run_training(total_timesteps: int, n_envs: int, seed: int, device: str | None = None):
    """Run PPO training with MLflow tracking."""
    # Ensure directories exist
    cfg.ensure_dirs()

    device = device or cfg.device
    if n_envs <= 0:
        n_envs = cfg.train.n_envs

    console.print(f"[bold]🚀 Starting Training on {device}[/bold]")

    # ── Startup Validation ──
    # Check if batch_size (128) evenly divides n_steps * n_envs (512 * n_envs)
    n_steps = 512
    batch_size = 128
    rollout_buffer = n_steps * n_envs
    if rollout_buffer % batch_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must evenly divide n_steps * n_envs "
            f"({n_steps} * {n_envs} = {rollout_buffer}). Please adjust n_envs."
        )

    # ── Environments ──
    console.print("Building environments...")
    train_env = build_train_env(n_envs=n_envs, seed=seed)
    eval_env = build_eval_env(seed=seed)

    # Approximately 50 eval points regardless of training length
    eval_freq = max(1, total_timesteps // (50 * n_envs))

    # ── MLflow Setup ──
    try:
        import mlflow
    except ImportError as e:
        raise RuntimeError("MLflow is required for training. Please install it.") from e

    from rl_mlops_demo.mlflow_tracker import MLFLOW_DB

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("rl-carracing-ppo")

    # Enable automatic logging of system metrics (CPU, RAM, GPU, etc.)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    run_name = f"ppo-{total_timesteps // 1000}k-seed{seed}"

    # ── Paths ──
    run_models_dir = MODELS_DIR / run_name
    run_models_dir.mkdir(parents=True, exist_ok=True)

    best_model_dir = run_models_dir / "best_model"
    best_model_path = best_model_dir / "best_model.zip"
    final_model_path = run_models_dir / "final_model.zip"
    vec_normalize_path = run_models_dir / "vec_normalize.pkl"

    with mlflow.start_run(run_name=run_name):
        # Log ALL hyperparameters for reproducibility
        hyperparams = {
            "algorithm": "PPO",
            "policy": "CnnPolicy",
            "n_envs": n_envs,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": "linear_schedule(1e-4)",
            "use_sde": True,
            "sde_sample_freq": 4,
            "activation_fn": "GELU",
            "ortho_init": False,
            "frame_stack": 2,
            "frame_skip": 2,
            "norm_obs": False,
            "norm_reward": True,
            "total_timesteps": total_timesteps,
            "seed": seed,
        }
        mlflow.log_params(hyperparams)

        system_tags = {
            "git_commit": get_git_sha(),
            "sb3_version": stable_baselines3.__version__,
            "gym_version": gym.__version__,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "device": str(device),
            "platform": f"{platform.system()}-{platform.machine()}",
        }
        mlflow.set_tags(system_tags)

        # ── Model Definition ──
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=linear_schedule(1e-4),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=True,
            sde_sample_freq=4,
            verbose=1,
            seed=seed,
            device=device,
            tensorboard_log=None,
            policy_kwargs=dict(
                log_std_init=-2,
                activation_fn=torch.nn.GELU,
                ortho_init=False,
                net_arch=dict(pi=[256], vf=[256]),
            ),
        )

        # ── Callbacks ──
        callbacks = CallbackList(
            [
                MLflowCallback(),
                NormalizeCheckpointCallback(
                    save_freq=eval_freq,
                    save_path=str(run_models_dir),
                    vec_normalize_env=train_env,
                ),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_model_dir),
                    log_path=str(LOGS_DIR),
                    eval_freq=eval_freq,
                    n_eval_episodes=5,
                    deterministic=True,
                    render=False,
                ),
                RichProgressCallback(total_timesteps=total_timesteps),
            ]
        )

        console.print(f"Starting training for {total_timesteps} timesteps...")
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=False,  # We use RichProgressCallback instead
            )
            console.print("\n[green]Training completed successfully![/green]")
        except KeyboardInterrupt:
            mlflow.set_tag("status", "interrupted")
            console.print("\n[yellow]Training interrupted. Saving artifacts...[/yellow]")

        # ── ALWAYS save artifacts ──
        console.print("Saving models and artifacts to MLflow...")
        model.save(str(final_model_path))
        train_env.save(str(vec_normalize_path))

        # Log to MLflow
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="model")
        if final_model_path.exists():
            mlflow.log_artifact(str(final_model_path), artifact_path="model")
        if vec_normalize_path.exists():
            mlflow.log_artifact(str(vec_normalize_path), artifact_path="model")
            # Also copy to best_model_dir so inference can find it easily
            import shutil

            shutil.copy2(vec_normalize_path, best_model_dir / "vec_normalize.pkl")
            mlflow.log_artifact(str(best_model_dir / "vec_normalize.pkl"), artifact_path="model")

        # Save exhaustive config JSON
        config_snapshot = {
            "algorithm": "PPO",
            "policy": "CnnPolicy",
            "n_envs": n_envs,
            "seed": seed,
            "device": str(device),
            "hyperparameters": hyperparams,
            "environment": {
                "env_id": "CarRacing-v3",
                "frame_skip": 2,
                "frame_stack": 2,
                "grayscale": True,
                "resize": [64, 64],
                "norm_obs": False,
                "norm_reward": True,
            },
            "versions": system_tags,
        }
        config_path = LOGS_DIR / f"training_config_{run_name}.json"
        config_path.write_text(json.dumps(config_snapshot, indent=2))
        mlflow.log_artifact(str(config_path), artifact_path="config")

    train_env.close()
    eval_env.close()
