"""Optuna hyperparameter tuning script for CarRacing PPO.

Demonstrates MLOps best practice of nested MLflow runs for tracking
hyperparameter sweeps. The parent run contains the sweep definition,
and child runs represent individual trials.

NOTE: This script is typically used to find the best hyperparameters,
and is then commented out or reserved for future model upgrades.
"""

try:
    import mlflow
    import optuna
except ImportError as e:
    raise RuntimeError("optuna and mlflow are required for tuning. Run `uv sync`") from e

import torch
from rich.console import Console
from stable_baselines3 import PPO

from rl_mlops_demo.config import cfg
from rl_mlops_demo.mlflow_tracker import MLFLOW_DB
from rl_mlops_demo.train import build_eval_env, build_train_env, linear_schedule

console = Console()


def run_tuning(
    n_trials: int, timesteps_per_trial: int, n_envs: int, seed: int, device: str | None = None
):
    """Run an Optuna hyperparameter sweep with nested MLflow tracking."""
    cfg.ensure_dirs()
    device = device or cfg.device
    if n_envs <= 0:
        n_envs = cfg.train.n_envs

    console.print(f"[bold]🔍 Starting Optuna sweep for {n_trials} trials[/bold]")

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("rl-carracing-optuna")

    with mlflow.start_run(run_name="optuna-sweep"):

        def objective(trial: optuna.Trial) -> float:
            # ── Suggest Hyperparameters ──
            lr_init = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024])
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)

            # Ensure divisibility
            rollout_buffer = n_steps * n_envs
            if rollout_buffer % batch_size != 0:
                raise optuna.exceptions.TrialPruned()

            # ── Start nested MLflow run ──
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(
                    {
                        "learning_rate": lr_init,
                        "n_steps": n_steps,
                        "batch_size": batch_size,
                        "ent_coef": ent_coef,
                    }
                )

                train_env = build_train_env(n_envs=n_envs, seed=seed)
                eval_env = build_eval_env(seed=seed)

                model = PPO(
                    policy="CnnPolicy",
                    env=train_env,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=linear_schedule(lr_init),
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=ent_coef,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    use_sde=True,
                    sde_sample_freq=4,
                    verbose=0,
                    seed=seed,
                    device=device,
                    policy_kwargs=dict(
                        activation_fn=torch.nn.GELU,
                        ortho_init=False,
                        net_arch=dict(pi=[256], vf=[256]),
                    ),
                )

                # Train the model (short run)
                model.learn(total_timesteps=timesteps_per_trial)

                # Evaluate manually
                from stable_baselines3.common.evaluation import evaluate_policy

                mean_reward, _ = evaluate_policy(
                    model, eval_env, n_eval_episodes=5, deterministic=True
                )

                mlflow.log_metric("eval_mean_reward", mean_reward)

                train_env.close()
                eval_env.close()

                return mean_reward

        # Run the study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Log best params to the parent run
        console.print("\n[green]Tuning complete![/green]")
        console.print(f"Best reward: {study.best_value}")
        console.print(f"Best params: {study.best_params}")

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_reward", study.best_value)
