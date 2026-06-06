"""Custom Stable-Baselines3 callbacks for MLOps tracking and monitoring.

Includes MLflow integration, checkpointing with VecNormalize stats,
and live progress bars.
"""

import os

try:
    import mlflow
except ImportError:
    mlflow = None

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

METRIC_KEYS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/value_loss",
    "train/policy_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/learning_rate",
]


class MLflowCallback(BaseCallback):
    """Stream SB3 training metrics to MLflow.

    Logs at _on_rollout_start() which fires AFTER PPO.train() has
    populated the logger dict from the previous cycle.
    """

    def __init__(self):
        super().__init__()
        self._last_logged_step = 0

    def _on_rollout_start(self) -> None:
        if not mlflow or not mlflow.active_run():
            return

        # Skip the very first rollout (no training data yet)
        if self.num_timesteps <= self._last_logged_step:
            return

        metrics = {}
        logger_dict = self.model.logger.name_to_value

        for key in METRIC_KEYS:
            if key in logger_dict:
                metrics[key.replace("/", ".")] = logger_dict[key]

        # Also log FPS and wallclock
        if "time/fps" in logger_dict:
            metrics["time.fps"] = logger_dict["time/fps"]

        if metrics:
            mlflow.log_metrics(metrics, step=self.num_timesteps)
            self._last_logged_step = self.num_timesteps

    def _on_step(self) -> bool:
        return True


class NormalizeCheckpointCallback(CheckpointCallback):
    """Extends CheckpointCallback to also save VecNormalize statistics.

    If training crashes, the recovered checkpoint's model is useless without
    its corresponding vec_normalize.pkl. This custom callback saves both together.
    """

    def __init__(self, save_freq: int, save_path: str, vec_normalize_env, **kwargs):
        super().__init__(save_freq=save_freq, save_path=save_path, **kwargs)
        self.vec_normalize_env = vec_normalize_env

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            norm_path = os.path.join(self.save_path, f"vecnorm_{self.num_timesteps}_steps.pkl")
            self.vec_normalize_env.save(norm_path)
        return result


class RichProgressCallback(BaseCallback):
    """Display a Rich progress bar during training.

    With `verbose=0` and no output, a 30-minute training run looks completely frozen.
    This provides a live progress bar updated at the end of each rollout.
    """

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress = None

    def _on_training_start(self):
        from rich.progress import BarColumn, Progress, TimeRemainingColumn

        self.progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        )
        self.task = self.progress.add_task("Training", total=self.total_timesteps)
        self.progress.start()

    def _on_step(self) -> bool:
        # Update progress bar
        if self.progress:
            self.progress.update(self.task, completed=self.num_timesteps)
        return True

    def _on_training_end(self):
        if self.progress:
            self.progress.stop()
