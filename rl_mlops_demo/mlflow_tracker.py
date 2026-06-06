"""MLflow tracking helpers and UI launcher."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from rl_mlops_demo.config import PROJECT_ROOT, cfg

if TYPE_CHECKING:
    from rl_mlops_demo.metrics import EpisodeRecord, RunSummary

console = Console()
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"


def _require_mlflow():
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow is not installed. Run `uv sync` after installing dependencies."
        ) from exc
    return mlflow


def log_evaluation_run(
    *,
    model_path: Path,
    records: list[EpisodeRecord],
    summary: RunSummary,
    csv_path: Path,
    deterministic: bool,
    run_type: str = "evaluation",
) -> str:
    """Persist an evaluation run as reviewable MLflow evidence."""
    mlflow = _require_mlflow()
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("rl-carracing-evaluation")

    with mlflow.start_run(run_name=f"{run_type}-{summary.run_id}") as run:
        resolved_model_path = model_path.resolve()
        mlflow.log_params(
            {
                "model_path": str(resolved_model_path.relative_to(PROJECT_ROOT)),
                "model_file": model_path.name,
                "env_id": cfg.env.env_id,
                "device": str(cfg.device),
                "deterministic": deterministic,
                "num_episodes": summary.num_episodes,
                "frame_skip": 2,
                "frame_stack": 2,
                "normalization_stats": any(model_path.parent.glob("vec*normalize*.pkl"))
                or any(model_path.parent.glob("vecnorm_*.pkl")),
                "model_origin": (
                    "external-pretrained"
                    if "hf-vukpetar-ppo-carracing-v0-v3" in str(model_path)
                    else "workshop-trained"
                ),
                "policy_contract": (
                    "raw-rgb-96x96-transposed"
                    if "hf-vukpetar-ppo-carracing-v0-v3" in str(model_path)
                    else "stacked-grayscale-2x64x64"
                ),
            }
        )
        mlflow.log_metrics(
            {
                "mean_reward": summary.mean_reward,
                "std_reward": summary.std_reward,
                "min_reward": summary.min_reward,
                "max_reward": summary.max_reward,
                "mean_episode_length": summary.mean_episode_length,
                "mean_latency_ms": summary.overall_mean_latency_ms,
                "p95_latency_ms": summary.overall_p95_latency_ms,
                "failure_rate": summary.failure_rate,
                "quality_gate_pass": float(summary.passed_quality_gate),
            }
        )
        mlflow.set_tags(
            {
                "run_type": run_type,
                "gate_result": "pass" if summary.passed_quality_gate else "fail",
                "model_family": "PPO",
                "workshop_run_id": summary.run_id,
            }
        )
        mlflow.log_artifact(str(csv_path), artifact_path="evaluation")

        config_path = csv_path.with_suffix(".config.json")
        config_path.write_text(
            json.dumps(
                {
                    "model_path": str(resolved_model_path),
                    "system": cfg.system_info,
                    "environment": asdict(cfg.env),
                    "thresholds": asdict(cfg.thresholds),
                    "deterministic": deterministic,
                    "records": len(records),
                },
                indent=2,
            )
        )
        mlflow.log_artifact(str(config_path), artifact_path="evaluation")
        return run.info.run_id


def launch_mlflow_ui() -> None:
    """Launch the MLflow UI pointed at the local file store."""
    _require_mlflow()
    MLARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    console.print("[bold]Launching MLflow UI at http://127.0.0.1:5000[/bold]")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            f"sqlite:///{MLFLOW_DB}",
            "--default-artifact-root",
            str(MLARTIFACTS_DIR),
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ],
        check=True,
    )
