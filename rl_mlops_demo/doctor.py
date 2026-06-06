"""Environment health checks for the demo.

The goal is to fail early with readable diagnostics before reaching the
live demo portion.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from rl_mlops_demo.config import MODELS_DIR, cfg

console = Console()


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _module_check(module_name: str, label: str | None = None) -> Check:
    label = label or module_name
    found = importlib.util.find_spec(module_name) is not None
    return Check(label, found, "installed" if found else "missing")


def run_doctor(*, deep: bool = False) -> bool:
    """Run setup checks and print a status table.

    Args:
        deep: When true, also create a CarRacing environment and step once.

    Returns:
        True when all required checks pass. Optional checks can fail.
    """
    checks: list[Check] = [
        Check("Python", sys.version_info >= (3, 11), sys.version.split()[0]),
        Check("Platform", True, f"{platform.system()}-{platform.machine()}"),
        Check("Torch device", True, str(cfg.device)),
        Check("Parallel envs", True, str(cfg.train.n_envs)),
        _module_check("torch"),
        _module_check("gymnasium"),
        _module_check("pygame"),
        _module_check("Box2D", "Box2D"),
        _module_check("stable_baselines3", "stable-baselines3"),
        _module_check("optuna"),
        _module_check("mlflow", "MLflow"),
    ]

    model_zips = sorted(MODELS_DIR.glob("**/*.zip"))
    model_detail = str(model_zips[0].relative_to(MODELS_DIR)) if model_zips else "no .zip found"
    checks.append(
        Check(
            "Cached model",
            bool(model_zips),
            model_detail,
        )
    )
    primary_model = MODELS_DIR / cfg.primary_model
    checks.append(
        Check(
            "Primary model",
            primary_model.exists(),
            cfg.primary_model,
        )
    )

    if deep:
        checks.append(_deep_env_check())
        checks.append(_deep_inference_contract_check())

    table = Table(title="RL MLOps Demo Doctor", show_header=True, header_style="bold cyan")
    table.add_column("Check", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Detail")

    for check in checks:
        status = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        table.add_row(check.name, status, check.detail)

    console.print(table)

    required = {
        "Python",
        "torch",
        "gymnasium",
        "pygame",
        "Box2D",
        "stable-baselines3",
    }
    if deep:
        required.add("Inference contract")
    failed_required = [c for c in checks if c.name in required and not c.passed]
    if failed_required:
        console.print("\n[red]Required checks failed.[/red] Run setup again.")
        return False

    if any(c.name == "MLflow" and not c.passed for c in checks):
        console.print(
            "\n[yellow]MLflow is missing.[/yellow] Core inference still works, "
            "but tracking demos need `uv sync` after adding the MLflow dependency."
        )

    console.print("\n[green]Core demo path is ready.[/green]")
    return True


def _deep_env_check() -> Check:
    try:
        import gymnasium as gym

        env = gym.make(cfg.env.env_id, render_mode=None)
        obs, _info = env.reset(seed=7)
        action = env.action_space.sample()
        env.step(action)
        env.close()
        return Check("CarRacing reset/step", True, f"obs_shape={getattr(obs, 'shape', '?')}")
    except Exception as exc:
        return Check("CarRacing reset/step", False, str(exc))


def _deep_inference_contract_check() -> Check:
    """Load the selected package and execute one step through the real wrapper chain."""
    vec_env = None
    try:
        from rl_mlops_demo.inference import (
            build_inference_env,
            find_latest_model,
            load_model,
        )

        model_path = find_latest_model()
        model = load_model(model_path)
        vec_env = build_inference_env(model, render_mode=None)

        obs = vec_env.reset()
        action, _state = model.predict(obs, deterministic=True)
        vec_env.step(action)
        return Check(
            "Inference contract",
            True,
            f"{model_path.name}; obs={tuple(obs.shape)}; action={tuple(action.shape)}",
        )
    except Exception as exc:
        detail = str(exc)
        if "spaces must have the same shape" in detail:
            detail = (
                f"package/preprocessing shape mismatch: {detail}. "
                "Replace the model and paired normalization stats together."
            )
        return Check("Inference contract", False, detail)
    finally:
        if vec_env is not None:
            vec_env.close()
