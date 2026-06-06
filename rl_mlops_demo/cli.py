"""CLI entrypoint — the single command interact with.

Subcommands:
    train     Train a PPO agent with MLflow tracking
    tune      Run Optuna hyperparameter sweep
    doctor    Check whether the demo environment is ready
    demo      Run the visual demo (pygame window)
    eval      Run headless evaluation (N episodes, no window)
    config    Print the current system and model configuration
    record    Record one deterministic evidence video
    mlflow-ui Launch the MLflow UI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from rl_mlops_demo.config import cfg

console = Console()


def _print_banner() -> None:
    """Print the workshop banner."""
    banner = r"""
[bold cyan]
  ╔══════════════════════════════════════════════════════════╗
  ║   🏎️  RL MLOps Demo From RL Agent to Production          ║
  ╚══════════════════════════════════════════════════════════╝
[/bold cyan]
"""
    console.print(banner, highlight=False)


# ── Subcommand: doctor ───────────────────────────────────────────


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run environment checks."""
    _print_banner()
    from rl_mlops_demo.doctor import run_doctor

    success = run_doctor(deep=args.deep)
    sys.exit(0 if success else 1)


# ── Subcommand: train ────────────────────────────────────────────


def cmd_train(args: argparse.Namespace) -> None:
    """Train a new model."""
    _print_banner()
    from rl_mlops_demo.train import run_training

    run_training(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
    )


# ── Subcommand: tune ─────────────────────────────────────────────


def cmd_tune(args: argparse.Namespace) -> None:
    """Run Optuna hyperparameter sweep."""
    _print_banner()
    from rl_mlops_demo.tune import run_tuning

    run_tuning(
        n_trials=args.n_trials,
        timesteps_per_trial=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
    )


# ── Subcommand: eval ─────────────────────────────────────────────


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation."""
    _print_banner()
    from rl_mlops_demo.inference import find_latest_model, load_model, run_evaluation

    # Optional logic to clear old logs
    if args.clean:
        from rl_mlops_demo.config import LOGS_DIR

        for f in LOGS_DIR.glob("*.csv"):
            f.unlink()
        console.print("[dim]Cleaned old CSV logs.[/dim]")

    model_path = find_latest_model()
    model = load_model(model_path)
    run_evaluation(
        model,
        model_path,
        num_episodes=args.n,
        deterministic=not args.stochastic,
    )


# ── Subcommand: demo ─────────────────────────────────────────────


def cmd_demo(args: argparse.Namespace) -> None:
    """Run visual demo."""
    _print_banner()
    from rl_mlops_demo.inference import find_latest_model, load_model, run_visual_demo

    if args.clean:
        from rl_mlops_demo.config import LOGS_DIR

        for f in LOGS_DIR.glob("*.csv"):
            f.unlink()

    model_path = find_latest_model()
    model = load_model(model_path)
    run_visual_demo(model, model_path, num_episodes=1)


def cmd_record(args: argparse.Namespace) -> None:
    """Record one evidence video using a selected package."""
    _print_banner()
    from rl_mlops_demo.inference import find_latest_model, load_model, record_evidence_video

    model_path = Path(args.model).resolve() if args.model else find_latest_model()
    model = load_model(model_path)
    record_evidence_video(model, Path(args.output).resolve(), max_steps=args.max_steps)


# ── Subcommand: config ───────────────────────────────────────────


def cmd_config(args: argparse.Namespace) -> None:
    """Print the active configuration."""
    _print_banner()
    from rich.pretty import pprint

    console.print("\n[bold]Current System & Inference Configuration:[/bold]")
    pprint(cfg)


# ── Subcommand: mlflow-ui ────────────────────────────────────────


def cmd_mlflow_ui(args: argparse.Namespace) -> None:
    """Launch MLflow UI."""
    _print_banner()
    from rl_mlops_demo.mlflow_tracker import launch_mlflow_ui

    launch_mlflow_ui()


# ── CLI Setup ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rl-demo",
        description="RL MLOps Demo CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor
    p_doctor = sub.add_parser("doctor", help="Check whether the demo environment is ready")
    p_doctor.add_argument("--deep", action="store_true", help="Run slow inference checks")
    p_doctor.set_defaults(func=cmd_doctor)

    # train
    p_train = sub.add_parser("train", help="Train a PPO agent")
    p_train.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps")
    p_train.add_argument(
        "--n-envs", type=int, default=0, help="Number of parallel environments (0=auto)"
    )
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps", "auto"],
        help="Override training device",
    )
    p_train.set_defaults(func=cmd_train)

    # tune
    p_tune = sub.add_parser("tune", help="Run Optuna hyperparameter sweep")
    p_tune.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    p_tune.add_argument("--timesteps", type=int, default=200_000, help="Timesteps per trial")
    p_tune.add_argument(
        "--n-envs", type=int, default=0, help="Number of parallel environments (0=auto)"
    )
    p_tune.add_argument("--seed", type=int, default=42, help="Random seed")
    p_tune.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps", "auto"],
        help="Override training device",
    )
    p_tune.set_defaults(func=cmd_tune)

    # eval
    p_eval = sub.add_parser("eval", help="Run headless evaluation")
    p_eval.add_argument("-n", type=int, default=5, help="Number of episodes")
    p_eval.add_argument("--stochastic", action="store_true", help="Use stochastic actions")
    p_eval.add_argument("--clean", action="store_true", help="Delete old CSVs first")
    p_eval.set_defaults(func=cmd_eval)

    # demo
    p_demo = sub.add_parser("demo", help="Run visual demo (pygame window)")
    p_demo.add_argument("--clean", action="store_true", help="Delete old CSVs first")
    p_demo.set_defaults(func=cmd_demo)

    # record
    p_record = sub.add_parser("record", help="Record one deterministic evidence video")
    p_record.add_argument("--model", type=str, help="Model path; defaults to selected package")
    p_record.add_argument("--output", type=str, required=True, help="Output MP4 path")
    p_record.add_argument("--max-steps", type=int, default=600, help="Maximum recorded steps")
    p_record.set_defaults(func=cmd_record)

    # config
    p_config = sub.add_parser("config", help="Print configuration")
    p_config.set_defaults(func=cmd_config)

    # mlflow-ui
    p_mlflow_ui = sub.add_parser("mlflow-ui", help="Launch the MLflow UI")
    p_mlflow_ui.set_defaults(func=cmd_mlflow_ui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
