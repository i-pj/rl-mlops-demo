# RL MLOps Demo: From Agent to Operated System

An educational MLOps workshop codebase built around a Reinforcement Learning agent for Gymnasium's `CarRacing-v3`. 

This repository provides a clean, professional environment for learning how to evaluate, track, and review ML models. It separates the "science" of training from the "engineering" of operating an ML system.

## What you will learn
- Setting up repeatable Python dependencies using `uv`.
- Establishing a reliable device-agnostic execution environment.
- Using MLflow to track parameters, metrics, and models.
- Evaluating model candidates using a deterministic, headless pipeline.
- Visualizing behavior and reviewing evidence for release gates.

## Student Setup

The project uses `uv` for strict dependency management.

```bash
# 1. Sync dependencies
uv sync

# 2. Verify the execution environment and model contract
uv run rl-demo doctor --deep
```

### Device Selection
The configuration defaults to `device = "auto"`. PyTorch will automatically select `cuda` if an NVIDIA GPU is available, fall back to `mps` on Apple Silicon, and use `cpu` otherwise. No code edits are required to run this on different machines.

## Provided Classroom Checkpoint
Training RL agents from scratch takes hours. To ensure a reliable workshop experience, this repository provides a known-good baseline model:
* **Location:** `models/provided-fallback/fallback-model.zip`
* **Note:** This is an external checkpoint downloaded from Hugging Face (`vukpetar/ppo-CarRacing-v0-v3`), retained explicitly for its stable performance across different environments. Refer to `models/MODEL_CARD.md` for full provenance details.

## Workshop Commands

The CLI exposes the following core operations:

```bash
# Verify environment readiness
uv run rl-demo doctor --deep

# Evaluate the model headlessly (generates evidence without a window)
uv run rl-demo eval -n 5

# Visually demo the model's behavior
uv run rl-demo demo

# Open MLflow to review the logged evidence
uv run rl-demo mlflow-ui
```

### Optional Training
If you wish to experiment with training a model yourself:
```bash
uv run rl-demo train
```
*Note: Depending on your hardware, a model may need 2M+ steps to converge to reasonable behavior.*

## Troubleshooting
- **No GUI / Video errors:** If `demo` fails, ensure you are not running in a headless SSH environment or try `eval` first to verify the inference logic works independently of the display.
- **MLflow errors:** Ensure you are in the `rl-mlops-demo` directory before launching `mlflow-ui`.
