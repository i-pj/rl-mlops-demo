"""Metrics collection, persistence, and analysis.

Captures per-episode telemetry (reward, latency distribution, metadata)
and writes it to append-only CSV files for later comparison.

MLOps concepts demonstrated:
    - Structured metric collection
    - Experiment tracking (each run gets a unique ID)
    - Evaluation statistics (mean, std, percentiles)
    - CSV as a lightweight experiment store
"""

from __future__ import annotations

import csv
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

from rl_mlops_demo.config import LOGS_DIR, cfg

# ─── Per-Episode Record ──────────────────────────────────────────


@dataclass
class EpisodeRecord:
    """Immutable snapshot of a single evaluation episode.

    Every field here is something you'd send to a production
    telemetry pipeline (Datadog, Grafana, CloudWatch, …).
    """

    run_id: str
    timestamp: str
    device: str
    env_id: str
    deterministic: bool
    episode_num: int
    total_reward: float
    episode_length: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    python_version: str
    torch_version: str
    platform_info: str


# ─── Run-Level Aggregation ────────────────────────────────────────


@dataclass
class RunSummary:
    """Aggregated statistics across all episodes in a single run."""

    run_id: str
    num_episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_episode_length: float
    overall_mean_latency_ms: float
    overall_p95_latency_ms: float
    failure_rate: float
    passed_quality_gate: bool

    def format_report(self) -> str:
        """Pretty-print the run summary for terminal output."""
        gate_icon = "✅ PASS" if self.passed_quality_gate else "❌ FAIL"
        return (
            "\n"
            "╔══════════════════════════════════════════════════════╗\n"
            "║              📊 Run Summary                        ║\n"
            "╠══════════════════════════════════════════════════════╣\n"
            f"║  Run ID:          {self.run_id[:16]}…\n"
            f"║  Episodes:        {self.num_episodes}\n"
            f"║  Mean Reward:     {self.mean_reward:>8.1f} ± {self.std_reward:.1f}\n"
            f"║  Reward Range:    [{self.min_reward:.0f}, {self.max_reward:.0f}]\n"
            f"║  Mean Length:     {self.mean_episode_length:.0f} steps\n"
            f"║  Mean Latency:    {self.overall_mean_latency_ms:.2f} ms\n"
            f"║  P95  Latency:    {self.overall_p95_latency_ms:.2f} ms\n"
            f"║  Failure Rate:    {self.failure_rate:>7.1%}\n"
            f"║  Quality Gate:    {gate_icon}"
            f" (threshold: {cfg.thresholds.min_deploy_reward})\n"
            "╚══════════════════════════════════════════════════════╝"
        )


# ─── Metrics Collector ────────────────────────────────────────────


class MetricsCollector:
    """Collects per-step latencies and produces an EpisodeRecord.

    Usage::

        mc = MetricsCollector(run_id, model_spec)
        mc.start_episode(episode_num)
        for step in loop:
            mc.record_step(latency_ms)
        record = mc.finish_episode(total_reward, episode_length)
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._step_latencies: list[float] = []
        self._episode_num: int = 0

    def start_episode(self, episode_num: int) -> None:
        """Reset per-episode accumulators."""
        self._episode_num = episode_num
        self._step_latencies = []

    def record_step(self, latency_ms: float) -> None:
        """Record a single inference-step latency."""
        self._step_latencies.append(latency_ms)

    def finish_episode(self, total_reward: float, episode_length: int) -> EpisodeRecord:
        """Compute statistics and return a completed record."""
        lats = np.array(self._step_latencies) if self._step_latencies else np.array([0.0])
        sys_info = cfg.system_info

        return EpisodeRecord(
            run_id=self.run_id,
            timestamp=datetime.now(tz=UTC).isoformat(),
            device=str(cfg.device),
            env_id=cfg.env.env_id,
            deterministic=cfg.deterministic,
            episode_num=self._episode_num,
            total_reward=round(total_reward, 2),
            episode_length=episode_length,
            mean_latency_ms=round(float(np.mean(lats)), 3),
            p50_latency_ms=round(float(np.percentile(lats, 50)), 3),
            p95_latency_ms=round(float(np.percentile(lats, 95)), 3),
            max_latency_ms=round(float(np.max(lats)), 3),
            min_latency_ms=round(float(np.min(lats)), 3),
            python_version=sys_info["python"],
            torch_version=sys_info["torch"],
            platform_info=sys_info["platform"],
        )


# ─── CSV Writer ───────────────────────────────────────────────────


def _csv_path(run_id: str) -> Path:
    return LOGS_DIR / f"run_{run_id}.csv"


def write_records(records: Sequence[EpisodeRecord], run_id: str) -> Path:
    """Append episode records to a run-specific CSV file.

    Returns the path so callers (dashboard, CLI) can reference it.
    """
    path = _csv_path(run_id)
    file_exists = path.exists()

    with path.open("a", newline="") as f:
        fieldnames = list(asdict(records[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    return path


def summarise_run(records: Sequence[EpisodeRecord]) -> RunSummary:
    """Compute aggregate statistics from a list of episode records."""
    rewards = np.array([r.total_reward for r in records])
    lengths = np.array([r.episode_length for r in records])
    mean_lats = np.array([r.mean_latency_ms for r in records])
    p95_lats = np.array([r.p95_latency_ms for r in records])

    mean_reward = float(np.mean(rewards))
    failures = rewards < cfg.thresholds.min_deploy_reward
    failure_rate = float(np.mean(failures))
    passed_quality_gate = (
        mean_reward >= cfg.thresholds.min_deploy_reward
        and failure_rate <= cfg.thresholds.max_failure_rate
        and float(np.mean(p95_lats)) <= cfg.thresholds.max_latency_ms
    )
    return RunSummary(
        run_id=records[0].run_id,
        num_episodes=len(records),
        mean_reward=round(mean_reward, 2),
        std_reward=round(float(np.std(rewards)), 2),
        min_reward=round(float(np.min(rewards)), 2),
        max_reward=round(float(np.max(rewards)), 2),
        mean_episode_length=round(float(np.mean(lengths)), 1),
        overall_mean_latency_ms=round(float(np.mean(mean_lats)), 3),
        overall_p95_latency_ms=round(float(np.mean(p95_lats)), 3),
        failure_rate=round(failure_rate, 4),
        passed_quality_gate=passed_quality_gate,
    )


def generate_run_id() -> str:
    """Short, timestamped, unique run identifier."""
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{ts}_{short_uuid}"
