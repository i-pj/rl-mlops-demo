from __future__ import annotations

from dataclasses import replace

from rl_mlops_demo.metrics import EpisodeRecord, summarise_run


def _record(*, reward: float, p95_latency_ms: float = 5.0) -> EpisodeRecord:
    return EpisodeRecord(
        run_id="test-run",
        timestamp="2026-06-06T00:00:00+00:00",
        device="cpu",
        env_id="CarRacing-v3",
        deterministic=True,
        episode_num=0,
        total_reward=reward,
        episode_length=1000,
        mean_latency_ms=3.0,
        p50_latency_ms=2.5,
        p95_latency_ms=p95_latency_ms,
        max_latency_ms=8.0,
        min_latency_ms=1.0,
        python_version="3.11",
        torch_version="2.0",
        platform_info="test",
    )


def test_quality_gate_requires_every_episode_to_clear_failure_threshold() -> None:
    summary = summarise_run([_record(reward=900), _record(reward=700)])

    assert summary.mean_reward == 800
    assert summary.failure_rate == 0.5
    assert not summary.passed_quality_gate


def test_quality_gate_includes_latency() -> None:
    records = [replace(_record(reward=900), p95_latency_ms=60.0) for _ in range(3)]

    summary = summarise_run(records)

    assert summary.failure_rate == 0
    assert not summary.passed_quality_gate


def test_quality_gate_passes_for_consistent_fast_run() -> None:
    summary = summarise_run([_record(reward=850), _record(reward=920)])

    assert summary.failure_rate == 0
    assert summary.passed_quality_gate
