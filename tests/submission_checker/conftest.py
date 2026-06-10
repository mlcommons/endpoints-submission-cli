"""Shared pytest fixtures and builder helpers for submission checker tests.

Pre-built, anonymised §8.1 submission directories live under test_submissions/.
Each sub_* directory was generated from real measurement data with org names,
system names, and model HF-org prefixes removed.

Module-level helpers (_config, _summary, etc.) are plain functions used directly
by the split test_checks_*.py files — they aren't fixtures because they take
arguments and don't need pytest lifecycle management.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from submission_checker.models import (
    AccuracyResult,
    CheckResult,
    Division,
    ModelContext,
    PercentileStats,
    PointConfig,
    PointSummary,
    RuntimeSettings,
    Severity,
    SystemDescription,
    compute_regions,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_NODE_TYPE = {
    "system_node_ensemble_id": 0,
    "number_of_nodes": 1,
    "host_processor_model_name": "AMD EPYC",
    "host_processors_per_node": 2,
    "host_processor_core_count": 64,
    "host_memory_capacity": "512 GB",
    "accelerator_model_name": "H100",
    "accelerators_per_node": 8,
    "accelerator_memory_capacity": "80 GB",
    "host_networking": "InfiniBand",
    "host_storage_type": "NVMe",
    "host_storage_capacity": "10 TB",
    "operating_system": "Ubuntu 22.04",
}

_M = 1024
_REGIONS = compute_regions(_M)

# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _system_desc(
    division: Division = Division.STANDARDIZED,
    max_supported_concurrency: int = 1024,
    **kwargs,
) -> SystemDescription:
    return SystemDescription(
        submitter_org_names="Test Org",
        system_name="test-sys",
        system_category="datacenter",
        system_availability_status="Available",
        max_supported_concurrency=max_supported_concurrency,
        serving_framework="vLLM",
        node_types=[_NODE_TYPE],
        division=division,
        **kwargs,
    )


def _config(concurrency: int = 64, stream: bool = True, lp_type: str = "concurrency") -> PointConfig:
    return PointConfig(
        concurrency=concurrency,
        dataset="mlperf-perf-dataset-v1",
        runtime_settings=RuntimeSettings(
            load_pattern=lp_type,
            min_duration_ms=1_200_000,
            stream_all_chunks=stream,
        ),
    )


def _summary(
    n_completed: int = 1000,
    n_issued: int = 1000,
    n_failed: int = 0,
    duration_ns: float = 1_200_000_000_000.0,
    total_tokens: float = 500_000.0,
) -> PointSummary:
    return PointSummary(
        n_samples_completed=n_completed,
        n_samples_issued=n_issued,
        n_samples_failed=n_failed,
        duration_ns=duration_ns,
        ttft=PercentileStats(total=0.0, percentiles={"50": 150_000_000.0, "95": 300_000_000.0}),
        output_sequence_lengths=PercentileStats(total=total_tokens),
    )


def _passed(results: list[CheckResult]) -> bool:
    return all(r.severity != Severity.ERROR for r in results)


def _model_ctx(
    tmp_path: Path,
    all_point_count: int = 7,
    valid_points: list[tuple[Path, PointConfig]] | None = None,
    loaded_points: list[tuple[PointConfig, PointSummary]] | None = None,
    system_desc: SystemDescription | None = None,
    model_name: str = "llama3-70b",
    accuracy_result: AccuracyResult | None = None,
) -> ModelContext:
    model_dir = tmp_path / model_name
    model_dir.mkdir(exist_ok=True)
    (model_dir / "points").mkdir(exist_ok=True)
    (model_dir / "results").mkdir(exist_ok=True)
    (model_dir / "accuracy").mkdir(exist_ok=True)
    return ModelContext(
        system_id="test-sys",
        system_desc=system_desc or _system_desc(),
        model_dir=model_dir,
        regions=_REGIONS,
        points_dir=model_dir / "points",
        accuracy_dir=model_dir / "accuracy",
        all_point_count=all_point_count,
        valid_points=valid_points or [],
        loaded_points=loaded_points or [],
        accuracy_result=accuracy_result,
    )

TEST_SUBMISSIONS = Path(__file__).parent.parent.parent / "test_submissions"


@pytest.fixture(scope="session")
def sub_a() -> Path:
    """MI355X 8-GPU, gpt-oss-120b (7 points, M=2048). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_a"


@pytest.fixture(scope="session")
def sub_b() -> Path:
    """MI355X 16-GPU, gpt-oss-120b (7 points, M=2048). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_b"


@pytest.fixture(scope="session")
def sub_c() -> Path:
    """TPU 4-chip, qwen3-coder-480b (7 points, M=512). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_c"


@pytest.fixture(scope="session")
def sub_d() -> Path:
    """TPU 8-chip, qwen3-coder-480b (8 points, M=1024). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_d"


@pytest.fixture(scope="session")
def sub_e() -> Path:
    """Gaudi DP=1, llama3-8b (11 points, M=1024). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_e"


@pytest.fixture(scope="session")
def sub_f() -> Path:
    """Gaudi DP=2, llama3-8b (11 points, M=1024). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_f"


@pytest.fixture(scope="session")
def sub_g() -> Path:
    """8-GPU vLLM, llama3-70b (10 points, M=2048). Missing LL and LT coverage."""
    return TEST_SUBMISSIONS / "sub_g"


@pytest.fixture(scope="session")
def sub_h() -> Path:
    """8-GPU SGLang, llama3-70b (10 points, M=2048). Missing LL and LT coverage."""
    return TEST_SUBMISSIONS / "sub_h"


@pytest.fixture(scope="session")
def sub_i() -> Path:
    """H200 8-GPU, deepseek-r1 (10 points, M=512). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_i"


@pytest.fixture(scope="session")
def sub_j() -> Path:
    """GB300 72-GPU, deepseek-r1 (10 points, M=16384). Missing LT coverage."""
    return TEST_SUBMISSIONS / "sub_j"


@pytest.fixture(scope="session")
def valid_standardized() -> Path:
    """Fully compliant synthetic Standardized submission — should pass all checks."""
    return TEST_SUBMISSIONS / "valid_standardized"


@pytest.fixture(scope="session")
def invalid_submission() -> Path:
    """Synthetic submission with deliberate violations (3 points, failed accuracy)."""
    return TEST_SUBMISSIONS / "invalid_submission"
