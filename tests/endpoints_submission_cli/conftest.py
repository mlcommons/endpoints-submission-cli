# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for endpoints-submission-cli tests."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Sample data paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_SAMPLE_SUBMISSIONS = _REPO_ROOT / "samplesubmissions" / "submissions"
_TEST_SUBMISSIONS = _REPO_ROOT / "test_submissions"


@pytest.fixture(scope="session")
def sample_google_run_dir() -> Path:
    """Path to a real Google run folder with system_info.json, config.yaml, result_summary.json."""
    return _SAMPLE_SUBMISSIONS / "google" / "Ironwood-192GBx4_vLLM" / "run_data" / "r1"


# ---------------------------------------------------------------------------
# Minimal run folder fixture (temp)
# ---------------------------------------------------------------------------

# Flat system_info dict used only in API mock responses (RUN_OUT).
_SYSTEM_INFO = {
    "system_name": "Test System",
    "system_category": "datacenter",
    "system_availability_status": "Available",
    "number_of_nodes": 1,
    "host_processor_model_name": "AMD EPYC 9575F",
    "host_processors_per_node": 2,
    "host_processor_core_count": 64,
    "host_memory_capacity": "512 GB",
    "host_storage_type": "NVMe SSD",
    "host_storage_capacity": "3.84 TB",
    "host_networking": "InfiniBand HDR",
    "host_networking_topology": "Single switch",
    "accelerators_per_node": 8,
    "accelerator_model_name": "NVIDIA H100 SXM5 80GB",
    "accelerator_memory_capacity": "80 GB",
    "operating_system": "Ubuntu 22.04",
    "framework": "vllm==0.4.2",
    "submitter_org_names": "TestOrg",
    "system_type_detail": "on-premise",
}

# New-format run folder files
_SYSTEM_DESC = {
    "organization_metadata": {"submitter_org_name": "TestOrg"},
    "system_under_test": {
        "system_metadata": {
            "system_name": "Test System",
            "system_category": "datacenter",
            "system_availability_status": "Available",
        },
        "serving_framework": None,
    },
}

_HW_INFO = {
    "hardware_ensemble": {
        "processor": {
            "host_processor_model_name": "AMD EPYC 9575F",
            "host_processors_per_node": 2,
            "host_processor_core_count": 64,
            "host_processor_vcpu_count": None,
        },
        "host_memory": {"host_memory_capacity": "512 GB"},
        "accelerator": {
            "accelerator_model_name": "NVIDIA H100 SXM5 80GB",
            "accelerators_per_node": 8,
            "accelerator_memory_capacity": "80 GB",
        },
        "networking": {
            "host_networking": "InfiniBand HDR",
            "host_network_card_count": "4x HDR",
        },
        "storage": {
            "host_storage_type": "NVMe SSD",
            "host_storage_capacity": "3.84 TB",
        },
    },
    "software_ensemble": {
        "operating_system": "Ubuntu 22.04",
        "other_software_stack": None,
    },
}

_SERVING_CONFIG = {"tensor_parallel": 1, "pipeline_parallel": 1, "framework": "vllm==0.4.2"}

_CONFIG = {
    "name": "test-benchmark",
    "version": "1.0",
    "type": "online",
    "model_params": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "temperature": 0.0,
        "max_new_tokens": 128,
    },
    "datasets": [{"name": "cnn_dailymail", "type": "performance"}],
    "settings": {
        "runtime": {
            "min_duration_ms": 600000,
            "max_duration_ms": 3600000,
        },
        "load_pattern": {
            "type": "concurrency",
            "target_concurrency": 4,
        },
    },
    "endpoint_config": {
        "endpoints": ["http://localhost:8080"],
        "api_type": "openai",
    },
}

_RESULT_SUMMARY = {
    "duration_ns": 475_700_000_000.0,
    "git_sha": "a1b2c3d",
    "n_samples_issued": 2000,
    "n_samples_completed": 2000,
    "qps": 4.2,
    "ttft": {
        "avg": 80_900_000.0,
        "total": 161_800_000_000.0,
        "percentiles": {
            "50": 80_500_000.0,
            "95": 150_000_000.0,
            "99": 200_000_000.0,
        },
    },
    "output_sequence_lengths": {
        "avg": 50.0,
        "total": 100_000.0,
        "percentiles": {"50": 48.0, "95": 90.0},
    },
    "tpot": {"avg": 8_700_000.0, "total": 17_400_000_000.0, "percentiles": {}},
}


@pytest.fixture
def run_folder(tmp_path: Path) -> Path:
    """Create a minimal valid run folder in a temp directory."""
    folder = tmp_path / "test_run"
    folder.mkdir()
    (folder / "system_desc.json").write_text(json.dumps(_SYSTEM_DESC))
    (folder / "mlperf-system-info-single-node-0.json").write_text(json.dumps(_HW_INFO))
    (folder / "serving_config.json").write_text(json.dumps(_SERVING_CONFIG))
    (folder / "config.yaml").write_text(yaml.dump(_CONFIG))
    (folder / "result_summary.json").write_text(json.dumps(_RESULT_SUMMARY))
    return folder


@pytest.fixture
def run_archive(run_folder: Path, tmp_path: Path) -> Path:
    """Create a .tar.gz archive of the run folder."""
    dest = tmp_path / "run.tar.gz"
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(run_folder, arcname=run_folder.name)
    return dest


# ---------------------------------------------------------------------------
# Mock API response fixtures
# ---------------------------------------------------------------------------

RUN_ID = "d5d9873e-5eca-4f8d-a487-4be1cb8b440c"
SUBMISSION_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

RUN_SUMMARY = {
    "id": RUN_ID,
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "concurrency": 4,
    "started_at": "2025-04-28T09:15:00",
    "finished_at": "2025-04-28T11:42:17",
}

RUN_OUT = {
    "id": RUN_ID,
    "user_id": "u_test",
    "benchmark_version": "a1b2c3d",
    "started_at": "2025-04-28T09:15:00",
    "finished_at": "2025-04-28T11:42:17",
    "expires_at": "2026-04-28T09:15:00",
    "pinned": False,
    "system_info": _SYSTEM_INFO,
    "config": _CONFIG,
    "result_summary": _RESULT_SUMMARY,
    "archive_uri": f"s3://bucket/runs/{RUN_ID}.tar.gz",
}

SUBMISSION_OUT = {
    "id": SUBMISSION_ID,
    "user_id": "u_test",
    "created_at": "2025-04-28T12:00:00",
    "status": "COMPLIANCE_CHECKING",
    "division": "standardized",
    "availability": "available",
    "early_publish": False,
    "publication_cycle": None,
    "target_availability_date": None,
    "availability_qualified_at": None,
    "compliance_passed_at": None,
    "first_published_at": None,
    "peer_review_started_at": None,
    "objection_resolution_started_at": None,
    "finalized_at": None,
    "withdrawn_at": None,
    "run_ids": [RUN_ID],
    "archive_uri": f"s3://bucket/submissions/{SUBMISSION_ID}.tar.gz",
    "pr_url": "https://github.com/mlcommons/submissions/pull/42",
    "pr_number": 42,
}
