# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Assemble a submission folder from downloaded run archives.

Transforms run-folder data (system_info.json, config.yaml, result_summary.json)
into the SubmissionChecker-compatible layout:

    <org>/
      systems/<system_id>.json
      pareto/<system_id>/<model>/points/point_<concurrency>.yaml
      pareto/<system_id>/<model>/results/point_<concurrency>/
          mlperf_endpoints_log_summary.json
          mlperf_endpoints_log_detail.json
      pareto/<system_id>/<model>/accuracy/
          accuracy_result.json
          accuracy.txt
"""

from __future__ import annotations

import json
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .exceptions import SubmissionBuildError

__all__ = ["build_submission_folder", "create_bundle_archive", "extract_archive"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_submission_folder(
    run_archives: list[tuple[str, Path]],
    division: str,
    work_dir: Path,
) -> Path:
    """Assemble a submission directory from a list of run archives.

    Args:
        run_archives: List of ``(run_id, archive_path)`` tuples.
        division: Submission division (e.g. ``"standardized"``).
        work_dir: Base directory in which to build the submission tree.

    Returns:
        Path to the assembled org-level submission directory.

    Raises:
        SubmissionBuildError: If any run archive is malformed or required fields
            are missing.
    """
    if not run_archives:
        raise SubmissionBuildError("At least one run archive is required")

    # Extract all archives into temp dirs and load their content
    run_data: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for run_id, archive_path in run_archives:
            run_dir = Path(tmp) / run_id
            extract_archive(archive_path, run_dir)
            data = _load_run_data(run_dir, run_id)
            run_data.append(data)

    # Determine org name from the first run's system_info
    org_name = _slugify(
        run_data[0]["system_info"].get("submitter_org_names", "org") or "org"
    )
    submission_dir = work_dir / org_name
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Group runs by system_id + model
    groups = _group_runs(run_data)

    for (system_id, model), runs in groups.items():
        _write_system_description(submission_dir, system_id, model, runs, division)
        _write_pareto_entries(submission_dir, system_id, model, runs)
        _write_accuracy_placeholders(submission_dir, system_id, model)

    return submission_dir


def create_bundle_archive(submission_dir: Path, dest: Path | None = None) -> Path:
    """Create a .tar.gz archive of *submission_dir*.

    Args:
        submission_dir: The assembled org-level directory.
        dest: Destination path. Defaults to ``<submission_dir.name>.tar.gz`` beside it.

    Returns:
        Path of the created archive.
    """
    if dest is None:
        dest = submission_dir.parent / f"{submission_dir.name}.tar.gz"
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(submission_dir, arcname=submission_dir.name)
    return dest


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive into *dest_dir*.

    Raises:
        SubmissionBuildError: If extraction fails.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Security: filter absolute paths and path traversal
            members = [m for m in tar.getmembers() if not m.name.startswith("/")]
            tar.extractall(dest_dir, members=members)
    except (tarfile.TarError, OSError) as exc:
        raise SubmissionBuildError(f"Failed to extract {archive_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_run_data(run_dir: Path, run_id: str) -> dict[str, Any]:
    """Find and load the three required files from an extracted run archive."""
    # Archives may contain a top-level directory wrapper
    candidates = list(run_dir.rglob("system_info.json"))
    if not candidates:
        raise SubmissionBuildError(
            f"Run {run_id}: archive does not contain system_info.json"
        )
    # Use the shallowest match
    system_info_path = min(candidates, key=lambda p: len(p.parts))
    base = system_info_path.parent

    config_path = base / "config.yaml"
    summary_path = base / "result_summary.json"

    for p in (config_path, summary_path):
        if not p.exists():
            raise SubmissionBuildError(
                f"Run {run_id}: archive is missing {p.name}"
            )

    system_info: dict[str, Any] = json.loads(system_info_path.read_text())
    config: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    result_summary: dict[str, Any] = json.loads(summary_path.read_text())

    return {
        "run_id": run_id,
        "system_info": system_info,
        "config": config,
        "result_summary": result_summary,
    }


def _group_runs(
    run_data: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group runs by (system_id, model) key."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for data in run_data:
        system_id = _extract_system_id(data["system_info"])
        model = _extract_model(data["config"])
        key = (system_id, model)
        groups.setdefault(key, []).append(data)
    return groups


def _extract_system_id(system_info: dict[str, Any]) -> str:
    name = system_info.get("system_name", "unknown_system") or "unknown_system"
    return _slugify(name)


def _extract_model(config: dict[str, Any]) -> str:
    model_params = config.get("model_params", {}) or {}
    name = model_params.get("name", "") or ""
    if name:
        # Use the last path component (e.g. "meta-llama/Llama-3.1-8B" → "Llama-3.1-8B")
        return _slugify(name.split("/")[-1])
    return "unknown_model"


def _extract_concurrency(config: dict[str, Any]) -> int:
    settings = config.get("settings", {}) or {}
    load_pattern = settings.get("load_pattern", {}) or {}
    concurrency = load_pattern.get("target_concurrency")
    if concurrency is not None:
        return int(concurrency)
    # Fallback: look at top-level target_concurrency
    return int(config.get("target_concurrency", 1))


def _write_system_description(
    submission_dir: Path,
    system_id: str,
    model: str,
    runs: list[dict[str, Any]],
    division: str,
) -> None:
    systems_dir = submission_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)

    # Derive max_supported_concurrency from the highest concurrency across all runs
    concurrencies = [_extract_concurrency(r["config"]) for r in runs]
    max_concurrency = max(concurrencies) if concurrencies else 64

    si = runs[0]["system_info"]
    cfg = runs[0]["config"]

    endpoint_url = ""
    ep_cfg = cfg.get("endpoint_config", {}) or {}
    endpoints = ep_cfg.get("endpoints", []) or []
    if endpoints:
        endpoint_url = str(endpoints[0])

    system_desc = {
        "division": _normalize_division(division),
        "publication_status": _normalize_publication_status(
            si.get("system_availability_status", "Available")
        ),
        "benchmark_model": model,
        "max_supported_concurrency": max(max_concurrency, 33),
        "endpoint_url": endpoint_url or "http://localhost:8080",
        "serving_framework": si.get("framework", "") or "",
        "submitter": si.get("submitter_org_names", "") or "",
        "system_name": si.get("system_name", system_id) or system_id,
        "system_type": si.get("system_category", "datacenter") or "datacenter",
        "system_type_detail": si.get("system_type_detail", "") or "",
        "number_of_nodes": int(si.get("number_of_nodes", 1) or 1),
        "host_processors_per_node": int(si.get("host_processors_per_node", 1) or 1),
        "host_processor_model_name": si.get("host_processor_model_name", "") or "",
        "host_processor_core_count": si.get("host_processor_core_count") or None,
        "host_processor_vcpu_count": si.get("host_processor_vcpu_count") or None,
        "host_memory_capacity": str(si.get("host_memory_capacity", "0 GB") or "0 GB"),
        "host_storage_type": si.get("host_storage_type", "") or "",
        "host_storage_capacity": si.get("host_storage_capacity", "") or "",
        "host_networking": si.get("host_networking", "") or "",
        "host_networking_topology": si.get("host_networking_topology", "") or "",
        "accelerators_per_node": int(si.get("accelerators_per_node", 0) or 0),
        "accelerator_model_name": si.get("accelerator_model_name", "") or "",
        "accelerator_memory_capacity": si.get("accelerator_memory_capacity", "") or "",
        "operating_system": si.get("operating_system", "") or "",
        "accelerator_host_interconnect": si.get("accelerator_host_interconnect", "") or "",
        "accelerator_interconnect": si.get("accelerator_interconnect", "") or "",
        "accelerator_memory_type": si.get("accelerator_memory_type", "") or "",
        "other_software_stack": si.get("other_software_stack", "") or "",
        "cooling": si.get("cooling", "") or "",
    }
    # Ensure at least one of core_count / vcpu_count is set
    if (
        system_desc["host_processor_core_count"] is None
        and system_desc["host_processor_vcpu_count"] is None
    ):
        system_desc["host_processor_core_count"] = 1

    (systems_dir / f"{system_id}.json").write_text(
        json.dumps(system_desc, indent=2), encoding="utf-8"
    )


def _write_pareto_entries(
    submission_dir: Path,
    system_id: str,
    model: str,
    runs: list[dict[str, Any]],
) -> None:
    for run in runs:
        concurrency = _extract_concurrency(run["config"])
        model_dir = submission_dir / "pareto" / system_id / model
        points_dir = model_dir / "points"
        result_dir = model_dir / "results" / f"point_{concurrency}"
        points_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        # point YAML
        point_cfg = {
            "concurrency": concurrency,
            "dataset": _extract_dataset(run["config"]),
            "runtime_settings": {
                "load_pattern": "concurrency",
                "min_duration_ms": _extract_min_duration_ms(run["config"]),
                "stream_all_chunks": True,
            },
        }
        (points_dir / f"point_{concurrency}.yaml").write_text(
            yaml.dump(point_cfg, default_flow_style=False), encoding="utf-8"
        )

        # log summary — map result_summary fields to PointSummary schema
        summary = _build_log_summary(run["result_summary"])
        (result_dir / "mlperf_endpoints_log_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        # detail log — empty object is acceptable
        (result_dir / "mlperf_endpoints_log_detail.json").write_text(
            "{}", encoding="utf-8"
        )


def _write_accuracy_placeholders(
    submission_dir: Path,
    system_id: str,
    model: str,
) -> None:
    accuracy_dir = submission_dir / "pareto" / system_id / model / "accuracy"
    accuracy_dir.mkdir(parents=True, exist_ok=True)
    (accuracy_dir / "accuracy.txt").write_text("Accuracy pending\n", encoding="utf-8")
    placeholder = {
        "metric": "rouge1",
        "score": 0.0,
        "quality_target": 0.0,
        "passed": False,
    }
    (accuracy_dir / "accuracy_result.json").write_text(
        json.dumps(placeholder, indent=2), encoding="utf-8"
    )


def _build_log_summary(result_summary: dict[str, Any]) -> dict[str, Any]:
    """Map raw result_summary.json to mlperf_endpoints_log_summary.json format."""
    ttft_raw = result_summary.get("ttft", {}) or {}
    osl_raw = result_summary.get("output_sequence_lengths", {}) or {}

    return {
        "n_samples_issued": result_summary.get("n_samples_issued", 0),
        "n_samples_completed": result_summary.get("n_samples_completed", 0),
        "n_samples_failed": 0,
        "duration_ns": float(result_summary.get("duration_ns", 0)),
        "ttft": {
            "total": float(ttft_raw.get("total", 0)),
            "percentiles": ttft_raw.get("percentiles", {}),
        },
        "output_sequence_lengths": {
            "total": float(osl_raw.get("total", 0)),
            "percentiles": osl_raw.get("percentiles", {}),
        },
    }


def _extract_dataset(config: dict[str, Any]) -> str:
    datasets = config.get("datasets", []) or []
    if datasets:
        return datasets[0].get("name", "") if isinstance(datasets[0], dict) else str(datasets[0])
    return ""


def _extract_min_duration_ms(config: dict[str, Any]) -> int:
    settings = config.get("settings", {}) or {}
    runtime = settings.get("runtime", {}) or {}
    return int(runtime.get("min_duration_ms", 600_000))


def _normalize_division(division: str) -> str:
    mapping = {
        "standardized": "Standardized",
        "serviced": "Serviced",
        "rdi": "RDI",
    }
    return mapping.get(division.lower(), division.title())


def _normalize_publication_status(status: str) -> str:
    mapping = {
        "available": "Available",
        "preview": "Preview",
        "rdi": "RDI",
    }
    return mapping.get(str(status).lower(), "Available")


def _slugify(name: str) -> str:
    """Convert a human-readable name to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\-]", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:64] or "unknown"
