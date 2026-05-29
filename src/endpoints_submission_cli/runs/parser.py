# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Parse a local run folder into a RunCreate API payload.

Run folder layout:

    <run_folder>/
        system_desc.json              — org/model/dataset metadata
        mlperf-system-info-*.json     — hardware/software info (structured)
        serving_config.json           — serving framework details
        config.yaml                   — benchmark configuration
        result_summary.json           — raw benchmark metrics
        run_metadata.json             — run-level metadata
        results.json                  — high-level result summary
        events.jsonl                  — per-event log
        report.txt                    — human-readable report
        sample_idx_map.json           — sample index mapping
        metrics/final_snapshot.json   — metrics snapshot
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from ..exceptions import RunFolderError

__all__ = ["parse_run_folder", "build_archive"]

_REQUIRED_FILES = ("config.yaml", "result_summary.json")


def parse_run_folder(path: Path) -> dict[str, Any]:
    """Parse *path* and return a dict suitable for ``POST /runs``.

    Args:
        path: Directory containing ``config.yaml`` and ``result_summary.json``,
              plus either ``system_info.json`` (old format) or
              ``system_desc.json`` / ``mlperf-system-info-*.json`` (new format).

    Returns:
        Dict with keys matching ``RunCreate`` schema fields.

    Raises:
        RunFolderError: If any required file is absent or malformed.
    """
    path = path.resolve()
    if not path.is_dir():
        raise RunFolderError(f"Run folder does not exist or is not a directory: {path}")

    _validate_required_files(path)

    system_info = _load_system_info(path)
    config = _load_yaml(path / "config.yaml")
    result_summary = _load_json(path / "result_summary.json")

    started_at, finished_at = _extract_timestamps(result_summary)
    benchmark_version = result_summary.get("git_sha") or "unknown"

    return {
        "benchmark_version": benchmark_version,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "system_info": system_info,
        "config": config,
        "result_summary": result_summary,
    }


def _validate_required_files(path: Path) -> None:
    missing = [f for f in _REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise RunFolderError(
            f"Run folder {path} is missing required file(s): {', '.join(missing)}"
        )
    has_system_info = (path / "system_desc.json").exists() or bool(
        list(path.glob("mlperf-system-info-*.json"))
    )
    if not has_system_info:
        raise RunFolderError(
            f"Run folder {path} is missing system info: "
            "expected system_desc.json or mlperf-system-info-*.json"
        )


def _load_system_info(path: Path) -> dict[str, Any]:
    """Load system info from the run folder (new format only).

    Merges system_desc.json + mlperf-system-info-*.json + serving_config.json
    into a flat dict of field names used by the rest of the code.
    """
    merged: dict[str, Any] = {}

    sd_path = path / "system_desc.json"
    if sd_path.exists():
        sd = _load_json(sd_path)
        org = sd.get("organization_metadata", {}) or {}
        sut = sd.get("system_under_test", {}) or {}
        sys_meta = sut.get("system_metadata", {}) or {}
        merged["submitter_org_names"] = org.get("submitter_org_name", "") or ""
        merged["system_name"] = sys_meta.get("system_name", "") or ""
        merged["system_category"] = sys_meta.get("system_category", "") or ""
        merged["system_availability_status"] = (
            sys_meta.get("system_availability_status", "") or ""
        )
        merged["framework"] = sut.get("serving_framework", "") or ""

    hw_paths = sorted(path.glob("mlperf-system-info-*.json"))
    if hw_paths:
        hw = _load_json(hw_paths[0])
        hw_ens = hw.get("hardware_ensemble", {}) or {}
        proc = hw_ens.get("processor", {}) or {}
        mem = hw_ens.get("host_memory", {}) or {}
        accel = hw_ens.get("accelerator", {}) or {}
        net = hw_ens.get("networking", {}) or {}
        storage = hw_ens.get("storage", {}) or {}
        sw = hw.get("software_ensemble", {}) or {}
        merged.update({
            "host_processor_model_name": proc.get("host_processor_model_name", "") or "",
            "host_processors_per_node": proc.get("host_processors_per_node", 1),
            "host_processor_core_count": proc.get("host_processor_core_count"),
            "host_processor_vcpu_count": proc.get("host_processor_vcpu_count"),
            "host_memory_capacity": mem.get("host_memory_capacity", "") or "",
            "accelerator_model_name": accel.get("accelerator_model_name", "") or "",
            "accelerators_per_node": accel.get("accelerators_per_node", 0),
            "accelerator_memory_capacity": accel.get("accelerator_memory_capacity", "") or "",
            "accelerator_memory_type": accel.get("accelerator_memory_type", "") or "",
            "accelerator_interconnect": accel.get("accelerator_interconnect", "") or "",
            "accelerator_host_interconnect": accel.get("accelerator_host_interconnect", "") or "",
            "host_networking": net.get("host_networking", "") or "",
            "host_network_card_count": net.get("host_network_card_count", "") or "",
            "host_storage_capacity": storage.get("host_storage_capacity", "") or "",
            "host_storage_type": storage.get("host_storage_type", "") or "",
            "cooling": hw_ens.get("cooling") or "",
            "operating_system": sw.get("operating_system", "") or "",
            "other_software_stack": sw.get("other_software_stack") or "",
        })

    sc_path = path / "serving_config.json"
    if sc_path.exists():
        sc = _load_json(sc_path)
        if sc.get("framework"):
            merged["framework"] = sc["framework"]

    if not merged:
        raise RunFolderError(
            f"Run folder {path} is missing system info: "
            "expected system_desc.json or mlperf-system-info-*.json"
        )

    return merged


def _load_json(file_path: Path) -> dict[str, Any]:
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunFolderError(f"Invalid JSON in {file_path.name}: {exc}") from exc


def _load_yaml(file_path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RunFolderError(
                f"{file_path.name} must be a YAML mapping, got {type(data).__name__}"
            )
        return data
    except yaml.YAMLError as exc:
        raise RunFolderError(f"Invalid YAML in {file_path.name}: {exc}") from exc


def _extract_timestamps(result_summary: dict[str, Any]) -> tuple[datetime, datetime]:
    """Derive started_at and finished_at from result_summary.

    Falls back to current time minus duration if no wall-clock timestamp is found.
    """
    now_utc = datetime.now(tz=timezone.utc)
    duration_s: float = result_summary.get("duration_ns", 0) / 1e9

    finished_at = now_utc
    started_at = finished_at - timedelta(seconds=duration_s)
    return started_at, finished_at


def build_archive(folder: Path, dest: Path | None = None) -> Path:
    """Create a .tar.gz archive of *folder*.

    Args:
        folder: Directory to archive.
        dest: Destination file path. Defaults to ``<folder.name>.tar.gz`` beside *folder*.

    Returns:
        Path of the created archive.
    """
    import tarfile

    if dest is None:
        dest = folder.parent / f"{folder.name}.tar.gz"
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)
    return dest
