# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Parse a local run folder into a RunCreate API payload.

Run folder layout:

    <run_folder>/
        system_desc.json              — org/system/model/dataset metadata (flat format)
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

_REQUIRED_FILES = ("system_desc.json", "config.yaml", "result_summary.json")


def parse_run_folder(path: Path) -> dict[str, Any]:
    """Parse *path* and return a dict suitable for ``POST /runs``.

    Args:
        path: Directory containing ``system_desc.json``, ``config.yaml``,
              and ``result_summary.json``.

    Returns:
        Dict with keys matching ``RunCreate`` schema fields.

    Raises:
        RunFolderError: If any required file is absent or malformed.
    """
    path = path.resolve()
    if not path.is_dir():
        raise RunFolderError(f"Run folder does not exist or is not a directory: {path}")

    _validate_required_files(path)

    system_info = _load_json(path / "system_desc.json")
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
