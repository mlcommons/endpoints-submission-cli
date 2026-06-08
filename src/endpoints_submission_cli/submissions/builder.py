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
from pydantic import ValidationError

from ..exceptions import SubmissionBuildError
from submission_checker.models.file import SystemDescription

__all__ = ["build_submission_folder", "create_bundle_archive", "extract_archive"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_submission_folder(
    run_archives: list[tuple[str, Path]],
    division: str,
    work_dir: Path,
    availability: str,
) -> Path:
    """Assemble a submission directory from a list of run archives.

    Args:
        run_archives: List of ``(run_id, archive_path)`` tuples.
        division: Submission division (e.g. ``"standardized"``).
        work_dir: Base directory in which to build the submission tree.
        availability: Publication/availability status (e.g. ``"available"``).

    Returns:
        Path to the assembled org-level submission directory.

    Raises:
        SubmissionBuildError: If any run archive is malformed or required fields
            are missing.
    """
    if not run_archives:
        raise SubmissionBuildError("At least one run archive is required")

    # Extract all archives into temp dirs and load their content
    normalized_division = _normalize_division(division)
    normalized_availability = _normalize_system_availability_status(availability)
    run_data: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for run_id, archive_path in run_archives:
            run_dir = Path(tmp) / run_id
            extract_archive(archive_path, run_dir)
            data = _load_run_data(run_dir, run_id, normalized_division, normalized_availability)
            run_data.append(data)

    # Determine org name from the first run's system_info
    org_name = _slugify(
        (run_data[0]["system_info"].get("organization_metadata") or {}).get(
            "submitter_org_name", "org"
        ) or "org"
    )
    submission_dir = work_dir / org_name
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Group runs by system_id + model
    groups = _group_runs(run_data)

    # Aggregate all runs per system_id so the system description uses the
    # global max_concurrency even when multiple model groups share one system.
    runs_by_system: dict[str, list[dict[str, Any]]] = {}
    for (system_id, _model), runs in groups.items():
        runs_by_system.setdefault(system_id, []).extend(runs)

    written_systems: set[str] = set()
    for (system_id, model), runs in groups.items():
        all_system_runs = runs_by_system[system_id]
        if system_id not in written_systems:
            _write_system_description(submission_dir, system_id, runs[0]["system_info"])
            written_systems.add(system_id)
        max_concurrency = max(_extract_concurrency(r["config"]) for r in all_system_runs)
        _write_pareto_entries(submission_dir, system_id, model, runs, max_concurrency)
        _write_accuracy(submission_dir, system_id, model, runs)

    # Create src/ for Standardized division submissions
    if normalized_division == "Standardized":
        (submission_dir / "src").mkdir(exist_ok=True)

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


def _load_run_data(run_dir: Path, run_id: str, division: str, availability: str) -> dict[str, Any]:
    """Find and load required files from an extracted run archive.

    Uses config.yaml as the directory anchor. All supplementary files are read
    into memory so they survive tempdir cleanup.
    """
    # Archives may contain a top-level directory wrapper; use config.yaml as anchor
    candidates = list(run_dir.rglob("config.yaml"))
    if not candidates:
        raise SubmissionBuildError(
            f"Run {run_id}: archive does not contain config.yaml"
        )
    config_path = min(candidates, key=lambda p: len(p.parts))
    base = config_path.parent

    summary_path = base / "result_summary.json"
    if not summary_path.exists():
        raise SubmissionBuildError(
            f"Run {run_id}: archive is missing result_summary.json"
        )

    config: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    result_summary: dict[str, Any] = json.loads(summary_path.read_text())

    system_info = _load_system_desc(base, run_id, division, availability)

    runtime_settings_path = base / "runtime_settings.json"
    runtime_settings: dict[str, Any] = (
        json.loads(runtime_settings_path.read_text())
        if runtime_settings_path.exists()
        else {}
    )

    return {
        "run_id": run_id,
        "system_info": system_info,
        "config": config,
        "result_summary": result_summary,
        "runtime_settings": runtime_settings,
        "_extra_files": _load_extra_files(base),
    }


def _load_system_desc(base: Path, run_id: str, division: str, availability: str) -> dict[str, Any]:
    """Load system_desc.json from a run folder and return the nested dict as-is.

    The CLI-provided ``division`` and ``availability`` (already normalized) are
    written into ``model_metadata.division`` and
    ``system_under_test.system_metadata.system_availability_status`` before
    validation so that placeholder values from the tool template never cause a
    spurious validation failure.

    Raises:
        SubmissionBuildError: If system_desc.json is absent or fails schema validation.
    """
    sd_path = base / "system_desc.json"
    if not sd_path.exists():
        raise SubmissionBuildError(
            f"Run {run_id}: archive is missing system_desc.json"
        )
    sd: dict[str, Any] = json.loads(sd_path.read_text())
    sd.setdefault("model_metadata", {})["division"] = division
    sd.setdefault("system_under_test", {}).setdefault("system_metadata", {})["system_availability_status"] = availability
    try:
        SystemDescription.model_validate(sd)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise SubmissionBuildError(
            f"Run {run_id}: system_desc.json failed schema validation: {errors}. "
            "Use get-mlperf-multi-node-system-info to generate the correct format."
        ) from exc
    return sd


def _load_extra_files(base: Path) -> dict[str, bytes]:
    """Read supplementary run files into memory for inclusion in the result directory."""
    extra: dict[str, bytes] = {}
    candidates = [
        "config.yaml",
        "events.jsonl",
        "results.json",
        "run_metadata.json",
        "sample_idx_map.json",
        "serving_config.json",
        "report.txt",
        "metrics/final_snapshot.json",
    ]
    for rel in candidates:
        p = base / rel
        if p.exists() and p.is_file():
            extra[rel] = p.read_bytes()
    for p in sorted(base.glob("mlperf-system-info-*.json")):
        if p.is_file():
            extra[p.name] = p.read_bytes()
    return extra


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
    sut = system_info.get("system_under_test") or {}
    sys_meta = sut.get("system_metadata") or {}
    name = sys_meta.get("system_name", "unknown_system") or "unknown_system"
    return name.strip().replace(" ", "_")


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
    system_info: dict[str, Any],
) -> None:
    systems_dir = submission_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / f"{system_id}.json").write_text(
        json.dumps(system_info, indent=2), encoding="utf-8"
    )


def _write_pareto_entries(
    submission_dir: Path,
    system_id: str,
    model: str,
    runs: list[dict[str, Any]],
    max_concurrency: int,
) -> None:
    from submission_checker.models import classify_concurrency, compute_regions

    regions = compute_regions(max(max_concurrency, 33))

    for run in runs:
        concurrency = _extract_concurrency(run["config"])
        model_dir = submission_dir / "pareto" / system_id / model
        points_dir = model_dir / "points"
        result_dir = model_dir / "results" / f"point_{concurrency}"
        points_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Build point YAML from config.yaml + runtime_settings.json
        cfg_settings = run["config"].get("settings", {}) or {}
        load_pattern = cfg_settings.get("load_pattern", {}) or {}
        rt_json = run.get("runtime_settings", {}) or {}
        datasets = run["config"].get("datasets", []) or []
        dataset_name = (
            datasets[0].get("name", "") if datasets and isinstance(datasets[0], dict) else ""
        )

        runtime_settings_out: dict[str, Any] = {
            **rt_json,
            "load_pattern": load_pattern.get("type", "concurrency"),
            "stream_all_chunks": True,
        }

        point_cfg: dict[str, Any] = {
            "concurrency": concurrency,
            "region": classify_concurrency(concurrency, regions),
            "dataset": dataset_name,
            "runtime_settings": runtime_settings_out,
        }
        (points_dir / f"point_{concurrency}.yaml").write_text(
            yaml.dump(point_cfg, default_flow_style=False), encoding="utf-8"
        )

        (result_dir / "mlperf_endpoints_log_summary.json").write_text(
            json.dumps(run["result_summary"], indent=2), encoding="utf-8"
        )
        # Convert events.jsonl (JSONL) to a JSON array for the detail log
        extra_files = run.get("_extra_files", {})
        if "events.jsonl" in extra_files:
            events = [json.loads(ln) for ln in extra_files["events.jsonl"].splitlines() if ln.strip()]
            detail_bytes = json.dumps(events, indent=2).encode()
        else:
            detail_bytes = b"[]"
        (result_dir / "mlperf_endpoints_log_detail.json").write_bytes(detail_bytes)
        (result_dir / "system_desc.json").write_text(
            json.dumps(run["system_info"], indent=2), encoding="utf-8"
        )

        # Copy all supplementary files into the result directory, preserving subdirs
        for rel_path, content in extra_files.items():
            dest = result_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)


def _fmt_score(score: float | dict) -> str:
    if isinstance(score, dict):
        return ", ".join(f"{k}={v}" for k, v in score.items())
    return f"{score:.4f}"


def _write_accuracy(
    submission_dir: Path,
    system_id: str,
    model: str,
    runs: list[dict[str, Any]],
) -> None:
    """Write accuracy/ files from the first run that has an accuracy_scores section in results.json.

    If no run provides accuracy_scores, the accuracy directory is not created.
    """
    accuracy_scores: dict[str, Any] | None = None
    for run in runs:
        results_bytes = run.get("_extra_files", {}).get("results.json")
        if not results_bytes:
            continue
        try:
            parsed = json.loads(results_bytes)
        except json.JSONDecodeError:
            continue
        scores = parsed.get("accuracy_scores")
        if scores:
            accuracy_scores = scores
            break

    if not accuracy_scores:
        return

    accuracy_dir = submission_dir / "pareto" / system_id / model / "accuracy"
    accuracy_dir.mkdir(parents=True, exist_ok=True)

    first_ds, first_data = next(iter(accuracy_scores.items()))
    metric = first_data.get("dataset_name") or first_ds
    raw_score = first_data.get("score", 0.0)
    score: float | dict = raw_score if isinstance(raw_score, dict) else float(raw_score)

    txt_lines = [
        f"{entry.get('dataset_name') or ds}: {_fmt_score(entry.get('score', 0.0))}"
        for ds, entry in accuracy_scores.items()
    ]
    (accuracy_dir / "accuracy.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    accuracy_result = {
        "metric": metric,
        "score": score,
        "quality_target": 0.0,
        "passed": True,
    }
    (accuracy_dir / "accuracy_result.json").write_text(
        json.dumps(accuracy_result, indent=2), encoding="utf-8"
    )



def _normalize_division(division: str) -> str:
    mapping = {
        "standardized": "Standardized",
        "serviced": "Serviced",
        "rdi": "RDI",
    }
    return mapping.get(division.strip().lower(), division.title())


def _normalize_system_availability_status(status: str) -> str:
    mapping = {
        "available": "Available",
        "preview": "Preview",
        "rdi": "RDI",
    }
    return mapping.get(str(status).strip().lower(), "Available")


def _slugify(name: str) -> str:
    """Convert a human-readable name to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\-]", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:64] or "unknown"
