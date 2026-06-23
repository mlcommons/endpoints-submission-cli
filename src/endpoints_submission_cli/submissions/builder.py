# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Assemble a submission folder from downloaded run archives.

Transforms run-folder data (system_info.json, config.yaml, result_summary.json)
into the SubmissionChecker-compatible layout:

    <org>/
      systems/<system_id>.json
      src/<benchmark_model>/
          <endpoint interface code>
      pareto/<system_id>/<model>/points/point_<concurrency>.yaml
      pareto/<system_id>/<model>/results/point_<concurrency>/
          results_summary.json
          config.yaml
          accuracy/
              results.json
      documentation/
"""

from __future__ import annotations

import json
import re
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from submission_checker.models.file import SystemDescription

from ..exceptions import SubmissionBuildError

__all__ = ["build_submission_folder", "create_bundle_archive", "extract_archive"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_submission_folder(
    run_archives: list[tuple[str, Path]],
    division: str,
    availability: str,
    work_dir: Path,
) -> Path:
    """Assemble a submission directory from a list of run archives.

    Args:
        run_archives: List of ``(run_id, archive_path)`` tuples.
        division: Submission division (e.g. ``"standardized"``).
        availability: Publication/availability status (e.g. ``"available"``).
        work_dir: Base directory in which to build the submission tree.

    Returns:
        Path to the assembled org-level submission directory.

    Raises:
        SubmissionBuildError: If any run archive is malformed or required fields
            are missing.
    """
    if not run_archives:
        raise SubmissionBuildError("At least one run archive is required")

    # Extract all archives into temp dirs and load their content.
    # division and availability normalization is handled by SystemDescription validators.
    run_data: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for run_id, archive_path in run_archives:
            run_dir = Path(tmp) / run_id
            extract_archive(archive_path, run_dir)
            data = _load_run_data(run_id, division, availability, run_dir)
            run_data.append(data)

    if len(run_data) > 1:
        first = run_data[0]["system_info"]
        if not all(r["system_info"] == first for r in run_data[1:]):
            warnings.warn(
                "Runs have inconsistent system_info; using the first run's data",
                stacklevel=2,
            )

    # Determine org name from the first run's system_info
    org_name = _slugify(run_data[0]["system_info"].get("submitter_org_names", "org") or "org")
    submission_dir = work_dir / org_name
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Group runs by system_id + model
    groups = _group_runs(run_data)

    runs_by_system: dict[str, list[dict[str, Any]]] = {}
    for (system_id, _model), runs in groups.items():
        runs_by_system.setdefault(system_id, []).extend(runs)

    # Validate max_supported_concurrency consistency per system before writing anything
    system_max_concurrency: dict[str, int] = {}
    for system_id, system_runs in runs_by_system.items():
        values = {r["system_info"].get("max_supported_concurrency") for r in system_runs}
        if None in values:
            raise SubmissionBuildError(
                f"System {system_id}: system_desc.json is missing max_supported_concurrency"
            )
        if len(values) > 1:
            raise SubmissionBuildError(
                f"System {system_id}: runs have inconsistent max_supported_concurrency"
                f" values: {sorted(values)}"
            )
        system_max_concurrency[system_id] = int(values.pop())

    max_tps = _compute_max_tps(run_data)

    written_systems: set[str] = set()
    for (system_id, model), runs in groups.items():
        if system_id not in written_systems:
            _write_system_description(submission_dir, system_id, runs[0]["system_info"])
            written_systems.add(system_id)
        _write_pareto_entries(
            submission_dir, system_id, model, runs, system_max_concurrency[system_id], max_tps
        )

    # Copy src/ for Standardized division submissions (mirrors documentation/ handling)
    if run_data[0]["system_info"].get("division") == "Standardized":
        _write_src(submission_dir, run_data)

    _write_documentation(submission_dir, run_data)

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


def _load_run_data(run_id: str, division: str, availability: str, run_dir: Path) -> dict[str, Any]:
    """Find and load required files from an extracted run archive.

    Uses config.yaml as the directory anchor. All supplementary files are read
    into memory so they survive tempdir cleanup.
    """
    # Archives may contain a top-level directory wrapper; use config.yaml as anchor
    candidates = list(run_dir.rglob("config.yaml"))
    if not candidates:
        raise SubmissionBuildError(f"Run {run_id}: archive does not contain config.yaml")
    config_path = min(candidates, key=lambda p: len(p.parts))
    base = config_path.parent

    summary_path = base / "result_summary.json"
    if not summary_path.exists():
        raise SubmissionBuildError(f"Run {run_id}: archive is missing result_summary.json")

    config: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    result_summary: dict[str, Any] = json.loads(summary_path.read_text())

    system_info = _load_system_desc(base, run_id, division, availability)

    return {
        "run_id": run_id,
        "system_info": system_info,
        "config": config,
        "result_summary": result_summary,
        "_extra_files": _load_extra_files(base),
    }


def _load_system_desc(base: Path, run_id: str, division: str, availability: str) -> dict[str, Any]:
    """Load system_desc.json from a run folder and return the validated flat dict.

    Applies the CLI-provided division and availability values, then validates
    against the SystemDescription schema.

    Raises:
        SubmissionBuildError: If system_desc.json is absent or fails schema validation.
    """
    sd_path = base / "system_desc.json"
    if not sd_path.exists():
        raise SubmissionBuildError("Run archive is missing system_desc.json")
    raw: dict[str, Any] = json.loads(sd_path.read_text())
    # CLI-provided values are authoritative for these fields
    raw["division"] = division
    raw["system_availability_status"] = availability
    try:
        sd = SystemDescription.model_validate(raw)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise SubmissionBuildError(
            f"Run {run_id}: system_desc.json failed schema validation: {errors}"
        ) from exc
    return sd.model_dump(mode="json")


def _load_extra_files(base: Path) -> dict[str, bytes]:
    """Read supplementary run files into memory for inclusion in the result directory."""
    extra: dict[str, bytes] = {}
    candidates = [
        "config.yaml",
        "results.json",
        "run_metadata.json",
        "sample_idx_map.json",
        "serving_config.json",
        "report.txt",
        "metrics/final_snapshot.json",
        "accuracy/results.json",
    ]
    for rel in candidates:
        p = base / rel
        if p.exists() and p.is_file():
            extra[rel] = p.read_bytes()
    for p in sorted(base.glob("mlperf-system-info-*.json")):
        if p.is_file():
            extra[p.name] = p.read_bytes()
    for subdir_name in ("documentation", "src"):
        subdir = base / subdir_name
        if subdir.is_dir():
            for p in sorted(subdir.rglob("*")):
                if p.is_file():
                    extra[str(p.relative_to(base))] = p.read_bytes()
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
    sd = SystemDescription.model_validate(system_info)
    return sd.system_name.strip().replace(" ", "_")


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


def _extract_run_type(config: dict[str, Any]) -> str:
    """Return 'accuracy' or 'performance' based on the first dataset type."""
    datasets = config.get("datasets", []) or []
    if datasets and isinstance(datasets[0], dict) and datasets[0].get("type") == "accuracy":
        return "accuracy"
    return "performance"


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


def _compute_max_tps(run_data: list[dict[str, Any]]) -> float | None:
    """Return the max system_tps across all runs, or None if unavailable."""
    values = []
    for run in run_data:
        meta_bytes = run.get("_extra_files", {}).get("run_metadata.json")
        if not meta_bytes:
            continue
        try:
            tps = json.loads(meta_bytes).get("system_tps")
            if tps is not None:
                values.append(float(tps))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return max(values) if values else None


def _write_pareto_entries(
    submission_dir: Path,
    system_id: str,
    model: str,
    runs: list[dict[str, Any]],
    max_concurrency: int,
    max_tps: float | None = None,
) -> None:
    from submission_checker.models import classify_concurrency, compute_regions

    regions = compute_regions(max_concurrency)

    # At most 1 perf + 1 acc run per concurrency (may change; stated here explicitly).
    seen: set[tuple[int, str]] = set()
    for run in runs:
        run_type = _extract_run_type(run["config"])
        c = _extract_concurrency(run["config"])
        key = (c, run_type)
        if key in seen:
            raise SubmissionBuildError(f"Duplicate {run_type} run at concurrency {c}")
        seen.add(key)

    for run in runs:
        concurrency = _extract_concurrency(run["config"])
        run_type = _extract_run_type(run["config"])
        model_dir = submission_dir / "pareto" / system_id / model
        points_dir = model_dir / "points"
        if run_type == "accuracy":
            result_dir = model_dir / "results" / f"point_{concurrency}" / "accuracy"
            yaml_dir = result_dir
        else:
            result_dir = model_dir / "results" / f"point_{concurrency}"
            yaml_dir = points_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        yaml_dir.mkdir(parents=True, exist_ok=True)

        # Build point YAML from config.yaml
        cfg_settings = run["config"].get("settings", {}) or {}
        load_pattern = cfg_settings.get("load_pattern", {}) or {}
        client_cfg = cfg_settings.get("client", {}) or {}
        runtime_cfg = cfg_settings.get("runtime", {}) or {}
        warmup_cfg = cfg_settings.get("warmup", {}) or {}
        datasets = run["config"].get("datasets", []) or []
        dataset_name = (
            datasets[0].get("name", "") if datasets and isinstance(datasets[0], dict) else ""
        )

        stream_all_chunks = client_cfg.get("stream_all_chunks")
        lp_from_config = load_pattern.get("type", "concurrency")

        runtime_settings_out: dict[str, Any] = {
            "load_pattern": lp_from_config,
            "stream_all_chunks": stream_all_chunks,
            "min_duration_ms": runtime_cfg.get("min_duration_ms"),
            # Drop the source config blocks in wholesale rather than cherry-picking keys.
            # The checker's Runtime/WarmupLoadgen models are extra="allow", so this carries
            # every field through while still satisfying the required seed disclosure
            # (runtime_settings.runtime) and exposing warmup salt for the salt check.
            "runtime": dict(runtime_cfg),
            "warmup": dict(warmup_cfg),
        }
        if runtime_cfg.get("n_samples_to_issue") is not None:
            runtime_settings_out["min_sample_count"] = runtime_cfg.get("n_samples_to_issue")

        # §8.3 warmup disclosure block — mapped from config.yaml settings.warmup.
        # Fields duration_s, requests_issued, requests_completed, data_source,
        # concurrency, and initialization_steps are submission metadata; populate
        # from config where available and leave unknown fields as null.
        warmup_out: dict[str, Any] = {
            "enabled": warmup_cfg.get("enabled", False),
            "duration_s": warmup_cfg.get("duration_s"),
            "requests_issued": warmup_cfg.get("requests_issued"),
            "requests_completed": warmup_cfg.get("requests_completed"),
            "data_source": warmup_cfg.get("data_source"),
            "concurrency": warmup_cfg.get("concurrency"),
            "initialization_steps": warmup_cfg.get("initialization_steps"),
        }

        point_cfg: dict[str, Any] = {
            "concurrency": concurrency,
            "region": classify_concurrency(concurrency, regions),
            "dataset": dataset_name,
            "runtime_settings": runtime_settings_out,
            "warmup": warmup_out,
        }
        (yaml_dir / f"point_{concurrency}.yaml").write_text(
            yaml.dump(point_cfg, default_flow_style=False), encoding="utf-8"
        )

        (result_dir / "results_summary.json").write_text(
            json.dumps(run["result_summary"], indent=2), encoding="utf-8"
        )
        extra_files = run.get("_extra_files", {})
        (result_dir / "system_desc.json").write_text(
            json.dumps(run["system_info"], indent=2), encoding="utf-8"
        )

        # Copy all supplementary files into the result directory, preserving subdirs.
        # For accuracy runs result_dir is already .../accuracy/, so strip the leading
        # "accuracy/" prefix from archive paths to avoid double-nesting.
        _acc_prefix = "accuracy/"
        for rel_path, content in extra_files.items():
            if rel_path == "run_metadata.json" and max_tps and max_tps > 0:
                try:
                    metadata = json.loads(content)
                    run_tps = metadata.get("system_tps")
                    if run_tps is not None:
                        metadata["tps_utilization"] = float(run_tps) / max_tps
                        content = json.dumps(metadata, indent=2).encode()
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            if rel_path == "results.json":
                content = _truncate_responses(content)
            dest_rel = (
                rel_path[len(_acc_prefix) :]
                if run_type == "accuracy" and rel_path.startswith(_acc_prefix)
                else rel_path
            )
            dest = result_dir / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        # Fallback: if the archive had no accuracy/results.json, derive from results.json
        if "accuracy/results.json" not in extra_files:
            _write_accuracy_fallback(result_dir, run)


_RESPONSES_LIMIT = 10 * 1024  # 10 KB


def _truncate_responses(content: bytes) -> bytes:
    """Truncate the responses list in a results.json payload to stay under 10 KB."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return content
    responses = data.get("responses")
    if not isinstance(responses, list) or not responses:
        return content
    # Walk items and stop as soon as adding the next one would exceed the limit.
    # Each item contributes its own bytes plus 2 for the ", " separator after the first.
    total = 2  # "[]"
    idx = 0
    for i, r in enumerate(responses):
        total += len(json.dumps(r).encode()) + (2 if i > 0 else 0)
        if total > _RESPONSES_LIMIT:
            break
        idx = i + 1
    data["responses"] = responses[:idx]
    return json.dumps(data, indent=2).encode()


def _write_accuracy_fallback(result_dir: Path, run: dict[str, Any]) -> None:
    """Write per-point accuracy/ files from results.json when the archive has no accuracy/ dir.

    Called only when accuracy/results.json is absent from the run's extra_files
    (i.e. the run archive did not include an accuracy/ directory). Sources accuracy_scores
    from results.json.

    The written results.json format:
        {"<dataset_name>": {"score": {...}, "num_samples": N, ...}}
    """
    results_bytes = run.get("_extra_files", {}).get("results.json")
    if not results_bytes:
        return
    try:
        parsed = json.loads(results_bytes)
    except json.JSONDecodeError:
        return
    accuracy_scores: dict[str, Any] | None = parsed.get("accuracy_scores")
    if not accuracy_scores:
        return

    accuracy_dir = result_dir / "accuracy"
    accuracy_dir.mkdir(parents=True, exist_ok=True)
    (accuracy_dir / "results.json").write_text(
        json.dumps(accuracy_scores, indent=2), encoding="utf-8"
    )


def _write_documentation(submission_dir: Path, run_data: list[dict[str, Any]]) -> None:
    """Merge documentation files from all runs into submission_dir/documentation/."""
    doc_dir = submission_dir / "documentation"
    doc_dir.mkdir(exist_ok=True)
    for run in run_data:
        for rel, content in run.get("_extra_files", {}).items():
            if not rel.startswith("documentation/"):
                continue
            dest = doc_dir / Path(rel).relative_to("documentation")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)


def _write_src(submission_dir: Path, run_data: list[dict[str, Any]]) -> None:
    """Copy src/ files from run archives into submission_dir/src/<model>/."""
    for run in run_data:
        model = _extract_model(run["config"])
        src_model_dir = submission_dir / "src" / model
        src_model_dir.mkdir(parents=True, exist_ok=True)
        for rel, content in run.get("_extra_files", {}).items():
            if not rel.startswith("src/"):
                continue
            dest = submission_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)


def _slugify(name: str) -> str:
    """Convert a human-readable name to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\-]", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:64] or "unknown"
