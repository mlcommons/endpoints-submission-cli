# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Assemble a submission folder from downloaded run archives.

Transforms run-folder data (system_info.json, config.yaml, result_summary.json)
into the MLC submission layout:

    <submitting_organization>/
      <submission_id>/
        src/<implementation>/           # SHARED — README.md + endpoint interface code
        docs/                           # SHARED — calibration, software disclosure, etc.
        results/<system>/
            system_desc_id.json         # one per system, not per point
            <benchmark_model>/
                r<N>/                   # one Pareto point per concurrency level
                    point.yaml
                    result_summary.json
                    accuracy_results.json
                    run_metadata.json
                    server_configs/     # OPTIONAL, point-specific, submitter-defined

The ``<submission_id>`` level is assigned by MLC. Callers that do not yet know it
(``submissions create`` only learns it from the API after the bundle is built and
checked) build under :data:`PENDING_SUBMISSION_ID` and call :func:`set_submission_id`
once the real value is known.
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
from ..truncation import truncate_responses

__all__ = [
    "PENDING_SUBMISSION_ID",
    "build_submission_folder",
    "create_bundle_archive",
    "extract_archive",
    "set_submission_id",
]

#: Placeholder directory name used when the MLC submission id is not yet known.
PENDING_SUBMISSION_ID = "pending-submission-id"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_submission_folder(
    run_archives: list[tuple[str, Path]],
    division: str,
    availability: str,
    work_dir: Path,
    submission_id: str | None = None,
) -> Path:
    """Assemble a submission directory from a list of run archives.

    Args:
        run_archives: List of ``(run_id, archive_path)`` tuples.
        division: Submission division (e.g. ``"standardized"``).
        availability: Publication/availability status (e.g. ``"available"``).
        work_dir: Base directory in which to build the submission tree.
        submission_id: MLC-assigned submission id, used as the directory level
            beneath the organization. Defaults to :data:`PENDING_SUBMISSION_ID`
            for callers that only learn the id after the bundle is built; use
            :func:`set_submission_id` to rename it afterwards.

    Returns:
        Path to the assembled org-level submission directory (the parent of the
        ``<submission_id>/`` level).

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
    org_dir = work_dir / org_name
    # <org>/<submission_id>/ — everything below is scoped to a single submission.
    submission_dir = org_dir / (submission_id or PENDING_SUBMISSION_ID)
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

    model_runs: dict[str, list[dict[str, Any]]] = {}
    for (_system_id, model), runs in groups.items():
        model_runs.setdefault(model, []).extend(runs)
    max_tps_by_model: dict[str, float | None] = {
        model: _compute_max_tps(runs) for model, runs in model_runs.items()
    }

    written_systems: set[str] = set()
    for (system_id, model), runs in groups.items():
        if system_id not in written_systems:
            _write_system_description(submission_dir, system_id, runs[0]["system_info"])
            written_systems.add(system_id)
        _write_point_dirs(
            submission_dir,
            system_id,
            model,
            runs,
            system_max_concurrency[system_id],
            max_tps_by_model[model],
        )

    _write_src(submission_dir, run_data)

    _write_documentation(submission_dir, run_data)

    return org_dir


def set_submission_id(org_dir: Path, submission_id: str) -> Path:
    """Rename the placeholder submission-id level under *org_dir* to *submission_id*.

    ``submissions create`` builds and checks the bundle before the API assigns an
    id, so the tree is written under :data:`PENDING_SUBMISSION_ID` and renamed here
    once the POST returns.

    Args:
        org_dir: The org-level directory returned by :func:`build_submission_folder`.
        submission_id: The MLC-assigned submission id.

    Returns:
        Path to the renamed ``<org>/<submission_id>/`` directory.

    Raises:
        SubmissionBuildError: If the placeholder directory is absent, or the
            destination already exists.
    """
    src = org_dir / PENDING_SUBMISSION_ID
    if not src.is_dir():
        raise SubmissionBuildError(
            f"No {PENDING_SUBMISSION_ID}/ directory under {org_dir}; nothing to rename"
        )
    dest = org_dir / submission_id
    if dest.exists():
        raise SubmissionBuildError(f"Submission directory already exists: {dest}")
    src.rename(dest)
    return dest


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
    for subdir_name in ("documentation", "src", "server_configs"):
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
    system_dir = submission_dir / "results" / system_id
    system_dir.mkdir(parents=True, exist_ok=True)
    # One per system, not per point.
    (system_dir / "system_desc_id.json").write_text(
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


def _write_point_dirs(
    submission_dir: Path,
    system_id: str,
    model: str,
    runs: list[dict[str, Any]],
    max_concurrency: int,
    max_tps: float | None = None,
) -> None:
    """Write one ``r<N>/`` Pareto-point directory per concurrency level.

    A performance run and an accuracy run at the same concurrency describe the
    same Pareto point, so they share a directory: the performance run supplies
    ``point.yaml`` / ``result_summary.json`` and the accuracy run contributes
    ``accuracy_results.json``.
    """
    from submission_checker.models import classify_concurrency, compute_regions

    regions = compute_regions(max_concurrency)

    # At most 1 perf + 1 acc run per concurrency (may change; stated here explicitly).
    seen: set[tuple[int, str]] = set()
    by_concurrency: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        run_type = _extract_run_type(run["config"])
        c = _extract_concurrency(run["config"])
        key = (c, run_type)
        if key in seen:
            raise SubmissionBuildError(f"Duplicate {run_type} run at concurrency {c}")
        seen.add(key)
        by_concurrency.setdefault(c, {})[run_type] = run

    model_dir = submission_dir / "results" / system_id / model
    for concurrency, runs_by_type in sorted(by_concurrency.items()):
        point_dir = model_dir / f"r{concurrency}"
        point_dir.mkdir(parents=True, exist_ok=True)

        # The performance run defines the point; fall back to the accuracy run when
        # a concurrency was measured for accuracy only.
        primary = runs_by_type.get("performance") or runs_by_type["accuracy"]
        accuracy_run = runs_by_type.get("accuracy")

        _write_point_yaml(point_dir, primary, concurrency, regions, classify_concurrency)

        (point_dir / "result_summary.json").write_text(
            json.dumps(primary["result_summary"], indent=2), encoding="utf-8"
        )

        _write_point_extra_files(point_dir, primary, max_tps)
        if accuracy_run is not None:
            _write_accuracy_results(point_dir, accuracy_run)


def _write_point_yaml(
    point_dir: Path,
    run: dict[str, Any],
    concurrency: int,
    regions: Any,
    classify_concurrency: Any,
) -> None:
    """Write ``point.yaml`` describing this Pareto point, derived from config.yaml."""
    cfg_settings = run["config"].get("settings", {}) or {}
    load_pattern = cfg_settings.get("load_pattern", {}) or {}
    client_cfg = cfg_settings.get("client", {}) or {}
    runtime_cfg = cfg_settings.get("runtime", {}) or {}
    warmup_cfg = cfg_settings.get("warmup", {}) or {}
    datasets = run["config"].get("datasets", []) or []
    dataset_name = datasets[0].get("name", "") if datasets and isinstance(datasets[0], dict) else ""

    runtime_settings_out: dict[str, Any] = {
        "load_pattern": load_pattern.get("type", "concurrency"),
        "stream_all_chunks": client_cfg.get("stream_all_chunks"),
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
        # Checker types this as a list; default to [] rather than null when absent.
        "initialization_steps": warmup_cfg.get("initialization_steps") or [],
    }

    point_cfg: dict[str, Any] = {
        "concurrency": concurrency,
        "region": classify_concurrency(concurrency, regions),
        "dataset": dataset_name,
        "runtime_settings": runtime_settings_out,
        "warmup": warmup_out,
    }
    (point_dir / "point.yaml").write_text(
        yaml.dump(point_cfg, default_flow_style=False), encoding="utf-8"
    )


def _write_point_extra_files(
    point_dir: Path,
    run: dict[str, Any],
    max_tps: float | None,
) -> None:
    """Copy a run's supplementary files into its point directory.

    ``src/`` and ``documentation/`` are shared across the whole submission and are
    written elsewhere; ``accuracy/`` is folded into ``accuracy_results.json``.
    """
    for rel_path, content in run.get("_extra_files", {}).items():
        if rel_path.startswith(("src/", "documentation/", "accuracy/")):
            continue
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
            content = truncate_responses(content)
        dest = point_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)


def _write_accuracy_results(point_dir: Path, accuracy_run: dict[str, Any]) -> None:
    """Write ``accuracy_results.json`` for a point from its accuracy run.

    Run archives carry accuracy either as a standalone ``accuracy/results.json`` or
    as ``accuracy_scores`` embedded in the run's own ``results.json``.
    """
    extras = accuracy_run.get("_extra_files", {})
    content = extras.get("accuracy/results.json") or extras.get("results.json")
    if content is None:
        return
    (point_dir / "accuracy_results.json").write_bytes(truncate_responses(content))


def _write_documentation(submission_dir: Path, run_data: list[dict[str, Any]]) -> None:
    """Merge documentation files from all runs into submission_dir/docs/."""
    doc_dir = submission_dir / "docs"
    doc_dir.mkdir(exist_ok=True)
    for run in run_data:
        for rel, content in run.get("_extra_files", {}).items():
            if not rel.startswith("documentation/"):
                continue
            dest = doc_dir / Path(rel).relative_to("documentation")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)


def _write_src(submission_dir: Path, run_data: list[dict[str, Any]]) -> None:
    """Populate the shared ``src/`` tree from the runs' ``src/`` folders.

    ``src/`` is shared across the whole submission and holds one directory per
    implementation (``trtllm/``, ``vllm/``, ``sglang/``, …), each documenting how to
    build the SUT and reproduce a point. Whatever the run archives provide is copied
    through verbatim and then validated — a missing implementation directory or a
    missing README is a defect in the submission, not something to paper over with a
    generated stub.

    Raises:
        SubmissionBuildError: If the submission ships no implementation directory, or
            if any implementation directory has no README.md.
    """
    src_dir = submission_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    for run in run_data:
        for rel, content in run.get("_extra_files", {}).items():
            if not rel.startswith("src/"):
                continue
            dest = src_dir / Path(rel).relative_to("src")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

    _validate_src(src_dir)


def _validate_src(src_dir: Path) -> None:
    """Check the assembled ``src/`` tree against the §2.2.1 source requirement.

    Enforced for every submission regardless of division for now; division-specific
    rulings are expected to relax this later.
    """
    impl_dirs = [d for d in sorted(src_dir.iterdir()) if d.is_dir()]

    if not impl_dirs:
        stray = sorted(p.name for p in src_dir.iterdir() if p.is_file())
        detail = f" (found loose file(s) at the top of src/: {', '.join(stray)})" if stray else ""
        raise SubmissionBuildError(
            "Submissions must ship source, but no implementation directory was found "
            f"under src/{detail}. Add src/<implementation>/ — e.g. src/trtllm/ — "
            "containing a README.md describing how to build/launch the SUT and reproduce "
            "a Pareto point."
        )

    missing = [d.name for d in impl_dirs if not _has_readme(d)]
    if missing:
        listed = ", ".join(f"src/{name}/" for name in missing)
        raise SubmissionBuildError(
            f"Missing README.md in {listed}. Each implementation directory must document "
            "how to build/launch the SUT and reproduce a Pareto point."
        )


def _has_readme(impl_dir: Path) -> bool:
    """True if *impl_dir* contains a README.md (matched case-insensitively)."""
    return any(p.is_file() and p.name.lower() == "readme.md" for p in impl_dir.iterdir())


def _slugify(name: str) -> str:
    """Convert a human-readable name to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\-]", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:64] or "unknown"
