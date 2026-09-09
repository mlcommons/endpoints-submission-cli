# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Assemble a submission folder from downloaded run archives.

Transforms run-folder data (system_desc.json, point.yaml, result_summary.json)
into the MLC submission layout (policies PR #119):

    <submitting_organization>/
      <submission_id>/
        src/<implementation>/           # SHARED — README.md + endpoint interface code
        docs/                           # SHARED — calibration, software disclosure, etc.
        results/<system>/
            <benchmark_model>/
                r<N>/                   # one Pareto point per concurrency level
                    point.yaml
                    result_summary.json
                    accuracy_results.json
                    system_desc.json    # §8.2, per point since PR #119
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

from submission_checker import layout
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

    # Validate max_supported_concurrency consistency per system before writing anything.
    # The value itself is no longer needed for the build — point.yaml declares its own
    # region — but disagreement across a system's runs still means a broken submission.
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

    # The shared trees are written first: a point's shared_src / shared_docs must
    # resolve to a directory that exists (§9.1), and _write_point_dirs validates or
    # fills in those pointers as it copies each disclosure.
    _write_src(submission_dir, run_data)
    _write_documentation(submission_dir, run_data)
    shared_src = _sole_implementation_path(submission_dir)

    # tps_utilization normalises each point against the peak of its own Pareto curve —
    # one system, one model (§8.5) — which is also how the checker recomputes it.
    # Normalising per model across systems would divide a small system's points by a
    # larger system's peak, so no submission with two systems could match its own file.
    for (system_id, model), runs in groups.items():
        _write_point_dirs(
            submission_dir,
            system_id,
            model,
            runs,
            _compute_max_tps(runs),
            shared_src,
        )

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

    Uses point.yaml as the directory anchor. All supplementary files are read
    into memory so they survive tempdir cleanup.

    point.yaml is required, not derived. The §8.3 disclosure it carries cannot be
    reconstructed from config.yaml for every harness, and guessing produced bundles
    the checker rejected — so the submitter supplies it and the builder copies it.
    That also makes it the natural anchor now that config.yaml is optional (v1.0).
    """
    # Archives may contain a top-level directory wrapper; use point.yaml as anchor.
    candidates = list(run_dir.rglob(layout.POINT_YAML))
    if not candidates:
        raise SubmissionBuildError(
            f"Run {run_id}: archive does not contain {layout.POINT_YAML}, which carries "
            "the §8.3 Pareto-point disclosure and is required for every run."
        )
    point_path = min(candidates, key=lambda p: len(p.parts))
    base = point_path.parent

    summary_path = _find_result_summary(base)
    if summary_path is None:
        raise SubmissionBuildError(
            f"Run {run_id}: archive is missing {layout.RESULT_SUMMARY_JSON}. endpoints writes"
            f" it to {layout.PERFORMANCE_SUBDIR}/{layout.RESULT_SUMMARY_JSON}."
        )

    point_bytes = point_path.read_bytes()
    point_config: dict[str, Any] = yaml.safe_load(point_bytes) or {}

    # config.yaml records what the harness was told to do. Optional as of v1.0: it
    # carries no disclosure of its own, so a run without one is still complete.
    config_path = base / layout.CONFIG_YAML
    config: dict[str, Any] = (
        (yaml.safe_load(config_path.read_text()) or {}) if config_path.exists() else {}
    )
    result_summary: dict[str, Any] = json.loads(summary_path.read_text())

    system_info = _load_system_desc(base, run_id, division, availability)

    return {
        "run_id": run_id,
        "system_info": system_info,
        "config": config,
        "point_config": point_config,
        "result_summary": result_summary,
        # Raw bytes, so the submitter's point.yaml lands in the bundle untouched.
        "point_yaml": point_bytes,
        "_extra_files": _load_extra_files(base),
    }


def _load_system_desc(base: Path, run_id: str, division: str, availability: str) -> dict[str, Any]:
    """Load system_desc.json from a run folder and return the validated flat dict.

    Applies the CLI-provided division and availability values, then validates
    against the SystemDescription schema.

    Raises:
        SubmissionBuildError: If system_desc.json is absent or fails schema validation.
    """
    sd_path = base / layout.SYSTEM_DESC_JSON
    if not sd_path.exists():
        raise SubmissionBuildError(f"Run archive is missing {layout.SYSTEM_DESC_JSON}")
    raw: dict[str, Any] = json.loads(sd_path.read_text())
    # CLI-provided values are authoritative for these fields. §8.2 renamed availability
    # to publication_status; the older spellings are dropped rather than left behind,
    # since SystemDescription rejects two spellings that disagree.
    raw["division"] = division
    for legacy in ("system_availability_status", "availability_status"):
        raw.pop(legacy, None)
    raw["publication_status"] = availability
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


def _find_result_summary(base: Path) -> Path | None:
    """Locate a run's performance summary, endpoints layout first.

    ``mlcommons/endpoints`` writes it to ``performance/result_summary.json``. The flat
    location is still read, because the builder consumes archives the API already holds
    — including ones uploaded before that layout was adopted. ``runs create`` is
    deliberately stricter: it validates *new* input and accepts only the real layout.
    """
    for candidate in (
        base / layout.PERFORMANCE_SUBDIR / layout.RESULT_SUMMARY_JSON,
        base / layout.RESULT_SUMMARY_JSON,
    ):
        if candidate.is_file():
            return candidate
    return None


def _load_extra_files(base: Path) -> dict[str, bytes]:
    """Read supplementary run files into memory for inclusion in the result directory."""
    extra: dict[str, bytes] = {}
    candidates = [
        "config.yaml",
        "results.json",
        "sample_idx_map.json",
        "serving_config.json",
        "report.txt",
        "metrics/final_snapshot.json",
        # Accuracy: the name endpoints writes, then the pre-#78 name.
        f"{layout.ACCURACY_SUBDIR}/{layout.ACCURACY_RESULTS_JSON}",
        f"{layout.ACCURACY_SUBDIR}/{layout.RESULTS_JSON}",
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
        model = _extract_model(data["config"], data["point_config"])
        key = (system_id, model)
        groups.setdefault(key, []).append(data)
    return groups


def _extract_system_id(system_info: dict[str, Any]) -> str:
    sd = SystemDescription.model_validate(system_info)
    return sd.system_name.strip().replace(" ", "_")


def _extract_model(config: dict[str, Any], point_config: dict[str, Any]) -> str:
    """Return the slugified benchmark-model name for a run's results directory.

    point.yaml wins. §8.1 names the directory ``results/<system>/<model_name>/``, and
    §8.2 defines ``model_name`` as the name from the round's supported-model list — so
    the disclosure is the authoritative source and config.yaml, which is optional as of
    v1.0, is only a fallback. Preferring the optional file put a name in the tree that
    §8.1 does not define (config.yaml carries a HuggingFace path, giving
    ``Llama-3_1-8B-Instruct`` where the spec asks for ``llama3.1-8b``).

    Note this is the opposite precedence from :func:`_extract_run_type`, deliberately:
    see that function for why.
    """
    model_params = config.get("model_params", {}) or {}
    name = point_config.get("model_name", "") or model_params.get("name", "") or ""
    if name:
        # Use the last path component (e.g. "meta-llama/Llama-3.1-8B" → "Llama-3.1-8B")
        return _slugify(str(name).split("/")[-1])
    return "unknown_model"


def _extract_concurrency(config: dict[str, Any], point_config: dict[str, Any]) -> int:
    """Return the run's target concurrency, which names its ``r<N>/`` directory.

    point.yaml wins: its ``concurrency`` is the §8.3 disclosure the checker validates,
    so deriving the directory name from anywhere else could contradict the bundle's
    own claim.
    """
    declared = point_config.get("concurrency")
    if declared is not None:
        return int(declared)
    settings = config.get("settings", {}) or {}
    load_pattern = settings.get("load_pattern", {}) or {}
    concurrency = load_pattern.get("target_concurrency")
    if concurrency is not None:
        return int(concurrency)
    # Fallback: look at top-level target_concurrency
    return int(config.get("target_concurrency", 1))


def _extract_run_type(config: dict[str, Any], point_config: dict[str, Any], run_id: str) -> str:
    """Return ``"accuracy"`` or ``"performance"`` for one run.

    config.yaml wins here, which is the opposite of :func:`_extract_model` and is not an
    oversight: ``datasets[].type`` is the harness stating what *this run* did, whereas
    §8.3's ``dataset_type`` describes what the *dataset* is used for. A dataset marked
    "Accuracy + Performance" says nothing about which of the two this run measured, so it
    cannot stand in for the config.

    When neither source answers, the run type is **unknown and not guessed**. Defaulting
    to "performance" was silently wrong in the one case that matters: an accuracy run
    shipped without a config.yaml would be filed as the performance run for its
    concurrency, so the real performance run at that concurrency would collide with it
    and the accuracy results would never reach ``accuracy_results.json``. A submission
    that loses a measurement is worse than one that fails to build.

    Raises:
        SubmissionBuildError: If neither config.yaml nor point.yaml identifies the run
            type unambiguously.
    """
    datasets = config.get("datasets", []) or []
    if datasets and isinstance(datasets[0], dict):
        declared = str(datasets[0].get("type") or "").strip().lower()
        if declared in ("accuracy", "performance"):
            return declared

    dataset_type = str(point_config.get("dataset_type") or "").strip().lower()
    if dataset_type in ("accuracy", "performance"):
        return dataset_type

    detail = (
        f"point.yaml declares dataset_type: {point_config['dataset_type']!r}, which covers"
        " both and so does not identify this run"
        if point_config.get("dataset_type")
        else "point.yaml declares no dataset_type"
    )
    raise SubmissionBuildError(
        f"Run {run_id}: cannot tell whether this is an accuracy or a performance run."
        f" config.yaml does not declare datasets[].type and {detail}."
        " Ship a config.yaml with the run, or set dataset_type to exactly 'Accuracy' or"
        " 'Performance' in point.yaml."
    )


def _run_system_tps(run: dict[str, Any]) -> float | None:
    """Derive a run's system TPS from its result summary.

    ``total_output_tokens / elapsed_seconds``, matching the checker's
    ``PointSummary.system_tps``. Read from ``result_summary.json`` rather than a
    submitter-declared field so the value cannot disagree with the measurement it
    normalises — policies PR #119 removed ``run_metadata.json``, which is where this
    used to be read from.
    """
    summary = run.get("result_summary") or {}
    try:
        duration_ns = float(summary.get("duration_ns") or 0.0)
        tokens = float((summary.get("output_sequence_lengths") or {}).get("total") or 0.0)
    except (TypeError, ValueError):
        return None
    if duration_ns <= 0:
        return None
    return tokens / (duration_ns / 1e9)


def _compute_max_tps(run_data: list[dict[str, Any]]) -> float | None:
    """Return the max system TPS across a curve's runs, or None if underivable.

    This is the denominator of ``tps_utilization`` (§8.2), which normalises each
    point against the peak of its own curve.
    """
    values = [tps for tps in (_run_system_tps(run) for run in run_data) if tps is not None]
    return max(values) if values else None


def _write_point_dirs(
    submission_dir: Path,
    system_id: str,
    model: str,
    runs: list[dict[str, Any]],
    max_tps: float | None = None,
    shared_src: str | None = None,
) -> None:
    """Write one ``r<N>/`` Pareto-point directory per concurrency level.

    A performance run and an accuracy run at the same concurrency describe the
    same Pareto point, so they share a directory: the performance run supplies
    ``point.yaml`` / ``result_summary.json`` and the accuracy run contributes
    ``accuracy_results.json``.

    ``point.yaml`` is copied from the run archive verbatim. It used to be derived from
    ``config.yaml``, which silently produced nulls for any §8.3 field a harness did not
    put in its config — the checker then rejected the result. Requiring the file keeps
    the builder simple and puts the disclosure where it belongs, with the submitter.
    """
    # At most 1 perf + 1 acc run per concurrency (may change; stated here explicitly).
    seen: set[tuple[int, str]] = set()
    by_concurrency: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        run_type = _extract_run_type(run["config"], run["point_config"], run["run_id"])
        c = _extract_concurrency(run["config"], run["point_config"])
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

        (point_dir / layout.POINT_YAML).write_bytes(
            _with_shared_pointers(primary["point_yaml"], submission_dir, shared_src)
        )

        (point_dir / layout.RESULT_SUMMARY_JSON).write_text(
            json.dumps(primary["result_summary"], indent=2), encoding="utf-8"
        )

        _write_point_system_desc(point_dir, primary, max_tps)

        _write_point_extra_files(point_dir, primary, max_tps)
        if accuracy_run is not None:
            _write_accuracy_results(point_dir, accuracy_run)


#: Marker delimiting the block the builder appends to a copied point.yaml.
_INJECTION_HEADER = "# --- added by endpoints-submission-cli: bundle-internal paths ---"


def _sole_implementation_path(submission_dir: Path) -> str | None:
    """The ``src/<impl>`` path to inject, or None when the submission is ambiguous.

    Returns None when ``src/`` holds more than one implementation: which one a given
    point was produced with is a fact only the submitter knows, so guessing would put a
    false claim in the disclosure. Such a submission must name ``shared_src`` itself.
    """
    src_dir = submission_dir / layout.SRC_DIR
    impl_dirs = [d for d in sorted(src_dir.iterdir()) if d.is_dir()] if src_dir.is_dir() else []
    if len(impl_dirs) != 1:
        return None
    return f"{layout.SRC_DIR}/{impl_dirs[0].name}"


def _with_shared_pointers(point_yaml: bytes, submission_dir: Path, shared_src: str | None) -> bytes:
    """Return *point_yaml* with ``shared_src`` / ``shared_docs`` present and resolvable.

    Issue #72 established that the builder does not derive §8.3 disclosure — a value the
    submitter measured must come from the submitter. These two keys are the exception,
    and deliberately so: they carry no measurement claim, and their referent does not
    exist until the builder creates it. ``src/<impl>/`` is the union of every run's
    ``src/`` folder, so no single run archive can name it. The same principle already
    governs ``tps_utilization``, which :func:`_write_point_system_desc` fills in because
    it is a function of the assembled curve. The rule is *the builder owns fields whose
    value is a function of the assembled bundle* — nothing more.

    Behaviour per key: validate if present, inject if absent, never overwrite. A
    submitter pointing at a different implementation is making a claim the builder has
    no standing to correct; a value that does not resolve is a build failure, because
    the checker would reject it (§9.1 "Shared path resolution") and failing here says so
    with the run in hand.

    The block is *appended* rather than round-tripped through ``safe_load``/``safe_dump``:
    a round trip discards comments and key order from a disclosure document. If the
    original does not survive being extended (flow style, multi-document), the function
    falls back to a full dump and warns.

    Raises:
        SubmissionBuildError: If a declared pointer does not resolve under the
            submission root, or if ``shared_src`` is absent and the submission ships
            more than one implementation directory.
    """
    try:
        current = yaml.safe_load(point_yaml) or {}
    except yaml.YAMLError as exc:
        raise SubmissionBuildError(f"Invalid YAML in {layout.POINT_YAML}: {exc}") from exc
    if not isinstance(current, dict):
        raise SubmissionBuildError(
            f"{layout.POINT_YAML} must be a YAML mapping, got {type(current).__name__}"
        )

    wanted = {"shared_src": shared_src, "shared_docs": layout.DOCS_DIR}
    additions: dict[str, str] = {}
    for key, default in wanted.items():
        declared = current.get(key)
        if declared not in (None, ""):
            if layout.resolve_shared_path(submission_dir, str(declared)) is None:
                raise SubmissionBuildError(
                    f"{layout.POINT_YAML} declares {key}: {declared!r}, which does not resolve"
                    " to a directory under the submission root. Paths must be relative to"
                    " the submission root and must not contain '..'."
                )
            continue
        if default is None:
            raise SubmissionBuildError(
                f"{layout.POINT_YAML} does not declare shared_src, and this submission ships"
                " more than one implementation directory under src/, so the builder cannot"
                " tell which one produced this point. Add shared_src: src/<implementation>"
                " to each point.yaml."
            )
        additions[key] = default

    if not additions:
        return point_yaml

    block = _INJECTION_HEADER + "\n" + yaml.safe_dump(additions, sort_keys=True)
    prefix = point_yaml if point_yaml.endswith(b"\n") else point_yaml + b"\n"
    extended = prefix + block.encode("utf-8")

    expected = {**current, **additions}
    try:
        round_tripped = yaml.safe_load(extended)
    except yaml.YAMLError:
        round_tripped = None
    if round_tripped != expected:
        # Flow style or a multi-document file cannot simply be extended. Dumping loses
        # the submitter's comments, so say so rather than doing it silently.
        warnings.warn(
            f"{layout.POINT_YAML} could not be extended in place (flow style or multiple"
            " documents); rewriting it, which drops comments and key order.",
            stacklevel=2,
        )
        return yaml.safe_dump(expected, sort_keys=False).encode("utf-8")
    return extended


def _write_point_system_desc(
    point_dir: Path,
    run: dict[str, Any],
    max_tps: float | None,
) -> None:
    """Write the §8.2 system description into a Pareto-point directory.

    Since policies PR #119 the system description is per point rather than one file
    per system, and it absorbed the fields that used to live in ``run_metadata.json``
    — including ``tps_utilization``, which is a function of the whole curve
    (``system_tps / max(system_tps)``) and so can only be filled in here, once every
    run in the curve has been read.
    """
    system_desc = dict(run["system_info"])
    if max_tps and max_tps > 0:
        run_tps = _run_system_tps(run)
        if run_tps is not None:
            system_desc["tps_utilization"] = run_tps / max_tps
    (point_dir / layout.SYSTEM_DESC_JSON).write_text(
        json.dumps(system_desc, indent=2), encoding="utf-8"
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
        if rel_path == layout.RESULTS_JSON:
            content = truncate_responses(content)
        dest = point_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)


def _write_accuracy_results(point_dir: Path, accuracy_run: dict[str, Any]) -> None:
    """Write ``accuracy_results.json`` for a point from its accuracy run.

    endpoints writes ``accuracy/accuracy_results.json``; older archives carry
    ``accuracy/results.json``, or ``accuracy_scores`` embedded in the run's own
    ``results.json``. All three are read so a bundle can still be assembled from runs
    uploaded before the layout settled.
    """
    extras = accuracy_run.get("_extra_files", {})
    content = (
        extras.get(f"{layout.ACCURACY_SUBDIR}/{layout.ACCURACY_RESULTS_JSON}")
        or extras.get(f"{layout.ACCURACY_SUBDIR}/{layout.RESULTS_JSON}")
        or extras.get(layout.RESULTS_JSON)
    )
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
