"""Submission checker — orchestrates §9.1 automated compliance checks.

Loading and structural validation live here; all rule logic lives in Pydantic
model validators on PointConfig, PointResult, and ModelContext.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

__all__ = ["SubmissionChecker"]

from . import layout
from .models import (
    AccuracyResult,
    CheckResult,
    ModelContext,
    ModelDir,
    PointConfig,
    PointResult,
    PointSummary,
    RegionPlacement,
    Regions,
    Report,
    SeedBinding,
    Severity,
    SrcDir,
    SubmissionDir,
    SystemDescription,
    compute_regions,
)
from .models import err as _err
from .models import ok as _ok
from .models import warn as _warn
from .models.loader import (
    load_accuracy_result,
    load_accuracy_scores,
    load_point_config,
    load_result_summary,
    load_system_description,
)
from .models.regions import ULTRA_LOW_CONCURRENCY_MAX
from .seed_sets import SeedSet, SeedSetError, load_seed_sets

if TYPE_CHECKING:
    from pathlib import Path

# Absolute tolerance for the tps_utilization consistency check.
_TPS_UTILIZATION_ABS_TOL = 0.1


# §2 — the only benchmark models accepted this submission round. system_desc.model_name
# must match one of these exactly.
_ALLOWED_MODEL_NAMES = ("llama3.1-8b", "gpt-oss-120b", "deepseek-r1")


def _results_has_accuracy_scores(path: Path) -> bool:
    """True if a results.json carries a non-empty ``accuracy_scores`` mapping."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("accuracy_scores"), dict)
        and bool(data["accuracy_scores"])
    )


#: Fields of ``system_desc.json`` allowed to differ between points of one curve.
#: ``tps_utilization`` is a per-point quantity that §8.2 nonetheless places in the
#: system description — worth raising with the WG, but not a submission defect.
_PER_POINT_SYSTEM_DESC_FIELDS = frozenset({"tps_utilization"})


@dataclass
class _LoadedPoint:
    """One Pareto point whose ``point.yaml`` parsed, before regions are known."""

    point_dir: Path
    yaml_path: Path
    config: PointConfig


def _system_desc_identity(data: dict[str, object]) -> dict[str, object]:
    """Strip the fields that legitimately vary between points of the same curve."""
    return {k: v for k, v in data.items() if k not in _PER_POINT_SYSTEM_DESC_FIELDS}


def _read_json(path: Path) -> dict[str, object] | None:
    """Parse *path* as a JSON object, returning None on any read or decode failure."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _as_float(value: object) -> float | None:
    """Coerce a JSON number to float, rejecting bools and everything non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _derived_system_tps(summary_path: Path) -> float | None:
    """Compute ``total output tokens / elapsed seconds`` from a result summary.

    Mirrors :attr:`~submission_checker.models.PointSummary.system_tps` but reads the
    raw JSON, so a summary that fails full validation still contributes its
    throughput to the curve's peak rather than silently shrinking the denominator.
    """
    data = _read_json(summary_path)
    if data is None:
        return None
    duration_ns = _as_float(data.get("duration_ns"))
    lengths = data.get("output_sequence_lengths")
    total = _as_float(lengths.get("total")) if isinstance(lengths, dict) else None
    if duration_ns is None or total is None or duration_ns <= 0:
        return None
    return total / (duration_ns / 1e9)


def _declared_tps_utilization(system_desc_path: Path) -> float | None:
    """Read ``tps_utilization`` from a point's ``system_desc.json``, or None."""
    data = _read_json(system_desc_path)
    return None if data is None else _as_float(data.get("tps_utilization"))


def _resolve_submission_root(path: Path) -> Path:
    """Return the ``<submission_id>/`` level, descending from an org root if needed.

    Submissions are laid out as ``<organisation>/<submission_id>/``, so a path may
    point at either level. A directory that already holds ``results/`` is the
    submission root; otherwise a single child holding ``results/`` is used. An
    ambiguous org root (several submissions) is returned unchanged so the caller
    gets the normal "missing results/" error rather than an arbitrary pick.
    """
    if not path.is_dir() or (path / "results").is_dir():
        return path
    candidates = [d for d in sorted(path.iterdir()) if d.is_dir() and (d / "results").is_dir()]
    return candidates[0] if len(candidates) == 1 else path


class SubmissionChecker:
    """Validates an MLPerf Endpoints submission directory against §9.1 rules.

    The *submission_path* should be a ``<submission_id>/`` directory containing
    ``results/`` and ``docs/`` as specified in §8.1. The submitting organisation's
    directory is also accepted — submissions are laid out as
    ``<organisation>/<submission_id>/`` and the single submission level below the
    org root is resolved automatically.

    Args:
        submission_path: Directory of the submission to validate.

    Example::

        checker = SubmissionChecker(Path("/submissions/acme_corp/sub-123"))
        report = checker.run()
        for err in report.errors:
            print(err.rule, err.message)
    """

    def __init__(self, submission_path: Path, seed_sets_path: Path | None = None) -> None:
        self.submission_path = _resolve_submission_root(submission_path)
        self.seed_sets_path = seed_sets_path
        self._seed_sets: dict[str, SeedSet] = {}
        self._seed_sets_error: str | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Report:
        """Run all §9.1 automated checks and return an aggregated report.

        Returns:
            A :class:`~submission_checker.models.Report` with every
            :class:`~submission_checker.models.CheckResult` produced.
        """
        report = Report(submission_path=self.submission_path)

        try:
            self._seed_sets = load_seed_sets(self.seed_sets_path)
        except SeedSetError as exc:
            # Without the registry every seed rule would report a defect the submitter
            # cannot fix, so say once that the checker is misconfigured and skip them.
            self._seed_sets = {}
            self._seed_sets_error = str(exc)
            report.results.append(_warn("seed-set-registry", str(exc), None, "#4.6"))

        if not self.submission_path.exists():
            report.results.append(
                _err(
                    "path-exists",
                    f"Submission path does not exist: {self.submission_path}",
                    self.submission_path,
                    "#1",
                )
            )
            return report
        report.results.append(
            _ok("path-exists", "Submission path exists", self.submission_path, "#1")
        )

        submission_dir = SubmissionDir(root=self.submission_path)
        report.results.extend(submission_dir._check_results)
        if any(r.severity == Severity.ERROR for r in submission_dir._check_results):
            return report

        results_dir = submission_dir.results_dir

        # Since policies PR #119 there is no per-system file: a system is simply a
        # directory under results/, and its description lives in every point's
        # system_desc.json.
        system_dirs = [d for d in sorted(results_dir.iterdir()) if d.is_dir()]
        if not system_dirs:
            report.results.append(
                _err(
                    "system-results-dir",
                    "No results/<system>/ directories found",
                    results_dir,
                    "#1",
                )
            )
            return report

        for system_dir in system_dirs:
            report.results.extend(self._check_system(system_dir))

        # Per-curve: tps_utilization must match system_tps / max(system_tps)
        # within each <system>/<benchmark_model> pareto curve.
        report.results.extend(self._check_tps_utilization(results_dir))

        # §15: at least one model must carry accuracy results — either as accuracy_scores
        # embedded in a results.json, or as a standalone accuracy/results.json.
        has_full_accuracy = any(True for _ in results_dir.rglob("accuracy_results.json")) or any(
            _results_has_accuracy_scores(p) for p in results_dir.rglob("results.json")
        )

        if has_full_accuracy:
            report.results.append(
                _ok(
                    "accuracy-present",
                    "At least one model has accuracy results",
                    results_dir,
                    "#15",
                )
            )
        else:
            report.results.append(
                _err(
                    "accuracy-present",
                    "No model in this submission has accuracy results "
                    "(accuracy_results.json or accuracy_scores in results.json)",
                    results_dir,
                    "#15",
                )
            )

        return report

    # ------------------------------------------------------------------
    # Submission-wide checks
    # ------------------------------------------------------------------

    def _check_tps_utilization(self, results_dir: Path) -> list[CheckResult]:
        """Verify each point's ``tps_utilization`` equals ``system_tps / max(system_tps)``.

        ``tps_utilization`` normalises a point to the peak ``system_tps`` of the
        system+model curve it belongs to — NOT across the whole submission.
        Normalising submission-wide is wrong when a submission contains more than
        one system: e.g. an ``MI355X_1x`` and an ``MI355X_8x`` config sharing a
        folder would force every 1x point to be divided by the 8x peak, so the
        smaller system can never match its stored (per-curve) values.

        The stored value lives in the point's ``system_desc.json`` (§8.2 absorbed it
        when policies PR #119 removed ``run_metadata.json``), while ``system_tps`` is
        derived from ``result_summary.json`` — the measurement itself — so a submitter
        cannot declare a throughput that disagrees with what they measured.

        Stored values are compared to the recomputed expectation within an absolute
        tolerance of ``_TPS_UTILIZATION_ABS_TOL``. Files that are missing or
        structurally invalid are left to the per-file presence and validity checks.
        """
        results: list[CheckResult] = []
        for system_dir, model_dir in layout.iter_curves(results_dir):
            entries: list[tuple[Path, float, float]] = []
            for point_dir in layout.iter_point_dirs(model_dir):
                tps = _derived_system_tps(point_dir / layout.RESULT_SUMMARY_JSON)
                sd_path = point_dir / layout.SYSTEM_DESC_JSON
                util = _declared_tps_utilization(sd_path)
                if tps is not None and util is not None:
                    entries.append((sd_path, tps, util))

            if not entries:
                continue
            max_tps = max(tps for _, tps, _ in entries)
            if max_tps <= 0:
                continue
            curve = f"{system_dir.name}/{model_dir.name}"
            for sd_path, tps, util in entries:
                expected = tps / max_tps
                if abs(util - expected) <= _TPS_UTILIZATION_ABS_TOL:
                    results.append(
                        _ok(
                            "tps-utilization",
                            f"tps_utilization {util:.4f} matches expected {expected:.4f}",
                            sd_path,
                            "#8.2",
                        )
                    )
                else:
                    results.append(
                        _err(
                            "tps-utilization",
                            f"tps_utilization {util} != expected {expected:.4f}"
                            f" (system_tps {tps:.4f} / curve max {max_tps:.4f} for"
                            f" {curve}; abs tol {_TPS_UTILIZATION_ABS_TOL})",
                            sd_path,
                            "#8.2",
                        )
                    )
        return results

    # ------------------------------------------------------------------
    # Per-system orchestration
    # ------------------------------------------------------------------

    def _check_system(self, system_dir: Path) -> list[CheckResult]:
        """Run every check scoped to one ``results/<system>/`` directory.

        The system description is no longer a per-system file, so a system contributes
        no checks of its own beyond holding at least one benchmark-model directory;
        everything else is per curve.
        """
        results: list[CheckResult] = []
        system_id = system_dir.name

        model_dirs = [d for d in sorted(system_dir.iterdir()) if d.is_dir()]
        if not model_dirs:
            results.append(
                _err(
                    "benchmark-model-dir",
                    f"No benchmark-model directories in results/{system_id}/",
                    system_dir,
                    "#1",
                )
            )
            return results

        for model_dir in model_dirs:
            results.extend(self._check_model(system_id, model_dir))

        return results

    # ------------------------------------------------------------------
    # Per benchmark-model orchestration
    # ------------------------------------------------------------------

    def _check_model(self, system_id: str, model_dir: Path) -> list[CheckResult]:
        """Run every check scoped to one Pareto curve (§8.5: one system, one model).

        Two-phase, because v1.0 derives ``C_min`` from the submitted points (§5.4)
        rather than taking it as a declaration: phase one parses every ``point.yaml``
        with no notion of regions, phase two computes the boundaries from what parsed
        and runs the region-dependent rules against them.
        """
        results: list[CheckResult] = []
        benchmark_model = model_dir.name

        src = SrcDir(root=self.submission_path)
        results.extend(src._check_results)

        model_structure = ModelDir(
            root=model_dir, system_id=system_id, benchmark_model=benchmark_model
        )
        results.extend(model_structure._check_results)
        if any(r.severity == Severity.ERROR for r in model_structure._check_results):
            return results

        point_dirs = model_structure.point_dirs

        # ── Phase 1: parse every point.yaml, with no region context ───────────
        loaded, load_results = self._load_point_configs(point_dirs)
        results.extend(load_results)
        if not loaded and not point_dirs:
            return results

        system_desc, sd_path, sd_results = self._load_curve_system_desc(point_dirs, model_dir)
        results.extend(sd_results)
        if system_desc is None or sd_path is None:
            return results

        results.extend(self._check_model_name(system_desc, sd_path))

        # ── Phase 1b: derive C_min, then the region boundaries ────────────────
        regions, region_results = self._derive_regions(
            system_desc, loaded, len(point_dirs), model_dir, sd_path
        )
        results.extend(region_results)

        # ── Phase 2: the region-dependent, per-point rules ────────────────────
        valid_points: list[tuple[Path, PointConfig]] = [(p.yaml_path, p.config) for p in loaded]
        loaded_points: list[tuple[PointConfig, PointSummary]] = []

        for point in loaded:
            results.extend(self._check_point(point, regions, loaded_points))

        results.extend(self._check_shared_paths(loaded))

        if self._seed_sets_error is None:
            seed_binding = SeedBinding(
                points=valid_points, registry=self._seed_sets, model_dir=model_dir
            )
            results.extend(seed_binding._check_results)

        accuracy_result, accuracy_dir, accuracy_results = self._load_curve_accuracy(loaded)
        results.extend(accuracy_results)

        # ModelContext validates point-count, coverage, config-consistency, accuracy-gate
        model_ctx = ModelContext(
            system_id=system_id,
            system_desc=system_desc,
            model_dir=model_dir,
            regions=regions,
            points_dir=model_dir,
            accuracy_dir=accuracy_dir,
            all_point_count=len(point_dirs),
            valid_points=valid_points,
            loaded_points=loaded_points,
            accuracy_result=accuracy_result,
        )
        results.extend(model_ctx._check_results)

        return results

    # ------------------------------------------------------------------
    # Phase 1 — parsing, before regions exist
    # ------------------------------------------------------------------

    def _load_point_configs(
        self, point_dirs: list[Path]
    ) -> tuple[list[_LoadedPoint], list[CheckResult]]:
        """Parse each point directory's ``point.yaml``; region rules run later."""
        results: list[CheckResult] = []
        loaded: list[_LoadedPoint] = []

        for point_dir in point_dirs:
            yaml_path = point_dir / layout.POINT_YAML
            if not yaml_path.exists():
                results.append(
                    _err(
                        "measurement-points-present",
                        f"Missing {layout.POINT_YAML} in"
                        f" {point_dir.relative_to(self.submission_path)}/",
                        yaml_path,
                        "#1",
                    )
                )
                continue

            config, config_results = load_point_config(yaml_path, context={"yaml_path": yaml_path})
            results.extend(config_results)
            if config is None:
                continue

            # ModelDir.point_dirs only yields names matching r<digits>, so this parses.
            dir_concurrency = layout.parse_point_dir(point_dir.name)
            if dir_concurrency is not None and dir_concurrency != config.concurrency:
                results.append(
                    _warn(
                        "point-dirname-concurrency",
                        f"{point_dir.name}/: directory concurrency {dir_concurrency}"
                        f" ≠ declared {config.concurrency}",
                        yaml_path,
                        "#1",
                    )
                )

            loaded.append(_LoadedPoint(point_dir=point_dir, yaml_path=yaml_path, config=config))

        if not loaded and point_dirs:
            results.append(
                _err(
                    "measurement-points-present",
                    f"No usable r<N>/{layout.POINT_YAML} files in"
                    f" {point_dirs[0].parent.relative_to(self.submission_path)}",
                    point_dirs[0].parent,
                    "#1",
                )
            )
        return loaded, results

    def _load_curve_system_desc(
        self, point_dirs: list[Path], model_dir: Path
    ) -> tuple[SystemDescription | None, Path | None, list[CheckResult]]:
        """Load the curve's system description from its points' ``system_desc.json``.

        Every point carries its own copy since policies PR #119, so the curve's
        description is whichever they agree on — and disagreeing on anything but
        :data:`_PER_POINT_SYSTEM_DESC_FIELDS` means the points do not describe one
        system, which §8.5 requires. Only the first copy is schema-validated: once the
        rest are byte-equivalent, validating them again would say nothing new.
        """
        results: list[CheckResult] = []
        rel = model_dir.relative_to(self.submission_path)
        present: list[tuple[Path, dict[str, object]]] = []

        for point_dir in point_dirs:
            sd_path = point_dir / layout.SYSTEM_DESC_JSON
            if not sd_path.exists():
                results.append(
                    _err(
                        "system-description-present",
                        f"Missing {layout.SYSTEM_DESC_JSON} in {rel}/{point_dir.name}/",
                        sd_path,
                        "#8.2",
                    )
                )
                continue
            data = _read_json(sd_path)
            if data is None:
                results.append(
                    _err(
                        "system-description-valid",
                        f"{layout.SYSTEM_DESC_JSON} is not readable as a JSON object",
                        sd_path,
                        "#8.2",
                    )
                )
                continue
            present.append((sd_path, data))

        if not present:
            results.append(
                _err(
                    "system-description-present",
                    f"No readable {layout.SYSTEM_DESC_JSON} under {rel}/",
                    model_dir,
                    "#8.2",
                )
            )
            return None, None, results

        ref_path, ref_data = present[0]
        results.extend(self._check_system_desc_consistency(present, model_dir))

        system_desc, load_results = load_system_description(ref_path)
        results.extend(load_results)
        if system_desc is None:
            return None, ref_path, results
        results.append(
            _ok(
                "system-description-valid",
                f"System description valid for {rel}",
                ref_path,
                "#8.2",
            )
        )
        return system_desc, ref_path, results

    def _check_system_desc_consistency(
        self, present: list[tuple[Path, dict[str, object]]], model_dir: Path
    ) -> list[CheckResult]:
        """§8.5: every point of a curve must describe the same system."""
        ref_path, ref_data = present[0]
        ref_identity = _system_desc_identity(ref_data)
        for sd_path, data in present[1:]:
            identity = _system_desc_identity(data)
            if identity == ref_identity:
                continue
            differing = sorted(
                key
                for key in set(identity) | set(ref_identity)
                if identity.get(key) != ref_identity.get(key)
            )
            return [
                _err(
                    "system-description-consistency",
                    f"{layout.SYSTEM_DESC_JSON} differs from {ref_path.parent.name}/ in:"
                    f" {', '.join(differing)}"
                    " — every point of a curve must describe the same system",
                    sd_path,
                    "#8.5",
                )
            ]
        return [
            _ok(
                "system-description-consistency",
                f"{len(present)} point(s) agree on the system description",
                model_dir,
                "#8.5",
            )
        ]

    def _derive_regions(
        self,
        system_desc: SystemDescription,
        loaded: list[_LoadedPoint],
        point_dir_count: int,
        model_dir: Path,
        sd_path: Path,
    ) -> tuple[Regions | None, list[CheckResult]]:
        """Compute the curve's region boundaries from ``C_max`` and a derived ``C_min``.

        Only points whose ``point.yaml`` parsed contribute to ``C_min``: a corrupt file
        has no trustworthy concurrency, and letting it set the floor would move every
        boundary and cascade spurious failures onto the points that are fine.
        """
        results: list[CheckResult] = []
        c_max = system_desc.max_supported_concurrency
        results.append(
            _ok("max-concurrency-declared", f"max_supported_concurrency = {c_max}", sd_path, "#7")
        )

        if not loaded:
            results.append(
                _err(
                    "region-basis",
                    f"No {layout.POINT_YAML} parsed — C_min cannot be derived, so no"
                    " region-dependent check can run for this curve",
                    model_dir,
                    "#5.4",
                )
            )
            return None, results

        lowest = min(point.config.concurrency for point in loaded)
        # A curve with no Ultra Low Concurrency point still gets usable boundaries;
        # ModelContext reports the missing coverage as its own error.
        c_min = min(lowest, ULTRA_LOW_CONCURRENCY_MAX)
        clamped = "" if c_min == lowest else f" (clamped from {lowest})"

        if len(loaded) < point_dir_count:
            results.append(
                _warn(
                    "region-basis",
                    f"C_min = {c_min}{clamped} derived from {len(loaded)} of"
                    f" {point_dir_count} points — the rest could not be parsed",
                    model_dir,
                    "#5.4",
                )
            )
        else:
            results.append(
                _ok(
                    "region-basis",
                    f"C_min = {c_min}{clamped} derived from all {point_dir_count} points",
                    model_dir,
                    "#5.4",
                )
            )

        try:
            regions = compute_regions(c_max, c_min)
        except ValueError as exc:
            results.append(_err("region-computation", str(exc), sd_path, "#5.5"))
            return None, results
        return regions, results

    # ------------------------------------------------------------------
    # Phase 2 — the region-dependent, per-point rules
    # ------------------------------------------------------------------

    def _check_point(
        self,
        point: _LoadedPoint,
        regions: Regions | None,
        loaded_points: list[tuple[PointConfig, PointSummary]],
    ) -> list[CheckResult]:
        """Run the per-point rules that need the curve's regions or its result summary.

        Appends to *loaded_points* as a side effect so :class:`ModelContext` sees every
        point whose config and summary both loaded.
        """
        results: list[CheckResult] = []

        if regions is not None:
            placement = RegionPlacement(
                config=point.config, regions=regions, yaml_path=point.yaml_path
            )
            results.extend(placement._check_results)

        summary_path = point.point_dir / layout.RESULT_SUMMARY_JSON
        if not summary_path.exists():
            results.append(
                _err(
                    "result-summary-present",
                    f"Missing {layout.RESULT_SUMMARY_JSON} for r{point.config.concurrency}:"
                    f" {summary_path.relative_to(self.submission_path)}",
                    summary_path,
                    "#1",
                )
            )
            return results

        summary, load_results = load_result_summary(summary_path)
        results.extend(load_results)
        if summary is None:
            return results

        # PointResult validates point-duration and metric-consistency
        point_result = PointResult.model_validate(
            {"config": point.config, "summary": summary, "yaml_path": point.yaml_path},
            context={"regions": regions, "summary_path": summary_path},
        )
        results.extend(point_result._check_results)
        loaded_points.append((point.config, summary))
        return results

    def _load_curve_accuracy(
        self, loaded: list[_LoadedPoint]
    ) -> tuple[AccuracyResult | None, Path | None, list[CheckResult]]:
        """Find the curve's accuracy results, preferring the standalone file.

        ``accuracy_results.json`` beside ``point.yaml`` wins; otherwise
        ``accuracy_scores`` embedded in the point's ``results.json`` is used. §4.3 says
        accuracy is a property of the curve, not of each point, so the first point that
        carries usable results supplies them for the whole curve — :meth:`run` enforces
        separately that at least one curve in the submission has any.
        """
        results: list[CheckResult] = []
        for point in loaded:
            point_dir = point.point_dir

            accuracy_json = point_dir / layout.ACCURACY_RESULTS_JSON
            if accuracy_json.exists():
                acc, acc_results = load_accuracy_result(accuracy_json)
                results.extend(acc_results)
                if acc is not None and not any(r.severity == Severity.ERROR for r in acc_results):
                    return acc, point_dir, results
                continue

            results_json = point_dir / layout.RESULTS_JSON
            if results_json.exists():
                acc, acc_results, present = load_accuracy_scores(results_json)
                if present:
                    results.extend(acc_results)
                    if acc is not None and not any(
                        r.severity == Severity.ERROR for r in acc_results
                    ):
                        return acc, point_dir, results
        return None, None, results

    # ------------------------------------------------------------------
    # Per-curve system-description rules
    # ------------------------------------------------------------------

    def _check_model_name(self, system_desc: SystemDescription, sd_path: Path) -> list[CheckResult]:
        """§2: ``model_name`` must be one of the accepted benchmark models, exactly."""
        if system_desc.model_name in _ALLOWED_MODEL_NAMES:
            return [
                _ok(
                    "model-name-valid",
                    f"model_name {system_desc.model_name!r} is an allowed model",
                    sd_path,
                    "#2",
                )
            ]
        return [
            _err(
                "model-name-valid",
                f"model_name {system_desc.model_name!r} is not an allowed model; "
                f"must be exactly one of: {', '.join(_ALLOWED_MODEL_NAMES)}",
                sd_path,
                "#2",
            )
        ]

    def _check_shared_paths(self, loaded: list[_LoadedPoint]) -> list[CheckResult]:
        """§9.1: each point's ``shared_src`` / ``shared_docs`` must resolve under the root.

        The pointers are how a point names the shared trees §8.1 places outside
        ``results/``. A value that does not resolve leaves the point's implementation
        and documentation unreviewable, which §9.1 answers with "Reject submission".
        """
        results: list[CheckResult] = []
        for point in loaded:
            for field_name in ("shared_src", "shared_docs"):
                value = getattr(point.config, field_name)
                if value is None:
                    continue  # absence is reported by point-disclosure-complete
                resolved = layout.resolve_shared_path(self.submission_path, value)
                if resolved is None:
                    results.append(
                        _err(
                            "shared-path-resolution",
                            f"{field_name} {value!r} does not resolve to a directory under the"
                            " submission root (paths must be root-relative and free of '..')",
                            point.yaml_path,
                            "#9.1",
                        )
                    )
                else:
                    results.append(
                        _ok(
                            "shared-path-resolution",
                            f"{field_name} {value!r} resolves to"
                            f" {resolved.relative_to(self.submission_path)}/",
                            point.yaml_path,
                            "#9.1",
                        )
                    )
        return results
