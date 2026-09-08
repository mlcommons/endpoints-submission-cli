"""Submission checker — orchestrates §9.1 automated compliance checks.

Loading and structural validation live here; all rule logic lives in Pydantic
model validators on PointConfig, PointResult, and ModelContext.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

__all__ = ["SubmissionChecker"]

from .models import (
    CheckResult,
    ModelContext,
    ModelDir,
    PointConfig,
    PointResult,
    PointSummary,
    Regions,
    Report,
    Severity,
    SrcDir,
    SubmissionDir,
    SystemDescription,
    SystemResults,
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
    load_run_metadata,
    load_system_description,
)

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

    def __init__(self, submission_path: Path) -> None:
        self.submission_path = _resolve_submission_root(submission_path)

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

        # One system_desc_id.json per system directory, not per point.
        system_jsons = sorted(results_dir.glob("*/system_desc_id.json"))
        if not system_jsons:
            report.results.append(
                _err(
                    "system-description-present",
                    "No results/<system>/system_desc_id.json files found",
                    results_dir,
                    "#1",
                )
            )
            return report

        for system_json in system_jsons:
            report.results.extend(self._check_system(system_json, results_dir))

        # Per-curve: tps_utilization must match system_tps / max(system_tps)
        # within each <system_desc_id>/<benchmark_model> pareto curve.
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
        """Verify each run's ``tps_utilization`` equals ``system_tps / max(system_tps)``.

        ``tps_utilization`` normalises a run to the peak ``system_tps`` of the
        system+model curve it belongs to — NOT across the whole submission.
        Normalising submission-wide is wrong when a submission contains more than
        one system: e.g. an ``MI355X_1x`` and an ``MI355X_8x`` config sharing a
        folder would force every 1x point to be divided by the 8x peak, so the
        smaller system can never match its stored (per-curve) values.

        Stored values are compared to the recomputed expectation within an
        absolute tolerance of ``_TPS_UTILIZATION_ABS_TOL``. Structurally invalid
        metadata (missing or non-numeric fields) is left to the per-file
        ``run-metadata-valid`` check.
        """
        # Group run metadata by its pareto curve: the first two path components
        # under results_dir are <system>/<benchmark_model>.
        curves: dict[tuple[str, ...], list[tuple[Path, float, float]]] = {}
        for md_path in sorted(results_dir.rglob("run_metadata.json")):
            try:
                data = json.loads(md_path.read_text())
            except (OSError, ValueError):
                continue
            tps = data.get("system_tps")
            util = data.get("tps_utilization")
            if (
                isinstance(tps, (int, float))
                and not isinstance(tps, bool)
                and isinstance(util, (int, float))
                and not isinstance(util, bool)
            ):
                rel = md_path.relative_to(results_dir).parts
                curve = rel[:2] if len(rel) >= 2 else rel
                curves.setdefault(curve, []).append((md_path, float(tps), float(util)))

        results: list[CheckResult] = []
        for curve in sorted(curves):
            entries = curves[curve]
            max_tps = max(tps for _, tps, _ in entries)
            if max_tps <= 0:
                continue
            for md_path, tps, util in entries:
                expected = tps / max_tps
                if abs(util - expected) <= _TPS_UTILIZATION_ABS_TOL:
                    results.append(
                        _ok(
                            "tps-utilization",
                            f"tps_utilization {util:.4f} matches expected {expected:.4f}",
                            md_path,
                            "#8.1",
                        )
                    )
                else:
                    results.append(
                        _err(
                            "tps-utilization",
                            f"tps_utilization {util} != expected {expected:.4f}"
                            f" (system_tps {tps} / curve max {max_tps} for"
                            f" {'/'.join(curve)}; abs tol {_TPS_UTILIZATION_ABS_TOL})",
                            md_path,
                            "#8.1",
                        )
                    )
        return results

    # ------------------------------------------------------------------
    # Per-system orchestration
    # ------------------------------------------------------------------

    def _check_system(self, system_json: Path, results_dir: Path) -> list[CheckResult]:
        results: list[CheckResult] = []
        # results/<system>/system_desc_id.json — the directory names the system.
        system_id = system_json.parent.name

        system_desc, load_results = load_system_description(system_json)
        results.extend(load_results)
        if system_desc is None:
            return results
        results.append(
            _ok(
                "system-description-valid",
                f"System description valid: {system_id}",
                system_json,
                "#1",
            )
        )

        # §2 — model_name must be one of the accepted benchmark models, exactly.
        if system_desc.model_name in _ALLOWED_MODEL_NAMES:
            results.append(
                _ok(
                    "model-name-valid",
                    f"model_name {system_desc.model_name!r} is an allowed model",
                    system_json,
                    "#2",
                )
            )
        else:
            results.append(
                _err(
                    "model-name-valid",
                    f"model_name {system_desc.model_name!r} is not an allowed model; "
                    f"must be exactly one of: {', '.join(_ALLOWED_MODEL_NAMES)}",
                    system_json,
                    "#2",
                )
            )

        M = system_desc.max_supported_concurrency
        results.append(
            _ok(
                "max-concurrency-declared",
                f"max_supported_concurrency = {M}",
                system_json,
                "#7",
            )
        )

        try:
            regions = compute_regions(M)
        except ValueError as exc:
            results.append(_err("region-computation", str(exc), system_json, "#7"))
            return results

        system_results = SystemResults(results_dir=results_dir, system_id=system_id)
        results.extend(system_results._check_results)
        if any(r.severity == Severity.ERROR for r in system_results._check_results):
            return results
        system_dir = system_results.system_dir
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
            results.extend(self._check_model(system_id, system_desc, regions, model_dir))

        return results

    # ------------------------------------------------------------------
    # Per benchmark-model orchestration
    # ------------------------------------------------------------------

    def _check_model(
        self,
        system_id: str,
        system_desc: SystemDescription,
        regions: Regions,
        model_dir: Path,
    ) -> list[CheckResult]:
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
        point_yamls = [d / "point.yaml" for d in point_dirs if (d / "point.yaml").exists()]
        missing_yaml = [d for d in point_dirs if not (d / "point.yaml").exists()]
        for d in missing_yaml:
            results.append(
                _err(
                    "measurement-points-present",
                    f"Missing point.yaml in {d.relative_to(self.submission_path)}/",
                    d / "point.yaml",
                    "#1",
                )
            )
        if not point_yamls:
            results.append(
                _err(
                    "measurement-points-present",
                    f"No r<N>/point.yaml files in {model_dir.relative_to(self.submission_path)}",
                    model_dir,
                    "#1",
                )
            )
            return results

        valid_points: list[tuple[Path, PointConfig]] = []
        loaded_points: list[tuple[PointConfig, PointSummary]] = []

        for yaml_path in point_yamls:
            config, config_results = load_point_config(
                yaml_path, context={"regions": regions, "yaml_path": yaml_path}
            )
            results.extend(config_results)
            if config is None:
                continue

            # directory-concurrency consistency warning. ModelDir.point_dirs only
            # yields names matching r<digits>, so the int() below cannot fail.
            point_result_dir = yaml_path.parent
            dir_concurrency = int(point_result_dir.name[1:])
            if dir_concurrency != config.concurrency:
                results.append(
                    _warn(
                        "point-dirname-concurrency",
                        f"{point_result_dir.name}/: directory concurrency {dir_concurrency}"
                        f" ≠ declared {config.concurrency}",
                        yaml_path,
                        "#1",
                    )
                )

            valid_points.append((yaml_path, config))

            summary_path = point_result_dir / "result_summary.json"

            if not summary_path.exists():
                results.append(
                    _err(
                        "result-file-present",
                        f"Missing result log for r{config.concurrency}:"
                        f" {summary_path.relative_to(self.submission_path)}",
                        summary_path,
                        "#1",
                    )
                )
                continue

            config_yaml_path = point_result_dir / "config.yaml"
            if not config_yaml_path.exists():
                results.append(
                    _err(
                        "result-file-present",
                        f"Missing config.yaml for r{config.concurrency}:"
                        f" {config_yaml_path.relative_to(self.submission_path)}",
                        config_yaml_path,
                        "#8.1",
                    )
                )

            run_metadata_path = point_result_dir / "run_metadata.json"
            if not run_metadata_path.exists():
                results.append(
                    _err(
                        "run-metadata-present",
                        f"Missing run_metadata.json for r{config.concurrency}:"
                        f" {run_metadata_path.relative_to(self.submission_path)}",
                        run_metadata_path,
                        "#8.1",
                    )
                )
            else:
                run_metadata, rm_results = load_run_metadata(run_metadata_path)
                results.extend(rm_results)
                if run_metadata is not None:
                    results.append(
                        _ok(
                            "run-metadata-valid",
                            f"run_metadata.json valid for r{config.concurrency}",
                            run_metadata_path,
                            "#8.1",
                        )
                    )

            summary, load_results = load_result_summary(summary_path)
            results.extend(load_results)
            if summary is None:
                continue

            # PointResult validates point-duration and metric-consistency
            point_result = PointResult.model_validate(
                {"config": config, "summary": summary, "yaml_path": yaml_path},
                context={"regions": regions, "summary_path": summary_path},
            )
            results.extend(point_result._check_results)
            loaded_points.append((config, summary))

        # Load accuracy per point, preferring the standalone accuracy_results.json
        # written beside point.yaml, then falling back to accuracy_scores embedded in
        # the point's results.json. run() enforces that at least one model has one.
        accuracy_dir: Path | None = None
        accuracy_result = None
        for yaml_path, _config in valid_points:
            if accuracy_result is not None:
                break
            point_dir = yaml_path.parent

            accuracy_json = point_dir / "accuracy_results.json"
            if accuracy_json.exists():
                loaded, acc_results = load_accuracy_result(accuracy_json)
                results.extend(acc_results)
                if loaded is not None and not any(
                    r.severity == Severity.ERROR for r in acc_results
                ):
                    accuracy_result, accuracy_dir = loaded, point_dir
                continue

            # Fallback: accuracy_scores embedded in the point's results.json.
            results_json = point_dir / "results.json"
            if results_json.exists():
                loaded, acc_results, present = load_accuracy_scores(results_json)
                if present:
                    results.extend(acc_results)
                    if loaded is not None and not any(
                        r.severity == Severity.ERROR for r in acc_results
                    ):
                        accuracy_result, accuracy_dir = loaded, point_dir

        # ModelContext validates point-count, regional-coverage, config-consistency, accuracy-gate
        model_ctx = ModelContext(
            system_id=system_id,
            system_desc=system_desc,
            model_dir=model_dir,
            regions=regions,
            points_dir=model_dir,
            accuracy_dir=accuracy_dir,
            all_point_count=len(point_yamls),
            valid_points=valid_points,
            loaded_points=loaded_points,
            accuracy_result=accuracy_result,
        )
        results.extend(model_ctx._check_results)

        return results
