"""Submission checker — orchestrates §9.1 automated compliance checks.

Loading and structural validation live here; all rule logic lives in Pydantic
model validators on PointConfig, PointResult, and ModelContext.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

__all__ = ["SubmissionChecker"]

from .models.loader import (
    load_accuracy_result,
    load_point_config,
    load_result_summary,
    load_system_description,
)
from .models import (
    CheckResult,
    ModelContext,
    PointConfig,
    PointResult,
    PointSummary,
    Regions,
    Report,
    RunDir,
    RuntimeSettings,
    Severity,
    SystemDescription,
    compute_regions,
)
from .models import ModelDir
from .models import err as _err
from .models import ok as _ok
from .models import warn as _warn

if TYPE_CHECKING:
    from pathlib import Path


class SubmissionChecker:
    """Validates an MLPerf Endpoints submission directory against §9.1 rules.

    The *submission_path* should be the submitting organisation's root directory.
    System directories are discovered by the presence of ``system_desc.json``
    inside each immediate subdirectory (§8.1).

    Args:
        submission_path: Root directory of the submission to validate.

    Example::

        checker = SubmissionChecker(Path("/submissions/acme_corp"))
        report = checker.run()
        for err in report.errors:
            print(err.rule, err.message)
    """

    def __init__(self, submission_path: Path) -> None:
        self.submission_path = submission_path

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Report:
        """Run all §9.1 automated checks and return an aggregated
        :class:`~submission_checker.models.Report`.

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

        system_dirs = [
            d
            for d in sorted(self.submission_path.iterdir())
            if d.is_dir() and (d / "system_desc.json").exists()
        ]
        if not system_dirs:
            report.results.append(
                _err(
                    "system-dir-present",
                    "No system directories found (each system directory must contain system_desc.json)",
                    self.submission_path,
                    "#1",
                )
            )
            return report

        for system_dir in system_dirs:
            report.results.extend(self._check_system(system_dir))

        return report

    # ------------------------------------------------------------------
    # Per-system orchestration
    # ------------------------------------------------------------------

    def _check_system(self, system_dir: Path) -> list[CheckResult]:
        results: list[CheckResult] = []
        system_name = system_dir.name
        system_json = system_dir / "system_desc.json"

        system_desc, load_results = load_system_description(system_json)
        results.extend(load_results)
        if system_desc is None:
            return results
        results.append(
            _ok(
                "system-description-valid",
                f"System description valid: {system_name}",
                system_json,
                "#1",
            )
        )

        M = system_desc.max_supported_concurrency
        results.append(
            _ok("max-concurrency-declared", f"max_supported_concurrency = {M}", system_json, "#7")
        )

        try:
            regions = compute_regions(M)
        except ValueError as exc:
            results.append(_err("region-computation", str(exc), system_json, "#7"))
            return results

        model_dirs = [
            d for d in sorted(system_dir.iterdir()) if d.is_dir() and d.name != "docs"
        ]
        if not model_dirs:
            results.append(
                _err(
                    "benchmark-model-dir",
                    f"No benchmark-model directories in {system_name}/",
                    system_dir,
                    "#1",
                )
            )
            return results

        for model_dir in model_dirs:
            results.extend(self._check_model(system_name, system_desc, regions, model_dir))

        return results

    # ------------------------------------------------------------------
    # Per benchmark-model orchestration
    # ------------------------------------------------------------------

    def _check_model(
        self,
        system_name: str,
        system_desc: SystemDescription,
        regions: Regions,
        model_dir: Path,
    ) -> list[CheckResult]:
        results: list[CheckResult] = []
        model_name = model_dir.name

        model_structure = ModelDir(
            root=model_dir, system_name=system_name, model_name=model_name
        )
        results.extend(model_structure._check_results)

        run_dirs = model_structure.run_dirs
        if not run_dirs:
            results.append(
                _err(
                    "measurement-points-present",
                    f"No r<N>/ run directories in {system_name}/{model_name}/",
                    model_dir,
                    "#1",
                )
            )
            return results

        valid_points: list[tuple[Path, PointConfig]] = []
        loaded_points: list[tuple[PointConfig, PointSummary]] = []
        accuracy_results = []

        for run_dir in run_dirs:
            run_structure = RunDir(
                root=run_dir, system_name=system_name, model_name=model_name
            )
            results.extend(run_structure._check_results)
            if any(r.severity == Severity.ERROR for r in run_structure._check_results):
                # Still try to load what we can for coverage checks
                pass

            point_yaml = run_dir / "point.yaml"
            config, config_results = load_point_config(
                point_yaml, context={"regions": regions, "yaml_path": point_yaml}
            )
            results.extend(config_results)
            if config is None:
                continue

            # dir-name concurrency consistency warning
            try:
                dir_concurrency = int(run_dir.name[1:])  # strip leading 'r'
                if dir_concurrency != config.concurrency:
                    results.append(
                        _warn(
                            "point-filename-concurrency",
                            f"{run_dir.name}/: directory concurrency {dir_concurrency}"
                            f" ≠ declared {config.concurrency}",
                            point_yaml,
                            "#1",
                        )
                    )
            except ValueError:
                pass

            valid_points.append((point_yaml, config))

            summary_path = run_dir / "mlperf_endpoints_log_summary.json"
            detail_path = run_dir / "mlperf_endpoints_log_detail.json"

            if not summary_path.exists():
                results.append(
                    _err(
                        "result-file-present",
                        f"Missing result log for {run_dir.name}:"
                        f" {summary_path.relative_to(self.submission_path)}",
                        summary_path,
                        "#1",
                    )
                )
                continue

            if not detail_path.exists():
                results.append(
                    _err(
                        "result-detail-present",
                        f"Missing detail log for {run_dir.name}:"
                        f" {detail_path.relative_to(self.submission_path)}",
                        detail_path,
                        "#1",
                    )
                )

            summary, load_results = load_result_summary(summary_path)
            results.extend(load_results)
            if summary is None:
                continue

            # PointResult validates point-duration and metric-consistency
            point_result = PointResult.model_validate(
                {"config": config, "summary": summary, "yaml_path": point_yaml},
                context={"regions": regions, "summary_path": summary_path},
            )
            results.extend(point_result._check_results)
            loaded_points.append((config, summary))

            # Per-run accuracy
            accuracy_dir = run_dir / "accuracy"
            txt_path = accuracy_dir / "accuracy.txt"
            if not txt_path.exists():
                results.append(
                    _err("accuracy-file", f"Missing accuracy/accuracy.txt in {run_dir.name}/", txt_path, "#15")
                )
            json_path = accuracy_dir / "accuracy_result.json"
            if not json_path.exists():
                results.append(
                    _err("accuracy-file", f"Missing accuracy/accuracy_result.json in {run_dir.name}/", json_path, "#15")
                )
                accuracy_results.append(None)
            else:
                acc_result, acc_results = load_accuracy_result(json_path)
                results.extend(acc_results)
                if any(r.severity == Severity.ERROR for r in acc_results):
                    acc_result = None
                accuracy_results.append(acc_result)

        # ModelContext validates point-count, regional-coverage, config-consistency, accuracy-gate
        model_ctx = ModelContext(
            system_id=system_name,
            system_desc=system_desc,
            model_dir=model_dir,
            regions=regions,
            all_point_count=len(run_dirs),
            valid_points=valid_points,
            loaded_points=loaded_points,
            accuracy_results=accuracy_results,
        )
        results.extend(model_ctx._check_results)

        return results
