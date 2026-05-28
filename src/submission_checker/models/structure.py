"""Pydantic models for submission directory structure validation (§8.1)."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, PrivateAttr, computed_field, model_validator

from .results import CheckResult, err, ok


class ModelDir(BaseModel):
    """Validates a <model>/ directory: sweep CSVs and at least one r<N>/ run dir."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    root: Path
    system_name: str
    model_name: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def run_dirs(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            d for d in self.root.iterdir() if d.is_dir() and re.match(r"^r\d+$", d.name)
        )

    @model_validator(mode="after")
    def _check_sweep_files(self) -> ModelDir:
        for fname in ("sweep_summary.csv", "sweep_distributions.csv"):
            path = self.root / fname
            if path.is_file():
                self._check_results.append(
                    ok(
                        "model-sweep-file",
                        f"Found {fname} in {self.system_name}/{self.model_name}/",
                        path,
                        "#1",
                    )
                )
            else:
                self._check_results.append(
                    err(
                        "model-sweep-file",
                        f"Missing {fname} in {self.system_name}/{self.model_name}/",
                        path,
                        "#1",
                    )
                )
        return self


class RunDir(BaseModel):
    """Validates an r<N>/ run directory: required files for a self-contained pareto point."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    root: Path
    system_name: str
    model_name: str

    @model_validator(mode="after")
    def _check_run_files(self) -> RunDir:
        label = f"{self.system_name}/{self.model_name}/{self.root.name}"

        point_yaml = self.root / "point.yaml"
        if point_yaml.is_file():
            self._check_results.append(
                ok("run-point-config", f"Found point.yaml in {label}/", point_yaml, "#1")
            )
        else:
            self._check_results.append(
                err("run-point-config", f"Missing point.yaml in {label}/", point_yaml, "#1")
            )

        accuracy_dir = self.root / "accuracy"
        if accuracy_dir.is_dir():
            self._check_results.append(
                ok("run-accuracy-dir", f"Found accuracy/ in {label}/", accuracy_dir, "#1")
            )
        else:
            self._check_results.append(
                err("run-accuracy-dir", f"Missing accuracy/ in {label}/", accuracy_dir, "#1")
            )

        for fname in ("mlperf_endpoints_log_summary.json", "mlperf_endpoints_log_detail.json"):
            path = self.root / fname
            if path.is_file():
                self._check_results.append(
                    ok("run-result-files", f"Found {fname} in {label}/", path, "#1")
                )
            else:
                self._check_results.append(
                    err("run-result-files", f"Missing {fname} in {label}/", path, "#1")
                )

        for fname, rule in (("run_metadata.json", "run-metadata"), ("report.txt", "run-report")):
            path = self.root / fname
            if path.is_file():
                self._check_results.append(ok(rule, f"Found {fname} in {label}/", path, "#1"))
            else:
                self._check_results.append(
                    err(rule, f"Missing {fname} in {label}/", path, "#1")
                )

        src_dir = self.root / "src"
        impl_dirs = [d for d in src_dir.iterdir() if d.is_dir()] if src_dir.is_dir() else []
        if src_dir.is_dir() and impl_dirs:
            self._check_results.append(
                ok("run-src-dir", f"Found src/<impl>/ in {label}/", src_dir, "#1")
            )
        else:
            self._check_results.append(
                err(
                    "run-src-dir",
                    f"Missing src/<implementation>/ in {label}/ (src/ must exist with at least one implementation subdirectory)",
                    src_dir,
                    "#1",
                )
            )

        return self
