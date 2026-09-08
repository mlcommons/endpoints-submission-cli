"""Pydantic models for submission directory structure validation (§8.1)."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, PrivateAttr, computed_field, model_validator

from .results import CheckResult, err, ok

#: A Pareto-point directory: "r" followed by the concurrency level (r1, r32, r256).
_POINT_DIR_RE = re.compile(r"r(\d+)")


class SubmissionDir(BaseModel):
    """Validates the submission directory: results/ and docs/ must exist."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    root: Path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def results_dir(self) -> Path:
        """Path to the results/ subdirectory."""
        return self.root / "results"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def docs_dir(self) -> Path:
        """Path to the docs/ subdirectory."""
        return self.root / "docs"

    @model_validator(mode="after")
    def _check_required_dirs(self) -> SubmissionDir:
        for name in ("results", "docs"):
            path = self.root / name
            if path.is_dir():
                self._check_results.append(
                    ok("required-dir", f"Found required directory: {name}/", path, "#1")
                )
            else:
                self._check_results.append(
                    err("required-dir", f"Missing required directory: {name}/", path, "#1")
                )
        return self


class SrcDir(BaseModel):
    """Validates the shared src/ tree (§2.2.1).

    src/ is shared across the whole submission and holds one directory per
    implementation (trtllm/, vllm/, sglang/, …). At least one such directory must
    exist and carry a README.md explaining how to build the SUT and reproduce a
    point.

    Enforced for every submission regardless of division for now; division-specific
    rulings are expected to relax this later.
    """

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    root: Path

    @model_validator(mode="after")
    def _check_src(self) -> SrcDir:
        src_dir = self.root / "src"
        if not src_dir.is_dir():
            self._check_results.append(
                err(
                    "src-dir",
                    "Missing src/ directory",
                    src_dir,
                    "#1",
                )
            )
            return self

        impl_dirs = [d for d in sorted(src_dir.iterdir()) if d.is_dir()]
        if not impl_dirs:
            self._check_results.append(
                err(
                    "src-dir",
                    "src/ contains no implementation directory",
                    src_dir,
                    "#1",
                )
            )
            return self

        self._check_results.append(
            ok(
                "src-dir",
                f"src/ present with {len(impl_dirs)} implementation "
                f"director{'y' if len(impl_dirs) == 1 else 'ies'}: "
                f"{', '.join(d.name for d in impl_dirs)}",
                src_dir,
                "#1",
            )
        )

        for impl_dir in impl_dirs:
            readme = next(
                (p for p in impl_dir.iterdir() if p.is_file() and p.name.lower() == "readme.md"),
                None,
            )
            if readme is not None:
                self._check_results.append(
                    ok("src-readme", f"src/{impl_dir.name}/README.md present", readme, "#1")
                )
            else:
                self._check_results.append(
                    err(
                        "src-readme",
                        f"Missing README.md in src/{impl_dir.name}/ "
                        "(each implementation must document how to reproduce a point)",
                        impl_dir / "README.md",
                        "#1",
                    )
                )
        return self


class SystemResults(BaseModel):
    """Validates results/<system_id>/ exists."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    results_dir: Path
    system_id: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def system_dir(self) -> Path:
        """Path to the results/<system_id>/ subdirectory."""
        return self.results_dir / self.system_id

    @model_validator(mode="after")
    def _check_dir_exists(self) -> SystemResults:
        path = self.system_dir
        if path.is_dir():
            self._check_results.append(
                ok("system-results-dir", f"Found results/{self.system_id}/", path, "#1")
            )
        else:
            self._check_results.append(
                err(
                    "system-results-dir",
                    f"No results/{self.system_id}/ directory found",
                    path,
                    "#1",
                )
            )
        return self


class ModelDir(BaseModel):
    """Validates a benchmark-model directory holds at least one r<N>/ point directory."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    root: Path
    system_id: str
    benchmark_model: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def point_dirs(self) -> list[Path]:
        """The r<N>/ Pareto-point directories, ordered by concurrency."""
        found = [d for d in self.root.iterdir() if d.is_dir() and _POINT_DIR_RE.fullmatch(d.name)]
        return sorted(found, key=lambda d: int(d.name[1:]))

    @model_validator(mode="after")
    def _check_point_dirs(self) -> ModelDir:
        rel = f"results/{self.system_id}/{self.benchmark_model}"
        if not self.root.is_dir():
            self._check_results.append(
                err("point-dirs", f"Missing benchmark-model directory: {rel}/", self.root, "#1")
            )
            return self
        dirs = self.point_dirs
        if dirs:
            self._check_results.append(
                ok(
                    "point-dirs",
                    f"Found {len(dirs)} Pareto point director"
                    f"{'y' if len(dirs) == 1 else 'ies'} in {rel}/: "
                    f"{', '.join(d.name for d in dirs)}",
                    self.root,
                    "#1",
                )
            )
        else:
            self._check_results.append(
                err(
                    "point-dirs",
                    f"No r<N>/ Pareto point directories in {rel}/",
                    self.root,
                    "#1",
                )
            )
        return self
