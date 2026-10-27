"""Pydantic models for submission directory structure validation (§8.1)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, PrivateAttr, computed_field, model_validator

from .. import layout
from .results import CheckResult, err, ok


class SubmissionDir(BaseModel):
    """Validates the submission directory: results/ and docs/ must exist."""

    _check_results: list[CheckResult] = PrivateAttr(default_factory=list)

    root: Path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def results_dir(self) -> Path:
        """Path to the results/ subdirectory."""
        return self.root / layout.RESULTS_DIR

    @computed_field  # type: ignore[prop-decorator]
    @property
    def docs_dir(self) -> Path:
        """Path to the docs/ subdirectory."""
        return self.root / layout.DOCS_DIR

    @model_validator(mode="after")
    def _check_required_dirs(self) -> SubmissionDir:
        hint = self._legacy_layout_hint()
        for name in (layout.RESULTS_DIR, layout.DOCS_DIR):
            path = self.root / name
            if path.is_dir():
                self._check_results.append(ok("required-dir", "pass", path, name=name))
            else:
                self._check_results.append(err("required-dir", "fail", path, name=name, hint=hint))
        return self

    def _legacy_layout_hint(self) -> str:
        """Name the v0.7 layout when that is what the submission uses.

        v0.7 kept results under ``pareto/`` and documentation under
        ``documentation/``; v1.0 (§8.1) renamed both. Without this, a v0.7-shaped
        submission reads as two unrelated missing directories.
        """
        legacy = [name for name in ("pareto", "documentation") if (self.root / name).is_dir()]
        if not legacy:
            return ""
        found = ", ".join(f"{name}/" for name in legacy)
        return (
            f" (found {found}: this looks like the v0.7 layout; v1.0 uses"
            " results/<system>/<benchmark_model>/r<N>/ and docs/ — see §8.1)"
        )


class SrcDir(BaseModel):
    """Validates the shared src/ tree (§8.1).

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
        src_dir = self.root / layout.SRC_DIR
        if not src_dir.is_dir():
            self._check_results.append(err("src-dir", "fail", src_dir))
            return self

        impl_dirs = [d for d in sorted(src_dir.iterdir()) if d.is_dir()]
        if not impl_dirs:
            self._check_results.append(err("src-dir", "fail-2", src_dir))
            return self

        self._check_results.append(
            ok(
                "src-dir",
                "pass",
                src_dir,
                impl_dirs_count=len(impl_dirs),
                value="y" if len(impl_dirs) == 1 else "ies",
                impl_dirs=", ".join(d.name for d in impl_dirs),
            )
        )

        for impl_dir in impl_dirs:
            readme = next(
                (p for p in impl_dir.iterdir() if p.is_file() and p.name.lower() == "readme.md"),
                None,
            )
            if readme is not None:
                self._check_results.append(
                    ok("src-readme", "pass", readme, impl_dir_name=impl_dir.name)
                )
            else:
                self._check_results.append(
                    err("src-readme", "fail", impl_dir / "README.md", impl_dir_name=impl_dir.name)
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
        return layout.iter_point_dirs(self.root)

    @model_validator(mode="after")
    def _check_point_dirs(self) -> ModelDir:
        rel = f"results/{self.system_id}/{self.benchmark_model}"
        if not self.root.is_dir():
            self._check_results.append(err("point-dirs", "fail", self.root, rel=rel))
            return self
        dirs = self.point_dirs
        if dirs:
            self._check_results.append(
                ok(
                    "point-dirs",
                    "pass",
                    self.root,
                    dirs_count=len(dirs),
                    value="y" if len(dirs) == 1 else "ies",
                    rel=rel,
                    dirs=", ".join(d.name for d in dirs),
                )
            )
        else:
            self._check_results.append(err("point-dirs", "fail-2", self.root, rel=rel))
        return self
