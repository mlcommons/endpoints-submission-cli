# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Approved speculative-decoding drafters (§2.9.4).

v1.0 replaced the single fixed drafter with "a curated approved-drafter-list model,
per benchmark". Two rules follow: a drafter must be *on* its benchmark's list, and it
may first be used in a submission whose ``target_cohort`` is at least two cohorts
after the one in which it was approved.

Like the seed sets, the list ships as data rather than code — §2.9.4 versions it per
submission round, so a checker release must not be on the critical path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .cohorts import Cohort

__all__ = [
    "APPROVED_DRAFTERS_ENV_VAR",
    "DRAFTER_APPROVAL_LEAD_COHORTS",
    "ApprovedDrafter",
    "DrafterListError",
    "bundled_drafters_path",
    "load_approved_drafters",
]

#: Environment variable naming a replacement drafter list.
APPROVED_DRAFTERS_ENV_VAR = "MLPERF_ENDPOINTS_APPROVED_DRAFTERS"

#: §2.9.4: "at least two cohorts after the cohort in which the drafter was approved".
DRAFTER_APPROVAL_LEAD_COHORTS = 2

_BUNDLED = Path(__file__).parent / "data" / "approved_drafters.yaml"


class DrafterListError(ValueError):
    """Raised when a drafter list cannot be read or is malformed."""


@dataclass(frozen=True)
class ApprovedDrafter:
    """One entry on a benchmark's approved list.

    §2.9.4 allows two identification forms, and an entry uses exactly one:

    * **Weight-identified** — a distinct draft model or head, by ``model_id`` and
      ``weight_checksum``.
    * **Configuration-identified** — a drafter introducing no separate weights (a
      self-speculative or early-exit pass), by ``target_checksum`` plus the
      ``configuration`` defining the draft pass.

    Attributes:
        benchmark: The benchmark model this entry is approved for.
        approved_cohort: The cohort in which the updated list was published.
        model_id: Draft model identifier, for a weight-identified entry.
        weight_checksum: Draft weight checksum, for a weight-identified entry.
        target_checksum: Target checksum, for a configuration-identified entry.
        configuration: The configuration defining the draft pass.
    """

    benchmark: str
    approved_cohort: str | None = None
    model_id: str | None = None
    weight_checksum: str | None = None
    target_checksum: str | None = None
    configuration: dict[str, object] = field(default_factory=dict)

    @property
    def is_configuration_identified(self) -> bool:
        """True when this entry introduces no separate weights (§2.9.4)."""
        return self.weight_checksum is None and self.target_checksum is not None

    def earliest_target_cohort(self) -> Cohort | None:
        """The first cohort a submission may use this drafter in (§2.9.4).

        Two cohorts after approval. ``None`` when the entry records no approval
        cohort, which leaves the lead-time rule unevaluable for it.
        """
        approved = Cohort.parse(self.approved_cohort) if self.approved_cohort else None
        if approved is None:
            return None
        cohort = approved
        for _ in range(DRAFTER_APPROVAL_LEAD_COHORTS):
            cohort = cohort.next()
        return cohort

    def matches(self, declared: dict[str, object]) -> bool:
        """True when a point's ``speculative_decoding`` block names this drafter.

        Matching is by checksum, per §9.1's "by weight checksum, or by target checksum
        plus configuration". A model ID alone is not enough — §2.9.4 identifies an
        entry by checksum precisely so a renamed or re-uploaded drafter cannot pass as
        an approved one.
        """
        if self.weight_checksum is not None:
            return declared.get("weight_checksum") == self.weight_checksum
        if self.target_checksum is not None:
            if declared.get("target_checksum") != self.target_checksum:
                return False
            stated = declared.get("configuration")
            return bool(self.configuration) and stated == self.configuration
        return False


def bundled_drafters_path() -> Path:
    """Path to the drafter list shipped with this checker."""
    return _BUNDLED


def load_approved_drafters(path: Path | None = None) -> dict[str, list[ApprovedDrafter]]:
    """Load the approved drafters, keyed by benchmark.

    Resolution order: the explicit *path*, then ``$MLPERF_ENDPOINTS_APPROVED_DRAFTERS``,
    then the bundled file.

    Returns:
        Entries grouped by benchmark. A benchmark absent from the mapping has no
        approved drafter, which §2.9.4 makes equivalent to disallowing speculative
        decoding for it.

    Raises:
        DrafterListError: If the chosen file is missing, unreadable, or malformed.
    """
    chosen = path or _env_path() or _BUNDLED
    try:
        raw = yaml.safe_load(chosen.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DrafterListError(f"Cannot read drafter list {chosen}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise DrafterListError(f"Invalid YAML in drafter list {chosen}: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict) or not isinstance(raw.get("drafters"), list):
        raise DrafterListError(f"{chosen} must be a mapping with a 'drafters' list")

    by_benchmark: dict[str, list[ApprovedDrafter]] = {}
    for index, entry in enumerate(raw["drafters"]):
        if not isinstance(entry, dict):
            raise DrafterListError(f"{chosen}: drafters[{index}] is not a mapping")
        benchmark = entry.get("benchmark")
        if not isinstance(benchmark, str) or not benchmark:
            raise DrafterListError(f"{chosen}: drafters[{index}] has no 'benchmark'")
        drafter = ApprovedDrafter(
            benchmark=benchmark,
            approved_cohort=_optional_str(entry.get("approved_cohort")),
            model_id=_optional_str(entry.get("model_id")),
            weight_checksum=_optional_str(entry.get("weight_checksum")),
            target_checksum=_optional_str(entry.get("target_checksum")),
            configuration=dict(entry.get("configuration") or {}),
        )
        if drafter.weight_checksum is None and drafter.target_checksum is None:
            raise DrafterListError(
                f"{chosen}: drafters[{index}] identifies no drafter — §2.9.4 needs a"
                " weight_checksum, or a target_checksum plus configuration"
            )
        by_benchmark.setdefault(benchmark, []).append(drafter)
    return by_benchmark


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _env_path() -> Path | None:
    """The drafter list named by the environment, if any."""
    value = os.environ.get(APPROVED_DRAFTERS_ENV_VAR)
    return Path(value) if value else None
