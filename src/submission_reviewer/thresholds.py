# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Heuristic tolerances, loaded from ``data/reviewer_thresholds.yaml``.

The packaged YAML is the single source of the values, so the models below have no
defaults: a key missing from it is a packaging bug and fails loudly. An override file
replaces only the keys it names.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .logs import TAIL_CHARS

__all__ = [
    "DistributionThresholds",
    "LogThresholds",
    "OutputThresholds",
    "ParetoThresholds",
    "Thresholds",
    "UpdateThresholds",
    "load_thresholds",
]

_PACKAGED = "reviewer_thresholds.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ParetoThresholds(_Section):
    """Tolerances for :mod:`submission_reviewer.checks.pareto_shape`."""

    dip_tolerance: float
    superlinear_tolerance: float
    per_user_rate_factor: float
    ttft_drop_note: float
    ttft_drop_attention: float
    tps_per_user_rise_note: float
    dominance_margin: float
    littles_law_high: float
    littles_law_low: float


class DistributionThresholds(_Section):
    """Tolerances for :mod:`submission_reviewer.checks.metric_distributions`."""

    uniform_ttft_min_span: float
    uniform_ttft_max_cv: float
    narrow_ratio: float
    identical_block_min_points: int
    round_step: float
    round_min_points: int


class OutputThresholds(_Section):
    """Tolerances for the aggregate output-length heuristics."""

    osl_deviation_note: float
    osl_min_samples: int
    length_finish_note: float


class LogThresholds(_Section):
    """Tolerances for the checks that read ``events.jsonl``."""

    max_text_samples: int
    max_text_chars: int
    ttft_trivial_note: float
    ttft_trivial_attention: float
    first_gap_ratio_note: float
    duplicate_containment: float
    duplicate_min_words: int
    duplicate_share_note: float
    duplicate_share_attention: float
    repetition_ratio: float
    repetition_min_words: int
    repetition_share_note: float
    repetition_share_attention: float
    filler_run: int = Field(gt=0, le=TAIL_CHARS)
    filler_share_note: float
    filler_share_attention: float
    offline_cross_pass_note: float
    offline_cross_pass_attention: float
    offline_same_sample_attention: float


class UpdateThresholds(_Section):
    """Tolerances for :mod:`submission_reviewer.checks.updates`."""

    metric_change: float


class Thresholds(_Section):
    """Every tolerance the checks use, one section per check family."""

    pareto: ParetoThresholds
    distributions: DistributionThresholds
    updates: UpdateThresholds
    outputs: OutputThresholds
    logs: LogThresholds


class ThresholdsError(ValueError):
    """An override file could not be read or does not fit the schema."""


def _read_yaml(text: str, source: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ThresholdsError(f"{source}: not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ThresholdsError(f"{source}: expected a mapping at the top level")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_thresholds(override: Path | None = None) -> Thresholds:
    """Return the packaged thresholds, with *override*'s keys applied on top.

    Args:
        override: Optional YAML file with the same shape as the packaged one.

    Raises:
        ThresholdsError: If *override* cannot be read or names an unknown key or a
            value of the wrong type.
    """
    packaged = resources.files("submission_reviewer").joinpath("data").joinpath(_PACKAGED)
    data = _read_yaml(packaged.read_text(encoding="utf-8"), _PACKAGED)
    if override is not None:
        try:
            text = override.read_text(encoding="utf-8")
        except OSError as exc:
            raise ThresholdsError(f"{override}: {exc}") from exc
        data = _merge(data, _read_yaml(text, str(override)))
    try:
        return Thresholds.model_validate(data)
    except ValueError as exc:
        raise ThresholdsError(f"{override or _PACKAGED}: {exc}") from exc
