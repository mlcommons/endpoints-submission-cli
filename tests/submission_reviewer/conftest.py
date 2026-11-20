# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Builders for synthetic submission trees the reviewer tests run on.

:func:`clean_curve` describes a frontier that a real serving system could produce:
throughput rises sublinearly and saturates, TPOT and TTFT grow with load, and no two
points share a distribution. Each test perturbs one value of it and expects the one
matching finding. ``point.yaml``, ``system_desc.json`` and ``system_power.json`` start
from the ``valid_standardized`` corpus submission, so they load with the checker's own
models, with its placeholder links and copied warmup replaced so the baseline produces
no metadata findings either.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from submission_reviewer import Finding, Level, ReviewReport, review

_CORPUS = Path(__file__).resolve().parents[2] / "test_submissions"
_TEMPLATE_POINT = _CORPUS / "valid_standardized/results/acme_h100x8_001/llama3_1-8b/r88"

#: Checks that need more than the submission itself: git history or per-request logs.
NEEDS_EXTRA_INPUT = frozenset(
    {"updates", "ttft_fragments", "output_padding_text", "offline_reordering", "warmup_prompts"}
)

SYSTEM = "sys_test"
MODEL = "llama3_1-8b"
CURVE = f"{SYSTEM}/{MODEL}"
DURATION_S = 1200.0
OSL = 250.0


@dataclass(frozen=True)
class PointSpec:
    """One point's metrics, in the units the rules use.

    ``ttft_spread`` and ``tpot_spread`` are p90 / p50. ``latency_ratio`` is the
    Little's-law in-flight count as a fraction of the concurrency; ``None`` writes no
    latency block.
    """

    concurrency: int
    system_tps: float
    ttft_p50_ms: float
    tpot_p50_ms: float
    ttft_spread: float = 1.8
    tpot_spread: float = 1.3
    offline: str | None = None
    latency_ratio: float | None = None
    overrides: dict[str, object] = field(default_factory=dict)


def clean_curve() -> list[PointSpec]:
    """A plausible frontier: sublinear, saturating, latency growing with load."""
    specs = []
    for c in (1, 4, 16, 64, 256, 1024):
        tpot = 5.0 * (1 + c / 100)
        specs.append(
            PointSpec(
                concurrency=c,
                system_tps=0.8 * c * 1000 / tpot,
                ttft_p50_ms=50.0 * (1 + c / 64),
                tpot_p50_ms=tpot,
            )
        )
    return specs


def with_point(specs: list[PointSpec], concurrency: int, **changes: object) -> list[PointSpec]:
    """Return *specs* with the point at *concurrency* changed."""
    return [replace(s, **changes) if s.concurrency == concurrency else s for s in specs]  # type: ignore[arg-type]


def _percentiles(p50: float, spread: float) -> dict[str, float]:
    return {"50": p50, "90": p50 * spread, "95": p50 * spread * 1.1, "99": p50 * spread * 1.3}


def summary_for(spec: PointSpec) -> dict[str, object]:
    """A ``result_summary.json`` body whose derived metrics equal *spec*'s."""
    duration_ns = DURATION_S * 1e9
    total_tokens = spec.system_tps * DURATION_S
    n = max(1, round(total_tokens / OSL))
    tpot_p90_ms = spec.tpot_p50_ms * spec.tpot_spread
    body: dict[str, object] = {
        "n_samples_issued": n,
        "n_samples_completed": n,
        "n_samples_failed": 0,
        "duration_ns": duration_ns,
        "ttft": {
            "total": spec.ttft_p50_ms * 1e6 * n,
            "percentiles": _percentiles(spec.ttft_p50_ms * 1e6, spec.ttft_spread),
        },
        "tpot": {
            "total": spec.tpot_p50_ms * 1e6 * n,
            "percentiles": _percentiles(spec.tpot_p50_ms * 1e6, spec.tpot_spread),
        },
        "output_sequence_lengths": {"total": total_tokens, "percentiles": {}},
        "system_tps": spec.system_tps,
        "tps_per_user": 1000.0 / tpot_p90_ms,
    }
    if spec.latency_ratio is not None:
        body["latency"] = {
            "total": spec.latency_ratio * spec.concurrency * duration_ns,
            "percentiles": {},
        }
    body.update(spec.overrides)
    return body


#: The corpus template's placeholders, replaced so the baseline has no metadata findings.
_CLEAN_LINKS = {
    "link_to_model": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    "link_to_model_transformation": "https://github.com/acme-ai/quantization/blob/main/README.md",
    "dataset_link": "https://github.com/mlcommons/endpoints/tree/main/datasets",
}

EditDesc = Callable[[dict[str, Any]], None]
EditConfig = Callable[[dict[str, Any], int], None]
EditPower = Callable[[dict[str, Any]], None]


def _clean_desc() -> dict[str, Any]:
    desc: dict[str, Any] = json.loads((_TEMPLATE_POINT / "system_desc.json").read_text())
    desc["node_config"] = "One node with 8x H100 SXM5, TP=8"
    desc["availability_url"] = "https://www.acme-servers.com/products/h100x8"
    desc["node_types"][0]["container_link"] = "https://hub.docker.com/r/acme/vllm-h100"
    return desc


def _clean_config(concurrency: int) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load((_TEMPLATE_POINT / "point.yaml").read_text())
    config.update(_CLEAN_LINKS, concurrency=concurrency)
    config["warmup"].update(
        concurrency=concurrency,
        data_source="synthetic prompts, 512 random tokens each",
        duration_s=30.0 + concurrency / 10,
        link_logs=f"https://storage.acme-servers.com/warmup/r{concurrency}.tar.gz",
    )
    return config


def write_submission(
    root: Path,
    specs: list[PointSpec],
    *,
    edit_desc: EditDesc | None = None,
    edit_config: EditConfig | None = None,
    edit_power: EditPower | None = None,
    docs: dict[str, str] | None = None,
    src: tuple[str, ...] = ("vllm",),
) -> Path:
    """Write a ``<submission_id>/`` tree with one frontier built from *specs*.

    The edit hooks change the clean baseline before it is written: *edit_desc* the
    ``system_desc.json`` of every point, *edit_config* each ``point.yaml`` (with the
    point's concurrency), *edit_power* the ``system_power.json``.
    """
    model_dir = root / "results" / SYSTEM / MODEL
    for spec in specs:
        point_dir = model_dir / f"r{spec.concurrency}"
        point_dir.mkdir(parents=True)
        config = _clean_config(spec.concurrency)
        if spec.offline is not None:
            config["offline"] = spec.offline
        else:
            config.pop("offline", None)
        if edit_config is not None:
            edit_config(config, spec.concurrency)
        desc = _clean_desc()
        if edit_desc is not None:
            edit_desc(desc)
        (point_dir / "point.yaml").write_text(yaml.safe_dump(config))
        (point_dir / "system_desc.json").write_text(json.dumps(desc))
        (point_dir / "result_summary.json").write_text(json.dumps(summary_for(spec)))

    power = json.loads((_TEMPLATE_POINT.parents[1] / "system_power.json").read_text())
    if edit_power is not None:
        edit_power(power)
    (model_dir.parent / "system_power.json").write_text(json.dumps(power))

    for impl in src:
        (root / "src" / impl).mkdir(parents=True)
        (root / "src" / impl / "README.md").write_text(f"# {impl}\n\nHow to launch the SUT.\n")
    (root / "docs").mkdir(exist_ok=True)
    for name, text in (docs or {}).items():
        (root / "docs" / name).write_text(text)
    return root


@pytest.fixture
def run_review(tmp_path: Path):
    """Write *specs* (default: the clean curve) as a submission and review it.

    Takes :func:`write_submission`'s keyword arguments.
    """
    counter = iter(range(1000))

    def _run(specs: list[PointSpec] | None = None, **kwargs: Any) -> ReviewReport:
        root = write_submission(tmp_path / f"sub{next(counter)}", specs or clean_curve(), **kwargs)
        return review(root)

    return _run


def checked(report: ReviewReport, check: str) -> list[Finding]:
    """Findings produced by *check*."""
    return [f for f in report.findings if f.check == check]


def titled(report: ReviewReport, title: str) -> list[Finding]:
    """Findings whose title contains *title*."""
    return [f for f in report.findings if title in f.title]


def attention(report: ReviewReport) -> list[Finding]:
    """Findings at :attr:`Level.ATTENTION`."""
    return [f for f in report.findings if f.level is Level.ATTENTION]
