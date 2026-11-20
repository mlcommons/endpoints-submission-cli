# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""§9.2 metadata checks: warmup, system description, division, speculative decoding."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest

from submission_checker.cohorts import Cohort
from submission_reviewer import Level, text
from submission_reviewer.text import (
    UrlProblem,
    accelerator_counts,
    accelerator_families,
    cohort_date,
    is_placeholder,
    name_key,
    read_docs_text,
    url_problem,
)

from .conftest import checked, titled


def test_clean_baseline_has_no_findings(run_review) -> None:
    assert run_review().findings == []


# ── Text helpers ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["TBD", " n/a ", "NA", "unknown", "???", "-", "<fill in>"])
def test_placeholders(value: str) -> None:
    assert is_placeholder(value)


@pytest.mark.parametrize("value", ["Ubuntu 22.04", "550.54", "none of the above", "", None])
def test_not_placeholders(value: object) -> None:
    assert not is_placeholder(value)


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("https://hub.docker.com/r/acme/vllm", None),
        ("https://example.com/model", UrlProblem.PLACEHOLDER),
        ("https://docs.example.org/x", UrlProblem.PLACEHOLDER),
        ("http://localhost:8000/v1", UrlProblem.PLACEHOLDER),
        ("http://10.0.0.5:8000/v1", UrlProblem.PRIVATE),
        ("https://inference.corp/v1", UrlProblem.PRIVATE),
        ("https://gpu-box/v1", UrlProblem.PRIVATE),
        ("http://api.acme.ai/v1", UrlProblem.NOT_HTTPS),
        ("s3://bucket/logs", UrlProblem.NOT_A_URL),
        ("TBD", UrlProblem.PLACEHOLDER),
        (None, None),
    ],
)
def test_url_problem(url: str | None, problem: str | None) -> None:
    assert url_problem(url) == problem


def test_accelerator_families_and_counts() -> None:
    assert accelerator_families("NVIDIA GB200 NVL72 with Grace") == {"GB200"}
    assert accelerator_families("sys_gb300_72gpu") == set()
    assert accelerator_families("8x MI355X, TPU v5p pods") == {"MI355X", "TPUV5P"}
    assert accelerator_counts("2 nodes, 8x NVIDIA H100 each") == [(8, "H100")]


def test_docs_reading_stops_at_the_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text, "_DOC_BUDGET", 10)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("a" * 6)
    (tmp_path / "docs" / "b.json").write_text("b" * 1000)
    (tmp_path / "docs" / "c.md").write_text("c" * 6)
    assert read_docs_text(tmp_path) == "aaaaaa\nbbbb"


def test_name_key() -> None:
    assert name_key("TensorRT-LLM") == name_key("tensorrt-llm") == "tensorrtllm"
    assert name_key("vLLM (TPU)") == name_key("vLLM 0.4.2") == "vllm"


@pytest.mark.parametrize(
    ("cohort", "date"),
    [("2026-10-C0", datetime.date(2026, 10, 7)), ("2026-10-C1", datetime.date(2026, 10, 21))],
)
def test_cohort_date(cohort: str, date: datetime.date) -> None:
    parsed = Cohort.parse(cohort)
    assert parsed is not None
    assert cohort_date(parsed) == date


# ── Warmup ────────────────────────────────────────────────────────────────────


def _warmup(**changes: Any):
    def edit(config: dict[str, Any], concurrency: int) -> None:
        config["warmup"].update(changes)

    return edit


def test_warmup_on_the_performance_dataset(run_review) -> None:
    report = run_review(edit_config=_warmup(data_source="llm-perf-dataset-v1, first 500 rows"))
    (found,) = checked(report, "warmup")
    assert found.level is Level.ATTENTION
    assert found.title == "Warmup declares the performance dataset"
    assert found.evidence["points"] == "r1,r4,r16,r64,r256,r1024"


def test_warmup_on_another_split_is_a_note(run_review) -> None:
    report = run_review(edit_config=_warmup(data_source="CNN/DailyMail validation split"))
    (found,) = checked(report, "warmup")
    assert (found.level, found.title) == (
        Level.NOTE,
        "Warmup uses a split of the performance dataset",
    )


def test_salted_warmup_is_context(run_review) -> None:
    def edit(config: dict[str, Any], concurrency: int) -> None:
        config["warmup"]["data_source"] = "llm-perf-dataset-v1"
        config["runtime_settings"]["warmup"] = {"salt": True}

    (found,) = checked(run_review(edit_config=edit), "warmup")
    assert (found.level, found.title) == (Level.INFO, "Salted warmup on the performance dataset")


def test_warmup_logs_without_a_link(run_review) -> None:
    report = run_review(edit_config=_warmup(logs_retained=True, link_logs=None))
    (found,) = checked(report, "warmup")
    assert found.title == "Warmup logs retained, but no link"


def test_warmup_log_link_placeholder(run_review) -> None:
    report = run_review(edit_config=_warmup(link_logs="https://example.com/warmup-logs"))
    (found,) = checked(report, "warmup")
    assert found.title == "Warmup log link can't be opened"
    assert "is a placeholder" in found.detail


def test_identical_warmup_everywhere(run_review) -> None:
    report = run_review(
        edit_config=_warmup(concurrency=64, duration_s=60.0, link_logs="https://acme.ai/w.tgz")
    )
    (found,) = checked(report, "warmup")
    assert found.title == "Identical warmup on every point"


def test_disabled_warmup_is_ignored(run_review) -> None:
    report = run_review(
        edit_config=_warmup(
            duration_s=0, requests_issued=0, requests_completed=0, data_source="llm-perf-dataset-v1"
        )
    )
    assert checked(report, "warmup") == []


# ── System description ────────────────────────────────────────────────────────


def _desc(**changes: Any):
    def edit(desc: dict[str, Any]) -> None:
        desc.update(changes)

    return edit


def _node(**changes: Any):
    def edit(desc: dict[str, Any]) -> None:
        desc["node_types"][0].update(changes)

    return edit


def test_placeholder_and_missing_fields(run_review) -> None:
    def edit(desc: dict[str, Any]) -> None:
        desc["node_types"][0].update(driver="TBD", hw_notes="N/A")
        del desc["node_types"][0]["container_link"]

    (found,) = titled(run_review(edit_desc=edit), "Missing or placeholder")
    assert found.level is Level.NOTE
    assert found.curve == "sys_test"
    assert "`driver` is “TBD”" in found.detail
    assert "`hw_notes` is “N/A”" in found.detail
    assert "`container_link` is missing" in found.detail


def test_placeholder_links(run_review) -> None:
    def edit(config: dict[str, Any], concurrency: int) -> None:
        config["link_to_model"] = "https://example.com/model"

    (found,) = titled(run_review(edit_config=edit), "Links a reviewer can't open")
    assert "link_to_model (https://example.com/model) is a placeholder" in found.detail


def test_prose_names_another_accelerator(run_review) -> None:
    report = run_review(edit_desc=_desc(node_config="8x H200 test node"))
    (found,) = titled(report, "Prose contradicts")
    assert found.level is Level.ATTENTION
    assert "names H200, but accelerator_info lists H100" in found.detail


def test_prose_count_matches_no_node_type(run_review) -> None:
    report = run_review(edit_desc=_desc(node_config="4x H100 per node"))
    (found,) = titled(report, "Prose contradicts")
    assert "says 4× H100, but the node types declare 8" in found.detail


def test_prose_total_count_is_accepted(run_review) -> None:
    def edit(desc: dict[str, Any]) -> None:
        desc["node_types"][0]["number_of_nodes"] = 2
        desc.update(system_node_ensemble_total=2, system_size="2 nodes", node_config="16x H100")

    def power(p: dict[str, Any]) -> None:
        p["node_sets"][0]["nodes_provisioned"] = 2

    assert titled(run_review(edit_desc=edit, edit_power=power), "Prose contradicts") == []


def test_framework_without_matching_src(run_review) -> None:
    report = run_review(edit_desc=_desc(serving_framework="SGLang"))
    (found,) = titled(report, "no matching code")
    assert "src/ holds vllm" in found.detail


def test_framework_alias_matches(run_review) -> None:
    report = run_review(edit_desc=_desc(serving_framework="TensorRT-LLM"), src=("trtllm",))
    assert titled(report, "no matching code") == []


def test_parallelism_beyond_the_hardware(run_review) -> None:
    def edit(desc: dict[str, Any]) -> None:
        desc["config_summary"]["tensor_parallel"] = 8
        desc["config_summary"]["data_parallel"] = 2

    (found,) = titled(run_review(edit_desc=edit), "Parallelism exceeds")
    assert found.level is Level.ATTENTION
    assert "TP×PP×DP = 16 on 8" in found.detail


def test_disaggregated_deployments_are_not_checked_for_size(run_review) -> None:
    def edit(desc: dict[str, Any]) -> None:
        desc["config_summary"].update(data_parallel=2, disaggregated=True)

    assert titled(run_review(edit_desc=edit), "Parallelism exceeds") == []


def test_config_summary_text_disagrees(run_review) -> None:
    def edit(desc: dict[str, Any]) -> None:
        desc["config_summary"] = "TP=4 PP=1 vllm defaults"
        desc["tensor_parallel"] = 8

    (found,) = titled(run_review(edit_desc=edit), "config_summary disagrees")
    assert "TP4 in text, 8 in fields" in found.detail


def test_system_size_disagrees(run_review) -> None:
    report = run_review(edit_desc=_desc(system_size="16 GPUs across 2 nodes"))
    (found,) = titled(report, "system_size disagrees")
    assert "names 16 accelerators" in found.detail
    assert "names 2 node(s)" in found.detail


def test_power_descriptor_describes_other_accelerators(run_review) -> None:
    def power(p: dict[str, Any]) -> None:
        p["node_sets"][0]["components"]["accelerator"].update(model="NVIDIA B200", count_per_node=4)

    (found,) = titled(run_review(edit_power=power), "Power descriptor describes")
    assert found.level is Level.ATTENTION
    assert "4 accelerators per node in system_power.json, 8 in system_desc.json" in found.detail
    assert "“NVIDIA B200” in system_power.json, H100 in system_desc.json" in found.detail


def test_power_descriptor_node_and_cpu_counts(run_review) -> None:
    def power(p: dict[str, Any]) -> None:
        p["node_sets"][0]["nodes_provisioned"] = 2
        p["node_sets"][0]["components"]["cpu"]["count_per_node"] = 1

    (found,) = titled(run_review(edit_power=power), "Power descriptor and system description")
    assert found.level is Level.NOTE
    assert "2 nodes provisioned" in found.detail
    assert "1 processors per node" in found.detail


# ── Division ──────────────────────────────────────────────────────────────────


def test_division_declared_inconsistently(run_review) -> None:
    def edit(config: dict[str, Any], concurrency: int) -> None:
        if concurrency == 64:
            config["division"] = "RDI"

    (found,) = titled(run_review(edit_config=edit), "Division declared inconsistently")
    assert found.level is Level.ATTENTION
    assert "r64 (RDI)" in found.detail


def test_rdi_division_with_available_status(run_review) -> None:
    def edit(config: dict[str, Any], concurrency: int) -> None:
        config["division"] = "RDI"

    report = run_review(edit_desc=_desc(division="RDI"), edit_config=edit)
    (found,) = titled(report, "RDI division with a non-RDI status")
    assert found.level is Level.ATTENTION


def _preview(date: str, **extra: Any):
    return _desc(system_availability_status="Preview", target_availability_date=date, **extra)


def test_preview_date_outside_the_window(run_review) -> None:
    # The template targets 2026-10-C1, 21 October 2026; 180 days later is 19 April 2027.
    report = run_review(
        edit_desc=_preview("2027-06-01"),
        docs={"preview.md": "The X100 accelerator begins shipping in May 2027."},
    )
    (found,) = checked(report, "division")
    assert found.title == "Preview availability date is outside the window"
    assert found.evidence["days"] == 223


def test_preview_date_already_passed(run_review) -> None:
    report = run_review(
        edit_desc=_preview("2026-09-01"), docs={"p.md": "General availability: Q3 2026"}
    )
    (found,) = checked(report, "division")
    assert found.title == "Preview availability date has already passed"


def test_preview_without_an_availability_statement(run_review) -> None:
    (found,) = checked(run_review(edit_desc=_preview("2027-01-15")), "division")
    assert found.title == "No statement of what is not yet available"


def test_preview_statement_in_notes_is_enough(run_review) -> None:
    def edit(desc: dict[str, Any]) -> None:
        _preview("2027-01-15")(desc)
        desc["node_types"][0]["hw_notes"] = "H100 NVL variant ships to customers in January 2027"

    assert checked(run_review(edit_desc=edit), "division") == []


def _serviced(url: str | None):
    def edit(desc: dict[str, Any]) -> None:
        desc.update(division="Serviced", endpoint_url=url)

    return edit


def _serviced_config(config: dict[str, Any], concurrency: int) -> None:
    config["division"] = "Serviced"


_SERVICED_DOCS = {
    "service.md": "Pricing: $0.20 per million tokens. Rate limit: 600 requests per minute. "
    "Our terms of service permit benchmarking."
}


def test_serviced_with_a_public_endpoint_and_disclosures(run_review) -> None:
    report = run_review(
        edit_desc=_serviced("https://api.acme.ai/v1/chat/completions"),
        edit_config=_serviced_config,
        docs=_SERVICED_DOCS,
    )
    assert checked(report, "division") == []


@pytest.mark.parametrize(
    ("url", "level", "phrase"),
    [
        (None, Level.ATTENTION, "is missing"),
        ("http://10.1.2.3:8000/v1", Level.ATTENTION, UrlProblem.PRIVATE),
        ("http://api.acme.ai/v1", Level.NOTE, UrlProblem.NOT_HTTPS),
    ],
)
def test_serviced_endpoint_problems(run_review, url: str | None, level: Level, phrase: str) -> None:
    report = run_review(edit_desc=_serviced(url), edit_config=_serviced_config, docs=_SERVICED_DOCS)
    (found,) = checked(report, "division")
    assert found.title == "Serviced endpoint URL is not a public endpoint"
    assert (found.level, phrase in found.detail) == (level, True)


def test_serviced_preview_tier_host(run_review) -> None:
    report = run_review(
        edit_desc=_serviced("https://beta.api.acme.ai/v1"),
        edit_config=_serviced_config,
        docs=_SERVICED_DOCS,
    )
    (found,) = checked(report, "division")
    assert found.title == "Serviced endpoint looks like a non-production tier"


def test_serviced_disclosures_missing(run_review) -> None:
    report = run_review(
        edit_desc=_serviced("https://api.acme.ai/v1"),
        edit_config=_serviced_config,
        docs={"service.md": "Pricing: $0.20 per million tokens."},
    )
    (found,) = checked(report, "division")
    assert found.evidence["missing"] == (
        "rate limits or quotas, terms of service permitting benchmarking"
    )


def test_available_with_a_placeholder_url(run_review) -> None:
    report = run_review(edit_desc=_desc(availability_url="https://example.com/systems"))
    (found,) = checked(report, "division")
    assert found.title == "Availability URL can't be checked"


# ── Speculative decoding ──────────────────────────────────────────────────────


def _specdec(by_point: dict[int, dict[str, Any]]):
    def edit(config: dict[str, Any], concurrency: int) -> None:
        if concurrency in by_point:
            config["speculative_decoding"] = by_point[concurrency]

    return edit


def test_no_speculative_decoding_no_findings(run_review) -> None:
    assert checked(run_review(), "specdec_info") == []


def test_speculative_decoding_is_listed(run_review) -> None:
    eagle = {"algorithm": "EAGLE-3", "model_id": "acme/eagle3-llama", "weight_checksum": "abc"}
    report = run_review(edit_config=_specdec({64: eagle, 256: {**eagle, "num_tokens": 4}}))
    (found,) = checked(report, "specdec_info")
    assert found.level is Level.INFO
    assert found.focus_area == "speculative_decoding"
    assert "r64: model_id=acme/eagle3-llama, weight_checksum=abc, algorithm=EAGLE-3" in found.detail


def test_drafter_differs_between_points(run_review) -> None:
    report = run_review(
        edit_config=_specdec({64: {"model_id": "acme/eagle-a"}, 256: {"model_id": "acme/eagle-b"}})
    )
    assert [f.title for f in checked(report, "specdec_info")] == [
        "Speculative decoding on 2 point(s)",
        "Drafter differs between points",
    ]
