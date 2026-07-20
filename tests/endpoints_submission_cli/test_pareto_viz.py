# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the terminal Pareto view (pareto_viz module)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from endpoints_submission_cli.pareto_viz import (
    Curve,
    Point,
    _nice_ticks,
    _pareto_frontier,
    collect_curves,
    render_pareto,
    render_pareto_from_result_dirs,
)


def _write_point(base: Path, system: str, model: str, concurrency: int, meta: dict) -> Path:
    """Create a pareto/<system>/<model>/results/point_<c>/run_metadata.json under *base*."""
    point_dir = base / "pareto" / system / model / "results" / f"point_{concurrency}"
    point_dir.mkdir(parents=True, exist_ok=True)
    (point_dir / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return point_dir


def _meta(concurrency: int, system_tps: float, ttft: float) -> dict:
    return {
        "concurrency": concurrency,
        "system_tps": system_tps,
        "tps_per_user": system_tps / concurrency,
        "ttft": ttft,
    }


@pytest.mark.unit
class TestPointAndCurve:
    def test_curve_stats(self) -> None:
        curve = Curve(
            system="sys",
            model="m",
            color="cyan",
            points=[
                Point(concurrency=16, system_tps=1300, tps_per_user=81.0, ttft=190),
                Point(concurrency=1000, system_tps=17600, tps_per_user=17.6, ttft=800),
            ],
        )
        assert curve.peak_throughput == 17600
        assert curve.best_per_user == 81.0
        assert curve.best_ttft == 190
        assert curve.max_concurrency == 1000
        assert curve.label == "sys"


@pytest.mark.unit
class TestParetoFrontier:
    def test_identifies_non_dominated_points(self) -> None:
        # (x, y) higher-is-better. (2, 2) dominates (1, 1); (3, 0) and (0, 3) are on it.
        pts = [(1.0, 1.0), (2.0, 2.0), (3.0, 0.0), (0.0, 3.0)]
        frontier = _pareto_frontier(pts)
        assert 0 not in frontier  # (1,1) dominated by (2,2)
        assert set(frontier) == {1, 2, 3}
        # Returned sorted by x ascending.
        xs = [pts[i][0] for i in frontier]
        assert xs == sorted(xs)

    def test_empty(self) -> None:
        assert _pareto_frontier([]) == []


@pytest.mark.unit
class TestNiceTicks:
    def test_round_numbers(self) -> None:
        ticks = _nice_ticks(0.0, 20000.0, 5)
        assert all(t % 1000 == 0 for t in ticks)
        assert ticks[0] >= 0.0
        assert ticks[-1] <= 20000.0

    def test_degenerate_range(self) -> None:
        assert _nice_ticks(5.0, 5.0, 5) == [5.0]


@pytest.mark.unit
class TestCollectCurves:
    def test_groups_by_system_and_model(self, tmp_path: Path) -> None:
        _write_point(tmp_path, "sysA", "llama3_1-8b", 16, _meta(16, 1300, 190))
        _write_point(tmp_path, "sysA", "llama3_1-8b", 1000, _meta(1000, 17600, 800))
        _write_point(tmp_path, "sysB", "llama3_1-8b", 64, _meta(64, 5000, 300))

        curves = collect_curves(
            [
                tmp_path / "pareto" / "sysA" / "llama3_1-8b" / "results" / "point_16",
                tmp_path / "pareto" / "sysA" / "llama3_1-8b" / "results" / "point_1000",
                tmp_path / "pareto" / "sysB" / "llama3_1-8b" / "results" / "point_64",
            ],
            tmp_path / "pareto",
        )
        assert len(curves) == 2
        by_system = {c.system: c for c in curves}
        assert len(by_system["sysA"].points) == 2
        assert len(by_system["sysB"].points) == 1
        # Points sorted by concurrency.
        assert [p.concurrency for p in by_system["sysA"].points] == [16, 1000]
        # Distinct colours and markers per curve.
        assert curves[0].color != curves[1].color
        assert curves[0].marker != curves[1].marker

    def test_skips_bad_metadata(self, tmp_path: Path) -> None:
        good = _write_point(tmp_path, "sysA", "m", 16, _meta(16, 1300, 190))
        bad = tmp_path / "pareto" / "sysA" / "m" / "results" / "point_32"
        bad.mkdir(parents=True)
        (bad / "run_metadata.json").write_text("{ not valid json", encoding="utf-8")
        missing = tmp_path / "pareto" / "sysA" / "m" / "results" / "point_64"
        missing.mkdir(parents=True)

        curves = collect_curves([good, bad, missing], tmp_path / "pareto")
        assert len(curves) == 1
        assert len(curves[0].points) == 1


@pytest.mark.unit
class TestRender:
    def _curves(self) -> list[Curve]:
        return [
            Curve(
                system="acme",
                model="m",
                color="cyan",
                marker="o",
                points=[
                    Point(16, 1300, 81.0, 190),
                    Point(256, 12000, 46.9, 400),
                    Point(1000, 17600, 17.6, 800),
                ],
            ),
            Curve(
                system="bluewave",
                model="m",
                color="yellow",
                marker="+",
                points=[
                    Point(16, 1500, 93.0, 210),
                    Point(256, 11000, 43.0, 500),
                    Point(1000, 15400, 15.4, 900),
                ],
            ),
        ]

    def test_render_produces_output(self) -> None:
        console = Console(width=100, file=io.StringIO(), record=True)
        render_pareto(self._curves(), console)
        text = console.export_text()
        assert "Pareto Frontier" in text
        assert "acme" in text and "bluewave" in text
        assert "efficient frontier" in text
        # The trade-off takeaway is present.
        assert "Trade-off" in text

    def test_render_single_curve(self) -> None:
        console = Console(width=100, file=io.StringIO(), record=True)
        render_pareto(self._curves()[:1], console)
        text = console.export_text()
        assert "acme" in text

    def test_render_flags_overlapping_points(self) -> None:
        # Two systems with identical points must collide in the same cells and be flagged.
        curves = [
            Curve(system="a", model="m", color="cyan", marker="o", points=[Point(16, 1000, 62, 100)]),
            Curve(system="b", model="m", color="yellow", marker="+", points=[Point(16, 1000, 62, 100)]),
        ]
        console = Console(width=100, file=io.StringIO(), record=True)
        render_pareto(curves, console)
        text = console.export_text()
        assert "overlap" in text  # the collision note is emitted

    def test_render_empty(self) -> None:
        console = Console(width=100, file=io.StringIO(), record=True)
        render_pareto([], console)
        assert "No run metadata" in console.export_text()

    def test_render_from_result_dirs(self, tmp_path: Path) -> None:
        _write_point(tmp_path, "sysA", "m", 16, _meta(16, 1300, 190))
        _write_point(tmp_path, "sysA", "m", 1000, _meta(1000, 17600, 800))
        dirs = [
            tmp_path / "pareto" / "sysA" / "m" / "results" / "point_16",
            tmp_path / "pareto" / "sysA" / "m" / "results" / "point_1000",
        ]
        console = Console(width=100, file=io.StringIO(), record=True)
        render_pareto_from_result_dirs(dirs, tmp_path / "pareto", console)
        assert "sysA" in console.export_text()
