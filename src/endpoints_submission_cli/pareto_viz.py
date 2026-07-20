# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Terminal Pareto-frontier view for a submission's measurement points.

Renders the same throughput-vs-interactivity trade-off shown by the MLPerf
endpoints web visualizer, but as a self-contained Unicode chart so submitters can
sanity-check their Pareto curves over SSH on a headless results box — no browser,
no local server.

The chart plots one curve per ``pareto/<system>/<model>`` directory:

    Y  = System throughput  (tokens/s, higher is better)
    X  = Tokens/s per user  (interactivity, higher is better)

so the ideal corner is top-right. As concurrency rises a system trades per-user
speed for aggregate throughput, tracing the characteristic falling curve.

Data is read from each point's ``run_metadata.json`` (the field the web
visualizer also consumes), so the terminal view matches the site.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

__all__ = ["render_pareto_from_result_dirs", "collect_curves", "Curve", "Point"]

# ANSI-standard colour names, chosen so curves stay distinct even on 8/16-colour
# terminals over SSH (where truecolor hexes can collapse together). The first three
# — cyan / yellow / magenta — are maximally separated for the common 1–3 curve case.
_PALETTE = [
    "cyan",
    "yellow",
    "magenta",
    "green",
    "red",
    "blue",
    "bright_white",
    "bright_black",
]

# Distinct ASCII marker glyphs, paired with the palette. Series identity is carried
# by BOTH colour and shape, so curves stay tellable apart in monochrome terminals,
# piped logs, and for colour-blind readers — not just on a colour display.
_MARKERS = ["o", "+", "x", "v", "#", "@", "^", "%"]

# Drawn when two different series' markers land in the same character cell, so an
# overlap is never silently hidden (reserved — not a series marker).
_COLLISION_MARKER = "*"

# Braille dot bitmasks, indexed [row 0..3][col 0..1]; char = chr(0x2800 + mask).
_BRAILLE_BITS = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


@dataclass
class Point:
    """One measurement point on a curve."""

    concurrency: int
    system_tps: float
    tps_per_user: float
    ttft: float


@dataclass
class Curve:
    """All points for one ``pareto/<system>/<model>`` directory."""

    system: str
    model: str
    color: str
    marker: str = "o"
    points: list[Point] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Human label — system name, plus model when several models are present."""
        return self.system

    @property
    def peak_throughput(self) -> float:
        """Highest system throughput across the curve's points."""
        return max((p.system_tps for p in self.points), default=0.0)

    @property
    def best_per_user(self) -> float:
        """Highest tokens/s-per-user across the curve's points."""
        return max((p.tps_per_user for p in self.points), default=0.0)

    @property
    def max_concurrency(self) -> int:
        """Highest concurrency level exercised."""
        return max((p.concurrency for p in self.points), default=0)

    @property
    def best_ttft(self) -> float:
        """Lowest (best) TTFT across the curve's points."""
        return min((p.ttft for p in self.points), default=0.0)


class _BrailleCanvas:
    """A braille sub-pixel canvas: each character cell holds a 2×4 dot grid.

    Coordinates are in dot space with the origin at the top-left; callers convert
    data to dot space via :meth:`_Plot`. Each cell keeps a single colour (the last
    series that drew into it wins), which is fine because curves rarely overlap.
    """

    def __init__(self, cols: int, rows: int) -> None:
        self.cols = cols
        self.rows = rows
        self.width = cols * 2
        self.height = rows * 4
        # (cx, cy) -> [mask, color]
        self._cells: dict[tuple[int, int], list[Any]] = {}
        # (cx, cy) -> [char, style]; drawn on top of braille (markers, labels).
        self._overlay: dict[tuple[int, int], list[str]] = {}
        # (cx, cy) -> series marker char already placed there (for collision detection).
        self._markers: dict[tuple[int, int], str] = {}
        # Count of cells where two different series markers overlapped (hidden points).
        self.collisions = 0

    def plot(self, x: int, y: int, color: str) -> None:
        """Light the dot at integer dot-coordinate (*x*, *y*) in *color*."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        cx, cy = x // 2, y // 4
        bit = _BRAILLE_BITS[y % 4][x % 2]
        cell = self._cells.get((cx, cy))
        if cell is None:
            self._cells[(cx, cy)] = [bit, color]
        else:
            cell[0] |= bit
            cell[1] = color

    def line(self, x0: int, y0: int, x1: int, y1: int, color: str) -> None:
        """Draw a straight line between two dot-coordinates (Bresenham)."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.plot(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def marker_cell(self, x: int, y: int) -> tuple[int, int]:
        """Return the character cell containing dot-coordinate (*x*, *y*)."""
        return x // 2, y // 4

    def put_marker(self, cx: int, cy: int, char: str, style: str) -> None:
        """Place a series marker, flagging (never hiding) an overlap with another series.

        If a different series' marker already occupies the cell, both would collapse to
        one glyph — so we swap in the reserved collision glyph and count it, instead of
        letting the later series silently overwrite the earlier one.
        """
        if not (0 <= cx < self.cols and 0 <= cy < self.rows):
            return
        existing = self._markers.get((cx, cy))
        if existing is not None and existing != char:
            self.collisions += 1
            self._markers[(cx, cy)] = _COLLISION_MARKER
            self._overlay[(cx, cy)] = [_COLLISION_MARKER, "bold white"]
            return
        self._markers[(cx, cy)] = char
        self._overlay[(cx, cy)] = [char, style]

    def put_label(self, cx: int, cy: int, text: str, style: str) -> None:
        """Write *text* starting at cell (*cx*, *cy*), skipping cells already occupied."""
        for i, ch in enumerate(text):
            col = cx + i
            if not (0 <= col < self.cols and 0 <= cy < self.rows):
                return
            if (col, cy) in self._overlay:
                return  # don't clobber a marker or another label
            self._overlay[(col, cy)] = [ch, style]

    def rows_markup(self) -> list[str]:
        """Render the canvas to a list of Rich-markup strings, one per character row."""
        out = []
        for cy in range(self.rows):
            parts = []
            for cx in range(self.cols):
                over = self._overlay.get((cx, cy))
                if over is not None:
                    parts.append(f"[{over[1]}]{over[0]}[/]")
                    continue
                cell = self._cells.get((cx, cy))
                if cell is None:
                    parts.append(" ")
                else:
                    parts.append(f"[{cell[1]}]{chr(0x2800 + cell[0])}[/]")
            out.append("".join(parts))
        return out


def _fmt(value: float) -> str:
    """Compact human formatting for axis ticks and stats."""
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def collect_curves(result_dirs: list[Path], pareto_root: Path) -> list[Curve]:
    """Group point directories into curves and load their run metadata.

    Args:
        result_dirs: ``point_*`` directories discovered under the submission.
        pareto_root: The submission's ``pareto/`` directory, used to derive the
            ``<system>/<model>`` curve key from each point's path.

    Returns:
        One :class:`Curve` per ``<system>/<model>``, ordered by first appearance,
        each with a stable palette colour and points sorted by concurrency.
    """
    curves: dict[tuple[str, str], Curve] = {}
    order: list[tuple[str, str]] = []

    for point_dir in result_dirs:
        meta_path = point_dir / "run_metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        try:
            rel = point_dir.resolve().relative_to(pareto_root.resolve()).parts
        except ValueError:
            rel = point_dir.parts
        system = rel[0] if len(rel) >= 1 else point_dir.name
        model = rel[1] if len(rel) >= 2 else ""
        key = (system, model)

        point = _point_from_meta(meta)
        if point is None:
            continue

        if key not in curves:
            idx = len(order)
            curves[key] = Curve(
                system=system,
                model=model,
                color=_PALETTE[idx % len(_PALETTE)],
                marker=_MARKERS[idx % len(_MARKERS)],
            )
            order.append(key)
        curves[key].points.append(point)

    result = [curves[k] for k in order]
    for curve in result:
        curve.points.sort(key=lambda p: p.concurrency)
    return result


def _point_from_meta(meta: dict[str, Any]) -> Point | None:
    """Build a :class:`Point` from a run_metadata dict, or None if fields are unusable."""
    try:
        return Point(
            concurrency=int(meta["concurrency"]),
            system_tps=float(meta["system_tps"]),
            tps_per_user=float(meta["tps_per_user"]),
            ttft=float(meta["ttft"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Return indices of non-dominated points (both axes higher-is-better), sorted by x.

    A point is dominated if another point is at least as good on both axes and
    strictly better on at least one. The returned indices trace the efficient
    frontier left-to-right.
    """
    keep: list[int] = []
    for i, (xi, yi) in enumerate(points):
        dominated = any(
            j != i and xj >= xi and yj >= yi and (xj > xi or yj > yi)
            for j, (xj, yj) in enumerate(points)
        )
        if not dominated:
            keep.append(i)
    keep.sort(key=lambda i: points[i][0])
    return keep


def _fmt_conc(c: int) -> str:
    """Compact concurrency label (e.g. 1000 -> '1k')."""
    return f"{c // 1000}k" if c >= 1000 and c % 1000 == 0 else str(c)


def _nice_ticks(lo: float, hi: float, count: int) -> list[float]:
    """Return ~*count* evenly spaced 'nice' round tick values spanning [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if raw <= step:
            break
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


def _build_chart(curves: list[Curve], cols: int, rows: int) -> Group:
    """Render the scatter/line chart with axes, frontier hull, and markers."""
    all_x = [p.tps_per_user for c in curves for p in c.points]
    all_y = [p.system_tps for c in curves for p in c.points]
    x_min, x_max = min(all_x), max(all_x)
    y_max = max(all_y)
    # Pad X so extreme points aren't glued to the border; anchor Y at 0 for an honest scale.
    x_pad = (x_max - x_min) * 0.08 or 1.0
    x_lo, x_hi = x_min - x_pad, x_max + x_pad
    y_lo, y_hi = 0.0, y_max * 1.08 or 1.0

    canvas = _BrailleCanvas(cols, rows)

    def to_dot(x: float, y: float) -> tuple[int, int]:
        dx = int((x - x_lo) / (x_hi - x_lo) * (canvas.width - 1))
        dy = int((y_hi - y) / (y_hi - y_lo) * (canvas.height - 1))
        return dx, dy

    # 1. Each curve's connecting line (braille, coloured).
    for curve in curves:
        pts = sorted(curve.points, key=lambda p: p.tps_per_user)
        for a, b in zip(pts, pts[1:], strict=False):
            x0, y0 = to_dot(a.tps_per_user, a.system_tps)
            x1, y1 = to_dot(b.tps_per_user, b.system_tps)
            canvas.line(x0, y0, x1, y1, curve.color)

    # 2. Identify the cross-system efficient frontier. We do NOT draw a connecting
    #    hull line — a single line across the chart slices through the other curves and
    #    clutters the crossing region. Instead the frontier is conveyed by emphasising
    #    the points themselves (bold marker + concurrency label) in step 3, so the best
    #    trade-offs read as the upper-right envelope of highlighted points.
    flat = [(p.tps_per_user, p.system_tps) for c in curves for p in c.points]
    flat_pts = [p for c in curves for p in c.points]
    frontier_ids = set(_pareto_frontier(flat))

    # 3. Measured points as per-series glyph markers (overlay, on top of every line).
    #    Frontier points are bold + white-labelled; the rest are dimmed so the frontier
    #    stands out. Concurrency labels go on frontier points plus each curve's own
    #    endpoints, so every curve's concurrency range is readable.
    frontier_point_ids = {id(flat_pts[i]) for i in frontier_ids}
    for curve in curves:
        endpoints = set()
        if curve.points:
            endpoints = {id(curve.points[0]), id(curve.points[-1])}
        for p in curve.points:
            dx, dy = to_dot(p.tps_per_user, p.system_tps)
            cx, cy = canvas.marker_cell(dx, dy)
            is_frontier = id(p) in frontier_point_ids
            style = f"bold {curve.color}" if is_frontier else curve.color
            canvas.put_marker(cx, cy, curve.marker, style)
            if is_frontier or id(p) in endpoints:
                label_style = f"bold {curve.color}" if is_frontier else "dim"
                canvas.put_label(cx + 1, cy, _fmt_conc(p.concurrency), label_style)

    body = canvas.rows_markup()
    gutter = 8

    lines: list[Text] = []

    # Y-axis label named once up top (vertical text is awkward in a terminal).
    yaxis = Text(" " * gutter, style="dim")
    yaxis.append("▲ System throughput — tokens/s (higher is better)", style="dim")
    lines.append(yaxis)

    # Y-axis tick rows (nice round numbers mapped to their pixel row).
    y_ticks = _nice_ticks(y_lo, y_hi, 5)
    row_label: dict[int, str] = {}
    for tv in y_ticks:
        r = round((y_hi - tv) / (y_hi - y_lo) * (rows - 1))
        if 0 <= r < rows:
            row_label[r] = _fmt(tv)
    for r in range(rows):
        gutter_txt = (row_label[r].rjust(gutter - 1) + " ") if r in row_label else " " * gutter
        line = Text(gutter_txt, style="dim")
        line.append("┤" if r in row_label else "│", style="dim")
        line.append_text(Text.from_markup(body[r]))
        lines.append(line)

    # X-axis rule.
    axis = Text(" " * gutter, style="dim")
    axis.append("└" + "─" * cols, style="dim")
    lines.append(axis)

    # X-axis tick labels at nice round numbers.
    buf = list(" " * cols)
    for tv in _nice_ticks(x_lo, x_hi, 6):
        cp = round((tv - x_lo) / (x_hi - x_lo) * (cols - 1))
        label = _fmt(tv)
        start = min(max(cp - len(label) // 2, 0), cols - len(label))
        if all(buf[start + k] == " " for k in range(len(label)) if start + k < cols):
            for k, ch in enumerate(label):
                if start + k < cols:
                    buf[start + k] = ch
    xlabels = Text(" " * (gutter + 1), style="dim")
    xlabels.append("".join(buf))
    lines.append(xlabels)

    xaxis = Text(" " * (gutter + 1), style="dim")
    xaxis.append("▶ Tokens/s per user — interactivity", style="dim")
    lines.append(xaxis)

    if canvas.collisions:
        note = Text(" " * (gutter + 1), style="dim")
        note.append(
            f"* = {canvas.collisions} cell(s) where points from different systems overlap "
            "(see the Pts column for each curve's true count).",
            style="dim",
        )
        lines.append(note)

    return Group(*lines)


def _build_legend(curves: list[Curve], multi: bool) -> Text:
    """One swatch per curve (glyph + colour), plus the frontier key."""
    legend = Text()
    for i, curve in enumerate(curves):
        if i:
            legend.append("    ")
        legend.append(f"{curve.marker} ", style=f"bold {curve.color}")
        legend.append(curve.label)
    if multi:
        legend.append("        ")
        legend.append("bold + labelled points", style="bold")
        legend.append(" = efficient frontier (best trade-offs)", style="dim")
    return legend


def _build_stats_table(curves: list[Curve]) -> Table:
    """Per-curve headline numbers, with the throughput leader flagged."""
    table = Table(show_edge=False, pad_edge=False, box=None, expand=False, padding=(0, 2, 0, 0))
    table.add_column("", no_wrap=True)
    table.add_column("System", no_wrap=True)
    table.add_column("Peak tput\n(tok/s)", justify="right", no_wrap=True)
    table.add_column("Best/user\n(tok/s)", justify="right", no_wrap=True)
    table.add_column("Best TTFT\n(ms)", justify="right", no_wrap=True)
    table.add_column("Max\nconc.", justify="right", no_wrap=True)
    table.add_column("Pts", justify="right", no_wrap=True)

    leader = max(curves, key=lambda c: c.peak_throughput, default=None)
    for curve in curves:
        name = Text()
        name.append(curve.label)
        if curve is leader and len(curves) > 1:
            name.append("  (peak)", style="bold yellow")
        table.add_row(
            Text(curve.marker, style=f"bold {curve.color}"),
            name,
            _fmt(curve.peak_throughput),
            _fmt(curve.best_per_user),
            _fmt(curve.best_ttft),
            str(curve.max_concurrency),
            str(len(curve.points)),
        )
    return table


def _crossover_note(curves: list[Curve]) -> Text | None:
    """Point out when no single system dominates (curves cross) — the interesting case."""
    if len(curves) < 2:
        return None
    tput_leader = max(curves, key=lambda c: c.peak_throughput)
    inter_leader = max(curves, key=lambda c: c.best_per_user)
    note = Text()
    if tput_leader is not inter_leader:
        note.append("  Trade-off: ", style="bold yellow")
        note.append(
            f"{tput_leader.label} wins on peak throughput, but "
            f"{inter_leader.label} delivers the best per-user interactivity — "
            "neither dominates, so the best choice depends on your latency budget."
        )
    else:
        note.append("  ", style="")
        note.append(
            f"{tput_leader.label} leads on both throughput and interactivity "
            "across the measured range.",
            style="bold green",
        )
    return note


def render_pareto(curves: list[Curve], console: Console) -> None:
    """Render the full Pareto view (chart + legend + stats) to *console*."""
    if not curves or not any(c.points for c in curves):
        console.print("[dim]No run metadata found to plot a Pareto view.[/dim]")
        return

    total_points = sum(len(c.points) for c in curves)

    # Size the chart to the terminal, within sensible bounds.
    width = getattr(console, "width", 100) or 100
    cols = max(40, min(90, width - 16))
    rows = 18

    chart = _build_chart(curves, cols, rows)
    legend = _build_legend(curves, multi=len(curves) > 1)
    table = _build_stats_table(curves)
    note = _crossover_note(curves)

    subtitle = f"{len(curves)} system(s) · {total_points} measurement points"
    blocks: list[Any] = [chart, Text(""), legend, Text(""), table]
    if note is not None:
        blocks += [Text(""), note]

    console.print(
        Panel(
            Group(*blocks),
            title="[bold]Pareto Frontier — Throughput vs Interactivity[/bold]",
            subtitle=f"[dim]{subtitle}[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def render_pareto_from_result_dirs(
    result_dirs: list[Path], pareto_root: Path, console: Console
) -> None:
    """Collect curves from *result_dirs* and render the Pareto view to *console*."""
    curves = collect_curves(result_dirs, pareto_root)
    render_pareto(curves, console)
