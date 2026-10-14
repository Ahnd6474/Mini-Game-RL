"""Dependency-free SVG learning-curve writer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _panel(
    records: Sequence[dict[str, object]],
    series: Sequence[tuple[str, str]],
    title: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    fixed_range: tuple[float, float] | None = None,
) -> str:
    available: list[tuple[str, str, list[tuple[float, float]]]] = []
    for key, label in series:
        points = []
        for record in records:
            value = record.get(key)
            iteration = record.get("iteration")
            if isinstance(value, (int, float)) and isinstance(iteration, (int, float)):
                if math.isfinite(float(value)):
                    points.append((float(iteration), float(value)))
        if points:
            available.append((key, label, points))
    if not available:
        return ""

    all_points = [point for _, _, points in available for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    if fixed_range is None:
        y_min = min(point[1] for point in all_points)
        y_max = max(point[1] for point in all_points)
        padding = max(1e-6, (y_max - y_min) * 0.12)
        y_min -= padding
        y_max += padding
    else:
        y_min, y_max = fixed_range
    x_span = max(1.0, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    left, top = x0 + 58, y0 + 34
    plot_w, plot_h = width - 78, height - 72
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="10" fill="#ffffff" stroke="#d1d5db"/>',
        f'<text x="{x0 + 16}" y="{y0 + 23}" class="title">{title}</text>',
    ]
    for tick in range(5):
        ratio = tick / 4
        y = top + plot_h * ratio
        value = y_max - ratio * y_span
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" class="grid"/>')
        parts.append(f'<text x="{left - 7}" y="{y + 4}" text-anchor="end" class="axis">{value:.3g}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axisline"/>')
    for tick in range(5):
        ratio = tick / 4
        x = left + plot_w * ratio
        value = x_min + ratio * x_span
        parts.append(f'<text x="{x}" y="{top + plot_h + 18}" text-anchor="middle" class="axis">{value:.0f}</text>')

    for index, (_, label, points) in enumerate(available):
        color = COLORS[index % len(COLORS)]
        coords = []
        for x_value, y_value in points:
            x = left + (x_value - x_min) / x_span * plot_w
            y = top + (y_max - y_value) / y_span * plot_h
            coords.append(f"{x:.2f},{y:.2f}")
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        for x_value, y_value in points:
            x = left + (x_value - x_min) / x_span * plot_w
            y = top + (y_max - y_value) / y_span * plot_h
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="{color}"/>')
        legend_x = left + index * 155
        parts.append(f'<line x1="{legend_x}" y1="{y0 + height - 13}" x2="{legend_x + 20}" y2="{y0 + height - 13}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 25}" y="{y0 + height - 9}" class="legend">{label}</text>')
    return "".join(parts)


def write_training_curves(records: Sequence[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        _panel(records, (("dvn_total", "train"), ("dvn_validation_total", "validation")), "DVN total loss", 20, 55, 580, 310),
        _panel(records, (("dpn_policy", "policy"), ("dpn_total", "total"), ("dpn_entropy", "entropy")), "DPN objectives", 620, 55, 580, 310),
        _panel(records, (("dvn_validation_cell_accuracy", "cell accuracy"), ("dvn_validation_decisive_accuracy", "winner accuracy"), ("dvn_validation_value_correlation", "value corr.")), "DVN held-out metrics", 20, 385, 580, 310, (0.0, 1.0)),
        _panel(records, (("tree_score", "match score"),), "DPN vs fixed tree benchmark", 620, 385, 580, 310, (0.0, 1.0)),
        _panel(records, (("dpn_approx_kl", "approx. KL"), ("dpn_clip_fraction", "clip fraction")), "PPO trust region", 20, 715, 580, 310),
        _panel(records, (("positions_per_second", "positions/s"),), "Training throughput", 620, 715, 580, 310),
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1220" height="1050" viewBox="0 0 1220 1050">
<style>.title{{font:600 15px system-ui;fill:#111827}}.axis{{font:11px system-ui;fill:#6b7280}}.legend{{font:11px system-ui;fill:#374151}}.grid{{stroke:#e5e7eb;stroke-width:1}}.axisline{{stroke:#9ca3af;stroke-width:1}}</style>
<rect width="1220" height="1050" fill="#f3f4f6"/>
<text x="20" y="32" style="font:700 22px system-ui;fill:#111827">Reversi DVN / DPN learning curves</text>
{''.join(panels)}</svg>'''
    path.write_text(svg, encoding="utf-8")
    return path
