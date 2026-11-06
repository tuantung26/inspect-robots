# ruff: noqa: E501
"""Render evaluation metrics as a compact, dependency-free HTML dashboard.

Run with:  uv run python examples/visualize_eval_log.py logs/run.json
"""

from __future__ import annotations

import argparse
import html
import json
import math
import webbrowser
from pathlib import Path
from typing import Any


def _value(value: Any) -> float | None:
    """Return a finite numeric value, or None when a score is absent or invalid."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _metric_card(label: str, value: str, detail: str) -> str:
    """Render one escaped summary card."""
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-detail">{html.escape(detail)}</div>'
        "</div>"
    )


def _bar_chart(
    samples: list[dict[str, Any]],
    *,
    key: str,
    title: str,
    color: str,
    precision: int,
    suffix: str = "",
    percent: bool = False,
) -> str:
    """Render per-scene reduced scores as a labeled horizontal SVG bar chart."""
    values = [_value(sample.get("reduced", {}).get(key)) for sample in samples]
    valid_values = [value for value in values if value is not None]
    scale_max = 1.0 if percent else max(valid_values, default=1.0)
    if scale_max <= 0:
        scale_max = 1.0

    width = 560
    height = 68 + 42 * max(len(samples), 1)
    left = 112
    plot_width = 330
    top = 40
    axis_bottom = top + 42 * len(samples)
    ticks = 4
    svg = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        f'<text class="chart-title" x="0" y="20">{html.escape(title)}</text>',
    ]

    for tick in range(ticks + 1):
        x = left + plot_width * tick / ticks
        tick_value = scale_max * tick / ticks
        tick_text = f"{tick_value * 100:.0f}%" if percent else f"{tick_value:.{precision}f}"
        svg.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{axis_bottom:.1f}"/>'
        )
        svg.append(
            f'<text class="axis-label" x="{x:.1f}" y="{axis_bottom + 20:.1f}" '
            f'text-anchor="middle">{tick_text}</text>'
        )

    for index, (sample, value) in enumerate(zip(samples, values, strict=True)):
        y = top + index * 42 + 5
        scene_id = html.escape(str(sample.get("scene_id", f"scene-{index}")))
        svg.append(
            f'<text class="scene-label" x="{left - 10}" y="{y + 13}" '
            f'text-anchor="end">{scene_id}</text>'
        )
        if value is None:
            svg.append(f'<text class="axis-label" x="{left}" y="{y + 13}">not scored</text>')
            continue
        clamped = min(max(value, 0.0), scale_max)
        bar_width = plot_width * clamped / scale_max
        display_value = f"{value * 100:.0f}%" if percent else f"{value:.{precision}f}{suffix}"
        svg.append(
            f'<rect class="bar" x="{left}" y="{y}" width="{bar_width:.1f}" '
            f'height="22" rx="6" fill="{color}"/>'
        )
        value_x = min(left + bar_width + 8, width - 58)
        svg.append(
            f'<text class="bar-label" x="{value_x:.1f}" y="{y + 15}">'
            f"{html.escape(display_value)}</text>"
        )

    svg.append("</svg>")
    return "".join(svg)


def render_report(data: dict[str, Any], source_name: str) -> str:
    """Build a single-page dashboard from a serialized Inspect Robots eval log."""
    eval_spec = data.get("eval", {})
    results = data.get("results", {})
    metrics = results.get("metrics", {})
    samples = data.get("samples", [])

    success = _value(metrics.get("success_at_end"))
    episode_length = _value(metrics.get("episode_length"))
    distance = _value(metrics.get("min_distance_to_goal"))
    success_text = "n/a" if success is None else f"{success * 100:.0f}%"
    length_text = "n/a" if episode_length is None else f"{episode_length:.1f}"
    distance_text = "n/a" if distance is None else f"{distance:.4f}"
    status = str(data.get("status", "unknown"))
    task = str(eval_spec.get("task", "Evaluation"))
    policy = str(eval_spec.get("policy", "unknown"))
    embodiment = str(eval_spec.get("embodiment", "unknown"))
    scene_count = results.get("total_scenes", len(samples))
    trial_count = results.get("total_trials", 0)
    errors = results.get("errored_trials", 0)

    cards = "".join(
        (
            _metric_card("Success rate", success_text, "higher is better"),
            _metric_card("Steps per scene", length_text, "average across scenes"),
            _metric_card("Closest distance", distance_text, "lower is better"),
            _metric_card("Trials", str(trial_count), f"{scene_count} scenes · {errors} errors"),
        )
    )
    status_class = "good" if status == "success" else "other"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(task)} evaluation</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Segoe UI", sans-serif; color: #172033; background: #f3f6fb; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 32px 18px; }}
main {{ max-width: 1040px; margin: auto; }}
header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; margin-bottom: 22px; }}
h1 {{ margin: 0 0 7px; font-size: clamp(24px, 4vw, 34px); letter-spacing: -.03em; }}
.subtitle {{ margin: 0; color: #647087; }}
.badge {{ border-radius: 999px; padding: 7px 12px; font-size: 13px; font-weight: 700; }}
.badge.good {{ color: #087443; background: #dcf7e9; }}
.badge.other {{ color: #8a4c07; background: #fff0cf; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
.metric-card, .panel, footer {{ background: white; border: 1px solid #e3e9f2; border-radius: 16px; box-shadow: 0 5px 18px #2339580a; }}
.metric-card {{ padding: 17px; }}
.metric-label {{ color: #657187; font-size: 13px; }}
.metric-value {{ margin: 7px 0 3px; font-size: 27px; font-weight: 750; }}
.metric-detail {{ color: #8390a3; font-size: 12px; }}
.charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.panel {{ padding: 18px 16px 12px; min-width: 0; }}
.panel h2 {{ margin: 0 0 2px; font-size: 16px; }}
.panel p {{ margin: 0; color: #758198; font-size: 12px; }}
.chart {{ display: block; width: 100%; height: auto; overflow: visible; }}
.chart-title {{ font-size: 12px; font-weight: 700; fill: #25324a; }}
.grid {{ stroke: #e8edf4; stroke-width: 1; }}
.axis-label {{ font-size: 10px; fill: #8a95a8; }}
.scene-label {{ font-size: 11px; fill: #536078; }}
.bar-label {{ font-size: 11px; font-weight: 650; fill: #25324a; }}
.bar {{ opacity: .9; }}
.legend {{ color: #8290a4; font-size: 12px; margin-top: 4px; }}
footer {{ margin-top: 14px; padding: 14px 17px; color: #647087; font-size: 12px; overflow-wrap: anywhere; }}
footer strong {{ color: #34425a; }}
@media (max-width: 700px) {{ body {{ padding: 20px 12px; }} .cards {{ grid-template-columns: repeat(2, 1fr); }} .charts {{ grid-template-columns: 1fr; }} header {{ align-items: center; }} }}
</style>
</head>
<body>
<main>
<header>
  <div><h1>{html.escape(task)}</h1><p class="subtitle">{html.escape(policy)} on {html.escape(embodiment)}</p></div>
  <span class="badge {status_class}">{html.escape(status)}</span>
</header>
<section class="cards">{cards}</section>
<section class="charts">
  <article class="panel"><h2>Per-scene performance</h2><p>Reduced scores across the configured epochs</p>
    {_bar_chart(samples, key="episode_length", title="Steps · lower is better", color="#6d7ef2", precision=1)}
    <div class="legend">Mean steps required for each scene.</div>
  </article>
  <article class="panel"><h2>Goal proximity</h2><p>Distance recorded during each trial</p>
    {_bar_chart(samples, key="min_distance_to_goal", title="Closest distance · lower is better", color="#21a67a", precision=4)}
    <div class="legend">The success radius in CubePick is 0.05.</div>
  </article>
  <article class="panel"><h2>Success by scene</h2><p>Fraction of trials ending with the environment's success signal</p>
    {_bar_chart(samples, key="success_at_end", title="Success rate", color="#f0a43b", precision=0, percent=True)}
    <div class="legend">A policy stopping by itself does not count as success.</div>
  </article>
  <article class="panel"><h2>How to read this</h2>
    <p>Each bar summarizes one scene after reducing its epochs. The summary cards show run-wide metrics. This CubePick case uses a scripted oracle, so it checks the evaluation pipeline rather than measuring a learned model.</p>
    <p class="legend">Source log: {html.escape(source_name)}</p>
  </article>
</section>
<footer><strong>Run details:</strong> seed {html.escape(str(eval_spec.get("seed", "n/a")))} ·
  {html.escape(str(scene_count))} scenes · {html.escape(str(trial_count))} trials ·
  maximum {html.escape(str(eval_spec.get("max_steps", "n/a")))} steps per trial.
</footer>
</main>
</body>
</html>
"""


def main() -> None:
    """Read an eval log and write a standalone HTML chart dashboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="path to an Inspect Robots EvalLog JSON file")
    parser.add_argument("-o", "--output", type=Path, help="HTML output path")
    parser.add_argument("--open", action="store_true", help="open the report in the browser")
    args = parser.parse_args()

    try:
        data = json.loads(args.log.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read evaluation log {args.log}: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        parser.error(f"{args.log} does not look like an Inspect Robots evaluation log")

    output = args.output or args.log.with_name(f"{args.log.stem}.charts.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(data, args.log.name), encoding="utf-8")
    print(f"wrote {output}")
    if args.open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
