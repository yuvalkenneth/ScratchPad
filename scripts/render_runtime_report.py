from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def point_path(values: list[float], *, width: int = 900, height: int = 260, padding: int = 24) -> str:
    if not values:
        return ""
    if len(values) == 1:
        x_values = [width // 2]
    else:
        x_values = [
            padding + index * (width - 2 * padding) / (len(values) - 1)
            for index in range(len(values))
        ]
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum or 1.0
    y_values = [
        height - padding - ((value - minimum) / span) * (height - 2 * padding)
        for value in values
    ]
    commands = [f"M {x_values[0]:.2f} {y_values[0]:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in zip(x_values[1:], y_values[1:]))
    return " ".join(commands)


def extract_rss_mib(report: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for snapshot in report.get("snapshots", []):
        process = snapshot.get("process") or {}
        rss_kib = process.get("rss_kib")
        if rss_kib is not None:
            values.append(round(float(rss_kib) / 1024, 3))
    return values


def latest_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    snapshots = report.get("snapshots", [])
    if not snapshots:
        return {}
    return snapshots[-1]


def endpoint_summary(report: dict[str, Any]) -> dict[str, Any]:
    snapshots = report.get("snapshots", [])
    return {
        "metrics_snapshots": sum(1 for snapshot in snapshots if snapshot.get("llama_metrics_raw")),
        "slots_snapshots": sum(1 for snapshot in snapshots if snapshot.get("llama_slots") is not None),
    }


def render_rss_section(rss_values: list[float]) -> str:
    if not rss_values:
        return """
  <section class="card warning">
    <h2>Process RSS MiB</h2>
    <p>No process RSS samples were collected. Pass <code>--pid</code> to
    <code>collect_llm_runtime.py</code>, or update the launcher to write a PID
    file. Without a PID this report can still show llama.cpp slot status, but
    cannot plot OS process memory.</p>
  </section>
"""
    path = point_path(rss_values)
    return f"""
  <section class="card">
    <h2>Process RSS MiB</h2>
    <svg viewBox="0 0 900 260" role="img" aria-label="RSS memory over snapshots">
      <path d="{path}" fill="none" stroke="#2563eb" stroke-width="3" />
    </svg>
  </section>
"""


def render_slots_section(snapshot: dict[str, Any]) -> str:
    slots = snapshot.get("llama_slots")
    if slots is None:
        return """
  <section class="card warning">
    <h2>Latest Slots</h2>
    <p>No <code>/slots</code> payload was collected. The server may have been
    down or the endpoint may not be available.</p>
  </section>
"""
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(slot.get('id', '')))}</td>"
        f"<td>{html.escape(str(slot.get('n_ctx', '')))}</td>"
        f"<td>{html.escape(str(slot.get('is_processing', '')))}</td>"
        f"<td>{html.escape(str(slot.get('speculative', '')))}</td>"
        "</tr>"
        for slot in slots
        if isinstance(slot, dict)
    )
    return f"""
  <section class="card">
    <h2>Latest Slots</h2>
    <table>
      <thead>
        <tr><th>slot</th><th>n_ctx</th><th>processing</th><th>speculative</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
"""


def render_html(report: dict[str, Any]) -> str:
    rss_values = extract_rss_mib(report)
    endpoints = endpoint_summary(report)
    latest = latest_snapshot(report)
    summary = {
        "snapshots": len(report.get("snapshots", [])),
        "base_url": report.get("base_url"),
        "pid": report.get("pid"),
        "rss_min_mib": min(rss_values) if rss_values else None,
        "rss_max_mib": max(rss_values) if rss_values else None,
        **endpoints,
    }
    rows = "\n".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{snapshot.get('timestamp_unix', ''):.3f}</td>"
        f"<td>{(snapshot.get('process') or {}).get('rss_kib', '')}</td>"
        f"<td>{html.escape(str((snapshot.get('process') or {}).get('cpu_percent', '')))}</td>"
        f"<td>{'yes' if snapshot.get('llama_metrics_raw') else 'no'}</td>"
        f"<td>{'yes' if snapshot.get('llama_slots') is not None else 'no'}</td>"
        "</tr>"
        for index, snapshot in enumerate(report.get("snapshots", []))
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Scratchpad LLM Runtime Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #18212f; background: #f8fafc; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .card {{ background: #fff; border: 1px solid #dbe3ed; border-radius: 12px; padding: 18px; margin: 18px 0; }}
    .warning {{ border-color: #f2c36b; background: #fff8e8; }}
    pre {{ background: #f4f6f8; padding: 16px; border-radius: 8px; overflow: auto; }}
    svg {{ width: 100%; max-width: 960px; border: 1px solid #d6dde6; border-radius: 8px; background: #fbfcfd; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 24px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5eaf0; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Scratchpad LLM Runtime Report</h1>
  <section class="card">
    <h2>Summary</h2>
    <pre>{html.escape(json.dumps(summary, indent=2))}</pre>
  </section>
  {render_rss_section(rss_values)}
  <section class="card">
    <h2>Endpoint Availability</h2>
    <p><code>/slots</code> snapshots: {endpoints["slots_snapshots"]}</p>
    <p><code>/metrics</code> snapshots: {endpoints["metrics_snapshots"]}</p>
    <p>If <code>/metrics</code> is zero, start llama.cpp with metrics enabled.</p>
  </section>
  {render_slots_section(latest)}
  <section class="card">
    <h2>Raw Snapshot Table</h2>
    <table>
      <thead>
        <tr><th>#</th><th>timestamp</th><th>rss_kib</th><th>cpu_percent</th><th>/metrics</th><th>/slots</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
</body>
</html>
"""


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a Scratchpad LLM runtime JSON report as HTML.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(report), encoding="utf-8")
    print(f"Wrote runtime HTML report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
