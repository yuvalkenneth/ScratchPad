"""Collect lightweight runtime snapshots from a local llama.cpp server."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def read_url(url: str, *, timeout: float = 2.0) -> str | None:
    """Read a URL as text, returning None when the endpoint is unavailable."""
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, URLError):
        return None


def read_json_url(url: str, *, timeout: float = 2.0) -> Any | None:
    """Read a URL and decode JSON, returning None for missing/invalid data."""
    raw = read_url(url, timeout=timeout)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def process_stats(pid: int | None) -> dict[str, Any] | None:
    """Read process memory and CPU stats through `ps` for an optional PID."""
    if pid is None:
        return None
    completed = subprocess.run(
        ["ps", "-o", "pid=,rss=,vsz=,%cpu=,%mem=,command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    parts = completed.stdout.strip().split(None, 5)
    if len(parts) < 6:
        return None
    return {
        "pid": int(parts[0]),
        "rss_kib": int(parts[1]),
        "vsz_kib": int(parts[2]),
        "cpu_percent": float(parts[3]),
        "mem_percent": float(parts[4]),
        "command": parts[5],
    }


def collect_snapshot(
    *,
    base_url: str,
    pid: int | None,
) -> dict[str, Any]:
    """Collect one endpoint/process snapshot."""
    normalized_base_url = base_url.rstrip("/")
    return {
        "timestamp_unix": time.time(),
        "base_url": normalized_base_url,
        "process": process_stats(pid),
        "llama_metrics_raw": read_url(f"{normalized_base_url}/metrics"),
        "llama_slots": read_json_url(f"{normalized_base_url}/slots"),
    }


def collect_series(
    *,
    base_url: str,
    pid: int | None,
    interval_seconds: float,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    """Collect snapshots over a duration at a fixed interval."""
    snapshots: list[dict[str, Any]] = []
    deadline = time.monotonic() + duration_seconds
    while True:
        snapshots.append(collect_snapshot(base_url=base_url, pid=pid))
        if time.monotonic() >= deadline:
            break
        time.sleep(interval_seconds)
    return snapshots


def run(argv: list[str] | None = None) -> int:
    """Run the runtime collector CLI."""
    parser = argparse.ArgumentParser(
        description="Collect local LLM runtime memory/process and llama.cpp monitoring snapshots."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--pid", type=int, help="Optional local server process id for ps memory stats.")
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.duration_seconds > 0:
        snapshots = collect_series(
            base_url=args.base_url,
            pid=args.pid,
            interval_seconds=args.interval_seconds,
            duration_seconds=args.duration_seconds,
        )
    else:
        snapshots = [collect_snapshot(base_url=args.base_url, pid=args.pid)]

    report = {
        "type": "scratchpad_llm_runtime_report",
        "base_url": args.base_url.rstrip("/"),
        "pid": args.pid,
        "interval_seconds": args.interval_seconds,
        "duration_seconds": args.duration_seconds,
        "snapshots": snapshots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote runtime report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
