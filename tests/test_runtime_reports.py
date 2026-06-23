from scripts.collect_llm_runtime import process_stats
from scripts.render_runtime_report import extract_rss_mib, render_html


def test_extract_rss_mib_from_runtime_report() -> None:
    report = {
        "snapshots": [
            {"process": {"rss_kib": 1024}},
            {"process": {"rss_kib": 2048}},
        ]
    }

    assert extract_rss_mib(report) == [1.0, 2.0]


def test_render_runtime_report_outputs_html_summary() -> None:
    html = render_html(
        {
            "base_url": "http://127.0.0.1:8080",
            "pid": 123,
            "snapshots": [
                {
                    "timestamp_unix": 1.0,
                    "process": {"rss_kib": 1024, "cpu_percent": 3.5},
                    "llama_metrics_raw": "metric 1",
                    "llama_slots": [],
                }
            ],
        }
    )

    assert "Scratchpad LLM Runtime Report" in html
    assert "rss_min_mib" in html
    assert "http://127.0.0.1:8080" in html


def test_process_stats_returns_none_for_missing_pid() -> None:
    assert process_stats(None) is None
