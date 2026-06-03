from __future__ import annotations

import json
from pathlib import Path

from src.cli.main import ReftypeCliApp
from src.local_usage_metrics import summarize_files, summarize_records


def test_summarizes_verdicts_abstentions_and_categories() -> None:
    summary = summarize_records(
        [
            {
                "file": "/repo/models/a.py",
                "verdict": "SAFE",
                "bugs": [],
            },
            {
                "file": "/repo/models/b.py",
                "verdict": "UNSAFE",
                "bugs": [{"category": "shape_incompatible"}],
            },
            {
                "file": "/repo/models/c.py",
                "verdict": "UNKNOWN",
                "abstained": True,
                "unknown_reasons": ["unsupported operator(s): torch.unique, custom_op"],
            },
        ],
        limit=2,
    )

    assert summary.reports == 3
    assert summary.analyzed_files == 3
    assert summary.verdicts == {"SAFE": 1, "UNSAFE": 1, "UNKNOWN": 1}
    assert summary.abstentions == 1
    assert summary.bug_categories == {"shape_incompatible": 1}
    assert summary.top_unsupported_ops == (("torch.unique", 1), ("custom_op", 1))


def test_json_summary_is_telemetry_free_and_path_redacted() -> None:
    summary = summarize_records([{"file": "/secret/model.py", "status": "SAFE"}])
    payload = summary.to_json_dict()

    assert payload["privacy"]["telemetry_free"] is True
    assert payload["privacy"]["source_code_included"] is False
    assert "/secret/model.py" not in json.dumps(payload)
    assert len(payload["redacted_files"][0]) == 16


def test_nested_report_files_are_loaded(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "file": "one.py",
                        "verdict": "SAFE",
                        "top_unsupported_ops": [{"op": "torch.linalg.eig", "count": 2}],
                    },
                    {"file": "two.py", "status": "UNSAFE", "bugs": [{"ruleId": "dtype"}]},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_files([report])

    assert summary.reports == 2
    assert summary.verdicts == {"SAFE": 1, "UNSAFE": 1}
    assert summary.top_unsupported_ops == (("torch.linalg.eig", 2),)
    assert summary.bug_categories == {"dtype": 1}


def test_cli_writes_markdown_summary(tmp_path: Path) -> None:
    report = tmp_path / "verify.json"
    out = tmp_path / "metrics.md"
    report.write_text(
        json.dumps(
            {
                "file": "model.py",
                "verdict": "UNKNOWN",
                "unknown_reasons": ["heuristic-tagged operator(s) used: einsum"],
            }
        ),
        encoding="utf-8",
    )

    rc = ReftypeCliApp().run(
        ["usage-metrics", str(report), "--format", "markdown", "-o", str(out)]
    )

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "Telemetry: **off**" in text
    assert "Files analyzed: 1" in text
    assert "`einsum`: 1" in text
