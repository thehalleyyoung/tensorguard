import json

from evaluation.sarif_trends import build_dashboard
from src.cli.main import ReftypeCliApp
from src.sarif_trend_dashboard import (
    SarifSnapshot,
    build_trend_dashboard,
    extract_alerts,
    render_markdown,
)


def _result(identity, *, rule="shape-incompatible", uri="model.py", line=10, msg="bug"):
    return {
        "ruleId": rule,
        "level": "error",
        "message": {"text": msg},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": uri},
                    "region": {"startLine": line},
                }
            }
        ],
        "partialFingerprints": {"tensorguard/v1": identity},
    }


def _sarif(*results):
    return {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "TensorGuard", "rules": []}},
                "results": list(results),
            }
        ],
    }


def test_extract_alerts_flattens_all_runs_and_keeps_identity():
    sarif = {
        "runs": [
            {"results": [_result("a", uri="a.py")]},
            {"results": [_result("b", uri="b.py", rule="dtype-mismatch")]},
        ]
    }
    alerts = extract_alerts(sarif)
    assert [a["identity"] for a in alerts] == [
        "fp:tensorguard/v1:a",
        "fp:tensorguard/v1:b",
    ]
    assert {a["uri"] for a in alerts} == {"a.py", "b.py"}


def test_duplicate_fingerprints_are_counted_not_collapsed():
    dashboard = build_trend_dashboard(
        [
            SarifSnapshot("r1", _sarif(_result("same", line=10), _result("same", line=20))),
            SarifSnapshot("r2", _sarif(_result("same", line=20))),
            SarifSnapshot("r3", _sarif(_result("same", line=10), _result("same", line=20))),
        ]
    )
    rows = dashboard["releases"]
    assert [r["open_total"] for r in rows] == [2, 1, 2]
    assert [r["closed"] for r in rows] == [0, 1, 0]
    assert [r["carried"] for r in rows] == [0, 1, 1]
    assert [r["opened"] for r in rows] == [2, 0, 1]
    assert [r["recurred"] for r in rows] == [0, 0, 1]
    assert dashboard["current_open_alerts"][0]["count"] == 2


def test_location_fallback_tracks_alert_when_fingerprints_absent():
    result = _result("ignored", uri="fallback.py", line=7)
    del result["partialFingerprints"]
    dashboard = build_trend_dashboard([SarifSnapshot("r1", _sarif(result))])
    assert dashboard["releases"][0]["open_total"] == 1
    assert dashboard["current_open_alerts"][0]["identity"].startswith("loc:")


def test_duplicate_release_names_are_rejected():
    try:
        build_trend_dashboard(
            [SarifSnapshot("r1", _sarif()), SarifSnapshot("r1", _sarif())]
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate release names should fail")


def test_markdown_renders_delta_table():
    dashboard = build_trend_dashboard(
        [SarifSnapshot("r1", _sarif(_result("a", msg="shape mismatch")))]
    )
    md = render_markdown(dashboard)
    assert "Code Scanning Trend Dashboard" in md
    assert "| r1 | 1 | 1 | 0 | 0 | 0 | 1 | shape-incompatible:1 |" in md
    assert "model.py:10" in md


def test_cli_command_writes_json_and_markdown(tmp_path):
    s1 = tmp_path / "r1.sarif"
    s1.write_text(json.dumps(_sarif(_result("a"))), encoding="utf-8")
    out_json = tmp_path / "trend.json"
    out_md = tmp_path / "trend.md"
    rc = ReftypeCliApp().run(
        [
            "sarif-trends",
            f"r1={s1}",
            "--output",
            str(out_json),
            "--markdown",
            str(out_md),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary"]["current_open"] == 1
    assert "Currently open alerts" in out_md.read_text(encoding="utf-8")


def test_real_code_artifact_tracks_open_closed_and_recurrence():
    dashboard = build_dashboard()
    rows = dashboard["releases"]
    assert [r["release"] for r in rows] == ["v0.1.0", "v0.1.1", "v0.1.2"]
    assert [r["open_total"] for r in rows] == [1, 0, 2]
    assert [r["closed"] for r in rows] == [0, 1, 0]
    assert [r["recurred"] for r in rows] == [0, 0, 1]
    assert dashboard["summary"]["current_open"] == 2
