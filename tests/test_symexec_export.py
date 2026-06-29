"""Step 68 — structured JSON / SARIF export of symexec results.

The symexec engine surfaces its findings through ``src.api.Bug`` (and thus the
repo's existing reporters), which flattens the symexec-specific signals down to a
single ``guard_evidence`` string.  :mod:`src.symexec.export` surfaces those richer
fields directly as a stable JSON object (:func:`result_to_dict`) and as a valid
SARIF 2.1.0 log (:func:`to_sarif`).  These tests pin the shape and invariants of
both, plus the ``--format sarif`` CLI surface.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from src.cli.main import main
from src.symexec import (
    EXPORT_SCHEMA_VERSION,
    SARIF_VERSION,
    analyze_source,
    bug_to_dict,
    result_to_dict,
    to_sarif,
)
from src.symexec.export import _sarif_level, result_to_sarif_run

_BUGGY = (
    "import torch\n"
    "def f():\n"
    "    a = torch.zeros(3)\n"
    "    b = torch.zeros(2)\n"
    "    return a + b\n"
)
_CLEAN = (
    "import torch\n"
    "def f():\n"
    "    a = torch.zeros(2, 3)\n"
    "    b = torch.zeros(3, 4)\n"
    "    return a @ b\n"
)


def _analyze(src, name="m.py"):
    return analyze_source(src, filename=name)


# -- result_to_dict ------------------------------------------------------


def test_result_to_dict_shape():
    d = result_to_dict(_analyze(_BUGGY, "buggy.py"), "buggy.py")
    assert d["schema_version"] == EXPORT_SCHEMA_VERSION
    assert d["file"] == "buggy.py"
    assert set(d) >= {
        "schema_version", "file", "functions_analyzed", "ran_main",
        "fingerprint", "bugs", "abstain_total", "abstain_coverage",
    }
    assert len(d["bugs"]) == 1
    assert d["bugs"][0]["kind"] == "broadcast_mismatch"


def test_result_to_dict_fingerprint_matches_result():
    sr = _analyze(_BUGGY, "buggy.py")
    assert result_to_dict(sr, "buggy.py")["fingerprint"] == sr.fingerprint()


def test_result_to_dict_clean_has_no_bugs():
    d = result_to_dict(_analyze(_CLEAN, "clean.py"), "clean.py")
    assert d["bugs"] == []


def test_bug_to_dict_is_superset_of_to_dict():
    sr = _analyze(_BUGGY, "buggy.py")
    bug = sr.bugs[0]
    full = bug_to_dict(bug)
    for k, v in bug.to_dict().items():
        assert full[k] == v


# -- SARIF structural validity ------------------------------------------


def test_to_sarif_top_level():
    log = to_sarif([("buggy.py", _analyze(_BUGGY, "buggy.py"))])
    assert log["version"] == SARIF_VERSION
    assert "$schema" in log
    assert isinstance(log["runs"], list) and len(log["runs"]) == 1


def test_sarif_one_run_per_file():
    items = [
        ("a.py", _analyze(_BUGGY, "a.py")),
        ("b.py", _analyze(_CLEAN, "b.py")),
    ]
    log = to_sarif(items)
    assert len(log["runs"]) == 2


def test_sarif_run_has_driver_and_rules():
    run = result_to_sarif_run(_analyze(_BUGGY, "buggy.py"), "buggy.py")
    driver = run["tool"]["driver"]
    assert driver["name"] == "TensorGuard-Symexec"
    rule_ids = [r["id"] for r in driver["rules"]]
    assert "broadcast_mismatch" in rule_ids


def test_sarif_result_object_fields():
    sr = _analyze(_BUGGY, "buggy.py")
    run = result_to_sarif_run(sr, "buggy.py")
    res = run["results"][0]
    bug = sr.bugs[0]
    assert res["ruleId"] == "broadcast_mismatch"
    assert res["level"] == "error"
    assert res["message"]["text"]
    region = res["locations"][0]["physicalLocation"]["region"]
    # SARIF columns are 1-based; symexec col is 0-based.
    assert region["startColumn"] == bug.col + 1
    assert region["startLine"] == max(bug.line, 1)
    # ruleIndex cross-references the driver rules list.
    assert run["tool"]["driver"]["rules"][res["ruleIndex"]]["id"] == res["ruleId"]


def test_sarif_result_properties_carry_symexec_fields():
    run = result_to_sarif_run(_analyze(_BUGGY, "buggy.py"), "buggy.py")
    props = run["results"][0]["properties"]
    assert props["kind"] == "broadcast_mismatch"
    assert 0.0 < props["confidence"] <= 0.99


def test_sarif_run_properties_carry_fingerprint_and_abstain():
    sr = _analyze(_BUGGY, "buggy.py")
    run = result_to_sarif_run(sr, "buggy.py")
    props = run["properties"]
    assert props["fingerprint"] == sr.fingerprint()
    assert props["abstain_total"] == sr.abstentions.total
    assert "abstain_coverage" in props


def test_sarif_clean_file_has_empty_results():
    run = result_to_sarif_run(_analyze(_CLEAN, "clean.py"), "clean.py")
    assert run["results"] == []
    assert run["tool"]["driver"]["rules"] == []


def test_sarif_is_deterministic():
    a = json.dumps(to_sarif([("buggy.py", _analyze(_BUGGY, "buggy.py"))]), sort_keys=True)
    b = json.dumps(to_sarif([("buggy.py", _analyze(_BUGGY, "buggy.py"))]), sort_keys=True)
    assert a == b


def test_sarif_level_mapping():
    assert _sarif_level("error") == "error"
    assert _sarif_level("warning") == "warning"
    assert _sarif_level("note") == "note"
    assert _sarif_level("") == "warning"


# -- SymResult convenience methods --------------------------------------


def test_symresult_to_dict_and_to_sarif():
    sr = _analyze(_BUGGY, "buggy.py")
    assert sr.to_dict("buggy.py")["fingerprint"] == sr.fingerprint()
    assert sr.to_sarif("buggy.py")["version"] == SARIF_VERSION


# -- CLI --format sarif -------------------------------------------------


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


def test_cli_sarif_format(tmp_path):
    p = tmp_path / "buggy.py"
    p.write_text(_BUGGY, encoding="utf-8")
    code, out = _run(["symexec", str(p), "--format", "sarif"])
    assert code == 1
    log = json.loads(out)
    assert log["version"] == SARIF_VERSION
    assert log["runs"][0]["results"][0]["ruleId"] == "broadcast_mismatch"


def test_cli_sarif_clean_exits_zero(tmp_path):
    p = tmp_path / "clean.py"
    p.write_text(_CLEAN, encoding="utf-8")
    code, out = _run(["symexec", str(p), "--format", "sarif"])
    assert code == 0
    log = json.loads(out)
    assert log["runs"][0]["results"] == []
