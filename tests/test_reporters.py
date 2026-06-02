"""Step 73 — JSON / JUnit-XML / GitHub reporters, proven on real findings."""

import json
import xml.etree.ElementTree as ET

from src.github_action import run_action
from src.reporters import (
    SCHEMA_VERSION,
    build_json,
    render,
    to_github_annotations,
    to_json,
    to_junit_xml,
    write_report,
)

# Real shape-mismatch model (fc1 -> 20, fc2 expects 30).
_BAD = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(10, 20)\n"
    "        self.fc2 = nn.Linear(30, 5)\n"
    "    def forward(self, x):\n"
    "        return self.fc2(self.fc1(x))\n"
)
_GOOD = (
    "import torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.conv1 = nn.Conv2d(3, 8, 3)\n"
    "        self.conv2 = nn.Conv2d(8, 16, 3)\n"
    "    def forward(self, x):\n"
    "        return self.conv2(self.conv1(x))\n"
)


def _results(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(_BAD, encoding="utf-8")
    res = run_action([str(bad)], input_shapes={"x": ("batch", 10)})
    assert res.total_issues == 1
    return res.results_by_file


def test_json_schema_and_findings(tmp_path):
    payload = build_json(_results(tmp_path))
    assert payload["schema"] == "tensorguard-report"
    assert payload["version"] == SCHEMA_VERSION
    assert payload["summary"]["files_checked"] == 1
    assert payload["summary"]["files_with_issues"] == 1
    assert payload["summary"]["total_findings"] == 1
    finding = payload["findings"][0]
    assert finding["line"] == 8
    assert finding["level"] == "error"
    assert "expects input dimension 30" in finding["message"]


def test_to_json_is_valid_and_sorted(tmp_path):
    text = to_json(_results(tmp_path))
    parsed = json.loads(text)  # must be valid JSON
    assert parsed["findings"][0]["file"].endswith("bad.py")


def test_junit_is_wellformed_with_failure(tmp_path):
    xml = to_junit_xml(_results(tmp_path))
    root = ET.fromstring(xml)  # must parse
    assert root.tag == "testsuites"
    suite = root.find("testsuite")
    assert suite.get("name") == "tensorguard"
    assert suite.get("failures") == "1"
    case = suite.find("testcase")
    assert case.get("name").endswith("bad.py")
    failure = case.find("failure")
    assert failure is not None
    assert "expects input dimension 30" in failure.get("message")


def test_junit_clean_file_has_no_failure(tmp_path):
    good = tmp_path / "ok.py"
    good.write_text(_GOOD, encoding="utf-8")
    res = run_action([str(good)])
    xml = to_junit_xml(res.results_by_file)
    root = ET.fromstring(xml)
    suite = root.find("testsuite")
    assert suite.get("failures") == "0"
    case = suite.find("testcase")
    assert case is not None
    assert case.find("failure") is None


def test_github_annotations_match_action(tmp_path):
    results = _results(tmp_path)
    text = to_github_annotations(results)
    assert text.startswith("::error ")
    assert "bad.py" in text
    assert "line=8" in text


def test_render_dispatch_and_unknown(tmp_path):
    results = _results(tmp_path)
    assert render(results, "json").lstrip().startswith("{")
    assert render(results, "junit").startswith("<?xml")
    assert render(results, "github").startswith("::error")
    try:
        render(results, "csv")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown report format" in str(e)


def test_write_report_files(tmp_path):
    results = _results(tmp_path)
    jp = tmp_path / "report.json"
    xp = tmp_path / "report.xml"
    write_report(str(jp), results, "json")
    write_report(str(xp), results, "junit")
    json.loads(jp.read_text(encoding="utf-8"))
    ET.fromstring(xp.read_text(encoding="utf-8"))


def test_xml_escaping_safe():
    # a message with XML-hostile characters must still produce parseable XML
    class _Ann:
        file = "m.py"
        line = 3
        col = None
        level = "error"
        message = "[SHAPE] bad <tensor> & \"quote\" mismatch"

        def render(self):
            return "::error::x"

    class _Result:
        diagnostics = []
        bugs = []

    # feed via a fake annotations_for_result by constructing results directly
    from src import reporters

    orig = reporters.annotations_for_result
    reporters.annotations_for_result = lambda f, r: [_Ann()]
    try:
        xml = reporters.to_junit_xml([("m.py", _Result())])
        root = ET.fromstring(xml)
        msg = root.find("testsuite").find("testcase").find("failure").get("message")
        assert "<tensor>" in msg and "&" in msg
    finally:
        reporters.annotations_for_result = orig
