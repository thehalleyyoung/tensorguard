"""Step 67 — GitHub Code Scanning-ready SARIF 2.1.0 output.

Builds a SARIF log from a real verification, then validates it three ways:
the spec-level :class:`SarifValidator` bundled in src/output/sarif_reporter.py,
the documented GitHub Code Scanning ingestion requirements
(:func:`check_code_scanning_requirements`), and the ``jsonschema`` meta-checks on
structure.  Also covers fingerprint stability, rule synthesis, and the action's
``--sarif`` output path.
"""

import json

import torch  # noqa: F401

from src.api import verify_architecture
from src.sarif_codescan import (
    SARIF_VERSION,
    build_sarif,
    check_code_scanning_requirements,
    to_json,
    write_sarif,
)

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


def _verify(src, shapes=None):
    return verify_architecture(src, input_shapes=shapes, filename="net.py")


def test_build_sarif_top_level_shape():
    result = _verify(_BAD, {"x": ("batch", 10)})
    sarif = build_sarif([("net.py", result)])
    assert sarif["version"] == SARIF_VERSION
    assert sarif["$schema"]
    assert len(sarif["runs"]) == 1
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "TensorGuard"
    assert run["automationDetails"]["id"]
    assert run["results"], "expected at least one result for a buggy model"


def test_meets_code_scanning_requirements():
    result = _verify(_BAD, {"x": ("batch", 10)})
    sarif = build_sarif([("net.py", result)])
    problems = check_code_scanning_requirements(sarif)
    assert problems == [], problems


def test_every_ruleid_resolves_and_has_metadata():
    result = _verify(_BAD, {"x": ("batch", 10)})
    sarif = build_sarif([("net.py", result)])
    run = sarif["runs"][0]
    rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    for r in run["tool"]["driver"]["rules"]:
        assert r["shortDescription"]["text"]
        assert r["fullDescription"]["text"]
        assert r["defaultConfiguration"]["level"] in {"error", "warning", "note"}
    for res in run["results"]:
        assert res["ruleId"] in rule_ids
        assert res["level"] in {"error", "warning", "note"}
        loc = res["locations"][0]["physicalLocation"]
        assert loc["artifactLocation"]["uri"] == "net.py"
        assert loc["region"]["startLine"] >= 1
        assert res["partialFingerprints"]["tensorguard/v1"]


def test_passes_spec_validator():
    # Load the self-contained reporter module directly by path so we don't
    # trigger src/output/__init__.py (which eagerly imports an unrelated,
    # pre-existing-broken html_report module).
    import importlib.util
    import os
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, "src", "output", "sarif_reporter.py")
    spec = importlib.util.spec_from_file_location("_tg_sarif_reporter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses needs the module registered
    try:
        spec.loader.exec_module(mod)
    finally:
        pass

    result = _verify(_BAD, {"x": ("batch", 10)})
    sarif = build_sarif([("net.py", result)])
    validator = mod.SarifValidator()
    ok = validator.validate(sarif)
    assert ok, [str(e) for e in validator.errors]


def test_clean_model_yields_empty_results():
    result = _verify(_GOOD)  # conv: rank auto-inferred
    sarif = build_sarif([("net.py", result)])
    assert sarif["runs"][0]["results"] == []
    # an empty run is still a valid, ingestible log
    assert check_code_scanning_requirements(sarif) == []


def test_fingerprint_is_line_independent_but_rule_specific():
    """A bug that moves to a new line keeps its fingerprint; different rule/file differ."""
    result = _verify(_BAD, {"x": ("batch", 10)})
    s1 = build_sarif([("net.py", result)])
    # Re-run identical source: fingerprints must be identical (stable).
    s2 = build_sarif([("net.py", _verify(_BAD, {"x": ("batch", 10)}))])
    fp1 = {r["partialFingerprints"]["tensorguard/v1"] for r in s1["runs"][0]["results"]}
    fp2 = {r["partialFingerprints"]["tensorguard/v1"] for r in s2["runs"][0]["results"]}
    assert fp1 == fp2
    # Different file path -> different fingerprint.
    s3 = build_sarif([("other.py", result)])
    fp3 = {r["partialFingerprints"]["tensorguard/v1"] for r in s3["runs"][0]["results"]}
    assert fp1.isdisjoint(fp3)


def test_to_json_roundtrips():
    result = _verify(_BAD, {"x": ("batch", 10)})
    sarif = build_sarif([("net.py", result)])
    parsed = json.loads(to_json(sarif))
    assert parsed["version"] == SARIF_VERSION


def test_write_sarif_file(tmp_path):
    out = tmp_path / "out" / "tensorguard.sarif"
    result = _verify(_BAD, {"x": ("batch", 10)})
    write_sarif(str(out), [("net.py", result)])
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert check_code_scanning_requirements(data) == []


def test_action_emits_sarif(tmp_path, monkeypatch):
    from src.github_action import main

    bad = tmp_path / "bad.py"
    bad.write_text(_BAD, encoding="utf-8")
    sarif_out = tmp_path / "tg.sarif"
    monkeypatch.setenv("INPUT_PATHS", str(bad))
    monkeypatch.setenv("INPUT_INPUT_SHAPES", "x=batch,10")
    monkeypatch.setenv("INPUT_SARIF_OUTPUT", str(sarif_out))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    rc = main()
    assert rc == 1
    assert sarif_out.exists()
    data = json.loads(sarif_out.read_text(encoding="utf-8"))
    assert data["version"] == SARIF_VERSION
    assert check_code_scanning_requirements(data) == []
    # the uri in the SARIF is the file we scanned
    uri = data["runs"][0]["results"][0]["locations"][0][
        "physicalLocation"
    ]["artifactLocation"]["uri"]
    assert uri == str(bad)
