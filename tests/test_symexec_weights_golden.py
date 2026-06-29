"""Roadmap step 5 — **golden-certificate regression suite**.

``tests/data/weights_golden/`` freezes, for a fixed set of small checkpoints (6
``proven_safe`` good files + the full 15-kind negative corpus), the *entire*
canonical JSON certificate (`dumps_weights_certificate`).  These tests assert:

* re-certifying each checkpoint reproduces its golden certificate **byte-for-byte**
  (the strongest possible determinism lock — tensors, findings, both fingerprints,
  and all flags are pinned, not just the finding kinds);
* certifying twice is byte-identical (determinism within a run);
* every golden certificate round-trips through `loads_weights_certificate`;
* the golden set covers both verdicts and all 15 finding kinds;
* the on-disk corpus is byte-identical to the generator and has no stray files.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _weights_golden_gen as gold  # noqa: E402

from src.symexec.weights import (  # noqa: E402
    dumps_weights_certificate,
    loads_weights_certificate,
    weights_certificate_to_dict,
)

_DATA_DIR = _HERE / "data" / "weights_golden"
_CERTS_DIR = _DATA_DIR / "certs"
_WEIGHTS_PY = _HERE.parent / "src" / "symexec" / "weights.py"


def _manifest() -> dict:
    return json.loads((_DATA_DIR / "manifest.json").read_text())


def _cases():
    return gold.cases()


def _code_finding_kinds() -> set[str]:
    tree = ast.parse(_WEIGHTS_PY.read_text())
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "WeightsFinding"):
            if node.args and isinstance(node.args[0], ast.Constant):
                kinds.add(node.args[0].value)
            for kw in node.keywords:
                if kw.arg == "kind" and isinstance(kw.value, ast.Constant):
                    kinds.add(kw.value.value)
    return kinds


# --------------------------------------------------------------------------- #
# Presence / structure.                                                         #
# --------------------------------------------------------------------------- #
def test_golden_dir_and_manifest():
    assert _DATA_DIR.is_dir()
    assert _CERTS_DIR.is_dir()
    entries = _manifest()["entries"]
    assert len(entries) == 21  # 6 good + 15 bad
    for e in entries:
        assert (_DATA_DIR / e["file"]).is_file()
        assert (_DATA_DIR / e["cert"]).is_file()


def test_no_stray_files():
    on_disk = {p.name for p in _DATA_DIR.iterdir()}
    expected = {c["file"] for c in _cases()} | {"manifest.json", "certs"}
    assert on_disk == expected, f"unexpected top-level: {on_disk ^ expected}"
    certs_on_disk = {p.name for p in _CERTS_DIR.iterdir()}
    expected_certs = {f"{c['name']}.json" for c in _cases()}
    assert certs_on_disk == expected_certs


# --------------------------------------------------------------------------- #
# THE lock: re-certification reproduces the golden bytes exactly.               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_golden_certificate_byte_identical(case):
    golden_path = _CERTS_DIR / f"{case['name']}.json"
    golden = golden_path.read_text()
    produced = gold.golden_text(str(_DATA_DIR), case)
    assert produced == golden, (
        f"{case['name']}: certificate drifted from golden file "
        f"{golden_path} (the certifier output changed)."
    )


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_certification_is_idempotent(case):
    a = gold.golden_text(str(_DATA_DIR), case)
    b = gold.golden_text(str(_DATA_DIR), case)
    assert a == b


# --------------------------------------------------------------------------- #
# Round-trip + canonical-form properties.                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_golden_round_trips(case):
    golden = (_CERTS_DIR / f"{case['name']}.json").read_text()
    cert = loads_weights_certificate(golden)
    # Re-serialising the parsed cert yields identical canonical JSON.
    assert dumps_weights_certificate(cert) + "\n" == golden
    # And the parsed dict matches the raw JSON.
    assert weights_certificate_to_dict(cert) == json.loads(golden)


def test_golden_json_is_canonical():
    """Every golden file is sorted-key, 2-space-indented, newline-terminated."""
    for c in _cases():
        text = (_CERTS_DIR / f"{c['name']}.json").read_text()
        assert text.endswith("\n")
        obj = json.loads(text)
        assert json.dumps(obj, indent=2, sort_keys=True) + "\n" == text


# --------------------------------------------------------------------------- #
# Coverage of verdicts + finding kinds.                                         #
# --------------------------------------------------------------------------- #
def test_golden_covers_both_verdicts():
    safe, unsafe = 0, 0
    for c in _cases():
        obj = json.loads((_CERTS_DIR / f"{c['name']}.json").read_text())
        if obj["proven_safe"]:
            safe += 1
            assert obj["findings"] == [], c["name"]
            assert c["good"]
        else:
            unsafe += 1
            assert obj["findings"], c["name"]
            assert not c["good"]
    assert safe >= 6 and unsafe >= 15


def test_golden_covers_all_finding_kinds():
    seen: set[str] = set()
    for c in _cases():
        obj = json.loads((_CERTS_DIR / f"{c['name']}.json").read_text())
        seen.update(f["kind"] for f in obj["findings"])
    code = _code_finding_kinds()
    assert len(code) == 15
    assert seen == code, f"golden kind coverage gap: {sorted(code ^ seen)}"


# --------------------------------------------------------------------------- #
# Regeneration lock: committed bytes == generator output.                       #
# --------------------------------------------------------------------------- #
def test_corpus_byte_identical_to_generator(tmp_path):
    regen = tmp_path / "weights_golden"
    gold.write_golden(str(regen))
    for c in _cases():
        assert (_DATA_DIR / c["file"]).read_bytes() == (regen / c["file"]).read_bytes(), \
            f"{c['file']} checkpoint drifted"
        rel = f"certs/{c['name']}.json"
        assert (_DATA_DIR / rel).read_text() == (regen / rel).read_text(), \
            f"{rel} golden drifted"
    # Manifest matches (order-insensitive on entries).
    key = lambda lst: sorted(lst, key=lambda e: e["name"])
    assert key(_manifest()["entries"]) == key(
        json.loads((regen / "manifest.json").read_text())["entries"])
