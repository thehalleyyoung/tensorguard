"""Step 90 — the labeled benchmark corpus as a standalone, citable contribution.

These tests pin the two new publication artifacts — a *Datasheets for Datasets*
datasheet and a `CITATION.cff` — and guard that they stay consistent with the
frozen, SHA-256-addressed manifest so the corpus cannot be cited with numbers
that have drifted from the data.
"""

import json
import os
import re
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_RB = os.path.join(_ROOT, "real_benchmarks")
DATASHEET = os.path.join(_RB, "DATASHEET.md")
CITATION = os.path.join(_ROOT, "CITATION.cff")
MANIFEST = os.path.join(_RB, "manifest.json")


def _manifest():
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def test_datasheet_and_citation_exist():
    assert os.path.exists(DATASHEET), "real_benchmarks/DATASHEET.md missing"
    assert os.path.exists(CITATION), "CITATION.cff missing"


def test_datasheet_is_fresh():
    env = dict(os.environ, PYTHONPATH=_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "real_benchmarks.build_datasheet", "--check"],
        cwd=_ROOT, capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, (
        f"DATASHEET.md is stale:\n{proc.stdout}\n{proc.stderr}"
    )


def test_datasheet_counts_match_manifest():
    m = _manifest()
    items = m["items"]
    text = open(DATASHEET, encoding="utf-8").read()

    n_clean = sum(1 for it in items if it["label"] == "clean")
    n_buggy = sum(1 for it in items if it["label"] == "buggy")
    assert f"{m['meta']['total']} models" in text
    assert f"{n_clean} clean / {n_buggy} buggy" in text

    # Per-label / per-domain counts are rendered as markdown rows.
    from collections import Counter
    for dom, n in Counter(it["domain"] for it in items).items():
        assert re.search(rf"\| `{re.escape(dom)}` \| {n} \|", text), (
            f"datasheet missing/incorrect domain row for {dom}={n}"
        )


def test_datasheet_documents_every_item():
    m = _manifest()
    text = open(DATASHEET, encoding="utf-8").read()
    for it in m["items"]:
        assert f"`{it['id']}`" in text, f"item {it['id']} not in datasheet"
        # The per-item table cites the first 12 hex of the content hash.
        assert it["sha256"][:12] in text, f"hash for {it['id']} not in datasheet"


def test_citation_has_required_fields():
    text = open(CITATION, encoding="utf-8").read()
    try:
        import yaml  # pyyaml is a dev/runtime dep in this repo
    except Exception:
        pytest.skip("pyyaml not available")
    data = yaml.safe_load(text)
    assert data.get("cff-version"), "CITATION.cff missing cff-version"
    assert data.get("title")
    assert data.get("authors"), "CITATION.cff needs at least one author"
    assert str(data.get("license")) == "MIT"
    # The dataset is referenced as a citable component.
    refs = data.get("references") or []
    assert any(r.get("type") == "dataset" for r in refs), (
        "CITATION.cff should reference the benchmark corpus as a dataset"
    )


def test_citation_version_matches_package():
    text = open(CITATION, encoding="utf-8").read()
    pyproject = open(os.path.join(_ROOT, "pyproject.toml"), encoding="utf-8").read()
    pv = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert pv, "could not find version in pyproject.toml"
    assert f'version: "{pv.group(1)}"' in text, (
        "CITATION.cff version is out of sync with pyproject.toml"
    )


def test_dataset_reference_version_matches_manifest():
    text = open(CITATION, encoding="utf-8").read()
    corpus_version = _manifest()["meta"]["version"]
    # The dataset reference block records the corpus version.
    assert f'version: "{corpus_version}"' in text, (
        "CITATION.cff dataset reference version != manifest meta.version"
    )


def test_datasheet_wired_into_reproduce_pipeline():
    repro = open(os.path.join(_ROOT, "reproducibility", "reproduce_all.py"),
                 encoding="utf-8").read()
    assert "real_benchmarks.build_datasheet" in repro
    assert "real_benchmarks/DATASHEET.md" in repro


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
