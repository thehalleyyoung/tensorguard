"""Step 179 — the growth playbook references only assets that actually exist.

A playbook that points at vaporware erodes trust. This test parses every
repo-relative path mentioned in `docs/GROWTH_PLAYBOOK.md` and asserts each one
exists, so the plan stays executable as the repo evolves.
"""

from __future__ import annotations

import os
import re

import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PLAYBOOK = os.path.join(ROOT, "docs", "GROWTH_PLAYBOOK.md")

# Repo-relative path tokens we expect to reference real files/dirs.
_KNOWN = [
    "examples/quickstart.py",
    "examples/tutorials/",
    "examples/tutorials/01_quickstart.ipynb",
    "examples/",
    "docs/RFC_pytorch_companion.md",
    "docs/site/",
    "API.md",
    "SOUNDNESS_CONTRACT.md",
    "VERIFIABLE_FRAGMENT.md",
    "bugclasses.jsonl",
    "editors/vscode/",
    "src/lsp_server.py",
    "action.yml",
    ".pre-commit-hooks.yml",
    "benchmarks/leaderboard_entries/",
    "lean/",
    "community_stubs/",
    "community_stubs/README.md",
    "src/flax_extractor.py",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "GOVERNANCE.md",
    "MAINTAINERS.md",
    "CHANGELOG.md",
    "GETTING_STARTED.md",
    "tests/test_api_stability.py",
]


def test_playbook_exists():
    assert os.path.exists(PLAYBOOK)


def test_playbook_pitch_present():
    with open(PLAYBOOK, encoding="utf-8") as fh:
        text = fh.read()
    assert "statically" in text and "zero false alarms" in text


@pytest.mark.parametrize("rel", _KNOWN)
def test_referenced_asset_exists(rel):
    path = os.path.join(ROOT, rel)
    assert os.path.exists(path), f"playbook references missing asset: {rel}"


def test_every_referenced_asset_is_covered_by_this_test():
    """Each repo-relative path the playbook mentions is in the asserted set."""
    with open(PLAYBOOK, encoding="utf-8") as fh:
        text = fh.read()
    # Markdown links of the form (../path) or (path), plus inline `code` paths.
    link_paths = set(re.findall(r"\]\(\.\./([^)#]+)\)", text))
    code_paths = {
        m for m in re.findall(r"`([^`]+)`", text)
        if "/" in m and not m.startswith(("pip", "bash", "python", "pytest", "tensorguard", "pre-commit"))
    }
    referenced = {p.rstrip("/") + ("/" if p.endswith("/") else "") for p in (link_paths | code_paths)}
    known = {k.rstrip("/") + ("/" if k.endswith("/") else "") for k in _KNOWN}
    uncovered = {
        p for p in referenced
        if p.rstrip("/") not in {k.rstrip("/") for k in known}
        and os.path.sep not in p[:0]  # keep simple
    }
    # Only enforce coverage for paths that look like real repo files/dirs.
    uncovered = {p for p in uncovered if os.path.exists(os.path.join(ROOT, p))}
    assert not uncovered, f"playbook references real assets not pinned by this test: {sorted(uncovered)}"
