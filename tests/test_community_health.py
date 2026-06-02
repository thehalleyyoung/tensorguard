"""Step 84 — validate the community-health files exist and are well-formed.

GitHub recognizes specific community-health files (CONTRIBUTING, CODE_OF_CONDUCT,
issue/PR templates, governance). These tests guard against accidental deletion
or malformed front matter so the project keeps its community surface intact.
"""

from __future__ import annotations

import os

import pytest

yaml = pytest.importorskip("yaml")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel: str) -> str:
    with open(os.path.join(_REPO, rel), "r", encoding="utf-8") as fh:
        return fh.read()


def test_core_docs_exist():
    for rel in (
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "GOVERNANCE.md",
        "MAINTAINERS.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
    ):
        path = os.path.join(_REPO, rel)
        assert os.path.exists(path), f"missing {rel}"
        assert os.path.getsize(path) > 200, f"{rel} looks empty"


def test_contributing_links_the_key_policies():
    text = _read("CONTRIBUTING.md")
    for ref in ("CODE_OF_CONDUCT.md", "SECURITY.md", "DEPRECATION_POLICY.md",
                "GOVERNANCE.md"):
        assert ref in text, f"CONTRIBUTING.md should reference {ref}"


def test_governance_documents_maintainer_rotation():
    text = _read("GOVERNANCE.md").lower()
    assert "rotation" in text
    assert "lead maintainer" in text


def test_maintainers_has_rotation_order():
    text = _read("MAINTAINERS.md").lower()
    assert "rotation" in text
    assert "lead" in text


def test_issue_templates_present_and_have_front_matter():
    tdir = os.path.join(_REPO, ".github", "ISSUE_TEMPLATE")
    assert os.path.isdir(tdir)
    md = [f for f in os.listdir(tdir) if f.endswith(".md")]
    assert len(md) >= 3, f"expected >=3 issue templates, found {md}"
    for f in md:
        text = _read(os.path.join(".github", "ISSUE_TEMPLATE", f))
        assert text.startswith("---"), f"{f} missing YAML front matter"
        fm = text.split("---", 2)[1]
        meta = yaml.safe_load(fm)
        assert meta.get("name"), f"{f} front matter needs a name"
        assert meta.get("about"), f"{f} front matter needs an about"


def test_issue_template_config_is_valid_yaml():
    cfg = _read(os.path.join(".github", "ISSUE_TEMPLATE", "config.yml"))
    data = yaml.safe_load(cfg)
    assert data["blank_issues_enabled"] is False
    urls = " ".join(link["url"] for link in data["contact_links"])
    assert "security" in urls.lower()


def test_pr_template_covers_soundness_and_testing():
    text = _read(".github/PULL_REQUEST_TEMPLATE.md").lower()
    assert "soundness" in text
    assert "test" in text
    assert "deprecation_policy.md" in text


def test_unsound_template_is_highest_severity():
    text = _read(os.path.join(".github", "ISSUE_TEMPLATE",
                              "unsound_result.md"))
    assert "SAFE" in text
    assert "soundness" in text.lower()
