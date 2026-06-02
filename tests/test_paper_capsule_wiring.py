"""Step 177 — assert the workshop submission is wired to the artifact capsule.

The paper claims its results are reproduced by the capsule; this check keeps
that promise honest by verifying, in CI, that (a) the capsule's one-command
entry point and its supporting files exist, and (b) the paper actually cites
them. If either side drifts, the test fails.
"""

from __future__ import annotations

import os

import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = os.path.join(ROOT, "workshop_fmai.tex")


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def test_capsule_entry_point_and_support_files_exist():
    for rel in (
        "capsule/reproduce.sh",
        "capsule/requirements.lock.txt",
        "reproducibility/reproduce_all.py",
        "docs/artifact/README.md",
        "docs/artifact/INSTALL.md",
        "docs/artifact/REQUIREMENTS.md",
        "docs/artifact/STATUS.md",
    ):
        assert os.path.exists(os.path.join(ROOT, rel)), f"missing {rel}"


def test_paper_exists_and_has_artifact_section():
    assert os.path.exists(PAPER)
    tex = _read(PAPER)
    assert "Artifact Availability" in tex
    assert "\\label{sec:artifact}" in tex


def test_paper_cites_the_capsule_one_command():
    tex = _read(PAPER)
    # The exact reproduce command the capsule exposes.
    assert "capsule/reproduce.sh" in tex
    # And points reviewers at the artifact appendix directory.
    assert "docs/artifact/" in tex


def test_paper_references_resolve_to_real_labels():
    tex = _read(PAPER)
    import re

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    missing = {r for r in refs if r not in labels}
    assert not missing, f"dangling \\ref(s): {sorted(missing)}"


def test_capsule_reproduce_invokes_the_audited_harness():
    sh = _read(os.path.join(ROOT, "capsule", "reproduce.sh"))
    # The capsule must run the same from-scratch + determinism check the paper
    # cites, so the paper's reproduction claim is backed by real code.
    assert "reproduce_all.py" in sh
    assert "--check" in sh
