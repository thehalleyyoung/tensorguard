"""Step 89 — reproducible artifact + Artifact-Evaluation package.

These tests guard that the Artifact-Evaluation documentation is present and,
crucially, *honest*: every regeneration command and every committed artifact it
references must actually exist, the significance harness must be wired into the
from-scratch reproduction pipeline, and the determinism contract must hold from
a clean tree.
"""

import os
import re
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_AE = os.path.join(_ROOT, "docs", "artifact")


def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def test_ae_docs_exist():
    for name in ("README.md", "STATUS.md", "INSTALL.md", "REQUIREMENTS.md"):
        path = os.path.join(_AE, name)
        assert os.path.exists(path), f"missing AE doc {name}"
        assert os.path.getsize(path) > 200, f"AE doc {name} is suspiciously short"


def test_status_requests_three_standard_badges():
    status = _read("docs", "artifact", "STATUS.md")
    for badge in ("Available", "Functional", "Reusable"):
        assert badge in status, f"STATUS.md does not mention the {badge} badge"


def test_claim_table_artifacts_and_scripts_all_exist():
    """Every artifact path and every `python ...py` script in the claim table
    must point at a real file in the repository."""
    readme = _read("docs", "artifact", "README.md")
    # Committed-artifact paths appear in backticks ending in .json/.md, and
    # scripts appear as `python ... path/to/file.py` or PYTHONPATH=... .
    referenced = set(re.findall(r"`[^`]*?([\w./-]+\.(?:json|py|md))`?", readme))
    # Pull explicit paths out of backticked spans to avoid false positives.
    paths = set()
    for span in re.findall(r"`([^`]+)`", readme):
        for tok in span.split():
            if re.search(r"\.(json|py|md)$", tok) and "/" in tok:
                paths.add(tok)
    # A few well-known artifacts that MUST resolve.
    required = {
        "reproducibility/reproduce_headline_60bug.json",
        "evaluation/confusion_matrices.json",
        "evaluation/significance.json",
        "evaluation/sound_mode_fp.json",
        "docs/formalization/type_system.md",
        "lean/TensorGuard/AxiomAudit.lean",
        "real_benchmarks/manifest.json",
        "reproducibility/numeric_claims_audit.json",
        "evaluation/significance.py",
        "evaluation/precision_recall.py",
    }
    missing_required = [p for p in required if not os.path.exists(os.path.join(_ROOT, p))]
    assert not missing_required, f"claim-table artifacts missing: {missing_required}"

    # And nothing referenced with a path separator may dangle.
    dangling = [p for p in paths
                if "/" in p and not os.path.exists(os.path.join(_ROOT, p))
                and not p.startswith("/work/")]
    assert not dangling, f"AE README references nonexistent paths: {dangling}"


def test_significance_is_wired_into_reproduce_pipeline():
    repro = _read("reproducibility", "reproduce_all.py")
    assert "evaluation/significance.py" in repro, (
        "significance harness not wired into reproduce_all STEPS"
    )
    assert "evaluation/significance.json" in repro, (
        "significance.json not declared byte-deterministic in reproduce_all"
    )


def test_reproduce_full_target_exists():
    makefile = _read("Makefile")
    assert re.search(r"^reproduce-full:", makefile, re.MULTILINE), (
        "Makefile is missing the reproduce-full target referenced by the AE docs"
    )


@pytest.mark.slow
def test_reproduce_all_check_passes_deterministically():
    """A clean-tree run of the from-scratch harness must reproduce + be
    byte-deterministic for the generated paths it owns."""
    env = dict(os.environ, PYTHONPATH=_ROOT)
    proc = subprocess.run(
        [sys.executable, "reproducibility/reproduce_all.py", "--check"],
        cwd=_ROOT, capture_output=True, text=True, timeout=600, env=env,
    )
    out = proc.stdout + proc.stderr
    # The harness may legitimately fail determinism ONLY if pre-existing
    # unrelated files are dirty; restrict the assertion to the paths it owns.
    assert "Reproduction PASS" in out, f"reproduction did not pass:\n{out[-3000:]}"
    if "DETERMINISM CHECK FAILED" in out:
        # Only tolerate diffs in files the harness does not own.
        owned = ("numeric_claims_audit.json", "significance.json",
                 "significance.md", "SOUNDNESS_CONTRACT.md",
                 "VERIFIABLE_FRAGMENT.md", "operator_confidence_table.json",
                 "real_benchmarks/manifest.json", "real_benchmarks/VERSION",
                 "real_benchmarks_audit.json")
        assert not any(o in out for o in owned), (
            f"a harness-owned artifact is non-deterministic:\n{out[-3000:]}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
