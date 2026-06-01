"""Regression tests for the numeric-claim audit harness (100_STEPS.md Step 4).

These pin two things:
  * the live audit over README.md / neurips.tex / workshop_fmai.tex passes
    (every registered headline number still matches its committed artifact and
    is still present in the prose, and every README ratio/percent token is
    covered by a registry claim);
  * the harness genuinely fails on MISMATCH / ORPHAN / SOURCE_MISSING, so a
    drifting number cannot pass silently.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MOD_PATH = REPO / "reproducibility" / "audit_numeric_claims.py"

spec = importlib.util.spec_from_file_location("audit_numeric_claims", MOD_PATH)
audit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_mod)


def test_live_audit_passes():
    audit = audit_mod.run_audit()
    bad = [r for r in audit["claims"]
           if r["status"] in ("MISMATCH", "ORPHAN", "SOURCE_MISSING")]
    assert not bad, f"audit found unbacked/mismatched claims: {bad}"
    assert not audit["readme_uncovered_tokens"], (
        f"README has ratio/percent tokens not covered by a registry claim: "
        f"{audit['readme_uncovered_tokens']}"
    )
    assert audit["passed"] is True


def test_every_registry_entry_classified():
    audit = audit_mod.run_audit()
    statuses = {r["status"] for r in audit["claims"]}
    allowed = {"VERIFIED", "QUALIFIED_ENV", "QUALIFIED_REGIME",
               "MISMATCH", "ORPHAN", "SOURCE_MISSING"}
    assert statuses <= allowed
    # the curated headline set must contain at least the core verified claims
    ids = {r["id"] for r in audit["claims"]}
    assert {"rp_53_of_60", "tg_32_of_34_fragmentfair",
            "hf_9_of_9_natural"} <= ids


def test_classify_detects_mismatch():
    entry = {
        "id": "fake_mismatch",
        "regime": "test",
        "claim": "wrong",
        "sources": [(str(REPO / "README.md"), r"TensorGuard")],
        "artifacts": ["reproduce_headline_60bug.json"],
        "compute": lambda: 999,
        "check": lambda a: a == 53,
    }
    assert audit_mod._classify(entry)["status"] == "MISMATCH"


def test_classify_detects_source_missing():
    entry = {
        "id": "fake_source_missing",
        "regime": "test",
        "claim": "x",
        "sources": [(str(REPO / "README.md"), r"this string is absolutely not in the readme zzzz")],
        "artifacts": [],
        "compute": lambda: 1,
        "check": lambda a: True,
    }
    assert audit_mod._classify(entry)["status"] == "SOURCE_MISSING"


def test_classify_detects_orphan():
    entry = {
        "id": "fake_orphan",
        "regime": "test",
        "claim": "x",
        "sources": [(str(REPO / "README.md"), r"TensorGuard")],
        "artifacts": ["does_not_exist_zzz.json"],
        "compute": lambda: audit_mod._art("does_not_exist_zzz.json"),
        "check": lambda a: True,
    }
    assert audit_mod._classify(entry)["status"] == "ORPHAN"


def test_pytea_regime_is_documented_not_silent():
    """The 25/34 Pytea number must be flagged QUALIFIED_REGIME (a stricter
    regime yields 22/34) rather than silently VERIFIED."""
    audit = audit_mod.run_audit()
    pytea = next(r for r in audit["claims"] if r["id"] == "pytea_25_of_34_fragmentfair")
    assert pytea["status"] == "QUALIFIED_REGIME"
