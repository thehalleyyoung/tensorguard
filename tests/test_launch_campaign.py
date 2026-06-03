"""Step 288 -- public launch campaign is grounded in real demos and policies."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import reproducibility.launch_campaign as lc


def test_every_channel_has_demo_evidence_rfc_and_support_anchor():
    audit = lc.build_audit()
    assert audit["summary"]["channel_count"] >= 4
    assert audit["summary"]["all_channels_have_required_anchors"] is True
    for channel in audit["channels"]:
        anchors = channel["anchors"]
        assert set(anchors) == {"demo", "evidence", "rfc", "support"}
        for kind in ("demo", "evidence", "rfc"):
            assert (lc.REPO / anchors[kind]).exists(), (channel["key"], kind, anchors[kind])
        assert anchors["support"] == "docs/launch/compatibility_support_promise.md"


def test_support_promise_is_anchored_in_existing_policy_surfaces():
    audit = lc.build_audit()
    assert audit["summary"]["support_promise_source_count"] >= 6
    for path in audit["support_sources"].values():
        assert (lc.REPO / path).exists(), path
    support = lc.OUT_SUPPORT.read_text(encoding="utf-8")
    for phrase in (
        "Current package version",
        "1.0-readiness",
        "does not claim the package has already shipped",
        "UNKNOWN is a supported outcome",
        "DEPRECATION_POLICY.md",
        "SECURITY.md",
    ):
        assert phrase in support


def test_campaign_is_honest_about_current_version():
    audit = lc.build_audit()
    assert audit["current_package_version"] != "1.0.0"
    assert audit["honest_versioning"] is True
    social = lc.OUT_SOCIAL.read_text(encoding="utf-8")
    assert "1.0-readiness" in social
    assert "shipped as `1.0.0`" not in social


def test_launch_copy_numeric_claims_are_source_derived():
    audit = lc.build_audit()
    allowed_numbers = {
        "1.0",  # campaign track, explicitly audited as readiness not release
        "1.0.0",
        audit["current_package_version"],
        audit["requires_python"].lstrip(">="),
        str(audit["summary"]["channel_count"]),
        str(audit["source_counts"]["model_gallery_entries"]),
        str(audit["source_counts"]["tutorial_notebooks"]),
        str(audit["source_counts"]["matrix_jobs"]),
        str(audit["source_counts"]["release_channels"]),
    }
    combined = "\n".join(path.read_text(encoding="utf-8") for path in lc.OUTPUTS if path.suffix == ".md")
    tokens = set(re.findall(r"\b\d+(?:\.\d+)*\b", combined))
    assert tokens <= allowed_numbers, tokens - allowed_numbers


def test_audit_covers_all_cited_paths_and_launch_gate():
    audit = lc.build_audit()
    assert audit["summary"]["all_cited_paths_exist"] is True
    assert all(audit["cited_paths"].values())
    assert audit["summary"]["release_ready_gate_passes"] is True
    assert audit["requires_python"].startswith(">=")


def test_generated_artifacts_are_byte_deterministic():
    first = lc.write_outputs()
    snapshots = {path: path.read_text(encoding="utf-8") for path in lc.OUTPUTS}
    second = lc.write_outputs()
    assert first == second
    assert {path: path.read_text(encoding="utf-8") for path in lc.OUTPUTS} == snapshots


def test_cli_check_passes_against_committed_artifacts():
    proc = subprocess.run(
        [sys.executable, "reproducibility/launch_campaign.py", "--check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_reproduce_all_and_makefile_own_launch_campaign():
    import reproducibility.reproduce_all as ra

    expected = {
        "docs/launch/one_point_zero_launch_campaign.md",
        "docs/launch/demo_script.md",
        "docs/launch/social_copy.md",
        "docs/launch/compatibility_support_promise.md",
        "reproducibility/launch_campaign_audit.json",
        "reproducibility/launch_campaign_audit.md",
    }
    assert expected <= set(ra.GENERATED_DETERMINISTIC)
    makefile = (lc.REPO / "Makefile").read_text(encoding="utf-8")
    assert "\nlaunch-campaign:" in makefile
    assert "reproducibility/launch_campaign.py --check" in makefile
