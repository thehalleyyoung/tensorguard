"""Step 286 - release-readiness checklist gates every release channel."""

from __future__ import annotations

import json

import reproducibility.release_readiness as rr


def test_manifest_gates_all_three_release_channels():
    data = rr.build_manifest()
    assert {channel["channel"] for channel in data["channels"]} == {"pypi", "conda", "docker"}
    assert data["all_channels_release_ready"] is True
    for channel in data["channels"]:
        assert channel["release_ready"] is True
        assert channel["passed_items"] == channel["required_items"]


def test_common_release_blockers_are_present_on_every_channel():
    data = rr.build_manifest()
    required = {
        "benchmark-dashboard",
        "deployment-dashboard",
        "numeric-claim-audit",
        "security-review",
    }
    for channel in data["channels"]:
        keys = {item["key"] for item in channel["items"]}
        assert required <= keys


def test_channel_specific_packaging_gates_are_present():
    data = rr.build_manifest()
    by_channel = {channel["channel"]: channel for channel in data["channels"]}
    assert {"source-artifact-package", "pypi-metadata"} <= {
        item["key"] for item in by_channel["pypi"]["items"]
    }
    assert {"conda-artifact-package", "conda-metadata"} <= {
        item["key"] for item in by_channel["conda"]["items"]
    }
    assert {"docker-artifact-package", "docker-metadata"} <= {
        item["key"] for item in by_channel["docker"]["items"]
    }


def test_release_gate_is_backed_by_real_evidence_paths():
    data = rr.build_manifest()
    for channel in data["channels"]:
        for item in channel["items"]:
            assert item["evidence"], item
            for path in item["evidence"]:
                assert (rr.REPO / path).exists(), (item["key"], path)
            assert item["command"], item


def test_rendered_artifacts_are_byte_identical_and_successful():
    assert rr.run(check=True) == 0
    committed = json.loads(rr.OUT_JSON.read_text(encoding="utf-8"))
    assert committed == rr.build_manifest()
