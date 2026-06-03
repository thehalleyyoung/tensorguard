"""Tests for the Step 257 multi-axis version-stability matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STAB_JSON = REPO / "reproducibility" / "cross_version_stability.json"

_VOLATILE = (
    "time", "elapsed", "timestamp", "wall", "clock",
    "_ms", "seconds", "duration", "date",
)
_HOST_DEPENDENT = ("available", "platform", "machine", "hostname", "host")


def _data():
    return json.loads(STAB_JSON.read_text())


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_artifact_no_volatile_or_host_specific_fields():
    data = _data()
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"
        assert not any(v in low for v in _HOST_DEPENDENT), (
            f"host-dependent key: {key}"
        )


def test_artifact_is_byte_deterministic():
    from reproducibility import cross_version_stability as cvs

    assert cvs.run(check=True) == 0


def test_static_blocked_import_proof_covers_torch_and_torchvision():
    data = _data()
    assert data["static_no_target_library_execution"] is True
    assert data["blocked_framework_imports_match_baseline"] is True
    assert data["torch_blocked_verdicts_match_baseline"] is True
    assert set(data["blocked_import_modules"]) == {"torch", "torchvision"}


def test_torch_version_range_covers_2_1_to_2_9():
    data = _data()
    versions = data["torch_versions_tested"]
    assert versions[0] == "2.1.0"
    assert versions[-1] == "2.9.1"
    assert len(versions) >= 9
    assert data["verdict_stable_across_torch_2_1_to_2_9"] is True
    for version, ok in data["per_torch_version_matches_baseline"].items():
        assert ok is True, f"verdict drift at torch {version}"


def test_torchvision_version_range_covers_0_16_to_0_24():
    data = _data()
    versions = data["torchvision_versions_tested"]
    assert versions[0] == "0.16.0"
    assert versions[-1] == "0.24.1"
    assert len(versions) >= 9
    assert data["verdict_stable_across_torchvision_0_16_to_0_24"] is True
    for version, ok in data["per_torchvision_version_matches_baseline"].items():
        assert ok is True, f"verdict drift at torchvision {version}"


def test_python_axis_links_to_cross_python_determinism_proof():
    data = _data()
    assert data["python_versions_qualified"] == [
        "3.9", "3.10", "3.11", "3.12", "3.13", "3.14",
    ]
    ev = data["python_determinism_evidence"]
    assert ev["artifact"] == "reproducibility/cross_python_determinism.json"
    assert ev["verdict_invariant_under_hash_randomization"] is True
    assert ev["deterministic_across_python_builds"] is True


def test_backend_and_os_axes_are_version_qualified():
    data = _data()
    assert data["backend_environments_qualified"] == ["cuda-less CPU", "MPS"]
    assert data["operating_systems_qualified"] == ["linux", "macos"]
    assert data["device_backend_verdicts_match"] is True
    assert data["os_verdict_stability_qualified"] is True


def test_environment_qualification_matrix_covers_required_axes():
    data = _data()
    axes = {row["axis"] for row in data["environment_qualification"]}
    assert axes == {
        "python", "pytorch", "torchvision", "backend", "operating_system",
    }
    for row in data["environment_qualification"]:
        assert row["values"]
        assert row["status"]
        assert row["evidence"]
        assert row["command"]


def test_torchvision_fixture_is_nonvacuous_and_blocked_import_stable():
    from reproducibility import cross_version_stability as cvs

    data = _data()
    fixture = data["torchvision_fixture"]
    assert fixture["id"] == cvs.TORCHVISION_FIXTURE.id
    assert fixture["references_torchvision"] is True
    assert "torchvision" in cvs.TORCHVISION_FIXTURE.source
    assert "v2." in cvs.TORCHVISION_FIXTURE.source
    assert fixture["blocked_import_matches"] is True


def test_cpu_mps_device_fixture_invariance():
    data = _data()
    normal = data["device_backend_fixture_verdicts"]
    blocked = data["device_backend_blocked_import_verdicts"]
    assert normal["cuda_less_cpu"] == normal["mps"]
    assert blocked["cuda_less_cpu"] == blocked["mps"]
    assert normal == blocked


def test_blocked_torch_and_torchvision_invariance_live():
    from reproducibility import cross_version_stability as cvs

    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "from torchvision.transforms import v2\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.t = v2.CenterCrop((8, 9))\n"
        "    def forward(self, x):\n"
        "        return self.t(x)\n"
    )
    fixture = cvs.StabilityFixture(
        "live_blocked_torchvision_fixture", src, {"x": (3, 16, 20)}
    )
    normal = cvs._verdict_map([fixture])
    blocked = cvs._score_with_imports_blocked([fixture])
    assert normal == blocked == {fixture.id: "SAFE"}


def test_overall_step_257_gate_passes():
    data = _data()
    assert data["all_framework_versions_stable"] is True
    assert data["overall_step_257_stability"] is True
