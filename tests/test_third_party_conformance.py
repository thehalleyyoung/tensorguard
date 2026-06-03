"""Step 281 — third-party stub/plugin conformance certification suite."""

from __future__ import annotations

import json
import textwrap

import pytest

from src.operator_plugin_abi import (
    ConformanceCase,
    OperatorTheoryContract,
    PluginProvenance,
    SecurityReview,
)
from src.shape_stub_registry import clear_user_stubs, get_shape_stub
from src.tensor_shapes import ShapeDim, TensorShape
from src.third_party_conformance import (
    ThirdPartyConformanceScenario,
    assert_conformance_passed,
    certify_plugin_contracts,
    certify_stub_manifests,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    clear_user_stubs()
    yield
    clear_user_stubs()


def _triple_last_dim(inp: TensorShape, _params):
    if inp.ndim < 1:
        return None, "TripleLastDim requires tensor input"
    last = inp.dims[-1]
    if last.is_symbolic:
        return TensorShape(inp.dims[:-1] + (ShapeDim("_triple"),)), None
    return TensorShape(inp.dims[:-1] + (ShapeDim(last.value * 3),)), None


def _plugin_contract(**overrides):
    data = {
        "class_name": "TripleLastDim",
        "transfer": _triple_last_dim,
        "conformance": (
            ConformanceCase(input_shape=("batch", 4), expected_output=("batch", 12)),
            ConformanceCase(input_shape=(2, 5), expected_output=(2, 15)),
        ),
        "provenance": PluginProvenance(
            package="acme-layers",
            version="1.2.3",
            source_url="https://example.com/acme-layers",
            license="MIT",
            author="Acme",
        ),
        "security_review": SecurityReview(
            reviewed_by="tg-maintainer",
            reviewed_on="2026-06-03",
            no_import_side_effects=True,
            no_network=True,
            no_filesystem_writes=True,
            deterministic=True,
            no_model_execution=True,
        ),
        "summary": "Triples the last tensor dimension.",
    }
    data.update(overrides)
    return OperatorTheoryContract(**data)


_PLUGIN_SAFE = textwrap.dedent(
    """
    import torch
    import torch.nn as nn
    from acme_layers import TripleLastDim
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = TripleLastDim()
            self.head = nn.Linear(12, 2)
        def forward(self, x):
            return self.head(self.block(x))
    """
)

_PLUGIN_BAD = _PLUGIN_SAFE.replace("nn.Linear(12, 2)", "nn.Linear(11, 2)")

_PLUGIN_HEURISTIC = textwrap.dedent(
    """
    import torch
    import torch.nn as nn
    from acme_layers import TripleLastDim
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = TripleLastDim()
            self.head = nn.Linear(12, 2)
        def forward(self, x):
            y = self.block(x)
            torch.unique(y)
            return self.head(y)
    """
)


def test_plugin_conformance_certifies_real_verifier_modes_and_preserves_registry():
    report = certify_plugin_contracts(
        [_plugin_contract()],
        [
            ThirdPartyConformanceScenario(
                name="safe-plugin-model",
                source=_PLUGIN_SAFE,
                input_shapes={"x": (3, 4)},
                expected_verdicts={"*": "SAFE"},
            ),
            ThirdPartyConformanceScenario(
                name="buggy-plugin-model",
                source=_PLUGIN_BAD,
                input_shapes={"x": (3, 4)},
                expected_verdicts={"*": "UNSAFE"},
                expected_bug_substrings=("Linear expects",),
            ),
            ThirdPartyConformanceScenario(
                name="heuristic-op-boundary",
                source=_PLUGIN_HEURISTIC,
                input_shapes={"x": (3, 4)},
                expected_verdicts={
                    "sound": "UNKNOWN",
                    "balanced": "SAFE",
                    "heuristic": "SAFE",
                },
            ),
        ],
        extension_name="acme-layers",
    )

    assert report.passed
    assert report.cases_checked == 2
    assert len(report.scenarios) == 9
    assert get_shape_stub("TripleLastDim") is None
    payload = json.loads(report.to_json())
    assert payload["passed"] is True
    assert payload["extension_name"] == "acme-layers"
    assert "heuristic-op-boundary" in report.to_markdown()
    assert_conformance_passed(report)


def _stub_manifest(**overrides):
    data = {
        "class_name": "FancyBlock",
        "kind": "last_dim_linear",
        "spec": {
            "in_arg": "in_features",
            "out_arg": "out_features",
            "arg_names": ["in_features", "out_features"],
        },
        "provenance": {
            "author": "Acme",
            "source_url": "https://example.com/fancy",
            "license": "MIT",
            "reviewed_by": "tg-maintainer",
        },
        "conformance": [
            {
                "input": ["batch", 8],
                "ctor_args": [8, 16],
                "expect": {"output": ["batch", 16]},
            }
        ],
    }
    data.update(overrides)
    return data


_STUB_SAFE = textwrap.dedent(
    """
    import torch.nn as nn
    from thirdparty import FancyBlock
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = FancyBlock(8, 16)
            self.head = nn.Linear(16, 4)
        def forward(self, x):
            return self.head(self.block(x))
    """
)

_STUB_BAD = _STUB_SAFE.replace("nn.Linear(16, 4)", "nn.Linear(99, 4)")


def test_stub_manifest_conformance_certifies_declarative_stubs_against_sound_modes():
    report = certify_stub_manifests(
        [_stub_manifest()],
        [
            ThirdPartyConformanceScenario(
                name="safe-stub-model",
                source=_STUB_SAFE,
                input_shapes={"x": (2, 8)},
                expected_verdicts={"*": "SAFE"},
            ),
            ThirdPartyConformanceScenario(
                name="buggy-stub-model",
                source=_STUB_BAD,
                input_shapes={"x": (2, 8)},
                expected_verdicts={"*": "UNSAFE"},
                expected_bug_substrings=("Linear expects",),
            ),
        ],
        extension_name="fancy-stub",
        modes=("sound", "balanced"),
    )

    assert report.passed
    assert report.cases_checked == 1
    assert len(report.scenarios) == 4
    assert get_shape_stub("FancyBlock") is None


def test_invalid_plugin_contract_fails_before_scenarios_run():
    report = certify_plugin_contracts(
        [_plugin_contract(abi_version="2.0")],
        [
            ThirdPartyConformanceScenario(
                name="would-run",
                source=_PLUGIN_SAFE,
                input_shapes={"x": (3, 4)},
                expected_verdicts={"*": "SAFE"},
            )
        ],
    )

    assert not report.passed
    assert report.scenarios == ()
    assert "validation failed" in report.errors[0]
    with pytest.raises(AssertionError):
        assert_conformance_passed(report)


def test_invalid_stub_manifest_fails_before_installing_or_running_scenarios():
    report = certify_stub_manifests(
        [_stub_manifest(provenance={})],
        [
            ThirdPartyConformanceScenario(
                name="would-run",
                source=_STUB_SAFE,
                input_shapes={"x": (2, 8)},
                expected_verdicts={"*": "SAFE"},
            )
        ],
    )

    assert not report.passed
    assert report.scenarios == ()
    assert get_shape_stub("FancyBlock") is None


def test_scenario_mismatch_is_reported_with_actionable_error():
    report = certify_plugin_contracts(
        [_plugin_contract()],
        [
            ThirdPartyConformanceScenario(
                name="wrong-expectation",
                source=_PLUGIN_BAD,
                input_shapes={"x": (3, 4)},
                expected_verdicts={"*": "SAFE"},
            )
        ],
        modes=("sound",),
    )

    assert not report.passed
    assert report.scenarios[0].verdict == "UNSAFE"
    assert "expected SAFE, got UNSAFE" in report.scenarios[0].errors


def test_invalid_soundness_mode_is_rejected():
    with pytest.raises(ValueError):
        certify_plugin_contracts([], [], modes=("strict",))
