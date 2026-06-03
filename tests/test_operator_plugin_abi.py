"""Step 280 — stable plugin ABI for executable third-party operator theories."""

from __future__ import annotations

import textwrap

import pytest

from src.model_checker import verify_model
from src.operator_plugin_abi import (
    ABI_VERSION,
    ConformanceCase,
    OperatorTheoryContract,
    PluginProvenance,
    SecurityReview,
    install_operator_theories,
    is_abi_compatible,
    validate_operator_theory,
)
from src.shape_stub_registry import clear_user_stubs, get_shape_stub
from src.tensor_shapes import ShapeDim, TensorShape


@pytest.fixture(autouse=True)
def _isolate_stubs():
    clear_user_stubs()
    yield
    clear_user_stubs()


def _provenance():
    return PluginProvenance(
        package="acme-tensor-layers",
        version="0.4.0",
        source_url="https://example.com/acme",
        license="MIT",
        author="Acme",
    )


def _review(**overrides):
    data = {
        "reviewed_by": "tg-maintainer",
        "reviewed_on": "2026-06-03",
        "no_import_side_effects": True,
        "no_network": True,
        "no_filesystem_writes": True,
        "deterministic": True,
        "no_model_execution": True,
    }
    data.update(overrides)
    return SecurityReview(**data)


def _last_dim_times_three(inp: TensorShape, params):
    if inp.ndim < 1:
        return None, "TripleLastDim requires a tensor input"
    last = inp.dims[-1]
    if last.is_symbolic:
        return TensorShape(inp.dims[:-1] + (ShapeDim("_triple"),)), None
    return TensorShape(inp.dims[:-1] + (ShapeDim(last.value * 3),)), None


def _contract(**overrides):
    data = {
        "class_name": "TripleLastDim",
        "transfer": _last_dim_times_three,
        "arg_names": (),
        "conformance": (
            ConformanceCase(input_shape=("batch", 4), expected_output=("batch", 12)),
        ),
        "provenance": _provenance(),
        "security_review": _review(),
        "summary": "Maps the last dim to three times its input size.",
    }
    data.update(overrides)
    return OperatorTheoryContract(**data)


def test_abi_version_accepts_same_major_and_rejects_future_major():
    assert ABI_VERSION == "1.0"
    assert is_abi_compatible("1.0")
    assert is_abi_compatible("1.7.3")
    assert not is_abi_compatible("2.0")
    assert not is_abi_compatible("not-a-version")


def test_valid_contract_installs_executable_transfer_and_verifies_real_model():
    reports = install_operator_theories([_contract()])
    assert reports[0].ok
    assert reports[0].cases_checked == 1
    assert get_shape_stub("TripleLastDim") is not None

    source = textwrap.dedent(
        """
        import torch.nn as nn
        from acme import TripleLastDim
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.triple = TripleLastDim()
                self.head = nn.Linear(12, 2)
            def forward(self, x):
                return self.head(self.triple(x))
        """
    )
    assert verify_model(source, input_shapes={"x": (5, 4)}).safe is True
    bad = source.replace("nn.Linear(12, 2)", "nn.Linear(11, 2)")
    assert verify_model(bad, input_shapes={"x": (5, 4)}).safe is False


def test_invalid_contract_does_not_install_by_default():
    bad = _contract(abi_version="2.0")
    with pytest.raises(ValueError):
        install_operator_theories([bad])
    assert get_shape_stub("TripleLastDim") is None


def test_security_attestation_is_required():
    report = validate_operator_theory(
        _contract(security_review=_review(no_network=False))
    )
    assert not report.ok
    assert any("no_network" in error for error in report.errors)


def test_provenance_is_required():
    report = validate_operator_theory(_contract(provenance=None))
    assert not report.ok
    assert any("provenance" in error for error in report.errors)


def test_conformance_mismatch_is_rejected_without_registry_leak():
    report = validate_operator_theory(
        _contract(
            conformance=(
                ConformanceCase(input_shape=("batch", 4), expected_output=("batch", 99)),
            )
        )
    )
    assert not report.ok
    assert any("expected output" in error for error in report.errors)
    assert get_shape_stub("TripleLastDim") is None


def test_transfer_crash_is_rejected():
    def crashing(_inp, _params):
        raise RuntimeError("boom")

    report = validate_operator_theory(
        _contract(transfer=crashing)
    )
    assert not report.ok
    assert any("transfer raised" in error for error in report.errors)


def test_no_entry_point_autoload_policy_is_documented_and_enforced_by_absence():
    import src.operator_plugin_abi as abi

    assert not hasattr(abi, "load_entry_points")
    assert not hasattr(abi, "autoload_plugins")
    docs = open("docs/plugins/operator_plugin_abi.md", encoding="utf-8").read()
    assert "never auto-discovered or auto-imported" in docs
    for phrase in (
        "no network access",
        "no filesystem writes",
        "no execution of the analyzed model",
    ):
        assert phrase in docs
