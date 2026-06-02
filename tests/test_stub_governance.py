"""Step 178 — community stub-registry governance, proven end-to-end.

Validates the declarative manifest format, the security guarantees (no
executable code, mandatory provenance), conformance-as-proof, the committed
example manifests, and the runtime load path that turns a third-party layer
from an UNKNOWN abstention into a precisely-checked block.
"""

from __future__ import annotations

import copy
import json

import pytest

from src.stub_governance import (
    validate_manifest,
    validate_directory,
    load_community_stubs,
)
from src.shape_stub_registry import (
    get_shape_stub,
    clear_user_stubs,
    registered_stub_names,
)

COMMUNITY_DIR = "community_stubs"


def _good_linear_manifest():
    return {
        "class_name": "MyCommunityLinear",
        "kind": "last_dim_linear",
        "spec": {
            "in_arg": "in_features",
            "out_arg": "out_features",
            "arg_names": ["in_features", "out_features"],
        },
        "provenance": {
            "author": "a",
            "source_url": "https://example.com/x.py",
            "license": "MIT",
            "reviewed_by": "maint",
        },
        "conformance": [
            {"input": ["batch", 16], "ctor_args": [16, 4],
             "expect": {"output": ["batch", 4]}},
            {"input": ["batch", 8], "ctor_args": [16, 4],
             "expect": {"error_contains": "expects last dim=16, got 8"}},
        ],
    }


@pytest.fixture(autouse=True)
def _isolate_registry():
    clear_user_stubs()
    yield
    clear_user_stubs()


def test_valid_manifest_passes_and_does_not_leak_into_registry():
    before = set(registered_stub_names())
    report = validate_manifest(_good_linear_manifest())
    assert report.ok, report.errors
    assert report.cases_checked == 2
    # Validation is side-effect free: nothing left registered.
    assert set(registered_stub_names()) == before
    assert get_shape_stub("MyCommunityLinear") is None


def test_missing_provenance_is_rejected():
    m = _good_linear_manifest()
    del m["provenance"]["license"]
    report = validate_manifest(m)
    assert not report.ok
    assert any("license" in e for e in report.errors)


def test_empty_provenance_field_is_rejected():
    m = _good_linear_manifest()
    m["provenance"]["author"] = "   "
    report = validate_manifest(m)
    assert not report.ok
    assert any("author" in e for e in report.errors)


@pytest.mark.parametrize("bad_field", ["transfer", "code", "python", "exec", "import"])
def test_code_bearing_fields_are_rejected(bad_field):
    m = _good_linear_manifest()
    m[bad_field] = "lambda x: x"
    report = validate_manifest(m)
    assert not report.ok
    assert any("declarative" in e or "forbidden" in e for e in report.errors)


def test_unsupported_kind_is_rejected():
    m = _good_linear_manifest()
    m["kind"] = "some_arbitrary_kind"
    report = validate_manifest(m)
    assert not report.ok
    assert any("unsupported kind" in e for e in report.errors)


def test_conformance_mismatch_is_rejected():
    m = _good_linear_manifest()
    # Claim a wrong output shape.
    m["conformance"][0]["expect"]["output"] = ["batch", 999]
    report = validate_manifest(m)
    assert not report.ok
    assert any("expected output" in e for e in report.errors)


def test_conformance_required():
    m = _good_linear_manifest()
    m["conformance"] = []
    report = validate_manifest(m)
    assert not report.ok
    assert any("conformance" in e for e in report.errors)


def test_shape_preserving_manifest():
    m = {
        "class_name": "MyNorm",
        "kind": "shape_preserving",
        "provenance": {"author": "a", "source_url": "u", "license": "MIT",
                       "reviewed_by": "m"},
        "conformance": [
            {"input": ["batch", "seq", 32], "expect": {"output": ["batch", "seq", 32]}},
        ],
    }
    assert validate_manifest(m).ok


def test_committed_example_manifests_validate():
    reports = validate_directory(COMMUNITY_DIR)
    assert reports, "expected example manifests in community_stubs/"
    for r in reports:
        assert r.ok, (r.source, r.errors)
        assert r.cases_checked >= 1


def test_load_community_stubs_registers_and_enables_precise_checking():
    before = set(registered_stub_names())
    loaded = load_community_stubs(COMMUNITY_DIR)
    assert "Linear8bitLt" in loaded and "T5LayerNorm" in loaded
    # Now the layer is known: a real shape contract is enforced.
    stub = get_shape_stub("Linear8bitLt")
    assert stub is not None
    params = stub.bind_params((768, 3072), {})
    from src.tensor_shapes import ShapeDim, TensorShape
    out, err = stub.transfer(TensorShape((ShapeDim("batch"), ShapeDim(768))), params)
    assert err is None and out.dims[-1].value == 3072
    # A wrong feature dim is now an error rather than a silent abstention.
    _, err2 = stub.transfer(TensorShape((ShapeDim("batch"), ShapeDim(512))), params)
    assert err2 and "768" in err2
    clear_user_stubs()
    assert set(registered_stub_names()) == before


def test_invalid_manifest_is_not_loaded():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        good = _good_linear_manifest()
        bad = _good_linear_manifest()
        bad["class_name"] = "BadOne"
        del bad["provenance"]
        with open(os.path.join(d, "good.json"), "w") as fh:
            json.dump(good, fh)
        with open(os.path.join(d, "bad.json"), "w") as fh:
            json.dump(bad, fh)
        loaded = load_community_stubs(d)
        assert "MyCommunityLinear" in loaded
        assert "BadOne" not in loaded
