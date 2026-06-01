"""Step 21 -- tests for the PyTorch operator-surface coverage matrix."""

from __future__ import annotations

import torch.nn as nn

from evaluation import operator_coverage as OC


# ---- live public-surface enumeration --------------------------------------
def test_enumerate_namespaces_nonempty_and_sorted():
    for ns in OC.NAMESPACES:
        names = OC.enumerate_namespace(ns)
        assert names, ns
        assert names == sorted(names)
        assert all(not n.startswith("_") for n in names)


def test_torch_nn_enumerates_only_module_subclasses():
    for n in OC.enumerate_namespace("torch.nn"):
        cls = getattr(nn, n)
        assert isinstance(cls, type) and issubclass(cls, nn.Module)


def test_enumerate_unknown_namespace_raises():
    import pytest
    with pytest.raises(ValueError):
        OC.enumerate_namespace("torch.bogus")


# ---- implemented operator census ------------------------------------------
def test_implemented_names_contains_known_ops():
    impl = OC.implemented_names()
    assert "matmul" in impl["torch"]
    assert "linear" in impl["torch.nn"]
    assert "relu" in impl["torch.nn.functional"]
    # Each namespace has a non-trivial set.
    for ns in OC.NAMESPACES:
        assert len(impl[ns]) > 0


# ---- coverage logic (deterministic, synthetic) ----------------------------
def test_coverage_for_synthetic():
    public = ["Alpha", "Beta", "Gamma", "Delta"]
    implemented = {"alpha", "gamma"}  # lowercase, case-insensitive match
    e = OC.coverage_for(public, implemented)
    assert e["total"] == 4
    assert e["covered"] == ["Alpha", "Gamma"]
    assert e["uncovered"] == ["Beta", "Delta"]
    assert e["covered_count"] == 2
    assert e["uncovered_count"] == 2
    assert e["coverage_ratio"] == 0.5


def test_coverage_for_empty():
    e = OC.coverage_for([], set())
    assert e["total"] == 0
    assert e["coverage_ratio"] == 0.0


def test_coverage_is_case_insensitive():
    e = OC.coverage_for(["MatMul"], {"matmul"})
    assert e["covered"] == ["MatMul"]


# ---- full matrix ----------------------------------------------------------
def test_build_matrix_structure_and_invariants():
    m = OC.build_matrix()
    assert set(m["namespaces"]) == set(OC.NAMESPACES)
    tot = cov = 0
    for ns in OC.NAMESPACES:
        e = m["namespaces"][ns]
        assert 0.0 <= e["coverage_ratio"] <= 1.0
        assert e["covered_count"] + e["uncovered_count"] == e["total"]
        assert e["covered_count"] <= e["total"]
        # covered names are a subset of the public surface
        public = set(OC.enumerate_namespace(ns))
        assert set(e["covered"]).issubset(public)
        tot += e["total"]
        cov += e["covered_count"]
    assert m["summary"]["total_public_operators"] == tot
    assert m["summary"]["total_covered"] == cov
    assert m["meta"]["torch_version"]


def test_build_matrix_is_deterministic():
    a = OC._dumps(OC.build_matrix())
    b = OC._dumps(OC.build_matrix())
    assert a == b


def test_nn_coverage_is_substantial():
    # The nn.Module layer map is the most complete surface; sanity-floor it so a
    # regression that drops layer recognition is caught.
    m = OC.build_matrix()
    assert m["namespaces"]["torch.nn"]["covered_count"] >= 50


# ---- committed artifact ---------------------------------------------------
def test_committed_matrix_check_passes():
    # Version-gated: returns 0 whether the local torch matches (byte-identical)
    # or differs (QUALIFIED skip).
    assert OC.run(check=True) == 0
