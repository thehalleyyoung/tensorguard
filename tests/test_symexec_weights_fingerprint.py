"""Roadmap step 6 — **structural-fingerprint collision audit**.

``_structural_fingerprint`` is the replay receipt that binds a certificate to an
exact structural reading (tensors + findings + which checks ran), independent of
file path.  Its soundness contract is two-directional:

* **Injective on what matters** — two *semantically different* readings must never
  share a fingerprint (otherwise a replay could accept the wrong reading).
* **Stable under cosmetics** — two *semantically identical* readings (e.g. tensors
  or findings supplied in a different order) must always share a fingerprint.

The canonical reading the fingerprint commits to is exactly::

    ( sorted [name, dtype, list(shape), begin, end] over tensors,
      sorted [kind, name or "", detail]            over findings,
      checked_finite, contract_checked )

These tests prove ``fp(a) == fp(b)  ⇔  canon(a) == canon(b)`` — adversarial pairs
are distinguished, cosmetic permutations coincide — both by targeted construction
and by randomized property testing, and they pin the one *documented* benign
identification (a finding ``name=None`` and ``name=""`` are treated as equal,
which is verdict-irrelevant).
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.symexec.weights import (
    WeightTensorInfo,
    WeightsFinding,
    _structural_fingerprint,
    certify_weights_file,
)

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _weights_golden_gen as gold  # noqa: E402


def _t(name, dtype, shape, begin, end) -> WeightTensorInfo:
    return WeightTensorInfo(name=name, dtype=dtype, shape=tuple(shape),
                            begin=begin, end=end)


def _f(kind, name, detail) -> WeightsFinding:
    return WeightsFinding(kind=kind, name=name, detail=detail)


def _fp(infos, findings, *, cf=True, cc=False) -> str:
    return _structural_fingerprint(infos, findings, checked_finite=cf,
                                   contract_checked=cc)


def _canon(infos, findings, cf, cc):
    """The exact equivalence class the fingerprint must commit to."""
    return (
        sorted([i.name, i.dtype, list(i.shape), i.begin, i.end] for i in infos),
        sorted([f.kind, f.name or "", f.detail] for f in findings),
        cf, cc,
    )


# --------------------------------------------------------------------------- #
# Stability: cosmetic permutations coincide.                                    #
# --------------------------------------------------------------------------- #
def test_tensor_input_order_irrelevant():
    ts = [_t("a", "U8", [1], 0, 1), _t("b", "F32", [2], 1, 9),
          _t("c", "I8", [3], 9, 12)]
    base = _fp(ts, [])
    for perm in itertools.permutations(ts):
        assert _fp(list(perm), []) == base


def test_finding_input_order_irrelevant():
    fs = [_f("storage_gap", "b", "gap"), _f("unknown_dtype", "a", "F7"),
          _f("non_finite_values", "c", "nan@0")]
    base = _fp([], fs)
    for perm in itertools.permutations(fs):
        assert _fp([], list(perm)) == base


def test_repeated_calls_deterministic():
    ts = [_t("a", "F32", [2, 3], 0, 24)]
    fs = [_f("storage_overlap", "a", "x")]
    assert _fp(ts, fs) == _fp(ts, fs) == _fp(ts, fs)


def test_real_certificate_fingerprint_is_path_independent(tmp_path):
    """Same bytes at two different paths ⇒ identical structural fingerprint."""
    data = gold.good_cases()[1]["data"]  # good_multi
    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "sub" / "b.safetensors"
    p2.parent.mkdir()
    p1.write_bytes(data)
    p2.write_bytes(data)
    c1 = certify_weights_file(str(p1))
    c2 = certify_weights_file(str(p2))
    assert c1.structural_fingerprint == c2.structural_fingerprint
    assert c1.filename != c2.filename  # path differs, fingerprint does not


# --------------------------------------------------------------------------- #
# Injectivity: every semantic field change flips the fingerprint.               #
# --------------------------------------------------------------------------- #
def test_each_tensor_field_change_flips_fingerprint():
    base = [_t("a", "F32", [2], 0, 8)]
    base_fp = _fp(base, [])
    variants = {
        "name":  [_t("z", "F32", [2], 0, 8)],
        "dtype": [_t("a", "F16", [2], 0, 8)],
        "shape": [_t("a", "F32", [4], 0, 8)],
        "shape_order": [_t("a", "F32", [2, 3], 0, 8)],
        "begin": [_t("a", "F32", [2], 1, 8)],
        "end":   [_t("a", "F32", [2], 0, 9)],
    }
    for label, v in variants.items():
        assert _fp(v, []) != base_fp, f"{label} change did not flip fingerprint"


def test_shape_dim_order_matters():
    assert _fp([_t("a", "F32", [2, 3], 0, 24)], []) != \
           _fp([_t("a", "F32", [3, 2], 0, 24)], [])


def test_begin_end_swap_flips_fingerprint():
    assert _fp([_t("a", "F32", [2], 0, 8)], []) != \
           _fp([_t("a", "F32", [2], 8, 0)], [])


def test_add_remove_tensor_flips_fingerprint():
    one = [_t("a", "U8", [1], 0, 1)]
    two = one + [_t("b", "U8", [1], 1, 2)]
    assert _fp(one, []) != _fp(two, [])


def test_each_finding_field_change_flips_fingerprint():
    base = [_f("storage_gap", "a", "gap [0,4)")]
    base_fp = _fp([], base)
    variants = {
        "kind":   [_f("storage_overlap", "a", "gap [0,4)")],
        "name":   [_f("storage_gap", "b", "gap [0,4)")],
        "detail": [_f("storage_gap", "a", "gap [4,8)")],
    }
    for label, v in variants.items():
        assert _fp([], v) != base_fp, f"{label} change did not flip fingerprint"


def test_add_remove_finding_flips_fingerprint():
    one = [_f("storage_gap", "a", "x")]
    two = one + [_f("unknown_dtype", "b", "y")]
    assert _fp([], one) != _fp([], two)


def test_flag_flips_change_fingerprint():
    ts = [_t("a", "F32", [2], 0, 8)]
    base = _fp(ts, [], cf=True, cc=False)
    assert _fp(ts, [], cf=False, cc=False) != base
    assert _fp(ts, [], cf=True, cc=True) != base
    assert _fp(ts, [], cf=False, cc=True) != base


# --------------------------------------------------------------------------- #
# Field-boundary adversarial: no cross-field bleed.                             #
# --------------------------------------------------------------------------- #
def test_tensor_name_cannot_impersonate_other_fields():
    """A tensor whose *name* contains JSON-looking field separators must not
    collide with a genuinely different multi-field/multi-tensor reading."""
    evil = _t('x","F32",[2],0,8', "F32", [2], 0, 8)
    benign = _t("x", "F32", [2], 0, 8)
    assert _fp([evil], []) != _fp([benign], [])
    # Nor with a two-tensor reading.
    two = [_t("x", "F32", [2], 0, 8), _t("y", "F32", [2], 8, 16)]
    assert _fp([evil], []) != _fp(two, [])


def test_finding_detail_cannot_impersonate_other_fields():
    evil = _f("k", "n", 'a","b","c')
    other = _f("k", "n", "a")
    assert _fp([], [evil]) != _fp([], [other])
    # Three separate single-char details must not equal one joined detail.
    joined = _f("k", "n", "abc")
    parts = [_f("k", "n", "a"), _f("k", "n", "b"), _f("k", "n", "c")]
    assert _fp([], [joined]) != _fp([], parts)


def test_value_moved_between_tensors_flips_fingerprint():
    """Moving a byte from one tensor's extent to another is a different reading."""
    a = [_t("a", "U8", [4], 0, 4), _t("b", "U8", [4], 4, 8)]
    b = [_t("a", "U8", [3], 0, 3), _t("b", "U8", [5], 3, 8)]
    assert _fp(a, []) != _fp(b, [])


# --------------------------------------------------------------------------- #
# Documented benign identification (verdict-irrelevant).                        #
# --------------------------------------------------------------------------- #
def test_finding_none_name_equals_empty_string_name():
    """A finding's ``name=None`` (whole-file) canonicalises to ``""`` exactly like
    a (degenerate) empty tensor name.  This identification is intentional and
    sound: ``name`` is descriptive metadata, never part of the safety verdict, and
    no certifier path emits an empty-string finding name in practice."""
    assert _fp([], [_f("k", None, "d")]) == _fp([], [_f("k", "", "d")])
    # It is the *only* such identification: a different detail still separates them.
    assert _fp([], [_f("k", None, "d")]) != _fp([], [_f("k", "", "d2")])


# --------------------------------------------------------------------------- #
# The full equivalence on real certificates.                                    #
# --------------------------------------------------------------------------- #
def test_golden_fingerprint_equivalence_is_exact():
    """Across the 21 golden checkpoints, two certificates share a structural
    fingerprint iff they share the canonical (tensors, findings, flags) reading."""
    certs = []
    for c in gold.cases():
        path = str(gold_dir() / c["file"])
        cert = certify_weights_file(
            path, check_finite=c["check_finite"],
            expected=gold.expected_contract(c["expected"]),
            contract_partial=c["partial"],
        )
        certs.append(cert)
    for x, y in itertools.combinations(certs, 2):
        same_fp = x.structural_fingerprint == y.structural_fingerprint
        same_canon = _canon(list(x.tensors), list(x.findings),
                            x.checked_finite, x.contract_checked) == \
                     _canon(list(y.tensors), list(y.findings),
                            y.checked_finite, y.contract_checked)
        assert same_fp == same_canon, (
            f"fingerprint/canon mismatch for {x.filename} vs {y.filename}"
        )


def gold_dir() -> Path:
    return _HERE / "data" / "weights_golden"


# --------------------------------------------------------------------------- #
# Randomized property: fp(a) == fp(b) ⇔ canon(a) == canon(b).                    #
# --------------------------------------------------------------------------- #
_NAMES = st.sampled_from(["a", "b", "c", "", "x|y", 'q","r'])
_DTYPES = st.sampled_from(["F32", "F16", "U8", "I8", "BOOL"])
_DIM = st.integers(min_value=0, max_value=5)
_OFF = st.integers(min_value=0, max_value=64)

_tensor = st.builds(
    lambda n, d, s, b, e: _t(n, d, s, b, e),
    _NAMES, _DTYPES, st.lists(_DIM, max_size=3), _OFF, _OFF,
)
_finding = st.builds(
    lambda k, n, d: _f(k, n, d),
    st.sampled_from(["storage_gap", "unknown_dtype", "non_finite_values"]),
    st.one_of(st.none(), _NAMES),
    st.sampled_from(["d0", "d1", "d2"]),
)
_reading = st.tuples(
    st.lists(_tensor, max_size=4),
    st.lists(_finding, max_size=4),
    st.booleans(),
    st.booleans(),
)


@settings(derandomize=True, deadline=None, database=None, max_examples=600,
          suppress_health_check=[HealthCheck.too_slow])
@given(_reading, _reading)
def test_fingerprint_iff_canonical(r1, r2):
    fp1 = _fp(r1[0], r1[1], cf=r1[2], cc=r1[3])
    fp2 = _fp(r2[0], r2[1], cf=r2[2], cc=r2[3])
    canon1 = _canon(r1[0], r1[1], r1[2], r1[3])
    canon2 = _canon(r2[0], r2[1], r2[2], r2[3])
    assert (fp1 == fp2) == (canon1 == canon2)


@settings(derandomize=True, deadline=None, database=None, max_examples=400,
          suppress_health_check=[HealthCheck.too_slow])
@given(_reading)
def test_permutation_invariance_property(r):
    infos, findings, cf, cc = r
    rng = random.Random(len(infos) * 31 + len(findings))
    pi, pf = infos[:], findings[:]
    rng.shuffle(pi)
    rng.shuffle(pf)
    assert _fp(infos, findings, cf=cf, cc=cc) == _fp(pi, pf, cf=cf, cc=cc)
