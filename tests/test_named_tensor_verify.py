"""Differential tests for named-tensor refine/align contracts.

The oracle is real PyTorch named tensors.  For every generated case, the static
checker must agree on success vs. RuntimeError and, when successful, the output
shape and names.
"""

from __future__ import annotations

import random
import textwrap
import warnings

import pytest

torch = pytest.importorskip("torch")

from src.named_tensor_verify import (  # noqa: E402
    NamedTensorSpec,
    find_named_tensor_bugs,
    verify_align_to,
    verify_named_tensor_source,
    verify_refine_names,
)


def _named(shape, names):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.randn(*shape).refine_names(*names)


def _real_refine(shape, names, requested):
    try:
        y = _named(shape, names).refine_names(*requested)
        return "ok", tuple(y.shape), tuple(y.names)
    except Exception:
        return "err", None, None


def _real_align(shape, names, target):
    try:
        y = _named(shape, names).align_to(*target)
        return "ok", tuple(y.shape), tuple(y.names)
    except Exception:
        return "err", None, None


@pytest.mark.parametrize(
    "shape,names,requested,expected_names",
    [
        ((2, 3), (None, None), ("N", "C"), ("N", "C")),
        ((2, 3), ("N", "C"), (..., "C"), ("N", "C")),
        ((2, 3, 4), ("N", None, "C"), ("N", "H", "C"), ("N", "H", "C")),
        ((2, 3), (None, None), ("...", "C"), (None, "C")),
    ],
)
def test_refine_valid_cases_match_torch(shape, names, requested, expected_names):
    real_status, real_shape, real_names = _real_refine(shape, names, requested)
    verdict = verify_refine_names(shape, names, requested)
    assert real_status == "ok"
    assert verdict.ok
    assert verdict.spec == NamedTensorSpec(real_shape, real_names)
    assert verdict.spec.names == expected_names


@pytest.mark.parametrize(
    "shape,names,requested,kind",
    [
        ((2, 3), ("N", "C"), ("N", "D"), "rename"),
        ((2, 3), ("N", "C"), (None, "C"), "demotion"),
        ((2, 3), (None, None), ("N",), "rank"),
        ((2, 3), (None, None), ("N", "N"), "duplicate"),
        ((2, 3, 4), ("N", "H", "C"), ("N", ..., "N"), "duplicate"),
        ((2,), (None,), ("1bad",), "invalid_name"),
    ],
)
def test_refine_invalid_cases_match_torch(shape, names, requested, kind):
    real_status, _, _ = _real_refine(shape, names, requested)
    verdict = verify_refine_names(shape, names, requested)
    assert real_status == "err"
    assert not verdict.ok
    assert verdict.error_kind == kind


@pytest.mark.parametrize(
    "shape,names,target,expected_shape,expected_names",
    [
        ((2, 3), ("N", "C"), ("C", "N"), (3, 2), ("C", "N")),
        ((2, 3), ("N", "C"), ("N", "C", "H"), (2, 3, 1), ("N", "C", "H")),
        ((2, 3, 4), ("N", None, "C"), ("H", ..., "C"), (1, 2, 3, 4), ("H", "N", None, "C")),
        ((2, 3), ("N", "C"), ("N", None, "C"), (2, 1, 3), ("N", None, "C")),
        ((2, 3), ("N", "C"), ("...", "C"), (2, 3), ("N", "C")),
    ],
)
def test_align_valid_cases_match_torch(shape, names, target, expected_shape, expected_names):
    real_status, real_shape, real_names = _real_align(shape, names, target)
    verdict = verify_align_to(shape, names, target)
    assert real_status == "ok"
    assert verdict.ok
    assert verdict.spec == NamedTensorSpec(real_shape, real_names)
    assert verdict.spec.shape == expected_shape
    assert verdict.spec.names == expected_names


@pytest.mark.parametrize(
    "shape,names,target,kind",
    [
        ((2, 3), ("N", "C"), ("N",), "missing_name"),
        ((2, 3), (None, "C"), ("N", "C"), "unnamed_dim"),
        ((2, 3), ("N", "C"), ("N", "N", "C"), "duplicate"),
        ((2, 3), ("N", "C"), (..., "N", ...), "ellipsis"),
        ((2,), ("N",), ("bad.name",), "invalid_name"),
    ],
)
def test_align_invalid_cases_match_torch(shape, names, target, kind):
    real_status, _, _ = _real_align(shape, names, target)
    verdict = verify_align_to(shape, names, target)
    assert real_status == "err"
    assert not verdict.ok
    assert verdict.error_kind == kind


def _random_valid_names(rng, rank):
    pool = ["N", "C", "H", "W", "B", "T"]
    rng.shuffle(pool)
    out = []
    used = 0
    for _ in range(rank):
        if rng.random() < 0.35:
            out.append(None)
        else:
            out.append(pool[used])
            used += 1
    return tuple(out)


def _random_tokens(rng, rank, existing_names):
    pool = [n for n in existing_names if n is not None] + ["X", "Y", "Z", None]
    count = rng.randint(0, rank + 2)
    tokens = [rng.choice(pool) for _ in range(count)]
    if rng.random() < 0.35:
        tokens.insert(rng.randint(0, len(tokens)), Ellipsis)
    if rng.random() < 0.05:
        tokens.append(Ellipsis)
    return tuple(tokens)


def test_refine_fuzz_matches_real_torch():
    rng = random.Random(1840)
    checked = 0
    for _ in range(300):
        rank = rng.randint(1, 4)
        shape = tuple(rng.randint(1, 4) for _ in range(rank))
        names = _random_valid_names(rng, rank)
        requested = _random_tokens(rng, rank, names)
        real_status, real_shape, real_names = _real_refine(shape, names, requested)
        verdict = verify_refine_names(shape, names, requested)
        assert ("ok" if verdict.ok else "err") == real_status, (shape, names, requested, verdict)
        if real_status == "ok":
            assert verdict.spec == NamedTensorSpec(real_shape, real_names)
        checked += 1
    assert checked == 300


def test_align_fuzz_matches_real_torch():
    rng = random.Random(1841)
    checked = 0
    for _ in range(300):
        rank = rng.randint(1, 4)
        shape = tuple(rng.randint(1, 4) for _ in range(rank))
        names = _random_valid_names(rng, rank)
        target = _random_tokens(rng, rank + 1, names)
        real_status, real_shape, real_names = _real_align(shape, names, target)
        verdict = verify_align_to(shape, names, target)
        assert ("ok" if verdict.ok else "err") == real_status, (shape, names, target, verdict)
        if real_status == "ok":
            assert verdict.spec == NamedTensorSpec(real_shape, real_names)
        checked += 1
    assert checked == 300


def test_symbolic_sizes_flow_without_false_positive():
    refined = verify_refine_names(("B", "C"), (None, None), ("batch", "channel"))
    assert refined.ok
    assert refined.spec.shape == ("B", "C")
    aligned = verify_align_to(("B", "C"), ("batch", "channel"), ("channel", "batch", "time"))
    assert aligned.ok
    assert aligned.spec.shape == ("C", "B", 1)
    assert aligned.spec.names == ("channel", "batch", "time")


CHAIN_OK = textwrap.dedent(
    '''
    def f(x):
        y = x.refine_names("N", "C")
        return y.align_to("C", "N")
    '''
)

ALIGN_BUG = textwrap.dedent(
    '''
    def f(x):
        return x.align_to("N")
    '''
)

REFINE_BUG = textwrap.dedent(
    '''
    def f(x):
        return x.refine_names("N", "D")
    '''
)


def _runtime_raises(source, shape, names):
    ns = {}
    exec(compile(source, "<named>", "exec"), ns)
    try:
        ns["f"](_named(shape, names))
        return False
    except Exception:
        return True


def test_source_checker_tracks_valid_chain_and_runtime_runs():
    assert not _runtime_raises(CHAIN_OK, (2, 3), (None, None))
    result = verify_named_tensor_source(CHAIN_OK, {"x": ((2, 3), (None, None))})
    assert result.status == "SAFE"
    assert result.bugs == []


@pytest.mark.parametrize(
    "source,shape,names,needle",
    [
        (ALIGN_BUG, (2, 3), ("N", "C"), "align_to"),
        (REFINE_BUG, (2, 3), ("N", "C"), "refine_names"),
    ],
)
def test_source_checker_catches_real_named_tensor_runtime_errors(source, shape, names, needle):
    assert _runtime_raises(source, shape, names)
    bugs = find_named_tensor_bugs(source, {"x": (shape, names)})
    assert len(bugs) == 1
    assert needle in bugs[0].message
    assert bugs[0].fix_suggestion


def test_unknown_source_shape_is_skipped_no_false_positive():
    assert find_named_tensor_bugs(ALIGN_BUG, {}) == []


def test_public_package_exports_named_tensor_checker():
    import tensorguard

    assert tensorguard.verify_refine_names is verify_refine_names
    assert tensorguard.verify_align_to is verify_align_to
    assert tensorguard.verify_named_tensor_source is verify_named_tensor_source
