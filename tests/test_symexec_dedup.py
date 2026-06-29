"""Step 70 — cross-engine de-duplication.

The FX/SMT shape path and the symbolic-execution engine reach overlapping
defects by independent routes, so ``api.analyze`` could surface a single
broadcast/matmul/reshape fault twice.  :func:`src.api._dedup_cross_engine`
collapses reports that agree on ``(file, line, column, category)`` — keeping the
first (FX) occurrence and enriching it with the duplicate's higher confidence and
any fix/evidence — while never merging genuinely distinct defects.

The unit tests drive the merge directly with synthetic ``Bug`` objects (torch
free); one integration test exercises the real two-engine pipeline (torch).
"""

from __future__ import annotations

import pytest

from src.api import (
    Bug,
    BugCategory,
    SourceLocation,
    _dedup_cross_engine,
)


def _bug(line, col, cat=BugCategory.TYPE_ERROR, conf=0.9, msg="m", file="f.py",
         fix=None, evidence=None):
    return Bug(
        category=cat,
        message=msg,
        location=SourceLocation(file=file, line=line, column=col),
        severity="error",
        confidence=conf,
        fix_suggestion=fix,
        guard_evidence=evidence,
    )


# -- unit: the merge key --------------------------------------------------


def test_same_site_same_category_merges():
    out = _dedup_cross_engine([_bug(5, 11, msg="fx"), _bug(5, 11, msg="sx")])
    assert len(out) == 1
    # first (FX) occurrence is the survivor
    assert out[0].message == "fx"


def test_different_column_not_merged():
    out = _dedup_cross_engine([_bug(5, 11), _bug(5, 12)])
    assert len(out) == 2


def test_different_category_not_merged():
    out = _dedup_cross_engine([
        _bug(5, 11, cat=BugCategory.TYPE_ERROR),
        _bug(5, 11, cat=BugCategory.NULL_DEREFERENCE),
    ])
    assert len(out) == 2


def test_different_line_not_merged():
    out = _dedup_cross_engine([_bug(5, 11), _bug(6, 11)])
    assert len(out) == 2


def test_different_file_not_merged():
    out = _dedup_cross_engine([_bug(5, 11, file="a.py"), _bug(5, 11, file="b.py")])
    assert len(out) == 2


# -- unit: enrichment of the survivor ------------------------------------


def test_confidence_is_maxed():
    out = _dedup_cross_engine([_bug(5, 11, conf=0.80), _bug(5, 11, conf=0.95)])
    assert len(out) == 1
    assert out[0].confidence == 0.95


def test_confidence_not_lowered():
    out = _dedup_cross_engine([_bug(5, 11, conf=0.95), _bug(5, 11, conf=0.80)])
    assert out[0].confidence == 0.95


def test_fix_suggestion_backfilled():
    out = _dedup_cross_engine([
        _bug(5, 11, fix=None),
        _bug(5, 11, fix="align dims"),
    ])
    assert out[0].fix_suggestion == "align dims"


def test_existing_fix_suggestion_preserved():
    out = _dedup_cross_engine([
        _bug(5, 11, fix="keep me"),
        _bug(5, 11, fix="overwrite"),
    ])
    assert out[0].fix_suggestion == "keep me"


def test_guard_evidence_backfilled():
    out = _dedup_cross_engine([
        _bug(5, 11, evidence=None),
        _bug(5, 11, evidence="prov chain"),
    ])
    assert out[0].guard_evidence == "prov chain"


# -- unit: ordering & no-op ----------------------------------------------


def test_order_preserved_for_distinct():
    bugs = [_bug(7, 1, msg="a"), _bug(3, 1, msg="b"), _bug(5, 2, msg="c")]
    out = _dedup_cross_engine(bugs)
    assert [b.message for b in out] == ["a", "b", "c"]


def test_empty_is_noop():
    assert _dedup_cross_engine([]) == []


def test_triplicate_collapses_to_one():
    out = _dedup_cross_engine([
        _bug(5, 11, conf=0.7), _bug(5, 11, conf=0.9), _bug(5, 11, conf=0.8),
    ])
    assert len(out) == 1
    assert out[0].confidence == 0.9


# -- integration: real two-engine pipeline -------------------------------

_DUP_SRC = (
    "import torch\n"
    "def f():\n"
    "    a = torch.zeros(3, 4)\n"
    "    b = torch.zeros(5, 6)\n"
    "    return a + b\n"
    "if __name__ == '__main__':\n"
    "    f()\n"
)


def test_pipeline_collapses_cross_engine_duplicate():
    pytest.importorskip("torch")
    from src.api import analyze, _run_shape_analysis, _run_symexec_analysis

    # Both engines must independently flag the same broadcast site for this test
    # to be meaningful.
    shape = _run_shape_analysis(_DUP_SRC, "m.py")
    sx = _run_symexec_analysis(_DUP_SRC, "m.py")
    if not (shape and sx):
        pytest.skip("engines did not both flag the site in this environment")

    r = analyze(_DUP_SRC, filename="m.py")
    sites = [(b.location.line, b.location.column, b.category) for b in r.bugs]
    # the duplicated (line, col, category) collapses to a single report
    assert len(sites) == len(set(sites))
    assert len(r.bugs) == 1
    # survivor carries the higher (symexec-calibrated) confidence
    assert r.bugs[0].confidence >= max(b.confidence for b in sx)
