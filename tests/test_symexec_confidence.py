"""Step 63 — confidence calibration.

Confidence is now a principled, evidence-driven calibration applied at the
``_emit`` choke point: a detector's declared value is a *prior* (the structural
certainty of the check) that corroborating evidence — an unconditional fault, an
SMT-forced path, a concrete witness, an algebraic certificate, a minimised
trace — can only *raise*, capped strictly below 1.0.  These tests verify the
pure calibration model and its end-to-end effect, and confirm calibration never
changes *which* bugs report (soundness/presentation separation).
"""

from __future__ import annotations

from src.symexec.bugs import SymBugKind
from src.symexec.confidence import (
    MAX_CONFIDENCE,
    ConfidenceSignals,
    calibrate,
)
from src.symexec.engine import analyze_source


# -- pure calibration model ---------------------------------------------


def test_no_signals_returns_prior_unchanged():
    assert calibrate(0.9, ConfidenceSignals()) == 0.9
    assert calibrate(0.85, ConfidenceSignals()) == 0.85


def test_calibration_is_monotone_in_evidence():
    prior = 0.85
    base = calibrate(prior, ConfidenceSignals())
    one = calibrate(prior, ConfidenceSignals(witness=True))
    two = calibrate(prior, ConfidenceSignals(witness=True, certificate=True))
    assert base < one < two


def test_calibration_never_below_prior():
    for sig in [
        ConfidenceSignals(),
        ConfidenceSignals(unconditional=True),
        ConfidenceSignals(witness=True, certificate=True, minimal_trace=True),
    ]:
        assert calibrate(0.9, sig) >= 0.9


def test_calibration_capped_below_one():
    maxed = calibrate(
        0.99,
        ConfidenceSignals(
            unconditional=True,
            smt_forced=True,
            witness=True,
            certificate=True,
            minimal_trace=True,
        ),
    )
    assert maxed == MAX_CONFIDENCE
    assert maxed < 1.0


def test_each_signal_contributes_a_positive_boost():
    prior = 0.8
    for kwargs in [
        {"unconditional": True},
        {"smt_forced": True},
        {"witness": True},
        {"certificate": True},
        {"minimal_trace": True},
    ]:
        assert calibrate(prior, ConfidenceSignals(**kwargs)) > prior


def test_certificate_outweighs_minimal_trace():
    prior = 0.8
    cert = calibrate(prior, ConfidenceSignals(certificate=True))
    trace = calibrate(prior, ConfidenceSignals(minimal_trace=True))
    assert cert > trace


# -- from_evidence derivation -------------------------------------------


def test_from_evidence_detects_witness_and_trace():
    ev = "concrete counterexample: shapes (3,) and (2,); minimal failing conditions: a != b"
    sig = ConfidenceSignals.from_evidence(
        ev, has_path_constraints=True, smt_checked=True
    )
    assert sig.witness and sig.minimal_trace and sig.smt_forced
    assert not sig.unconditional
    assert not sig.certificate


def test_from_evidence_detects_certificate():
    sig = ConfidenceSignals.from_evidence(
        "certified counterexample: a.shape=(2, 3) @ b.shape=(4, 5)",
        has_path_constraints=False,
        smt_checked=False,
    )
    assert sig.certificate
    assert sig.unconditional  # no path constraints


def test_from_evidence_none_evidence():
    sig = ConfidenceSignals.from_evidence(
        None, has_path_constraints=False, smt_checked=False
    )
    assert not sig.witness and not sig.certificate and not sig.minimal_trace
    assert sig.unconditional


# -- end-to-end ----------------------------------------------------------


def test_unconditional_concrete_bug_gets_a_boost():
    # A forced concrete broadcast failure with no path constraints: the
    # "unconditional" signal raises confidence above the 0.9 prior.
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(3)\n"
        "    b = torch.zeros(2)\n"
        "    return a + b\n"
    )
    res = analyze_source(src)
    bcast = [b for b in res.bugs if b.kind is SymBugKind.BROADCAST_MISMATCH]
    assert bcast
    assert bcast[0].confidence > 0.9
    assert bcast[0].confidence <= MAX_CONFIDENCE


def test_matmul_certificate_stays_high_confidence():
    # Concrete matmul mismatch carries a concretization certificate → high.
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(4, 5)\n"
        "    return a @ b\n"
    )
    res = analyze_source(src)
    mm = [b for b in res.bugs if b.kind is SymBugKind.MATMUL_DIM_MISMATCH]
    assert mm
    assert mm[0].confidence >= 0.95


def test_calibration_does_not_change_which_bugs_report():
    # The set of reported bug kinds is identical to the structural expectation;
    # calibration only touches confidence values.
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(3)\n"
        "    b = torch.zeros(2)\n"
        "    return a + b\n"
    )
    res = analyze_source(src)
    assert {b.kind for b in res.bugs} == {SymBugKind.BROADCAST_MISMATCH}
    # all confidences are valid probabilities strictly below the reserved 1.0
    for b in res.bugs:
        assert 0.0 < b.confidence <= MAX_CONFIDENCE


def test_syntax_error_keeps_full_confidence():
    # The "won't even parse" defect is reported with confidence 1.0, outside the
    # calibration path (reserved ceiling).
    res = analyze_source("def f(:\n    pass\n")
    assert res.bugs
    assert res.bugs[0].confidence == 1.0
