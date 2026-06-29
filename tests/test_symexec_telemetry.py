"""Tests for telemetry & confidence feedback (roadmap Step 87).

Telemetry must be (1) opt-in — a disabled sink ingests nothing; (2) anonymous —
records carry no source content; (3) advisory & side-effect-free — enabling it
never changes analysis or fingerprints.  Plus the calibration maths (reliability
diagram, ECE, Brier score, per-kind precision, prior-adjustment suggestions) and
the string-in/string-out serialization must be correct and deterministic.
"""

import pytest

from src.symexec import analyze_source, SymConfig
from src.symexec.bugs import SymBug, SymBugKind
from src.symexec.telemetry import (
    CalibrationRecord,
    CalibrationReport,
    PriorSuggestion,
    ReliabilityBin,
    TelemetrySink,
    brier_score,
    calibration_report,
    expected_calibration_error,
    precision_by_kind,
    records_from_jsonl,
    records_to_jsonl,
    reliability_table,
    suggest_prior_adjustments,
)


_PROVEN_SRC = """
import torch
def f():
    a = torch.randn(2, 3)
    b = torch.randn(4, 5)
    return a @ b
"""


def _bug(kind=SymBugKind.MATMUL_DIM_MISMATCH, conf=0.9, evidence=None):
    return SymBug(
        kind=kind, message="m", line=1, col=0, function="f",
        confidence=conf, evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# Opt-in behaviour                                                            #
# --------------------------------------------------------------------------- #

def test_sink_disabled_by_default():
    assert TelemetrySink().enabled is False


def test_disabled_sink_ingests_nothing():
    sink = TelemetrySink()
    sink.record_bug(_bug())
    assert sink.record_result(analyze_source(_PROVEN_SRC)) == 0
    assert sink.records == []


def test_enable_then_record():
    sink = TelemetrySink().enable()
    r = analyze_source(_PROVEN_SRC)
    n = sink.record_result(r, outcome=True)
    assert n == len(r.bugs) >= 1
    assert all(rec.outcome is True for rec in sink.records)


def test_disable_stops_ingestion():
    sink = TelemetrySink(enabled=True)
    sink.record_bug(_bug())
    sink.disable()
    sink.record_bug(_bug())
    assert len(sink.records) == 1


def test_clear():
    sink = TelemetrySink(enabled=True)
    sink.record_bug(_bug())
    sink.clear()
    assert sink.records == []


# --------------------------------------------------------------------------- #
# Anonymity & evidence extraction                                             #
# --------------------------------------------------------------------------- #

def test_record_carries_no_source_content():
    rec = CalibrationRecord.from_bug(_bug())
    d = rec.to_dict()
    assert set(d) == {
        "kind", "confidence", "has_witness", "has_certificate",
        "has_minimal_trace", "outcome",
    }
    # no file/line/col/message anywhere
    assert "message" not in d and "line" not in d


def test_evidence_flags_extracted():
    rec = CalibrationRecord.from_bug(
        _bug(evidence="concrete counterexample x=1; minimal failing conditions: a")
    )
    assert rec.has_witness is True
    assert rec.has_minimal_trace is True
    assert rec.has_certificate is False


def test_kind_is_enum_value_string():
    rec = CalibrationRecord.from_bug(_bug(kind=SymBugKind.BROADCAST_MISMATCH))
    assert rec.kind == "broadcast_mismatch"


def test_with_outcome():
    rec = CalibrationRecord.from_bug(_bug())
    assert rec.outcome is None
    assert rec.with_outcome(True).outcome is True
    assert rec.with_outcome(False).outcome is False


# --------------------------------------------------------------------------- #
# Side-effect freedom: telemetry never changes analysis                       #
# --------------------------------------------------------------------------- #

def test_telemetry_does_not_change_fingerprint():
    before = analyze_source(_PROVEN_SRC).fingerprint()
    sink = TelemetrySink(enabled=True)
    r = analyze_source(_PROVEN_SRC)
    sink.record_result(r, outcome=True)
    after = analyze_source(_PROVEN_SRC).fingerprint()
    assert before == after == r.fingerprint()


# --------------------------------------------------------------------------- #
# Calibration metrics                                                         #
# --------------------------------------------------------------------------- #

def _rec(kind, conf, outcome):
    return CalibrationRecord(kind=kind, confidence=conf, outcome=outcome)


def test_reliability_table_buckets_and_precision():
    recs = [_rec("k", 0.95, True) for _ in range(8)] + [
        _rec("k", 0.95, False) for _ in range(2)
    ]
    table = reliability_table(recs, bins=10)
    assert len(table) == 1
    b = table[0]
    assert b.count == 10
    assert b.lo == pytest.approx(0.9)
    assert b.hi == pytest.approx(1.0)
    assert b.mean_confidence == pytest.approx(0.95)
    assert b.empirical_precision == pytest.approx(0.8)


def test_reliability_ignores_unlabelled():
    recs = [_rec("k", 0.9, True), CalibrationRecord("k", 0.9, outcome=None)]
    table = reliability_table(recs)
    assert sum(b.count for b in table) == 1


def test_confidence_one_falls_in_last_bucket():
    table = reliability_table([_rec("k", 1.0, True)], bins=10)
    assert len(table) == 1
    assert table[0].lo == pytest.approx(0.9)


def test_ece_perfect_calibration_is_zero():
    # mean confidence 0.8 with exactly 80% precision -> ECE 0
    recs = [_rec("k", 0.8, True) for _ in range(8)] + [
        _rec("k", 0.8, False) for _ in range(2)
    ]
    assert expected_calibration_error(recs) == pytest.approx(0.0)


def test_ece_none_without_labels():
    assert expected_calibration_error([CalibrationRecord("k", 0.9)]) is None


def test_brier_score():
    recs = [_rec("k", 1.0, True), _rec("k", 0.0, False)]
    assert brier_score(recs) == pytest.approx(0.0)
    recs2 = [_rec("k", 0.0, True)]
    assert brier_score(recs2) == pytest.approx(1.0)


def test_brier_none_without_labels():
    assert brier_score([CalibrationRecord("k", 0.5)]) is None


def test_precision_by_kind():
    recs = [
        _rec("a", 0.9, True), _rec("a", 0.9, False),
        _rec("b", 0.5, False),
    ]
    by = precision_by_kind(recs)
    assert by["a"].count == 2 and by["a"].empirical_precision == pytest.approx(0.5)
    assert by["b"].empirical_precision == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Prior-adjustment suggestions (the feedback loop)                            #
# --------------------------------------------------------------------------- #

def test_suggestions_respect_min_count():
    recs = [_rec("a", 0.9, False) for _ in range(3)]
    assert suggest_prior_adjustments(recs, min_count=5) == []
    out = suggest_prior_adjustments(recs, min_count=3)
    assert len(out) == 1


def test_suggestion_delta_sign_and_order():
    recs = (
        [_rec("over", 0.9, False) for _ in range(10)]          # over-confident
        + [_rec("under", 0.6, True) for _ in range(10)]        # under-confident
    )
    out = suggest_prior_adjustments(recs, min_count=5)
    by = {s.kind: s for s in out}
    assert by["over"].delta == pytest.approx(-0.9)   # precision 0 - conf 0.9
    assert by["under"].delta == pytest.approx(0.4)   # precision 1 - conf 0.6
    # ordered by descending |delta|
    assert out[0].kind == "over"


# --------------------------------------------------------------------------- #
# Bundled report                                                              #
# --------------------------------------------------------------------------- #

def test_calibration_report_bundle():
    recs = [_rec("k", 0.9, True) for _ in range(6)] + [
        _rec("k", 0.9, False) for _ in range(4)
    ]
    rep = calibration_report(recs)
    assert isinstance(rep, CalibrationReport)
    assert rep.total == 10 and rep.labelled == 10
    assert rep.ece is not None and rep.brier is not None
    assert "telemetry: 10 records" in rep.summary()


def test_report_summary_handles_no_labels():
    rep = calibration_report([CalibrationRecord("k", 0.9)])
    assert rep.labelled == 0
    assert "ECE=n/a" in rep.summary() and "Brier=n/a" in rep.summary()


# --------------------------------------------------------------------------- #
# Serialization round-trip                                                    #
# --------------------------------------------------------------------------- #

def test_jsonl_round_trip():
    recs = [
        CalibrationRecord("a", 0.9, has_witness=True, outcome=True),
        CalibrationRecord("b", 0.5, has_minimal_trace=True, outcome=False),
        CalibrationRecord("c", 0.7, outcome=None),
    ]
    text = records_to_jsonl(recs)
    assert text.count("\n") == 2  # 3 records, 2 separators
    assert records_from_jsonl(text) == recs


def test_jsonl_tolerates_blank_lines():
    recs = [CalibrationRecord("a", 0.9, outcome=True)]
    assert records_from_jsonl(records_to_jsonl(recs) + "\n\n") == recs


def test_jsonl_is_deterministic():
    recs = [CalibrationRecord("a", 0.9, outcome=True)]
    assert records_to_jsonl(recs) == records_to_jsonl(recs)


def test_reliability_rejects_nonpositive_bins():
    with pytest.raises(ValueError):
        reliability_table([_rec("k", 0.5, True)], bins=0)


# --------------------------------------------------------------------------- #
# End-to-end with the engine                                                  #
# --------------------------------------------------------------------------- #

def test_end_to_end_calibration_from_results():
    sink = TelemetrySink(enabled=True)
    # Confirmed real bug.
    sink.record_result(analyze_source(_PROVEN_SRC), outcome=True)
    # A heuristic suspicion labelled a false positive.
    heur = analyze_source(
        "import torch\ndef g(n):\n    return torch.randn(7) + torch.randn(n)\n",
        config=SymConfig.heuristic(),
    )
    sink.record_result(heur, outcome=False)
    rep = sink.report()
    assert rep.total == len(sink.records) >= 2
    assert rep.labelled == rep.total
    # the heuristic suspicion is over-confident relative to its (false) outcome
    assert rep.brier is not None
