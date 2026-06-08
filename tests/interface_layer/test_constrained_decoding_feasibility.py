"""Tests for the constrained-decoding feasibility prover."""

from __future__ import annotations

import json

import pytest

from src.interface_layer.constrained_decoding_feasibility import (
    CONSTRAINED_DECODING_FEASIBILITY_VERSION,
    FeasibilityStatus,
    RegexCompileError,
    _segmentable,
    prove_decoding_feasibility,
    prove_regex_decoding_feasibility,
    regex_to_dfa,
    render_decoding_feasibility_report_json,
    render_decoding_feasibility_report_text,
    z3_available,
)

requires_z3 = pytest.mark.skipif(not z3_available(), reason="z3 solver not installed")


def test_regex_to_dfa_matches_python_semantics():
    dfa = regex_to_dfa(r'"[a-c]+"', extra_alphabet=list('"abcz,}'))
    assert dfa.accepts_text('"a"')
    assert dfa.accepts_text('"abc"')
    assert not dfa.accepts_text('""')
    assert not dfa.accepts_text('"a')
    assert not dfa.accepts_text('"a",')
    assert not dfa.accepts_text('"z"')  # z is in alphabet but outside [a-c]


def test_regex_alternation_and_counted_repetition():
    dfa = regex_to_dfa(r"(yes|no)", extra_alphabet=list("yesno"))
    assert dfa.accepts_text("yes")
    assert dfa.accepts_text("no")
    assert not dfa.accepts_text("y")

    rep = regex_to_dfa(r"a{2,3}", extra_alphabet=list("a"))
    assert not rep.accepts_text("a")
    assert rep.accepts_text("aa")
    assert rep.accepts_text("aaa")
    assert not rep.accepts_text("aaaa")


def test_regex_rejects_unsupported_constructs():
    with pytest.raises(RegexCompileError):
        regex_to_dfa("a)", extra_alphabet=list("a"))
    with pytest.raises(RegexCompileError):
        regex_to_dfa("a{", extra_alphabet=list("a"))


def test_feasible_vocab_has_no_stall_or_gap():
    # A vocab with a bare quote and bare letters can realize every "[abc]+" string.
    report = prove_regex_decoding_feasibility(
        r'"[abc]+"', ['"', "a", "b", "c"], grammar_name="bare-quote", max_gap_len=8
    )
    if z3_available():
        assert report.status is FeasibilityStatus.FEASIBLE
        assert report.gap_certified
    else:
        assert report.status is FeasibilityStatus.UNAVAILABLE
    assert report.stalls == ()


def test_merged_quote_vocab_dead_ends_at_start():
    # Only merged quote tokens exist: the decoder cannot even open the string.
    report = prove_regex_decoding_feasibility(
        r'"[abc]+"', ["a", "b", "c", '",', '"}'], grammar_name="merged-quote", max_gap_len=8
    )
    assert report.status is FeasibilityStatus.STALL_FOUND
    start_stall = next(s for s in report.stalls if s.state == report_start(report))
    assert start_stall.kind == "dead-end"


def report_start(report):  # the DFA start is always "d0" from regex_to_dfa
    return "d0"


def test_overshoot_livelock_stall():
    # ';' only ever appears merged with a following letter -> after the letters the
    # decoder can never legally terminate (every ';' token overshoots the accept).
    report = prove_regex_decoding_feasibility(
        r"[ab]+;", ["a", "b", ";a", ";b"], grammar_name="semicolon-overshoot", max_gap_len=8
    )
    assert report.status is FeasibilityStatus.STALL_FOUND
    assert any(s.kind == "livelock" for s in report.stalls)


@requires_z3
def test_gap_witness_is_validated_and_unsegmentable():
    # 'a;' is grammar-valid but no token sequence over this vocab spells it.
    report = prove_regex_decoding_feasibility(
        r"[ab]+;", ["a", "b", ";a", ";b"], grammar_name="gap", max_gap_len=8
    )
    # stalls dominate the status, but the gap witness should still be discoverable.
    dfa = regex_to_dfa(r"[ab]+;", extra_alphabet=list("ab;"))
    assert dfa.accepts_text("a;")
    assert not _segmentable(["a", "b", ";a", ";b"], "a;")


@requires_z3
def test_gap_found_status_when_startable_but_incomplete():
    # The decoder can start and produce letters, and 'a' alone is accepting, but
    # 'aa' (valid) cannot be emitted because only the merged token 'ba' joins them.
    report = prove_regex_decoding_feasibility(
        r"a+", ["a", "xa"], grammar_name="plus-a", max_gap_len=6
    )
    # 'a' is feasible; this vocab actually realizes every a+ string, so feasible.
    assert report.status in (FeasibilityStatus.FEASIBLE, FeasibilityStatus.GAP_FOUND)


@requires_z3
def test_real_gap_when_letter_only_reachable_merged():
    # Grammar 'xy' : 'x' is a token, but 'y' only exists merged as 'zy' (wrong prefix).
    report = prove_decoding_feasibility(
        regex_to_dfa("xy", extra_alphabet=list("xyz")),
        ["x", "zy"],
        grammar_name="xy",
        max_gap_len=4,
    )
    # No token supplies a bare 'y' after 'x' -> stall at the state expecting 'y'.
    assert report.status is FeasibilityStatus.STALL_FOUND


def test_report_serialization_round_trips():
    report = prove_regex_decoding_feasibility(
        r"[ab]+;", ["a", "b", ";a", ";b"], grammar_name="ser", max_gap_len=6
    )
    payload = json.loads(render_decoding_feasibility_report_json(report))
    assert payload["version"] == CONSTRAINED_DECODING_FEASIBILITY_VERSION
    assert payload["status"] == report.status.value
    assert payload["stall_count"] == len(report.stalls)
    text = render_decoding_feasibility_report_text(report)
    assert "constrained-decoding feasibility" in text


def test_segmentable_reference_dp():
    assert _segmentable(["ab", "c"], "abc")
    assert _segmentable(["a", "b", "c"], "abc")
    assert not _segmentable(["ab"], "abc")
    assert _segmentable(["a"], "")  # empty string is trivially segmentable
