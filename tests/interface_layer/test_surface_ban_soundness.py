"""Tests for surface-ban soundness (suppress_tokens / bad_words_ids bypass)."""

from __future__ import annotations

import json

from src.interface_layer.surface_ban_soundness import (
    SURFACE_BAN_SOUNDNESS_VERSION,
    BanSoundnessStatus,
    _BadWordAutomaton,
    naive_substring_suppression,
    prove_surface_ban,
    render_surface_ban_report_json,
    render_surface_ban_report_text,
)

# id, surface
VOCAB = [(0, "a"), (1, "b"), (2, "ab"), (3, "c"), (4, "x"), (5, "bomb"), (6, "om")]


def test_suppressing_merged_token_is_bypassed_by_pieces():
    report = prove_surface_ban(VOCAB, "ab", suppressed_ids=[2])
    assert report.status is BanSoundnessStatus.BYPASS_FOUND
    assert report.witness is not None
    assert report.witness.token_ids == (0, 1)
    assert "ab" in report.witness.decoded_text


def test_suppressing_all_constituents_is_sound_unbounded():
    # Ban the merged token AND both single chars -> no allowed sequence spells 'ab'.
    report = prove_surface_ban(VOCAB, "ab", suppressed_ids=[0, 1, 2])
    assert report.status is BanSoundnessStatus.PROVEN_SOUND
    assert report.sound


def test_single_token_containing_target_is_a_bypass():
    # 'om' appears inside the allowed token 'bomb' (id 5).
    report = prove_surface_ban(VOCAB, "om", suppressed_ids=[6])
    assert report.status is BanSoundnessStatus.BYPASS_FOUND
    assert report.witness is not None
    assert "om" in report.witness.decoded_text


def test_bad_word_sequence_blocks_piecewise_but_merged_survives():
    # Ban the id-tuple (a, b); the merged 'ab' token (id 2) still spells it.
    report = prove_surface_ban(VOCAB, "ab", bad_word_id_seqs=[[0, 1]])
    assert report.status is BanSoundnessStatus.BYPASS_FOUND
    assert report.witness is not None
    assert report.witness.token_ids == (2,)


def test_bad_word_plus_suppress_can_be_sound():
    # Minimal vocab: only ways to make 'ab' are merged id2 or pieces (0,1).
    vocab = [(0, "a"), (1, "b"), (2, "ab"), (3, "c")]
    report = prove_surface_ban(vocab, "ab", suppressed_ids=[2], bad_word_id_seqs=[[0, 1]])
    assert report.status is BanSoundnessStatus.PROVEN_SOUND


def test_naive_substring_suppression_lists_containing_tokens():
    ids = naive_substring_suppression(VOCAB, "om")
    # 'om' (id6) and 'bomb' (id5) both contain 'om'.
    assert set(ids) == {5, 6}


def test_bad_word_automaton_masks_only_completing_id():
    ac = _BadWordAutomaton([[0, 1]])
    s0 = 0
    s1 = ac.step(s0, 0)  # 'a' -> partial
    assert s1 is not None and s1 != 0
    assert ac.step(s1, 1) is None  # completing 'b' is masked
    assert ac.step(s0, 1) is not None  # lone 'b' is fine


def test_bad_word_automaton_trivial_when_empty():
    ac = _BadWordAutomaton([])
    assert ac.trivial
    assert ac.step(0, 999) == 0


def test_report_serialization_round_trips():
    report = prove_surface_ban(VOCAB, "ab", suppressed_ids=[2])
    payload = json.loads(render_surface_ban_report_json(report))
    assert payload["version"] == SURFACE_BAN_SOUNDNESS_VERSION
    assert payload["status"] == report.status.value
    assert payload["target"] == "ab"
    text = render_surface_ban_report_text(report)
    assert "surface-ban soundness" in text
    assert "BYPASS" in text


def test_overlapping_bad_word_tuples():
    # Two tuples sharing a prefix; ensure AC fail-links don't crash and both block.
    vocab = [(0, "a"), (1, "b"), (2, "c"), (3, "abc")]
    # Ban (a,b) and (b,c). Spelling 'abc' piecewise hits a banned tuple; merged id3 works.
    report = prove_surface_ban(vocab, "abc", bad_word_id_seqs=[[0, 1], [1, 2]])
    assert report.status is BanSoundnessStatus.BYPASS_FOUND
    assert report.witness.token_ids == (3,)
    # Suppress the merged token too -> sound.
    report2 = prove_surface_ban(vocab, "abc", suppressed_ids=[3], bad_word_id_seqs=[[0, 1], [1, 2]])
    assert report2.status is BanSoundnessStatus.PROVEN_SOUND
