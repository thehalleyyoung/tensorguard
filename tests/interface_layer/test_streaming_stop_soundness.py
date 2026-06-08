from src.interface_layer.streaming_stop_soundness import (
    StopHazardKind,
    StopSoundnessStatus,
    prove_streaming_stop,
    render_streaming_stop_report_json,
    render_streaming_stop_report_text,
)


def test_overshoot_single_token_carries_past_stop():
    vocab = [(0, "\n\nThe"), (1, "a"), (2, "b")]
    report = prove_streaming_stop(vocab, "\n\n")
    assert report.status is StopSoundnessStatus.HAZARDS_FOUND
    assert report.has_overshoot
    h = next(h for h in report.hazards if h.kind is StopHazardKind.STOP_OVERSHOOT)
    assert h.token_ids == (0,)
    assert "\n\n" in h.decoded


def test_split_stop_across_two_tokens():
    vocab = [(0, "</"), (1, "s>"), (2, "x")]
    report = prove_streaming_stop(vocab, "</s>")
    assert report.status is StopSoundnessStatus.HAZARDS_FOUND
    assert report.has_split
    h = next(h for h in report.hazards if h.kind is StopHazardKind.SPLIT_STOP)
    assert h.decoded.count("</s>") >= 1
    # no single token surface contains the whole stop
    assert all("</s>" not in s for s in h.surfaces)


def test_sound_when_stop_is_exact_suffix_token():
    # Only realization of the stop is the atomic token, ending on a boundary.
    vocab = [(0, "</s>"), (1, "ab"), (2, "c")]
    report = prove_streaming_stop(vocab, "</s>")
    assert report.status is StopSoundnessStatus.PROVEN_SOUND
    assert not report.hazards


def test_single_char_stop_never_splits_but_can_overshoot():
    over = prove_streaming_stop([(0, "\n\n"), (1, "a")], "\n")
    assert over.has_overshoot and not over.has_split
    sound = prove_streaming_stop([(0, "\n"), (1, "ab")], "\n")
    assert sound.status is StopSoundnessStatus.PROVEN_SOUND


def test_empty_inputs_abstain():
    assert prove_streaming_stop([(0, "a")], "").status is StopSoundnessStatus.ABSTAINED
    assert prove_streaming_stop([], "x").status is StopSoundnessStatus.ABSTAINED


def test_split_requires_no_single_token_containment():
    # Both a split route (</ + s>) AND an atomic token exist; overshoot present too
    # via the atomic-containing-trailing token. We just assert split is still found
    # because a cross-token realization is reachable.
    vocab = [(0, "</"), (1, "s>"), (2, "y")]
    report = prove_streaming_stop(vocab, "</s>")
    assert report.has_split


def test_serialization_roundtrip():
    report = prove_streaming_stop([(0, "</"), (1, "s>")], "</s>")
    js = render_streaming_stop_report_json(report)
    assert '"streaming-stop-soundness"' in js
    txt = render_streaming_stop_report_text(report)
    assert "streaming-stop soundness" in txt


def test_witness_decodes_to_contain_stop():
    vocab = [(0, "foo<"), (1, "/tool"), (2, "_call>tail")]
    # stop is "</tool_call>" -> need a leading "<" then "/tool" then "_call>"
    vocab = [(0, "a<"), (1, "/tool"), (2, "_call>")]
    report = prove_streaming_stop(vocab, "</tool_call>")
    assert report.has_split
    h = next(h for h in report.hazards if h.kind is StopHazardKind.SPLIT_STOP)
    assert "</tool_call>" in h.decoded
    assert all("</tool_call>" not in s for s in h.surfaces)
