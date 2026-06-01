"""Step 14 regression tests -- latent-bug recall above the strongest baseline.

These tests pin the invariants that make the hard-recall comparison fair and
meaningful:

* every model in the corpus is a *genuine* bug (the latent fault really fails
  when exercised);
* the strongest dynamic baseline (``runtime_backward``) is *silent* on every
  model -- i.e. it predicts clean -- so the corpus really is "latent";
* TensorGuard's recall is *strictly greater* than the baseline's;
* every model TensorGuard misses carries a root-cause tag;
* the generator and the on-disk artifact are deterministic.
"""

from __future__ import annotations

import os

import pytest

from evaluation import hard_recall


@pytest.fixture(scope="module")
def artifact():
    return hard_recall.run(check=False)


def test_corpus_nonempty_and_three_families():
    corpus = hard_recall.build_corpus()
    assert len(corpus) >= 6
    fams = {m["family"] for m in corpus}
    assert fams == {"phase_eval", "path_flag", "silent_freeze"}


def test_every_model_is_a_genuine_bug():
    for model in hard_recall.build_corpus():
        genuine, detail = hard_recall.is_genuine_bug(model)
        assert genuine, "%s should be a genuine bug, got %s" % (model["id"], detail)


def test_strongest_baseline_is_silent_on_all():
    """The whole point: a single seeded train()+backward sees none of these."""
    for model in hard_recall.build_corpus():
        buggy, detail = hard_recall.runtime_backward_silent(model)
        assert not buggy, "%s must be silent under runtime_backward (%s)" % (
            model["id"], detail)


def test_tensorguard_recall_strictly_above_baseline(artifact):
    s = artifact["summary"]
    assert s["runtime_backward_recall"] == 0.0
    assert s["tensorguard_recall"] > s["runtime_backward_recall"]
    assert s["recall_advantage"] == pytest.approx(
        s["tensorguard_recall"] - s["runtime_backward_recall"])


def test_tensorguard_catches_phase_and_path_families(artifact):
    by_fam = artifact["by_family"]
    # Static analysis sees both eval-only and untaken-branch faults in full.
    assert by_fam["phase_eval"]["tg_caught"] == by_fam["phase_eval"]["total"]
    assert by_fam["path_flag"]["tg_caught"] == by_fam["path_flag"]["total"]


def test_every_miss_is_root_cause_tagged(artifact):
    for entry in artifact["per_model"]:
        if not entry["tensorguard"]["caught"]:
            assert "tensorguard_miss_root_cause" in entry
            assert entry["tensorguard_miss_root_cause"].strip()
    for miss in artifact["tensorguard_misses"]:
        assert miss["root_cause"].strip()


def test_silent_freeze_misses_have_specific_tag(artifact):
    misses = {m["id"]: m for m in artifact["tensorguard_misses"]}
    for sid in ("silent_freeze_fc1", "silent_freeze_conv"):
        assert sid in misses
        assert "requires_grad" in misses[sid]["root_cause"]


def test_generator_is_deterministic():
    a = hard_recall.build_corpus()
    b = hard_recall.build_corpus()
    assert [m["id"] for m in a] == [m["id"] for m in b]
    assert [m["source"] for m in a] == [m["source"] for m in b]


def test_artifact_on_disk_is_up_to_date():
    """`--check` must pass against the committed JSON (byte-identical)."""
    assert os.path.exists(hard_recall.OUT_JSON)
    hard_recall.run(check=True)
