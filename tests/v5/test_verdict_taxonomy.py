"""Tests for src/v5/verdict_taxonomy.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from src.v5.verdict_taxonomy import (
    Verdict,
    classify_refutation,
    is_proof_verdict,
    summarize,
)


# ---------------------------------------------------------------------------
# Verdict enum string forms
# ---------------------------------------------------------------------------

def test_verdict_str_forms():
    assert str(Verdict.VERIFIED) == "Verified"
    assert str(Verdict.REFUTED_PROOF) == "Refuted-Proof"
    assert str(Verdict.CONTRACT_VIOLATION) == "Contract-Violation"
    assert str(Verdict.LIBRARY_WARN) == "Library-Warn"
    assert str(Verdict.ABSTAIN) == "Abstain"
    assert str(Verdict.NA) == "N/A"


def test_verdict_enum_members_exist():
    members = {v.name for v in Verdict}
    assert members == {
        "VERIFIED",
        "REFUTED_PROOF",
        "CONTRACT_VIOLATION",
        "LIBRARY_WARN",
        "ABSTAIN",
        "NA",
    }


# ---------------------------------------------------------------------------
# is_proof_verdict
# ---------------------------------------------------------------------------

def test_is_proof_verdict_true_only_for_refuted_proof():
    assert is_proof_verdict(Verdict.REFUTED_PROOF) is True


def test_is_proof_verdict_false_for_others():
    for v in Verdict:
        if v is not Verdict.REFUTED_PROOF:
            assert is_proof_verdict(v) is False, f"Expected False for {v}"


# ---------------------------------------------------------------------------
# classify_refutation — bug-corpus item
# ---------------------------------------------------------------------------

def test_classify_refutation_bug_corpus_item_is_refuted_proof():
    item = {
        "id": "bug_003",
        "category": "view_reshape_total_size",
        "is_buggy_gt": True,
        "bucket": "Refuted",
    }
    assert classify_refutation(item) is Verdict.REFUTED_PROOF


def test_classify_refutation_bug_corpus_flag_via_corpus_field():
    item = {"id": "bug_042", "corpus": "bug", "bucket": "Refuted"}
    assert classify_refutation(item) is Verdict.REFUTED_PROOF


# ---------------------------------------------------------------------------
# classify_refutation — block-corpus CONTRACT_VIOLATION
# ---------------------------------------------------------------------------

def test_classify_refutation_config_pattern_contract_violation():
    item = {
        "id": "transformers__BertAttention__abc",
        "library": "transformers",
        "category": "attention",
        "bucket": "Refuted",
    }
    source = (
        "class BertAttention(nn.Module):\n"
        "    def __init__(self, config):\n"
        "        self.num_heads = self.config.num_attention_heads\n"
        "        self.head_dim = self.config.hidden_size // self.num_heads\n"
    )
    assert classify_refutation(item, source=source) is Verdict.CONTRACT_VIOLATION


def test_classify_refutation_kwargs_pattern_contract_violation():
    item = {"id": "some__Mod__xyz", "library": "diffusers", "bucket": "Refuted"}
    source = "def forward(self, x, **kwargs):\n    return self.layer(x)\n"
    assert classify_refutation(item, source=source) is Verdict.CONTRACT_VIOLATION


def test_classify_refutation_getattr_config_contract_violation():
    item = {"id": "mod__Foo__111", "bucket": "Refuted"}
    source = "hidden = getattr(self.config, 'hidden_size', 768)\n"
    assert classify_refutation(item, source=source) is Verdict.CONTRACT_VIOLATION


# ---------------------------------------------------------------------------
# classify_refutation — block-corpus LIBRARY_WARN
# ---------------------------------------------------------------------------

def test_classify_refutation_no_config_patterns_library_warn():
    item = {
        "id": "torchvision__AnyStage__8a5bd23f",
        "library": "torchvision",
        "category": "vision_cnn",
        "bucket": "Refuted",
    }
    source = (
        "class AnyStage(nn.Sequential):\n"
        "    def __init__(self, width_in, width_out, stride, depth, block_constructor):\n"
        "        super().__init__()\n"
        "        for i in range(depth):\n"
        "            self.add_module(str(i), block_constructor(width_in, width_out, stride))\n"
    )
    assert classify_refutation(item, source=source) is Verdict.LIBRARY_WARN


def test_classify_refutation_no_source_defaults_library_warn():
    item = {
        "id": "torchvision__SomeLayer__999",
        "library": "torchvision",
        "category": "vision_cnn",
        "bucket": "Refuted",
    }
    assert classify_refutation(item) is Verdict.LIBRARY_WARN


# ---------------------------------------------------------------------------
# summarize round trip
# ---------------------------------------------------------------------------

def test_summarize_round_trip():
    verdicts = [
        Verdict.REFUTED_PROOF,
        Verdict.REFUTED_PROOF,
        Verdict.CONTRACT_VIOLATION,
        Verdict.LIBRARY_WARN,
        Verdict.VERIFIED,
        Verdict.ABSTAIN,
    ]
    counts = summarize(verdicts)
    assert counts["Refuted-Proof"] == 2
    assert counts["Contract-Violation"] == 1
    assert counts["Library-Warn"] == 1
    assert counts["Verified"] == 1
    assert counts["Abstain"] == 1
    assert "N/A" not in counts


def test_summarize_empty():
    assert summarize([]) == {}
