"""Tests for VerificationKnowledgeBase and cross-session knowledge transfer."""

from __future__ import annotations

import json
import os
import tempfile
import textwrap
from typing import Any, Dict, List

import pytest

import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.knowledge_base import (
    VerificationKnowledgeBase,
    ProofSchema,
    FamilyRecord,
    TransferredKnowledge,
    VerificationStrategy,
    FailureMode,
    compute_arch_hash,
    anti_unify_proof_certificates,
    _extract_layer_sequence,
    _extract_forward_pattern,
)
from src.shape_cegar import ShapeCEGARLoop


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures / helpers
# ═══════════════════════════════════════════════════════════════════════════════

RESNET18_SOURCE = textwrap.dedent("""\
    import torch.nn as nn

    class ResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = x.mean(dim=[2, 3])
            x = self.fc(x)
            return x
""")

RESNET50_SOURCE = textwrap.dedent("""\
    import torch.nn as nn

    class ResNet50(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 256, kernel_size=7)
            self.bn1 = nn.BatchNorm2d(256)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(256, 1000)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = x.mean(dim=[2, 3])
            x = self.fc(x)
            return x
""")

VGG_SOURCE = textwrap.dedent("""\
    import torch.nn as nn

    class VGG(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
            self.fc = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.fc(x)
            return x
""")


def _make_cert(model_name: str, steps: List[Dict], cert_hash: str = "") -> Dict[str, Any]:
    """Helper to create a serialised proof certificate dict."""
    return {
        "model_name": model_name,
        "properties": ["shape_safety"],
        "steps": steps,
        "root_step": len(steps) - 1,
        "theories_used": ["arith"],
        "verification_conditions": [],
        "certificate_hash": cert_hash or f"hash_{model_name}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. KB creation
# ═══════════════════════════════════════════════════════════════════════════════

class TestKBCreation:

    def test_create_empty_kb(self):
        kb = VerificationKnowledgeBase()
        assert kb.family_count == 0
        assert kb.total_predicates == 0

    def test_repr(self):
        kb = VerificationKnowledgeBase()
        assert "0 families" in repr(kb)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Save / Load
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveLoad:

    def test_save_and_load_roundtrip(self, tmp_path):
        kb = VerificationKnowledgeBase()
        kb.record("hash1", predicates=["x.shape[-1] == 768"])
        path = str(tmp_path / "kb.json")
        kb.save(path)

        kb2 = VerificationKnowledgeBase.load(path)
        assert kb2.family_count == 1
        assert kb2.total_predicates == 1

    def test_load_nonexistent_returns_empty(self, tmp_path):
        kb = VerificationKnowledgeBase.load(str(tmp_path / "no_such_file.json"))
        assert kb.family_count == 0

    def test_save_creates_directories(self, tmp_path):
        kb = VerificationKnowledgeBase()
        path = str(tmp_path / "subdir" / "kb.json")
        kb.save(path)
        assert os.path.exists(path)

    def test_load_corrupt_json_returns_empty(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("{{{invalid json")
        kb = VerificationKnowledgeBase.load(path)
        assert kb.family_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Merge
# ═══════════════════════════════════════════════════════════════════════════════

class TestMerge:

    def test_merge_disjoint_families(self):
        kb1 = VerificationKnowledgeBase()
        kb1.record("hash1", predicates=["p1"])
        kb2 = VerificationKnowledgeBase()
        kb2.record("hash2", predicates=["p2"])

        kb1.merge(kb2)
        assert kb1.family_count == 2
        assert kb1.has_family("hash1")
        assert kb1.has_family("hash2")

    def test_merge_overlapping_families(self):
        kb1 = VerificationKnowledgeBase()
        kb1.record("hash1", predicates=["p1", "p2"])
        kb2 = VerificationKnowledgeBase()
        kb2.record("hash1", predicates=["p2", "p3"])

        kb1.merge(kb2)
        assert kb1.family_count == 1
        record = kb1.get_family_record("hash1")
        assert record is not None
        assert set(record.predicates) == {"p1", "p2", "p3"}

    def test_merge_preserves_session_count(self):
        kb1 = VerificationKnowledgeBase()
        kb1.record("h", predicates=["p1"])
        kb2 = VerificationKnowledgeBase()
        kb2.record("h", predicates=["p2"])
        kb1.merge(kb2)
        # Each record() increments session_count by 1
        record = kb1.get_family_record("h")
        assert record is not None
        assert record.session_count >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Architectural pattern hashing
# ═══════════════════════════════════════════════════════════════════════════════

class TestArchHash:

    def test_same_arch_same_hash(self):
        """ResNet-18 and ResNet-50 with same layer skeleton → same hash."""
        h1 = compute_arch_hash(RESNET18_SOURCE)
        h2 = compute_arch_hash(RESNET50_SOURCE)
        assert h1 == h2

    def test_different_arch_different_hash(self):
        """ResNet vs VGG → different hash."""
        h1 = compute_arch_hash(RESNET18_SOURCE)
        h3 = compute_arch_hash(VGG_SOURCE)
        assert h1 != h3

    def test_hash_is_deterministic(self):
        h1 = compute_arch_hash(RESNET18_SOURCE)
        h2 = compute_arch_hash(RESNET18_SOURCE)
        assert h1 == h2

    def test_hash_is_hex_string(self):
        h = compute_arch_hash(RESNET18_SOURCE)
        assert len(h) == 64  # SHA-256
        assert all(c in "0123456789abcdef" for c in h)

    def test_extract_layer_sequence(self):
        layers = _extract_layer_sequence(RESNET18_SOURCE)
        assert "Conv2d" in layers
        assert "BatchNorm2d" in layers
        assert "ReLU" in layers
        assert "Linear" in layers

    def test_extract_forward_pattern(self):
        calls = _extract_forward_pattern(RESNET18_SOURCE)
        assert "conv1" in calls
        assert "bn1" in calls
        assert "fc" in calls

    def test_empty_source_hash(self):
        h = compute_arch_hash("")
        assert isinstance(h, str) and len(h) == 64

    def test_invalid_syntax_hash(self):
        h = compute_arch_hash("def {{{{ invalid")
        assert isinstance(h, str) and len(h) == 64


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Anti-unification over proof schemas
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiUnification:

    def test_anti_unify_identical_certs(self):
        steps = [
            {"rule": "asserted", "conclusion": "x > 0", "premises": []},
            {"rule": "mp", "conclusion": "y > 0", "premises": [0]},
        ]
        cert1 = _make_cert("M1", steps, "h1")
        cert2 = _make_cert("M2", steps, "h2")
        schema = anti_unify_proof_certificates([cert1, cert2])
        assert len(schema.rule_skeleton) == 2
        # No variables needed — steps are identical
        assert all(len(v) == 0 for v in schema.variable_positions)

    def test_anti_unify_different_conclusions(self):
        steps1 = [{"rule": "asserted", "conclusion": "x > 0", "premises": []}]
        steps2 = [{"rule": "asserted", "conclusion": "x > 5", "premises": []}]
        cert1 = _make_cert("M1", steps1, "h1")
        cert2 = _make_cert("M2", steps2, "h2")
        schema = anti_unify_proof_certificates([cert1, cert2])
        assert len(schema.rule_skeleton) == 1
        # The conclusion field should be a variable
        assert schema.variable_positions[0].get("conclusion") is not None

    def test_anti_unify_preserves_common_rules(self):
        steps1 = [
            {"rule": "asserted", "conclusion": "A", "premises": []},
            {"rule": "mp", "conclusion": "B", "premises": [0]},
        ]
        steps2 = [
            {"rule": "asserted", "conclusion": "C", "premises": []},
            {"rule": "mp", "conclusion": "D", "premises": [0]},
        ]
        cert1 = _make_cert("M1", steps1, "h1")
        cert2 = _make_cert("M2", steps2, "h2")
        schema = anti_unify_proof_certificates([cert1, cert2])
        # Rule is common → preserved
        assert schema.rule_skeleton[0]["rule"] == "asserted"
        assert schema.rule_skeleton[1]["rule"] == "mp"
        # Conclusions differ → variables
        assert "conclusion" in schema.variable_positions[0]
        assert "conclusion" in schema.variable_positions[1]

    def test_anti_unify_empty_list(self):
        schema = anti_unify_proof_certificates([])
        assert len(schema.rule_skeleton) == 0

    def test_anti_unify_single_cert(self):
        steps = [{"rule": "asserted", "conclusion": "x > 0", "premises": []}]
        cert = _make_cert("M1", steps, "h1")
        schema = anti_unify_proof_certificates([cert])
        assert len(schema.rule_skeleton) == 1
        assert schema.source_count == 1

    def test_proof_schema_serialization(self):
        schema = ProofSchema(
            rule_skeleton=[{"rule": "mp", "conclusion": "?V0"}],
            variable_positions=[{"conclusion": "?V0"}],
            source_count=3,
            arch_hash="abc123",
        )
        d = schema.to_dict()
        schema2 = ProofSchema.from_dict(d)
        assert schema2.source_count == 3
        assert schema2.arch_hash == "abc123"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Knowledge transfer / lookup
# ═══════════════════════════════════════════════════════════════════════════════

class TestKnowledgeTransfer:

    def test_lookup_empty_kb(self):
        kb = VerificationKnowledgeBase()
        result = kb.lookup("nonexistent")
        assert not result.has_knowledge
        assert result.predicates == []

    def test_record_and_lookup(self):
        kb = VerificationKnowledgeBase()
        kb.record("h1", predicates=["x.shape[-1] == 768", "x.shape[0] >= 1"])
        transferred = kb.lookup("h1")
        assert transferred.has_knowledge
        assert len(transferred.predicates) == 2

    def test_transferred_predicates(self):
        kb = VerificationKnowledgeBase()
        kb.record("h1", predicates=["p1", "p2"])
        preds = kb.get_transferred_predicates("h1")
        assert preds == ["p1", "p2"]

    def test_transferred_predicates_empty(self):
        kb = VerificationKnowledgeBase()
        assert kb.get_transferred_predicates("nope") == []

    def test_record_deduplicates_predicates(self):
        kb = VerificationKnowledgeBase()
        kb.record("h1", predicates=["p1", "p2"])
        kb.record("h1", predicates=["p2", "p3"])
        record = kb.get_family_record("h1")
        assert record is not None
        assert record.predicates == ["p1", "p2", "p3"]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Cross-session predicate transfer
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossSessionTransfer:

    def test_save_load_preserves_predicates(self, tmp_path):
        """Predicates survive serialisation across sessions."""
        kb = VerificationKnowledgeBase()
        arch = compute_arch_hash(RESNET18_SOURCE)
        kb.record(arch, predicates=["x.shape[-1] == 64"])
        path = str(tmp_path / "kb.json")
        kb.save(path)

        kb2 = VerificationKnowledgeBase.load(path)
        transferred = kb2.lookup(arch)
        assert "x.shape[-1] == 64" in transferred.predicates

    def test_same_family_gets_transferred_predicates(self):
        """Recording for ResNet-18 and looking up by ResNet-50 hash works."""
        kb = VerificationKnowledgeBase()
        h18 = compute_arch_hash(RESNET18_SOURCE)
        kb.record(h18, predicates=["x.shape[-1] == 64"])

        h50 = compute_arch_hash(RESNET50_SOURCE)
        # Same arch hash → same family
        assert h18 == h50
        transferred = kb.lookup(h50)
        assert transferred.has_knowledge
        assert "x.shape[-1] == 64" in transferred.predicates


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Integration: neurosym_refinement with KB
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeurosymIntegration:

    def test_refinement_loop_accepts_kb_path(self, tmp_path):
        """NeurosymRefinementLoop accepts knowledge_base_path."""
        from src.neurosym_refinement import NeurosymRefinementLoop

        kb_path = str(tmp_path / "kb.json")
        loop = NeurosymRefinementLoop(
            max_iterations=1,
            knowledge_base_path=kb_path,
            llm_call=lambda s, e: None,
        )
        assert loop.knowledge_base_path == kb_path
        assert loop._kb is not None

    def test_refinement_loop_without_kb(self):
        """Works fine without KB (backward compat)."""
        from src.neurosym_refinement import NeurosymRefinementLoop

        loop = NeurosymRefinementLoop(max_iterations=1, llm_call=lambda s, e: None)
        assert loop._kb is None


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Integration: ShapeCEGARLoop with KB
# ═══════════════════════════════════════════════════════════════════════════════

class TestCEGARIntegration:

    def test_cegar_loop_accepts_knowledge_base(self):
        """ShapeCEGARLoop accepts knowledge_base parameter."""
        kb = VerificationKnowledgeBase()
        loop = ShapeCEGARLoop(
            RESNET18_SOURCE,
            input_shapes={"x": ("batch", 3, 32, 32)},
            knowledge_base=kb,
        )
        assert loop._knowledge_base is kb

    def test_cegar_loop_without_kb(self):
        """Works fine without KB (backward compat)."""
        loop = ShapeCEGARLoop(
            RESNET18_SOURCE,
            input_shapes={"x": ("batch", 3, 32, 32)},
        )
        assert loop._knowledge_base is None


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Repair context and misc
# ═══════════════════════════════════════════════════════════════════════════════

class TestRepairContext:

    def test_repair_context_includes_predicates(self):
        kb = VerificationKnowledgeBase()
        kb.record("h1", predicates=["x.shape[-1] == 768"])
        ctx = kb.get_repair_context("h1")
        assert "x.shape[-1] == 768" in ctx

    def test_repair_context_includes_failure_modes(self):
        kb = VerificationKnowledgeBase()
        kb.record("h1", failure_modes=[{
            "description": "dimension mismatch at fc layer",
            "fix_description": "adjust linear input dim",
        }])
        ctx = kb.get_repair_context("h1")
        assert "dimension mismatch" in ctx
        assert "adjust linear input dim" in ctx

    def test_repair_context_empty_for_unknown(self):
        kb = VerificationKnowledgeBase()
        assert kb.get_repair_context("unknown") == ""

    def test_get_all_arch_hashes(self):
        kb = VerificationKnowledgeBase()
        kb.record("h1", predicates=["p1"])
        kb.record("h2", predicates=["p2"])
        hashes = kb.get_all_arch_hashes()
        assert set(hashes) == {"h1", "h2"}

    def test_family_record_merge(self):
        r1 = FamilyRecord(arch_hash="h1", predicates=["p1"], session_count=1)
        r2 = FamilyRecord(arch_hash="h1", predicates=["p2"], session_count=1)
        r1.merge(r2)
        assert set(r1.predicates) == {"p1", "p2"}
        assert r1.session_count == 2

    def test_record_with_proof_certificate(self):
        kb = VerificationKnowledgeBase()
        cert = _make_cert("M1", [
            {"rule": "asserted", "conclusion": "x>0", "premises": []},
        ], "cert_hash_1")
        kb.record("h1", proof_certificate=cert)
        record = kb.get_family_record("h1")
        assert record is not None
        assert len(record.proof_certificates) == 1

    def test_record_two_certs_creates_schema(self):
        kb = VerificationKnowledgeBase()
        steps1 = [{"rule": "asserted", "conclusion": "A", "premises": []}]
        steps2 = [{"rule": "asserted", "conclusion": "B", "premises": []}]
        kb.record("h1", proof_certificate=_make_cert("M1", steps1, "c1"))
        kb.record("h1", proof_certificate=_make_cert("M2", steps2, "c2"))
        record = kb.get_family_record("h1")
        assert record is not None
        assert record.proof_schema is not None

    def test_record_strategies(self):
        kb = VerificationKnowledgeBase()
        kb.record("h1", strategies=[{"propagator_type": "shape_cegar", "iteration_count": 5}])
        transferred = kb.lookup("h1")
        assert len(transferred.strategies) == 1
        assert transferred.strategies[0]["propagator_type"] == "shape_cegar"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. _parse_predicate_string (in shape_cegar.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParsePredicateString:

    def test_parse_dim_eq(self):
        from src.shape_cegar import _parse_predicate_string, PredicateKind
        p = _parse_predicate_string("x.shape[-1] == 768")
        assert p is not None
        assert p.kind == PredicateKind.DIM_EQ
        assert p.tensor == "x"
        assert p.axis == -1
        assert p.value == 768

    def test_parse_dim_ge(self):
        from src.shape_cegar import _parse_predicate_string, PredicateKind
        p = _parse_predicate_string("x.shape[0] >= 1")
        assert p is not None
        assert p.kind == PredicateKind.DIM_GE

    def test_parse_dim_gt(self):
        from src.shape_cegar import _parse_predicate_string, PredicateKind
        p = _parse_predicate_string("x.shape[2] > 5")
        assert p is not None
        assert p.kind == PredicateKind.DIM_GT

    def test_parse_dim_divisible(self):
        from src.shape_cegar import _parse_predicate_string, PredicateKind
        p = _parse_predicate_string("x.shape[1] % 8 == 0")
        assert p is not None
        assert p.kind == PredicateKind.DIM_DIVISIBLE
        assert p.divisor == 8

    def test_parse_ndim_eq(self):
        from src.shape_cegar import _parse_predicate_string, PredicateKind
        p = _parse_predicate_string("len(x.shape) == 4")
        assert p is not None
        assert p.kind == PredicateKind.NDIM_EQ
        assert p.value == 4

    def test_parse_dim_match(self):
        from src.shape_cegar import _parse_predicate_string, PredicateKind
        p = _parse_predicate_string("x.shape[-1] == w.shape[0]")
        assert p is not None
        assert p.kind == PredicateKind.DIM_MATCH
        assert p.match_tensor == "w"
        assert p.match_axis == 0

    def test_parse_invalid_returns_none(self):
        from src.shape_cegar import _parse_predicate_string
        assert _parse_predicate_string("not a predicate") is None
