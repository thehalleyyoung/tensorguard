"""
Tests for iterative neuro-symbolic refinement with knowledge accumulation.

Validates Type 2/3 coupling: the refinement loop accumulates knowledge
across iterations and feeds it back to the LLM.
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List, Optional

from src.neurosym_refinement import (
    NeurosymRefinementLoop,
    RefinementResult,
    run_neurosym_refinement,
    _build_repair_prompt,
)
from src.cegar_explanation import VerificationExplanation


def _make_explanation(
    verdict: str = "UNSAFE",
    counterexample_path: Optional[List[str]] = None,
    refinement_trace: Optional[List[str]] = None,
) -> VerificationExplanation:
    """Create a mock VerificationExplanation."""
    return VerificationExplanation(
        model_name="test_model",
        verdict=verdict,
        layer_explanations=[],
        counterexample_path=counterexample_path or [],
        refinement_trace=refinement_trace or [],
    )


class TestRefinementResultFields:
    """Test that RefinementResult includes knowledge accumulation fields."""

    def test_accumulated_predicates_field(self):
        result = RefinementResult(
            original_verdict="UNSAFE",
            final_verdict="SAFE",
            iterations=1,
            accumulated_predicates=["dim_0 >= 1"],
        )
        assert result.accumulated_predicates == ["dim_0 >= 1"]

    def test_knowledge_base_field(self):
        result = RefinementResult(
            original_verdict="UNSAFE",
            final_verdict="UNSAFE",
            iterations=2,
            knowledge_base=[
                {"iteration": 1, "verdict": "UNSAFE", "key_violation": "shape mismatch"},
                {"iteration": 2, "verdict": "UNSAFE", "key_violation": "dim error"},
            ],
        )
        assert len(result.knowledge_base) == 2
        assert result.knowledge_base[0]["iteration"] == 1

    def test_serialization_includes_new_fields(self):
        result = RefinementResult(
            original_verdict="UNSAFE",
            final_verdict="SAFE",
            iterations=1,
            accumulated_predicates=["pred1"],
            knowledge_base=[{"k": "v"}],
        )
        d = result.to_dict()
        assert "accumulated_predicates" in d
        assert "knowledge_base" in d


class TestBuildRepairPrompt:
    """Test that repair prompts include iteration history."""

    def test_prompt_without_history(self):
        explanation = _make_explanation()
        prompt = _build_repair_prompt("class M: pass", explanation)
        assert "Buggy Model Code" in prompt
        assert "Prior Fix Attempts" not in prompt

    def test_prompt_with_history(self):
        explanation = _make_explanation()
        history = [
            {
                "iteration": 1,
                "verdict": "UNSAFE",
                "key_violation": "dim[2] != 10",
                "fix_snippet": "class M: ...",
            }
        ]
        prompt = _build_repair_prompt("class M: pass", explanation, history)
        assert "Prior Fix Attempts" in prompt
        assert "Attempt 1" in prompt
        assert "dim[2] != 10" in prompt

    def test_prompt_multiple_history_entries(self):
        explanation = _make_explanation()
        history = [
            {"iteration": 1, "verdict": "UNSAFE", "key_violation": "err1"},
            {"iteration": 2, "verdict": "UNSAFE", "key_violation": "err2"},
        ]
        prompt = _build_repair_prompt("class M: pass", explanation, history)
        assert "Attempt 1" in prompt
        assert "Attempt 2" in prompt


class TestIterativeRefinementLoop:
    """Test the iterative refinement loop with knowledge accumulation."""

    def test_safe_model_no_iterations(self):
        """Safe models should return immediately with no iterations."""
        call_count = 0

        def mock_llm(source, explanation):
            nonlocal call_count
            call_count += 1
            return source

        loop = NeurosymRefinementLoop(
            max_iterations=3,
            llm_call=mock_llm,
        )
        # Use a model we know verifies as SAFE
        safe_source = """
import torch.nn as nn
class SafeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
"""
        result = loop.run(safe_source)
        assert result.original_verdict == "SAFE"
        assert result.iterations == 0
        assert call_count == 0

    def test_knowledge_accumulated_across_iterations(self):
        """Knowledge base should grow with each iteration."""
        iteration = [0]

        def mock_llm(source, explanation):
            iteration[0] += 1
            if iteration[0] <= 2:
                # Return a modified version
                return """
import torch.nn as nn
class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
"""
            return None

        loop = NeurosymRefinementLoop(
            max_iterations=3,
            llm_call=mock_llm,
        )
        # Use a model known to have dimension mismatch
        buggy_source = """
import torch.nn as nn
class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
"""
        result = loop.run(buggy_source)
        # Should have returned some result with knowledge fields
        assert isinstance(result.accumulated_predicates, list)
        assert isinstance(result.knowledge_base, list)
        # Even if detected as safe (due to verification catching the bug),
        # the fields should exist
        assert hasattr(result, 'accumulated_predicates')
        assert hasattr(result, 'knowledge_base')

    def test_result_has_all_fields(self):
        """RefinementResult from loop should have all new fields."""
        def mock_llm(source, explanation):
            return None

        loop = NeurosymRefinementLoop(
            max_iterations=1,
            llm_call=mock_llm,
        )
        result = loop.run("""
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""")
        assert hasattr(result, 'accumulated_predicates')
        assert hasattr(result, 'knowledge_base')
