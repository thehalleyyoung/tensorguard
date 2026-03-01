"""
Tests for the neuro-symbolic CEGAR refinement loop.

Covers:
  - Refinement loop with mock LLM responses
  - Explanation formatting in prompts
  - Iteration limit enforcement
  - RefinementResult serialization
  - Safe model short-circuit
  - LLM returning None (no fix available)
"""

from __future__ import annotations

import json
import pytest

from src.neurosym_refinement import (
    NeurosymRefinementLoop,
    RefinementResult,
    run_neurosym_refinement,
    _build_repair_prompt,
    _extract_code_block,
)
from src.cegar_explanation import VerificationExplanation


# ═══════════════════════════════════════════════════════════════════════════════
# Test model source snippets
# ═══════════════════════════════════════════════════════════════════════════════

SAFE_MODEL = """\
import torch.nn as nn

class SafeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 256)

    def forward(self, x):
        return self.fc(x)
"""

BUGGY_MODEL = """\
import torch.nn as nn

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
"""

FIXED_MODEL = """\
import torch.nn as nn

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Mock LLM helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_mock_llm(responses):
    """Create a mock LLM callable that returns pre-defined responses in order."""
    call_count = [0]

    def mock_llm(source, explanation):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(responses):
            return responses[idx]
        return None

    return mock_llm


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Refinement loop with mock LLM
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeurosymRefinementLoop:

    def test_safe_model_no_iterations(self):
        """A safe model should short-circuit with 0 iterations."""
        result = run_neurosym_refinement(
            SAFE_MODEL,
            input_shapes={"x": ("batch", 768)},
            llm_call=make_mock_llm([]),
        )
        assert result.original_verdict == "SAFE"
        assert result.final_verdict == "SAFE"
        assert result.iterations == 0
        assert result.fixes_proposed == []
        assert result.fixes_verified == []

    def test_buggy_model_fixed_first_try(self):
        """LLM fixes the bug on the first attempt."""
        mock_llm = make_mock_llm([FIXED_MODEL])
        result = run_neurosym_refinement(
            BUGGY_MODEL,
            input_shapes={"x": ("batch", 768)},
            llm_call=mock_llm,
        )
        assert result.original_verdict in ("UNSAFE", "UNKNOWN", "TIMEOUT")
        assert result.final_verdict == "SAFE"
        assert result.iterations == 1
        assert len(result.fixes_proposed) == 1
        assert result.fixes_verified == [True]

    def test_buggy_model_fixed_second_try(self):
        """LLM fails first attempt, succeeds on second."""
        mock_llm = make_mock_llm([BUGGY_MODEL, FIXED_MODEL])
        result = run_neurosym_refinement(
            BUGGY_MODEL,
            input_shapes={"x": ("batch", 768)},
            llm_call=mock_llm,
        )
        assert result.original_verdict in ("UNSAFE", "UNKNOWN", "TIMEOUT")
        assert result.final_verdict == "SAFE"
        assert result.iterations == 2
        assert len(result.fixes_proposed) == 2
        assert result.fixes_verified[0] is False
        assert result.fixes_verified[1] is True

    def test_iteration_limit_respected(self):
        """Loop should stop after max_iterations even if not fixed."""
        mock_llm = make_mock_llm([BUGGY_MODEL] * 10)
        result = run_neurosym_refinement(
            BUGGY_MODEL,
            max_iterations=3,
            input_shapes={"x": ("batch", 768)},
            llm_call=mock_llm,
        )
        assert result.iterations == 3
        assert result.final_verdict != "SAFE"
        assert len(result.fixes_proposed) == 3

    def test_llm_returns_none_stops_loop(self):
        """If LLM returns None, loop should stop immediately."""
        mock_llm = make_mock_llm([None])
        result = run_neurosym_refinement(
            BUGGY_MODEL,
            input_shapes={"x": ("batch", 768)},
            llm_call=mock_llm,
        )
        assert result.iterations == 0
        assert result.fixes_proposed == []

    def test_explanation_trace_populated(self):
        """Explanation trace should contain at least the initial explanation."""
        mock_llm = make_mock_llm([FIXED_MODEL])
        result = run_neurosym_refinement(
            BUGGY_MODEL,
            input_shapes={"x": ("batch", 768)},
            llm_call=mock_llm,
        )
        # Initial explanation + one re-verification explanation
        assert len(result.explanation_trace) >= 2
        # Each trace entry should be a non-empty string
        for trace in result.explanation_trace:
            assert isinstance(trace, str)
            assert len(trace) > 0

    def test_llm_model_recorded(self):
        """The LLM model name should be recorded in results."""
        result = run_neurosym_refinement(
            SAFE_MODEL,
            input_shapes={"x": ("batch", 768)},
            llm_model="gpt-5-chat-latest",
            llm_call=make_mock_llm([]),
        )
        assert result.llm_model == "gpt-5-chat-latest"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Explanation formatting in prompts
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptFormatting:

    def test_repair_prompt_includes_source(self):
        """Repair prompt should include the original model source."""
        explanation = VerificationExplanation(
            model_name="TestModel",
            verdict="UNSAFE",
            layer_explanations=["✗ Layer fc1 (Linear 768→256): Input requires last dim = 768."],
            counterexample_path=["  Input dimensions: batch=1, d=-1"],
            refinement_trace=["  Iteration 1: Added predicate x.shape[-1] == 768 (from CEGAR discovery)"],
        )
        prompt = _build_repair_prompt(BUGGY_MODEL, explanation)
        assert "class BuggyModel" in prompt
        assert "nn.Linear(768, 256)" in prompt

    def test_repair_prompt_includes_explanation(self):
        """Repair prompt should include TensorGuard explanation."""
        explanation = VerificationExplanation(
            model_name="TestModel",
            verdict="UNSAFE",
            layer_explanations=["✗ Layer fc1 (Linear 768→256): shape mismatch"],
            counterexample_path=["  ✗ Step 2: shape_incompatible"],
            refinement_trace=["  Iteration 1: Added predicate x.shape[-1] == 768"],
        )
        prompt = _build_repair_prompt(BUGGY_MODEL, explanation)
        assert "TensorGuard Verification Report" in prompt
        assert "Counterexample Trace" in prompt
        assert "CEGAR Refinement Predicates" in prompt

    def test_repair_prompt_includes_counterexample_details(self):
        """Counterexample path should be prominently included."""
        explanation = VerificationExplanation(
            model_name="TestModel",
            verdict="UNSAFE",
            counterexample_path=[
                "  Input dimensions: batch=1, dim=100",
                "  ✗ Step 1 (Linear 768→256): shape_incompatible",
                "    Got shape (1, 100), expected compatible with (768,)",
            ],
        )
        prompt = _build_repair_prompt(BUGGY_MODEL, explanation)
        assert "batch=1, dim=100" in prompt
        assert "Got shape (1, 100)" in prompt

    def test_extract_code_block_fenced(self):
        """Should extract code from fenced code blocks."""
        response = '```python\nclass Fixed(nn.Module):\n    pass\n```'
        assert _extract_code_block(response) == "class Fixed(nn.Module):\n    pass"

    def test_extract_code_block_unfenced(self):
        """Should return raw text when no code block present."""
        response = "class Fixed(nn.Module):\n    pass"
        assert _extract_code_block(response) == response


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Result dataclass serialization
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefinementResultSerialization:

    def test_to_dict(self):
        """to_dict should produce a JSON-serializable dictionary."""
        result = RefinementResult(
            original_verdict="UNSAFE",
            final_verdict="SAFE",
            iterations=2,
            fixes_proposed=["fix1", "fix2"],
            fixes_verified=[False, True],
            llm_model="gpt-4.1-nano",
            explanation_trace=["trace1", "trace2", "trace3"],
        )
        d = result.to_dict()
        assert d["original_verdict"] == "UNSAFE"
        assert d["final_verdict"] == "SAFE"
        assert d["iterations"] == 2
        assert len(d["fixes_proposed"]) == 2
        assert d["fixes_verified"] == [False, True]
        # Ensure JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict(r)) should reproduce the original result."""
        original = RefinementResult(
            original_verdict="UNSAFE",
            final_verdict="SAFE",
            iterations=1,
            fixes_proposed=["fix_code"],
            fixes_verified=[True],
            llm_model="gpt-4.1-nano",
            explanation_trace=["t1", "t2"],
        )
        roundtripped = RefinementResult.from_dict(original.to_dict())
        assert roundtripped.original_verdict == original.original_verdict
        assert roundtripped.final_verdict == original.final_verdict
        assert roundtripped.iterations == original.iterations
        assert roundtripped.fixes_proposed == original.fixes_proposed
        assert roundtripped.fixes_verified == original.fixes_verified
        assert roundtripped.llm_model == original.llm_model
        assert roundtripped.explanation_trace == original.explanation_trace

    def test_empty_result_serialization(self):
        """Default/empty result should serialize cleanly."""
        result = RefinementResult(
            original_verdict="SAFE",
            final_verdict="SAFE",
            iterations=0,
        )
        d = result.to_dict()
        assert d["fixes_proposed"] == []
        assert d["fixes_verified"] == []
        json.dumps(d)  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: NeurosymRefinementLoop class directly
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeurosymRefinementLoopClass:

    def test_default_model_name(self):
        """Default LLM model should be gpt-4.1-nano."""
        loop = NeurosymRefinementLoop()
        assert loop.llm_model == "gpt-4.1-nano"

    def test_custom_model_name(self):
        """Should accept custom model names."""
        loop = NeurosymRefinementLoop(llm_model="gpt-5-chat-latest")
        assert loop.llm_model == "gpt-5-chat-latest"

    def test_max_iterations_default(self):
        """Default max_iterations should be 5."""
        loop = NeurosymRefinementLoop()
        assert loop.max_iterations == 5

    def test_run_returns_refinement_result(self):
        """run() should return a RefinementResult."""
        loop = NeurosymRefinementLoop(
            llm_call=make_mock_llm([]),
            input_shapes={"x": ("batch", 768)},
        )
        result = loop.run(SAFE_MODEL)
        assert isinstance(result, RefinementResult)
