"""
Counterexample-Guided LLM Refinement (Neuro-Symbolic CEGAR Loop).

Implements closed-loop neuro-symbolic coupling (Kautz Type 2/3) where
formal verification output from TensorGuard is fed back to an LLM to
guide code repair.  The loop:

    verify → explain → repair (LLM) → re-verify

Each iteration uses the structured explanation from ``explain_verification()``
(layer-by-layer safety analysis, counterexample traces, CEGAR refinement
predicates) as prompt context, creating a genuine feedback loop between
formal tools and neural generation.

Usage::

    from src.neurosym_refinement import run_neurosym_refinement

    result = run_neurosym_refinement(buggy_source, max_iterations=5)
    print(result.final_verdict, result.fixes_verified)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import textwrap
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.cegar_explanation import (
    VerificationExplanation,
    explain_verification,
)
from src.model_checker import verify_model
from src.shape_cegar import CEGARVerdict
from src.knowledge_base import VerificationKnowledgeBase

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RefinementResult:
    """Result of a neuro-symbolic refinement loop run.

    Attributes
    ----------
    original_verdict : str
        Verdict on the original (unmodified) model code.
    final_verdict : str
        Verdict after the last refinement iteration.
    iterations : int
        Number of refinement iterations performed.
    fixes_proposed : list of str
        Source code of each fix proposed by the LLM.
    fixes_verified : list of bool
        Whether each proposed fix was verified as SAFE.
    llm_model : str
        Name of the LLM model used.
    explanation_trace : list of str
        The rendered explanation fed to the LLM at each iteration.
    """
    original_verdict: str
    final_verdict: str
    iterations: int
    fixes_proposed: List[str] = field(default_factory=list)
    fixes_verified: List[bool] = field(default_factory=list)
    llm_model: str = ""
    explanation_trace: List[str] = field(default_factory=list)
    accumulated_predicates: List[str] = field(default_factory=list)
    knowledge_base: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RefinementResult":
        """Deserialize from a dictionary."""
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM integration
# ═══════════════════════════════════════════════════════════════════════════════

_REPAIR_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a PyTorch model repair assistant. You receive a buggy nn.Module
    class together with a formal verification report from TensorGuard that
    explains exactly which tensor shape constraint is violated and the
    counterexample trace showing the failing dimension assignment.

    Your job: fix the model so that all shape constraints are satisfied.
    Output ONLY the corrected Python class (including imports). No explanation.
""")


def _build_repair_prompt(
    model_source: str,
    explanation: VerificationExplanation,
    iteration_history: Optional[List[Dict[str, Any]]] = None,
    kb_context: str = "",
) -> str:
    """Build the user-side repair prompt with formal verification context.

    When iteration_history is provided, includes prior failed fixes and
    the specific constraints they violated, enabling the LLM to learn
    from prior attempts (Type 2/3 neuro-symbolic coupling).

    When kb_context is provided, includes historical knowledge from
    prior verification sessions for similar architectures.
    """
    parts = [
        "## Buggy Model Code\n```python",
        model_source.strip(),
        "```\n",
        "## TensorGuard Verification Report",
        explanation.render(),
        "",
    ]
    # Include knowledge base context from prior sessions
    if kb_context:
        parts.append("## Historical Knowledge (from prior verification sessions)")
        parts.append(kb_context)
        parts.append("")
    # Include counterexample path details prominently
    if explanation.counterexample_path:
        parts.append("## Counterexample Trace (failing dimension assignment)")
        for line in explanation.counterexample_path:
            parts.append(line)
        parts.append("")
    # Include CEGAR refinement predicates
    if explanation.refinement_trace:
        parts.append("## CEGAR Refinement Predicates Discovered")
        for line in explanation.refinement_trace:
            parts.append(line)
        parts.append("")
    # Include iteration history for Type 2/3 coupling
    if iteration_history:
        parts.append("## Prior Fix Attempts (learn from these failures)")
        for entry in iteration_history:
            parts.append(f"### Attempt {entry.get('iteration', '?')}")
            parts.append(f"Verdict: {entry.get('verdict', 'unknown')}")
            if entry.get('key_violation'):
                parts.append(f"Key violation: {entry['key_violation']}")
            if entry.get('fix_snippet'):
                parts.append(f"Fix attempted:\n```python\n{entry['fix_snippet'][:500]}\n```")
            parts.append("")
    parts.append(
        "Fix the model so that all shape constraints are satisfied. "
        "Output ONLY the corrected Python class (with imports), no explanation."
    )
    return "\n".join(parts)


def _extract_code_block(response: str) -> str:
    """Extract Python code from an LLM response (handles ```python blocks)."""
    # Try to find a fenced code block
    import re
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(response)
    if match:
        return match.group(1).strip()
    # If no code block, return the whole response stripped
    return response.strip()


def _call_openai(
    model_source: str,
    explanation: VerificationExplanation,
    llm_model: str,
    api_key: str,
    iteration_history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Call OpenAI API to get a repair suggestion.

    Returns the extracted code or None on failure.
    """
    try:
        import openai
    except ImportError:
        logger.warning("openai package not installed; cannot call LLM")
        return None

    client = openai.OpenAI(api_key=api_key)
    user_prompt = _build_repair_prompt(model_source, explanation, iteration_history)

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        content = response.choices[0].message.content or ""
        return _extract_code_block(content)
    except Exception as e:
        logger.error("OpenAI API call failed: %s", e)
        return None


def _get_api_key() -> Optional[str]:
    """Try to get OPENAI_API_KEY from environment or bashrc."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    # Try sourcing bashrc
    try:
        result = subprocess.run(
            ["bash", "-c", "source ~/.bashrc 2>/dev/null && echo $OPENAI_API_KEY"],
            capture_output=True, text=True, timeout=5,
        )
        key = result.stdout.strip()
        if key:
            return key
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Core refinement loop
# ═══════════════════════════════════════════════════════════════════════════════

class NeurosymRefinementLoop:
    """Counterexample-guided LLM refinement loop.

    Implements a CEGAR-style loop where:
    1. TensorGuard verifies model source code
    2. If UNSAFE: ``explain_verification()`` generates a structured explanation
    3. The explanation (layer analysis, counterexample trace, CEGAR predicates)
       is fed to an LLM as prompt context
    4. The LLM proposes a fix
    5. The fix is re-verified
    6. Repeat up to ``max_iterations``

    Parameters
    ----------
    max_iterations : int
        Maximum number of verify→explain→repair→re-verify cycles.
    llm_model : str
        OpenAI model name to use for repair suggestions.
    llm_call : callable, optional
        Override for the LLM call function (for testing / mocking).
        Signature: ``(model_source, explanation) -> Optional[str]``
    input_shapes : dict, optional
        Input shapes to pass to ``verify_model`` / ``explain_verification``.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        llm_model: str = "gpt-4.1-nano",
        llm_call: Optional[Callable[[str, VerificationExplanation], Optional[str]]] = None,
        input_shapes: Optional[Dict[str, tuple]] = None,
        knowledge_base_path: Optional[str] = None,
    ):
        self.max_iterations = max_iterations
        self.llm_model = llm_model
        self._llm_call = llm_call
        self.input_shapes = input_shapes
        self.knowledge_base_path = knowledge_base_path
        self._kb: Optional[VerificationKnowledgeBase] = None
        if knowledge_base_path:
            self._kb = VerificationKnowledgeBase.load(knowledge_base_path)

    def _verify_and_explain(self, source: str) -> VerificationExplanation:
        """Run verification and generate explanation."""
        return explain_verification(
            source,
            input_shapes=self.input_shapes,
        )

    def _get_llm_fix(
        self,
        source: str,
        explanation: VerificationExplanation,
        iteration_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Get a fix from the LLM, using the formal explanation as context."""
        if self._llm_call is not None:
            return self._llm_call(source, explanation)
        # Real OpenAI call
        api_key = _get_api_key()
        if not api_key:
            logger.warning("No OPENAI_API_KEY available; cannot call LLM")
            return None
        return _call_openai(source, explanation, self.llm_model, api_key, iteration_history)

    def run(self, model_source: str) -> RefinementResult:
        """Execute the neuro-symbolic refinement loop.

        Implements Type 2/3 neuro-symbolic coupling:
        - Each iteration feeds the formal verification explanation to the LLM
        - Prior failed fix attempts and their specific violations are
          accumulated and included in subsequent prompts
        - CEGAR predicates discovered across iterations are tracked
        - The LLM receives progressively more context about what doesn't work

        Parameters
        ----------
        model_source : str
            Python source code of the (possibly buggy) nn.Module.

        Returns
        -------
        RefinementResult
            Contains the original/final verdicts, proposed fixes,
            accumulated knowledge, and the explanation trace.
        """
        fixes_proposed: List[str] = []
        fixes_verified: List[bool] = []
        explanation_trace: List[str] = []
        accumulated_predicates: List[str] = []
        knowledge_base: List[Dict[str, Any]] = []
        iteration_history: List[Dict[str, Any]] = []

        # Load transferred knowledge from KB if available
        kb_context = ""
        arch_hash = ""
        if self._kb is not None:
            arch_hash = self._kb.compute_arch_hash(model_source)
            transferred = self._kb.lookup(arch_hash)
            if transferred.has_knowledge:
                kb_context = self._kb.get_repair_context(arch_hash)
                # Prime accumulated predicates with transferred ones
                accumulated_predicates.extend(transferred.predicates)
                logger.info(
                    "KB: transferred %d predicates for arch %s",
                    len(transferred.predicates), arch_hash[:12],
                )

        # Initial verification
        explanation = self._verify_and_explain(model_source)
        original_verdict = explanation.verdict
        explanation_trace.append(explanation.render())

        # Accumulate CEGAR predicates from initial verification
        if explanation.refinement_trace:
            accumulated_predicates.extend(explanation.refinement_trace)

        if original_verdict == "SAFE":
            return RefinementResult(
                original_verdict=original_verdict,
                final_verdict="SAFE",
                iterations=0,
                fixes_proposed=fixes_proposed,
                fixes_verified=fixes_verified,
                llm_model=self.llm_model,
                explanation_trace=explanation_trace,
                accumulated_predicates=accumulated_predicates,
                knowledge_base=knowledge_base,
            )

        current_source = model_source
        current_verdict = original_verdict

        for i in range(self.max_iterations):
            # Get LLM fix guided by formal explanation + iteration history
            fix = self._get_llm_fix(current_source, explanation, iteration_history)
            if fix is None:
                break

            fixes_proposed.append(fix)

            # Re-verify the proposed fix
            explanation = self._verify_and_explain(fix)
            current_verdict = explanation.verdict
            explanation_trace.append(explanation.render())

            # Accumulate CEGAR predicates from this iteration
            if explanation.refinement_trace:
                for pred in explanation.refinement_trace:
                    if pred not in accumulated_predicates:
                        accumulated_predicates.append(pred)

            is_safe = current_verdict == "SAFE"
            fixes_verified.append(is_safe)

            # Build knowledge entry for this iteration
            key_violation = ""
            if explanation.counterexample_path:
                key_violation = explanation.counterexample_path[0] if explanation.counterexample_path else ""
            knowledge_entry = {
                "iteration": i + 1,
                "verdict": current_verdict,
                "key_violation": key_violation,
                "fix_snippet": fix[:500],
                "predicates_discovered": len(accumulated_predicates),
            }
            knowledge_base.append(knowledge_entry)

            if not is_safe:
                # Add to iteration history for subsequent prompts
                iteration_history.append(knowledge_entry)

            if is_safe:
                break

            # Update source for next iteration
            current_source = fix

        # Save updated knowledge to KB
        if self._kb is not None and arch_hash:
            failure_modes_list = []
            for entry in knowledge_base:
                if entry.get("verdict") != "SAFE" and entry.get("key_violation"):
                    failure_modes_list.append({
                        "description": entry["key_violation"],
                        "fix_description": "",
                        "predicates_needed": [],
                    })
            self._kb.record(
                arch_hash,
                predicates=accumulated_predicates,
                failure_modes=failure_modes_list if failure_modes_list else None,
            )
            if self.knowledge_base_path:
                self._kb.save(self.knowledge_base_path)

        return RefinementResult(
            original_verdict=original_verdict,
            final_verdict=current_verdict,
            iterations=len(fixes_proposed),
            fixes_proposed=fixes_proposed,
            fixes_verified=fixes_verified,
            llm_model=self.llm_model,
            explanation_trace=explanation_trace,
            accumulated_predicates=accumulated_predicates,
            knowledge_base=knowledge_base,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════════════════

def run_neurosym_refinement(
    model_source: str,
    max_iterations: int = 5,
    llm_model: str = "gpt-4.1-nano",
    llm_call: Optional[Callable[[str, VerificationExplanation], Optional[str]]] = None,
    input_shapes: Optional[Dict[str, tuple]] = None,
) -> RefinementResult:
    """Run counterexample-guided LLM refinement on a model.

    This is a convenience wrapper around ``NeurosymRefinementLoop``.

    Parameters
    ----------
    model_source : str
        Python source of a (possibly buggy) nn.Module.
    max_iterations : int
        Max verify→explain→repair→re-verify cycles (default 5).
    llm_model : str
        OpenAI model to use.  Default ``"gpt-4.1-nano"`` (weak LLM benchmark).
        Also supports ``"gpt-5-chat-latest"`` for stronger results.
    llm_call : callable, optional
        Override for LLM integration (for testing / mocking).
    input_shapes : dict, optional
        Input shapes for verification.

    Returns
    -------
    RefinementResult
    """
    loop = NeurosymRefinementLoop(
        max_iterations=max_iterations,
        llm_model=llm_model,
        llm_call=llm_call,
        input_shapes=input_shapes,
    )
    return loop.run(model_source)
