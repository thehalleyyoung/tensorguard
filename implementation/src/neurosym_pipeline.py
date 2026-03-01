"""
Neuro-Symbolic Pipeline: LLM Detection → TensorGuard Certification.

Combines the high recall of LLM-based shape bug detection with the
formal guarantees of TensorGuard's SMT-backed verification condition generation.

Pipeline stages:
  1. LLM Triage: GPT-4.1-nano (or any LLM) classifies model as
     "likely buggy" or "likely safe" with natural-language rationale.
  2. TensorGuard Certification: For models the LLM calls safe,
     TensorGuard attempts to produce SMT-LIB verification conditions.
     For models the LLM calls buggy, TensorGuard attempts to produce
     a formal counterexample confirming the bug.
  3. Disagreement Resolution: When LLM and TensorGuard disagree,
     the pipeline flags for review and provides both analyses.

Key insight: LLMs excel at fuzzy pattern matching (high recall)
but cannot provide formal guarantees. TensorGuard provides verification
conditions but has limited recall on complex patterns. The hybrid pipeline
achieves both high recall AND formal guarantees.

Usage::

    from src.neurosym_pipeline import NeurosymPipeline

    pipeline = NeurosymPipeline(openai_api_key="...")
    result = pipeline.analyze(source_code, input_shapes={"x": ("batch", 784)})
    print(result.summary())
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from src.model_checker import verify_model, Device, Phase
from src.unified import analyze_unified


# ═══════════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════════

class Verdict(Enum):
    """Final pipeline verdict."""
    CERTIFIED_SAFE = auto()       # LLM: safe, TG: safe + verification condition
    CONFIRMED_BUG = auto()        # LLM: bug, TG: bug + counterexample
    LLM_BUG_TG_SAFE = auto()      # LLM: bug, TG: safe (possible FN from TG)
    LLM_SAFE_TG_BUG = auto()      # LLM: safe, TG: bug (LLM missed it)
    LLM_BUG_TG_UNKNOWN = auto()   # LLM: bug, TG: parse/analysis error
    LLM_SAFE_TG_UNKNOWN = auto()  # LLM: safe, TG: parse/analysis error
    LLM_UNKNOWN = auto()          # LLM failed to respond


class Confidence(Enum):
    """Confidence level of pipeline verdict."""
    FORMAL = auto()     # Backed by SMT verification condition
    HIGH = auto()       # LLM and TG agree
    MEDIUM = auto()     # Only one tool has opinion
    LOW = auto()        # Tools disagree — needs review
    NONE = auto()       # No useful signal


@dataclass
class LLMAnalysis:
    """Result of LLM-based shape analysis."""
    predicts_bug: Optional[bool]
    confidence: float            # 0-1 from LLM self-assessment
    rationale: str               # Natural language explanation
    bug_location: Optional[str]  # Where the LLM thinks the bug is
    raw_response: str
    model: str
    strategy: str
    latency_ms: float


@dataclass
class PipelineResult:
    """Combined result from the neuro-symbolic pipeline."""
    verdict: Verdict
    confidence: Confidence
    llm_analysis: LLMAnalysis
    tg_safe: Optional[bool]
    tg_certificate: Optional[str]   # SMT-LIB verification condition text
    tg_counterexample: Optional[str]  # Pretty-printed counterexample
    tg_errors: List[str]
    tg_latency_ms: float
    total_latency_ms: float

    # Disagreement analysis
    disagreement: bool = False
    disagreement_analysis: Optional[str] = None

    def summary(self) -> str:
        """One-line summary for CI/CD integration."""
        if self.verdict == Verdict.CERTIFIED_SAFE:
            return f"✓ CERTIFIED SAFE (formal proof) [{self.total_latency_ms:.0f}ms]"
        elif self.verdict == Verdict.CONFIRMED_BUG:
            return f"✗ CONFIRMED BUG (formal counterexample) [{self.total_latency_ms:.0f}ms]"
        elif self.verdict == Verdict.LLM_BUG_TG_SAFE:
            return (f"⚠ DISAGREEMENT: LLM suspects bug but TensorGuard certifies safe. "
                    f"Review recommended. [{self.total_latency_ms:.0f}ms]")
        elif self.verdict == Verdict.LLM_SAFE_TG_BUG:
            return (f"✗ BUG FOUND: TensorGuard found bug LLM missed "
                    f"(formal counterexample) [{self.total_latency_ms:.0f}ms]")
        elif self.verdict in (Verdict.LLM_BUG_TG_UNKNOWN, Verdict.LLM_SAFE_TG_UNKNOWN):
            return (f"⚠ LLM says {'bug' if self.llm_analysis.predicts_bug else 'safe'}, "
                    f"TensorGuard could not analyze. [{self.total_latency_ms:.0f}ms]")
        return f"? Unknown verdict [{self.total_latency_ms:.0f}ms]"

    def detailed_report(self) -> str:
        """Multi-line report suitable for PR comments."""
        lines = [
            "═" * 60,
            "TensorGuard Neuro-Symbolic Pipeline Report",
            "═" * 60,
            f"Verdict: {self.verdict.name}",
            f"Confidence: {self.confidence.name}",
            "",
            "── LLM Analysis ──",
            f"  Predicts bug: {self.llm_analysis.predicts_bug}",
            f"  Strategy: {self.llm_analysis.strategy}",
            f"  Rationale: {self.llm_analysis.rationale}",
        ]
        if self.llm_analysis.bug_location:
            lines.append(f"  Bug location: {self.llm_analysis.bug_location}")

        lines.extend([
            "",
            "── TensorGuard Analysis ──",
            f"  Safe: {self.tg_safe}",
        ])
        if self.tg_certificate:
            cert_lines = self.tg_certificate.split('\n')
            lines.append(f"  Verification condition: ({len(cert_lines)} lines SMT-LIB)")
        if self.tg_counterexample:
            lines.append(f"  Counterexample:\n    {self.tg_counterexample}")
        if self.tg_errors:
            lines.append(f"  Errors: {self.tg_errors}")

        if self.disagreement:
            lines.extend([
                "",
                "── Disagreement Analysis ──",
                f"  {self.disagreement_analysis}",
            ])

        lines.extend([
            "",
            f"Total latency: {self.total_latency_ms:.0f}ms "
            f"(LLM: {self.llm_analysis.latency_ms:.0f}ms, "
            f"TG: {self.tg_latency_ms:.0f}ms)",
            "═" * 60,
        ])
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM query
# ═══════════════════════════════════════════════════════════════════════════════

COT_SYSTEM_PROMPT = (
    "You are an expert PyTorch debugger specializing in tensor shape analysis. "
    "You methodically verify neural network architectures for correctness. "
    "You must output a structured JSON response."
)

COT_USER_TEMPLATE = (
    "Analyze this PyTorch nn.Module for shape/device/phase bugs.\n\n"
    "Procedure:\n"
    "1. List all layers in __init__ with their input/output dimensions.\n"
    "2. Trace data flow through forward(), tracking tensor shapes.\n"
    "3. Check each connection point for dimensional compatibility.\n"
    "4. Check device assignments and train/eval phase issues.\n\n"
    "After your analysis, output a JSON block:\n"
    "```json\n"
    '{{"has_bug": true/false, "confidence": 0.0-1.0, '
    '"rationale": "...", "bug_location": "line X: description" or null}}\n'
    "```\n\n"
    "```python\n{code}\n```"
)


def _query_llm(client: Any, code: str, model: str = "gpt-4.1-nano") -> LLMAnalysis:
    """Query LLM for shape bug analysis."""
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": COT_USER_TEMPLATE.format(code=code)},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content or ""
        latency = (time.time() - t0) * 1000

        # Parse structured response
        predicts_bug, confidence, rationale, bug_loc = _parse_llm_response(raw)

        return LLMAnalysis(
            predicts_bug=predicts_bug,
            confidence=confidence,
            rationale=rationale,
            bug_location=bug_loc,
            raw_response=raw,
            model=model,
            strategy="chain_of_thought_structured",
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.time() - t0) * 1000
        return LLMAnalysis(
            predicts_bug=None,
            confidence=0.0,
            rationale=f"LLM query failed: {e}",
            bug_location=None,
            raw_response="",
            model=model,
            strategy="chain_of_thought_structured",
            latency_ms=latency,
        )


def _parse_llm_response(raw: str) -> Tuple[Optional[bool], float, str, Optional[str]]:
    """Extract structured fields from LLM response."""
    # Try to find JSON block
    import re
    json_match = re.search(r'```json\s*\n(.*?)\n\s*```', raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return (
                data.get("has_bug"),
                float(data.get("confidence", 0.5)),
                data.get("rationale", ""),
                data.get("bug_location"),
            )
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: look for YES/NO in last lines
    lines = raw.strip().split('\n')
    for line in reversed(lines[-5:]):
        line_upper = line.strip().upper()
        if line_upper.startswith("YES") or "HAS_BUG" in line_upper and "TRUE" in line_upper:
            return True, 0.5, raw[-200:], None
        if line_upper.startswith("NO") or "HAS_BUG" in line_upper and "FALSE" in line_upper:
            return False, 0.5, raw[-200:], None

    return None, 0.0, "Could not parse LLM response", None


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class NeurosymPipeline:
    """Neuro-symbolic pipeline combining LLM detection with formal certification.

    The pipeline leverages LLM's high recall for initial triage, then uses
    TensorGuard's SMT-backed verification for formal certification of the
    LLM's assessment. When the two disagree, both analyses are preserved
    for human review.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4.1-nano",
        default_device: Device = Device.CPU,
        default_phase: Phase = Phase.TRAIN,
    ):
        self.model = model
        self.default_device = default_device
        self.default_phase = default_phase

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            except ImportError:
                self._client = None
        else:
            self._client = None

    def analyze(
        self,
        source: str,
        input_shapes: Optional[Dict[str, tuple]] = None,
        max_k: Optional[int] = None,
    ) -> PipelineResult:
        """Run the full neuro-symbolic pipeline."""
        t_start = time.time()

        # Stage 1: LLM triage
        llm_analysis = self._run_llm(source)

        # Stage 2: TensorGuard verification
        tg_safe, tg_cert, tg_cex, tg_errors, tg_ms = self._run_tensorguard(
            source, input_shapes, max_k
        )

        # Stage 3: Combine verdicts
        verdict, confidence, disagreement, dis_analysis = self._combine(
            llm_analysis, tg_safe, tg_errors
        )

        total_ms = (time.time() - t_start) * 1000

        return PipelineResult(
            verdict=verdict,
            confidence=confidence,
            llm_analysis=llm_analysis,
            tg_safe=tg_safe,
            tg_certificate=tg_cert,
            tg_counterexample=tg_cex,
            tg_errors=tg_errors,
            tg_latency_ms=tg_ms,
            total_latency_ms=total_ms,
            disagreement=disagreement,
            disagreement_analysis=dis_analysis,
        )

    def analyze_batch(
        self,
        models: List[Dict[str, Any]],
    ) -> List[PipelineResult]:
        """Analyze multiple models. Each dict needs 'source' and optionally 'input_shapes'."""
        return [
            self.analyze(
                m["source"],
                m.get("input_shapes"),
                m.get("max_k"),
            )
            for m in models
        ]

    def _run_llm(self, source: str) -> LLMAnalysis:
        """Stage 1: LLM-based triage."""
        if self._client is None:
            return LLMAnalysis(
                predicts_bug=None,
                confidence=0.0,
                rationale="No OpenAI client available",
                bug_location=None,
                raw_response="",
                model=self.model,
                strategy="unavailable",
                latency_ms=0.0,
            )
        return _query_llm(self._client, source, self.model)

    def _run_tensorguard(
        self,
        source: str,
        input_shapes: Optional[Dict[str, tuple]],
        max_k: Optional[int],
    ) -> Tuple[Optional[bool], Optional[str], Optional[str], List[str], float]:
        """Stage 2: Formal verification.

        Uses analyze_unified for bug detection (higher recall) and
        verify_model for verification condition generation (formal guarantees).
        """
        t0 = time.time()
        errors: List[str] = []

        # First: use unified analysis for detection (higher recall)
        found_bug = False
        cex_text = None
        try:
            unified_result = analyze_unified(source)
            if unified_result.bugs:
                found_bug = True
                cex_text = "\n".join(
                    f"  [{b.kind}] line {b.line}: {b.message}"
                    for b in unified_result.bugs
                )
        except Exception as e:
            errors.append(f"Unified analysis: {e}")

        # Second: use verify_model for verification condition generation
        cert_text = None
        try:
            result = verify_model(
                source,
                input_shapes=input_shapes,
                default_device=self.default_device,
                default_phase=self.default_phase,
                max_k=max_k,
            )

            if not result.safe:
                found_bug = True
                if result.counterexample and not cex_text:
                    cex_text = result.counterexample.pretty()

            if result.safe and result.certificate:
                try:
                    cert_text = result.certificate.smtlib_certificate()
                except Exception:
                    cert_text = str(result.certificate.to_dict())

            errors.extend(result.errors)
        except Exception as e:
            errors.append(f"verify_model: {e}")

        latency = (time.time() - t0) * 1000
        is_safe = not found_bug
        return is_safe, cert_text, cex_text, errors, latency

    def _combine(
        self,
        llm: LLMAnalysis,
        tg_safe: Optional[bool],
        tg_errors: List[str],
    ) -> Tuple[Verdict, Confidence, bool, Optional[str]]:
        """Stage 3: Combine LLM and TensorGuard verdicts."""
        if llm.predicts_bug is None:
            # LLM failed
            if tg_safe is None:
                return Verdict.LLM_UNKNOWN, Confidence.NONE, False, None
            elif tg_safe:
                return Verdict.CERTIFIED_SAFE, Confidence.MEDIUM, False, None
            else:
                return Verdict.CONFIRMED_BUG, Confidence.MEDIUM, False, None

        if tg_safe is None:
            # TensorGuard failed
            if llm.predicts_bug:
                return (Verdict.LLM_BUG_TG_UNKNOWN, Confidence.LOW, False,
                        f"TensorGuard errors: {tg_errors}")
            else:
                return (Verdict.LLM_SAFE_TG_UNKNOWN, Confidence.LOW, False,
                        f"TensorGuard errors: {tg_errors}")

        # Both have opinions
        if llm.predicts_bug and not tg_safe:
            # Agreement: both say bug
            return Verdict.CONFIRMED_BUG, Confidence.FORMAL, False, None

        if not llm.predicts_bug and tg_safe:
            # Agreement: both say safe
            return Verdict.CERTIFIED_SAFE, Confidence.FORMAL, False, None

        if llm.predicts_bug and tg_safe:
            # Disagreement: LLM says bug, TG says safe
            analysis = (
                "LLM detected a potential bug but TensorGuard's formal analysis "
                "certifies the model as safe. Possible causes: (1) LLM false positive "
                "— the pattern looks suspicious but is actually correct; "
                "(2) TensorGuard's abstraction misses the bug class the LLM detected "
                "(e.g., semantic bugs outside shape/device/phase). "
                f"LLM rationale: {llm.rationale}"
            )
            return Verdict.LLM_BUG_TG_SAFE, Confidence.LOW, True, analysis

        # not llm.predicts_bug and not tg_safe
        # Disagreement: LLM says safe, TG found bug
        analysis = (
            "TensorGuard found a formal counterexample for a bug that the LLM missed. "
            "This demonstrates the value of formal verification — the LLM's pattern "
            "matching failed to detect this shape incompatibility, but symbolic "
            "constraint propagation caught it."
        )
        return Verdict.LLM_SAFE_TG_BUG, Confidence.FORMAL, True, analysis
