"""
Contrastive and social explanation generation for CEGAR verification results.

Implements Miller (2019) "Explanation in Artificial Intelligence: Insights
from the Social Sciences" — explanations should be *contrastive* ("Why X
and not Y?"), *selective* (highlight surprising constraints), and
*social* (calibrated to the developer's epistemic state).

This module lifts CEGAR traces from system logs into developer-facing
narratives by:

1. **ContrastiveExplainer** — generates "Why X and not Y?" explanations
   using Craig interpolants to isolate the distinguishing constraint
   between the buggy configuration (the "fact") and the closest valid
   configuration (the "foil").

2. **ExplanationCalibrator** — models the developer's epistemic state
   to avoid redundant explanations and prioritise surprising constraints.

3. **NarrativeGenerator** — converts raw CEGAR iteration records into
   template-based natural-language narratives suitable for IDE display.

References
----------
* Miller, T. "Explanation in Artificial Intelligence: Insights from the
  Social Sciences", Artificial Intelligence 267 (2019) 1–38.
* Lipton, P. "Contrastive Explanation", Royal Institute of Philosophy
  Supplement 27 (1990) 247–266.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional solver imports
# ---------------------------------------------------------------------------

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

# ---------------------------------------------------------------------------
# Internal imports (graceful)
# ---------------------------------------------------------------------------

from src.shape_cegar import (
    ShapePredicate,
    PredicateKind,
    ShapeCEGARResult,
    CEGARVerdict,
    IterationRecord,
)
from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    SafetyViolation,
    LayerDef,
    LayerKind,
    OpKind,
    extract_computation_graph,
)
from src.cegar_explanation import (
    _layer_description,
    _LAYER_TYPE_NAMES,
    VerificationExplanation,
)

try:
    from src.craig_interpolation import (
        InterpolationPredicateDiscovery,
        InterpolationMethod,
        DimMapping,
        LinearComboPredicate,
    )
    HAS_INTERPOLATION = True
except ImportError:
    HAS_INTERPOLATION = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ContrastiveFoil:
    """A hypothetical valid configuration contrasted against the actual bug.

    Attributes
    ----------
    foil_condition : str
        Human-readable description of the change that would fix the error.
    distinguishing_constraint : str
        The constraint that separates the buggy config from the foil.
    distance : int
        Edit distance (number of dimension changes) from bug to foil.
    foil_dims : Dict[str, int]
        The concrete dimension values in the foil configuration.
    """
    foil_condition: str
    distinguishing_constraint: str
    distance: int
    foil_dims: Dict[str, int] = field(default_factory=dict)


@dataclass
class ContrastiveExplanation:
    """A complete contrastive explanation for a shape error.

    Follows Miller (2019): explains *why the fact and not the foil*.
    """
    fact_description: str
    foil: ContrastiveFoil
    interpolant_constraint: Optional[str] = None
    full_text: str = ""

    def render(self) -> str:
        return self.full_text


@dataclass
class NarrativeStep:
    """A single step in a CEGAR narrative."""
    iteration: int
    text: str
    is_root_cause: bool = False


@dataclass
class CEGARNarrative:
    """A complete natural-language narrative of a CEGAR trace."""
    model_name: str
    steps: List[NarrativeStep] = field(default_factory=list)
    root_cause: Optional[str] = None
    summary: str = ""

    def render(self) -> str:
        lines = [f"TensorGuard verification narrative for {self.model_name}:"]
        lines.append("")
        for step in self.steps:
            prefix = "  → " if step.is_root_cause else "  • "
            lines.append(f"{prefix}{step.text}")
        if self.root_cause:
            lines.append("")
            lines.append(f"  Root cause: {self.root_cause}")
        if self.summary:
            lines.append("")
            lines.append(f"  {self.summary}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "steps": [
                {"iteration": s.iteration, "text": s.text,
                 "is_root_cause": s.is_root_cause}
                for s in self.steps
            ],
            "root_cause": self.root_cause,
            "summary": self.summary,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ContrastiveExplainer
# ═══════════════════════════════════════════════════════════════════════════════

class ContrastiveExplainer:
    """Generates 'Why X and not Y?' explanations for shape errors.

    Given a shape error (the *fact*), identifies the closest valid
    configuration (the *foil*) and uses Craig interpolants to isolate
    the distinguishing constraint.  Output follows the template:

        "The shape error occurs because [fact].
         If instead [foil_condition], the operation would succeed."

    Parameters
    ----------
    timeout_ms : int
        Z3 solver timeout for foil search (default 3000).
    max_foils : int
        Maximum number of foil candidates to generate (default 3).
    """

    def __init__(
        self,
        timeout_ms: int = 3000,
        max_foils: int = 3,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.max_foils = max_foils
        self._interpolation: Optional[Any] = None
        if HAS_INTERPOLATION:
            self._interpolation = InterpolationPredicateDiscovery(
                timeout_ms=timeout_ms,
                method=InterpolationMethod.AUTO,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        cegar_result: ShapeCEGARResult,
        graph: Optional[ComputationGraph] = None,
    ) -> List[ContrastiveExplanation]:
        """Generate contrastive explanations for all bugs in *cegar_result*.

        Returns one ``ContrastiveExplanation`` per real bug found.
        """
        if cegar_result.verdict == CEGARVerdict.SAFE:
            return []

        if graph is None and cegar_result.verification_result:
            graph = cegar_result.verification_result.graph

        explanations: List[ContrastiveExplanation] = []

        # Collect bug violations
        bugs = cegar_result.real_bugs or []
        if not bugs and cegar_result.verification_result:
            vr = cegar_result.verification_result
            if vr.counterexample and vr.counterexample.violations:
                bugs = vr.counterexample.violations

        if not bugs:
            # Synthesise from discovered predicates
            explanations.append(self._explain_from_predicates(
                cegar_result, graph,
            ))
            return explanations

        for bug in bugs:
            expl = self._explain_single_bug(bug, cegar_result, graph)
            explanations.append(expl)

        return explanations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _explain_single_bug(
        self,
        bug: SafetyViolation,
        cegar_result: ShapeCEGARResult,
        graph: Optional[ComputationGraph],
    ) -> ContrastiveExplanation:
        """Build a contrastive explanation for a single safety violation."""
        # --- Fact description ---
        fact_desc = self._describe_fact(bug, graph)

        # --- Find foil (closest valid configuration) ---
        foil = self._find_foil(bug, cegar_result, graph)

        # --- Try Craig interpolant for distinguishing constraint ---
        interpolant_str = self._interpolant_for_bug(bug, cegar_result)

        # --- Assemble full text ---
        full_text = (
            f"The shape error occurs because {fact_desc}. "
            f"If instead {foil.foil_condition}, the operation would succeed."
        )
        if interpolant_str:
            full_text += (
                f" The distinguishing constraint is: {interpolant_str}."
            )

        return ContrastiveExplanation(
            fact_description=fact_desc,
            foil=foil,
            interpolant_constraint=interpolant_str,
            full_text=full_text,
        )

    def _explain_from_predicates(
        self,
        cegar_result: ShapeCEGARResult,
        graph: Optional[ComputationGraph],
    ) -> ContrastiveExplanation:
        """Fallback: explain from discovered predicates when no explicit
        bug violations are available."""
        preds = cegar_result.discovered_predicates
        if preds:
            fact_desc = (
                f"the model requires {preds[0].pretty()} "
                f"but the input does not satisfy this constraint"
            )
            foil_cond = f"the input satisfied {preds[0].pretty()}"
        else:
            fact_desc = "the input shape is incompatible with the model"
            foil_cond = "the input shape matched the model's expectations"

        foil = ContrastiveFoil(
            foil_condition=foil_cond,
            distinguishing_constraint=preds[0].pretty() if preds else "unknown",
            distance=1,
        )
        full_text = (
            f"The shape error occurs because {fact_desc}. "
            f"If instead {foil.foil_condition}, the operation would succeed."
        )
        return ContrastiveExplanation(
            fact_description=fact_desc,
            foil=foil,
            full_text=full_text,
        )

    def _describe_fact(
        self,
        bug: SafetyViolation,
        graph: Optional[ComputationGraph],
    ) -> str:
        """Produce a human-readable description of the bug (the 'fact')."""
        parts: List[str] = []

        # Layer context
        step = bug.step
        if step.layer_ref and graph:
            layer = graph.layers.get(step.layer_ref)
            if layer:
                parts.append(
                    f"layer {step.layer_ref} ({_layer_description(layer)})"
                )
        elif step.layer_ref:
            parts.append(f"layer {step.layer_ref}")

        # Shape details from the violation
        if hasattr(bug, "shape_a") and bug.shape_a and hasattr(bug, "shape_b") and bug.shape_b:
            parts.append(
                f"receives shape {bug.shape_a} "
                f"but expects shape compatible with {bug.shape_b}"
            )
        elif bug.message:
            parts.append(bug.message)
        else:
            parts.append("a shape mismatch was detected")

        return " ".join(parts)

    def _find_foil(
        self,
        bug: SafetyViolation,
        cegar_result: ShapeCEGARResult,
        graph: Optional[ComputationGraph],
    ) -> ContrastiveFoil:
        """Identify the closest valid configuration (the foil).

        Uses discovered predicates and layer definitions to synthesise
        the minimal change that would make the operation succeed.
        """
        step = bug.step

        # Strategy 1: Use layer-level dimension info
        if step.layer_ref and graph:
            layer = graph.layers.get(step.layer_ref)
            if layer:
                return self._foil_from_layer(layer, step, bug)

        # Strategy 2: Use discovered predicates
        preds = cegar_result.discovered_predicates
        if preds:
            return self._foil_from_predicates(preds, bug)

        # Fallback
        return ContrastiveFoil(
            foil_condition="the upstream layer produced a compatible shape",
            distinguishing_constraint="shape compatibility",
            distance=1,
        )

    def _foil_from_layer(
        self,
        layer: LayerDef,
        step: ComputationStep,
        bug: SafetyViolation,
    ) -> ContrastiveFoil:
        """Synthesise a foil from the layer's declared dimensions."""
        if layer.kind == LayerKind.LINEAR and layer.in_features:
            return ContrastiveFoil(
                foil_condition=(
                    f"the input's last dimension were {layer.in_features}"
                ),
                distinguishing_constraint=(
                    f"input.shape[-1] == {layer.in_features}"
                ),
                distance=1,
                foil_dims={"input_last_dim": layer.in_features},
            )
        if layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D) and layer.in_channels:
            return ContrastiveFoil(
                foil_condition=(
                    f"the input had {layer.in_channels} channels"
                ),
                distinguishing_constraint=(
                    f"input.shape[1] == {layer.in_channels}"
                ),
                distance=1,
                foil_dims={"input_channels": layer.in_channels},
            )
        if layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D):
            nf = layer.num_features or layer.params.get("normalized_shape", "?")
            return ContrastiveFoil(
                foil_condition=(
                    f"the input had {nf} features"
                ),
                distinguishing_constraint=(
                    f"input.shape[1] == {nf}"
                ),
                distance=1,
                foil_dims={"input_features": nf} if isinstance(nf, int) else {},
            )

        return ContrastiveFoil(
            foil_condition=(
                f"the input shape matched {_layer_description(layer)}'s requirements"
            ),
            distinguishing_constraint="shape compatibility",
            distance=1,
        )

    def _foil_from_predicates(
        self,
        predicates: List[ShapePredicate],
        bug: SafetyViolation,
    ) -> ContrastiveFoil:
        """Synthesise a foil from CEGAR-discovered predicates."""
        # Pick the most specific predicate
        for pred in predicates:
            if pred.kind == PredicateKind.DIM_EQ and pred.value is not None:
                return ContrastiveFoil(
                    foil_condition=(
                        f"{pred.tensor}.shape[{pred.axis}] were {pred.value}"
                    ),
                    distinguishing_constraint=pred.pretty(),
                    distance=1,
                    foil_dims={f"{pred.tensor}_dim{pred.axis}": pred.value},
                )
            if pred.kind == PredicateKind.DIM_MATCH:
                return ContrastiveFoil(
                    foil_condition=(
                        f"{pred.tensor}.shape[{pred.axis}] equalled "
                        f"{pred.match_tensor}.shape[{pred.match_axis}]"
                    ),
                    distinguishing_constraint=pred.pretty(),
                    distance=1,
                )

        first = predicates[0]
        return ContrastiveFoil(
            foil_condition=f"the constraint {first.pretty()} were satisfied",
            distinguishing_constraint=first.pretty(),
            distance=1,
        )

    def _interpolant_for_bug(
        self,
        bug: SafetyViolation,
        cegar_result: ShapeCEGARResult,
    ) -> Optional[str]:
        """Attempt to compute a Craig interpolant that isolates the
        distinguishing constraint between fact and foil.

        Returns a pretty-printed constraint string or None.
        """
        if not HAS_Z3 or not HAS_INTERPOLATION:
            return None
        if self._interpolation is None:
            return None

        # Use interpolation stats if available from the CEGAR run
        if cegar_result.interpolation_stats:
            succeeded = cegar_result.interpolation_stats.get(
                "interpolations_succeeded", 0
            )
            if succeeded > 0:
                # Predicates discovered via interpolation
                for pred in cegar_result.discovered_predicates:
                    if "interpolation" in pred.provenance or "craig" in pred.provenance:
                        return pred.pretty()

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ExplanationCalibrator
# ═══════════════════════════════════════════════════════════════════════════════

class ExplanationCalibrator:
    """Models developer epistemic state per Miller (2019).

    Tracks which constraints the developer has already seen, avoids
    redundant explanations, and prioritises surprising / unexpected
    constraints.

    Attributes
    ----------
    seen_constraints : set of str
        Pretty-printed constraints already shown to the developer.
    common_knowledge : set of str
        Constraints considered "obvious" (e.g., batch dimension > 0).
    """

    def __init__(self) -> None:
        self.seen_constraints: Set[str] = set()
        self.common_knowledge: Set[str] = {
            "batch > 0",
            "batch >= 1",
            "input.shape[0] > 0",
            "input.shape[0] >= 1",
        }
        self._surprise_scores: Dict[str, float] = {}

    def mark_seen(self, constraint: str) -> None:
        """Record that the developer has seen *constraint*."""
        self.seen_constraints.add(constraint)

    def is_redundant(self, constraint: str) -> bool:
        """True if the developer already knows this constraint."""
        return (
            constraint in self.seen_constraints
            or constraint in self.common_knowledge
        )

    def add_common_knowledge(self, constraint: str) -> None:
        """Mark a constraint as common knowledge (never show)."""
        self.common_knowledge.add(constraint)

    def surprise_score(self, predicate: ShapePredicate) -> float:
        """Compute a surprise score ∈ [0, 1] for a predicate.

        Higher scores indicate more surprising / informative predicates.
        Heuristic based on predicate kind and whether the developer has
        encountered similar constraints before.
        """
        pretty = predicate.pretty()

        # Already seen → zero surprise
        if self.is_redundant(pretty):
            return 0.0

        # Cache
        if pretty in self._surprise_scores:
            return self._surprise_scores[pretty]

        score = 0.5  # baseline

        # Kind-based adjustments
        if predicate.kind == PredicateKind.DIM_DIVISIBLE:
            score += 0.3  # divisibility is often non-obvious
        elif predicate.kind == PredicateKind.DIM_MATCH:
            score += 0.2  # cross-tensor constraints are informative
        elif predicate.kind == PredicateKind.DIM_EQ:
            score += 0.1

        # Provenance-based: interpolation-derived are more surprising
        if "interpolation" in predicate.provenance or "craig" in predicate.provenance:
            score += 0.2

        score = min(score, 1.0)
        self._surprise_scores[pretty] = score
        return score

    def calibrate(
        self,
        explanations: List[ContrastiveExplanation],
    ) -> List[ContrastiveExplanation]:
        """Filter and reorder explanations based on developer state.

        Removes redundant explanations and sorts by surprise score.
        """
        filtered: List[ContrastiveExplanation] = []
        for expl in explanations:
            constraint = expl.foil.distinguishing_constraint
            if not self.is_redundant(constraint):
                filtered.append(expl)
                self.mark_seen(constraint)

        return filtered

    def prioritise_predicates(
        self,
        predicates: List[ShapePredicate],
    ) -> List[ShapePredicate]:
        """Sort predicates by surprise score (highest first).

        Filters out predicates the developer already knows.
        """
        scored = [
            (self.surprise_score(p), p)
            for p in predicates
            if not self.is_redundant(p.pretty())
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  NarrativeGenerator
# ═══════════════════════════════════════════════════════════════════════════════

# Template strings for narrative generation
_TEMPLATES = {
    "discovered_shape_mismatch": (
        "TensorGuard discovered that layer {layer} expects shape {expected} "
        "but receives {actual}"
    ),
    "root_cause": (
        "The root cause is at layer {layer} where {operation} assumes "
        "{assumption}"
    ),
    "predicate_added": (
        "TensorGuard discovered the constraint: {predicate}"
    ),
    "predicate_from_interpolation": (
        "Craig interpolation isolated the constraint: {predicate}"
    ),
    "iteration_safe": (
        "Iteration {n}: all shapes verified — no violations found"
    ),
    "iteration_violations": (
        "Iteration {n}: found {count} shape violation(s), "
        "{spurious} spurious and {real} confirmed"
    ),
    "verdict_safe": (
        "Verification complete: model is shape-safe after {iterations} "
        "iteration(s) and {predicates} discovered constraint(s)"
    ),
    "verdict_unsafe": (
        "Verification complete: confirmed shape bug found after "
        "{iterations} iteration(s)"
    ),
    "verdict_timeout": (
        "Verification inconclusive after {iterations} iteration(s) — "
        "iteration budget exhausted"
    ),
}


class NarrativeGenerator:
    """Converts CEGAR traces into natural-language narratives.

    Uses template-based generation (no LLM needed) to produce
    developer-friendly descriptions of the verification process.

    Parameters
    ----------
    calibrator : ExplanationCalibrator, optional
        If provided, narratives respect the developer's epistemic state.
    """

    def __init__(
        self,
        calibrator: Optional[ExplanationCalibrator] = None,
    ) -> None:
        self.calibrator = calibrator

    def generate(
        self,
        cegar_result: ShapeCEGARResult,
        graph: Optional[ComputationGraph] = None,
        model_name: Optional[str] = None,
    ) -> CEGARNarrative:
        """Convert a CEGAR result into a natural-language narrative.

        Parameters
        ----------
        cegar_result : ShapeCEGARResult
            Result from the CEGAR verification loop.
        graph : ComputationGraph, optional
            Computation graph for layer-level detail.
        model_name : str, optional
            Override model name in the narrative.
        """
        if model_name is None:
            if graph:
                model_name = graph.class_name
            elif (cegar_result.verification_result
                  and cegar_result.verification_result.graph):
                model_name = cegar_result.verification_result.graph.class_name
            else:
                model_name = "Model"

        if graph is None and cegar_result.verification_result:
            graph = cegar_result.verification_result.graph

        steps: List[NarrativeStep] = []
        root_cause: Optional[str] = None

        # --- Per-iteration narrative ---
        for record in cegar_result.iteration_log:
            step_texts = self._narrate_iteration(record, graph)
            for text, is_root in step_texts:
                steps.append(NarrativeStep(
                    iteration=record.iteration + 1,
                    text=text,
                    is_root_cause=is_root,
                ))

        # --- Bug narrative ---
        if cegar_result.real_bugs and graph:
            for bug in cegar_result.real_bugs:
                rc_text = self._narrate_bug(bug, graph)
                if rc_text:
                    root_cause = rc_text
                    steps.append(NarrativeStep(
                        iteration=cegar_result.iterations,
                        text=rc_text,
                        is_root_cause=True,
                    ))

        # --- Verdict summary ---
        summary = self._narrate_verdict(cegar_result)

        return CEGARNarrative(
            model_name=model_name,
            steps=steps,
            root_cause=root_cause,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal template-filling methods
    # ------------------------------------------------------------------

    def _narrate_iteration(
        self,
        record: IterationRecord,
        graph: Optional[ComputationGraph],
    ) -> List[Tuple[str, bool]]:
        """Generate narrative lines for a single CEGAR iteration.

        Returns list of (text, is_root_cause) pairs.
        """
        results: List[Tuple[str, bool]] = []
        n = record.iteration + 1

        if record.num_violations == 0:
            results.append((
                _TEMPLATES["iteration_safe"].format(n=n),
                False,
            ))
        else:
            results.append((
                _TEMPLATES["iteration_violations"].format(
                    n=n,
                    count=record.num_violations,
                    spurious=record.num_spurious,
                    real=record.num_real,
                ),
                False,
            ))

        # Narrate each discovered predicate
        for pred in record.predicates_added:
            pretty = pred.pretty()

            # Respect epistemic calibration
            if self.calibrator and self.calibrator.is_redundant(pretty):
                continue

            if "interpolation" in pred.provenance or "craig" in pred.provenance:
                template = _TEMPLATES["predicate_from_interpolation"]
            else:
                template = _TEMPLATES["predicate_added"]

            results.append((template.format(predicate=pretty), False))

            if self.calibrator:
                self.calibrator.mark_seen(pretty)

        return results

    def _narrate_bug(
        self,
        bug: SafetyViolation,
        graph: ComputationGraph,
    ) -> Optional[str]:
        """Generate a root-cause narrative for a confirmed bug."""
        step = bug.step

        layer_name = step.layer_ref or f"step {bug.step_index}"
        layer = graph.layers.get(step.layer_ref) if step.layer_ref else None

        if layer:
            layer_desc = _layer_description(layer)
            if hasattr(bug, "shape_a") and bug.shape_a and hasattr(bug, "shape_b") and bug.shape_b:
                return _TEMPLATES["discovered_shape_mismatch"].format(
                    layer=f"{layer_name} ({layer_desc})",
                    expected=bug.shape_b,
                    actual=bug.shape_a,
                )

        # Build root cause from bug message
        if bug.message:
            operation = step.op.name if hasattr(step.op, "name") else str(step.op)
            return _TEMPLATES["root_cause"].format(
                layer=layer_name,
                operation=operation,
                assumption=bug.message,
            )

        return None

    def _narrate_verdict(self, cegar_result: ShapeCEGARResult) -> str:
        """Generate the final verdict narrative line."""
        verdict = cegar_result.verdict
        iterations = cegar_result.iterations
        n_preds = len(cegar_result.discovered_predicates)

        if verdict == CEGARVerdict.SAFE:
            return _TEMPLATES["verdict_safe"].format(
                iterations=iterations,
                predicates=n_preds,
            )
        elif verdict == CEGARVerdict.UNSAFE:
            return _TEMPLATES["verdict_unsafe"].format(
                iterations=iterations,
            )
        else:
            return _TEMPLATES["verdict_timeout"].format(
                iterations=iterations,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Convenience entry point
# ═══════════════════════════════════════════════════════════════════════════════

def explain_contrastively(
    cegar_result: ShapeCEGARResult,
    graph: Optional[ComputationGraph] = None,
    model_name: Optional[str] = None,
    calibrator: Optional[ExplanationCalibrator] = None,
) -> Dict[str, Any]:
    """One-call entry point: run contrastive + narrative explanation.

    Returns a dict with keys:
        - ``contrastive``: list of ContrastiveExplanation dicts
        - ``narrative``: CEGARNarrative dict
        - ``calibrated``: bool (whether calibrator was used)
    """
    explainer = ContrastiveExplainer()
    narrator = NarrativeGenerator(calibrator=calibrator)

    contrastive = explainer.explain(cegar_result, graph=graph)

    if calibrator:
        contrastive = calibrator.calibrate(contrastive)

    narrative = narrator.generate(
        cegar_result, graph=graph, model_name=model_name,
    )

    return {
        "contrastive": [
            {
                "fact": e.fact_description,
                "foil_condition": e.foil.foil_condition,
                "distinguishing_constraint": e.foil.distinguishing_constraint,
                "foil_distance": e.foil.distance,
                "interpolant": e.interpolant_constraint,
                "full_text": e.full_text,
            }
            for e in contrastive
        ],
        "narrative": narrative.to_dict(),
        "calibrated": calibrator is not None,
    }
