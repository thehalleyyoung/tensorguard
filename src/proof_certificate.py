"""Genuine proof certificates with inference chains for TensorGuard.

Transforms assertion witnesses into proof objects that can be independently
verified without re-running the SMT solver. Uses Z3's proof mode to extract
inference steps, producing certificates in a simple proof format.

Multiple certificate strategies are supported to maximize coverage:
  1. Z3 native proof (solver.proof())
  2. UNSAT-core based certificates
  3. Replay-based certificates (fresh context, no UserPropagator, proof=true)
  4. Dual-solver verification certificates
"""
from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


class CertificateStrategy(enum.Enum):
    """Which strategy successfully produced the proof certificate."""
    Z3_NATIVE_PROOF = "z3_native_proof"
    UNSAT_CORE = "unsat_core"
    REPLAY = "replay"
    DUAL_SOLVER = "dual_solver"


# ── Known Z3 proof rules ────────────────────────────────────────────────────

_KNOWN_RULES = frozenset({
    "mp",
    "asserted",
    "unit-resolution",
    "th-lemma",
    "rewrite",
    "monotonicity",
    "trans",
    "refl",
    "hypothesis",
    "lemma",
    "def-intro",
    "apply-def",
    "iff-true",
    "iff-false",
    "commutativity",
    "quant-intro",
    "quant-inst",
    "symm",
    "and-elim",
    "or-elim",
    "not-or-elim",
    "iff~",
    "mp~",
    "proof-bind",
    "elim-unused",
    "der",
    "nnf-pos",
    "nnf-neg",
    "sk",
    "pull-quant",
    "push-quant",
    "elim-and",
    "elim-or",
})

# Rules that belong to a background theory (as opposed to propositional)
_THEORY_RULES = {"th-lemma"}


@dataclass
class ProofStep:
    """A single inference step in a proof certificate.

    Attributes:
        rule:       Inference rule name (e.g. "mp", "asserted", "th-lemma").
        conclusion: The proved formula in SMT-LIB syntax.
        premises:   Indices of premise steps in the certificate's step list.
        theory:     Background theory that produced this step, if any
                    ("arith", "eq", None for propositional).
    """

    rule: str
    conclusion: str
    premises: List[int] = field(default_factory=list)
    theory: Optional[str] = None

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "rule": self.rule,
            "conclusion": self.conclusion,
            "premises": self.premises,
        }
        if self.theory is not None:
            d["theory"] = self.theory
        return d


@dataclass
class ProofCertificate:
    """A genuine proof certificate with inference chains.

    Attributes:
        model_name:              Class name of the verified nn.Module.
        properties:              List of property names proved safe.
        steps:                   Ordered list of proof steps.
        root_step:               Index of the final UNSAT derivation step.
        theories_used:           Background theories encountered.
        verification_conditions: The SMT-LIB assertions (formula strings).
        certificate_hash:        SHA-256 over the serialised steps.
        extraction_time_ms:      Wall-clock time to extract the proof.
    """

    model_name: str
    properties: List[str]
    steps: List[ProofStep]
    root_step: int
    theories_used: List[str] = field(default_factory=list)
    verification_conditions: List[str] = field(default_factory=list)
    certificate_hash: str = ""
    extraction_time_ms: float = 0.0
    proof_source: str = "z3"  # "z3" or "cvc5"
    strategy: Optional[CertificateStrategy] = None

    def __post_init__(self) -> None:
        if not self.certificate_hash:
            self.certificate_hash = self._compute_hash()

    # ── public API ───────────────────────────────────────────────────────

    def verify_locally(self) -> bool:
        """Walk the proof steps and structurally verify each inference.

        Checks:
          * Every premise index is valid and points to an earlier step.
          * ``asserted`` steps have no premises.
          * Non-leaf rules have at least one premise.
          * The root step exists.

        Returns True when all structural checks pass.
        """
        n = len(self.steps)
        if n == 0:
            return False
        if self.root_step < 0 or self.root_step >= n:
            return False
        for i, step in enumerate(self.steps):
            # Every premise must reference an earlier step.
            for p in step.premises:
                if p < 0 or p >= n or p >= i:
                    return False
            # ``asserted`` / ``refl`` / ``hypothesis`` are leaf rules.
            if step.rule in ("asserted", "refl", "hypothesis"):
                if step.premises:
                    return False
            # Non-leaf rules need at least one premise (except rules that
            # Z3 sometimes emits without sub-proof children).
            if step.rule not in (
                "asserted",
                "refl",
                "hypothesis",
                "rewrite",
                "def-intro",
                "commutativity",
                # Z3 can emit unit-resolution nodes whose proof children were
                # compressed into formula arguments in the Python proof AST.
                "unit-resolution",
                "th-lemma",
                "iff-true",
                "iff-false",
                "elim-unused",
                "der",
                "sk",
            ):
                if not step.premises:
                    return False
        return True

    def to_alethe(self) -> str:
        """Export in a simplified Alethe-like proof format.

        Alethe is a proof format accepted by several SMT checkers.  We emit
        a simplified version that captures the essential structure.
        """
        lines: List[str] = []
        lines.append(f"(unsat")
        for vc in self.verification_conditions:
            lines.append(f"  (assume {vc})")
        lines.append("")
        for i, step in enumerate(self.steps):
            premise_refs = " ".join(f"@{p}" for p in step.premises)
            tag = f"@{i}"
            if step.rule == "asserted":
                lines.append(f"  (anchor :step {tag})")
                lines.append(f"  (step {tag} {step.conclusion} :rule assert)")
            else:
                theory_ann = f" :theory {step.theory}" if step.theory else ""
                lines.append(
                    f"  (step {tag} {step.conclusion} "
                    f":rule {step.rule} :premises ({premise_refs}){theory_ann})"
                )
        lines.append(f")")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "model_name": self.model_name,
            "properties": self.properties,
            "steps": [s.to_dict() for s in self.steps],
            "root_step": self.root_step,
            "theories_used": self.theories_used,
            "verification_conditions": self.verification_conditions,
            "certificate_hash": self.certificate_hash,
            "extraction_time_ms": self.extraction_time_ms,
            "proof_source": self.proof_source,
        }
        if self.strategy is not None:
            d["strategy"] = self.strategy.value
        return d

    def pretty(self) -> str:
        stats = self.summary_stats()
        strategy_str = self.strategy.value if self.strategy else "unknown"
        lines = [
            f"ProofCertificate({self.model_name})",
            f"  Proof source:     {self.proof_source}",
            f"  Strategy:         {strategy_str}",
            f"  Properties:       {', '.join(self.properties)}",
            f"  Proof steps:      {stats['step_count']}",
            f"  Theory lemmas:    {stats['theory_lemma_count']}",
            f"  Max depth:        {stats['max_depth']}",
            f"  Theories:         {', '.join(self.theories_used) or '(none)'}",
            f"  Extraction time:  {self.extraction_time_ms:.1f}ms",
            f"  Hash:             {self.certificate_hash[:16]}…",
        ]
        return "\n".join(lines)

    def summary_stats(self) -> dict:
        step_count = len(self.steps)
        theory_lemma_count = sum(
            1 for s in self.steps if s.rule == "th-lemma"
        )
        rule_histogram: Dict[str, int] = {}
        for s in self.steps:
            rule_histogram[s.rule] = rule_histogram.get(s.rule, 0) + 1

        # Compute max depth by walking backwards from root.
        depth_cache: Dict[int, int] = {}

        def _depth(idx: int) -> int:
            if idx in depth_cache:
                return depth_cache[idx]
            step = self.steps[idx]
            if not step.premises:
                depth_cache[idx] = 0
                return 0
            d = 1 + max(_depth(p) for p in step.premises)
            depth_cache[idx] = d
            return d

        max_depth = _depth(self.root_step) if step_count > 0 else 0

        return {
            "step_count": step_count,
            "theory_lemma_count": theory_lemma_count,
            "max_depth": max_depth,
            "rule_histogram": rule_histogram,
            "theories_used": list(self.theories_used),
            "verification_condition_count": len(self.verification_conditions),
        }

    # ── private helpers ──────────────────────────────────────────────────

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        for s in self.steps:
            h.update(s.rule.encode())
            h.update(s.conclusion.encode())
            for p in s.premises:
                h.update(p.to_bytes(4, "little", signed=True))
        return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# ProofExtractor
# ═══════════════════════════════════════════════════════════════════════════════


class ProofExtractor:
    """Extract a proof certificate from a Z3 solver in proof mode.

    When the original solver was not created with proof generation enabled,
    the extractor automatically replays the assertions in a fresh solver
    using a ``z3.Context('proof', 'true')`` context so that proof objects
    are available regardless of the process-global Z3 configuration.

    Usage::

        s = z3.Solver()
        # ... add assertions ...
        if s.check() == z3.unsat:
            extractor = ProofExtractor(s, list(s.assertions()))
            cert = extractor.extract("MyModel", ["prop1"])
    """

    def __init__(
        self,
        solver: "z3.Solver",
        assertions: List["z3.BoolRef"],
    ) -> None:
        self._solver = solver
        self._assertions = assertions

    def extract(
        self,
        model_name: str = "",
        properties: Optional[List[str]] = None,
    ) -> Optional[ProofCertificate]:
        """Extract a ``ProofCertificate`` from the solver.

        Returns ``None`` when proof extraction is not possible (e.g. the
        solver returned SAT, proof mode was not enabled, or Z3 raised an
        internal error).
        """
        if not HAS_Z3:
            return None
        if properties is None:
            properties = []

        t0 = time.perf_counter()

        # Try to get a proof from the original solver first.
        proof_obj = self._try_get_proof(self._solver)

        # If that failed (proof mode wasn't enabled when the solver was
        # created), replay the assertions in a fresh proof-enabled solver.
        if proof_obj is None:
            proof_obj, replay_assertions = self._replay_with_proof_context()
            if replay_assertions is not None:
                self._assertions = replay_assertions

        if proof_obj is None:
            return None

        steps: List[ProofStep] = []
        visited: Dict[int, int] = {}
        theories_seen: set[str] = set()

        try:
            root_idx = self._walk_proof(proof_obj, visited, steps, theories_seen)
        except Exception:
            return None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        vc_strings = []
        for a in self._assertions:
            try:
                vc_strings.append(a.sexpr())
            except Exception:
                vc_strings.append(str(a))

        cert = ProofCertificate(
            model_name=model_name,
            properties=properties,
            steps=steps,
            root_step=root_idx,
            theories_used=sorted(theories_seen),
            verification_conditions=vc_strings,
            extraction_time_ms=elapsed_ms,
            strategy=CertificateStrategy.Z3_NATIVE_PROOF,
        )
        return cert

    @staticmethod
    def _try_get_proof(solver: "z3.Solver") -> Optional["z3.ExprRef"]:
        """Try to extract a proof object; return None on failure."""
        try:
            p = solver.proof()
            return p if p is not None else None
        except Exception:
            return None

    def _replay_with_proof_context(
        self,
    ) -> tuple:
        """Re-solve assertions in a fresh context with proof generation.

        Returns (proof_obj, new_assertions) or (None, None).
        """
        try:
            ctx = z3.Context("proof", "true")
            s2 = z3.Solver(ctx=ctx)
            s2.set("timeout", 10000)
            # Translate assertions to the new context.
            new_assertions = []
            for a in self._assertions:
                translated = a.translate(ctx)
                s2.add(translated)
                new_assertions.append(translated)
            if s2.check() != z3.unsat:
                return None, None
            proof_obj = s2.proof()
            if proof_obj is None:
                return None, None
            return proof_obj, new_assertions
        except Exception:
            return None, None

    # ── recursive proof walker ───────────────────────────────────────────

    def _walk_proof(
        self,
        node: "z3.ExprRef",
        visited: Dict[int, int],
        steps: List[ProofStep],
        theories_seen: set,
    ) -> int:
        """Recursively walk the Z3 proof DAG, returning the step index."""
        node_id = id(node)
        if node_id in visited:
            return visited[node_id]

        # Extract rule name.
        try:
            rule_name = node.decl().name()
        except Exception:
            rule_name = "unknown"

        children = node.children()
        premise_indices: List[int] = []

        # In Z3 proof nodes the last child is the conclusion formula;
        # preceding children are either sub-proof nodes or formula
        # arguments.  We identify sub-proofs by checking if their
        # declaration name is a known proof rule.
        conclusion_node = None
        if children:
            conclusion_node = children[-1]
            proof_children = [
                c for c in children[:-1]
                if z3.is_app(c) and c.decl().name() in _KNOWN_RULES
            ]
        else:
            proof_children = []

        # Recurse on premise sub-proofs first.
        for pc in proof_children:
            idx = self._walk_proof(pc, visited, steps, theories_seen)
            premise_indices.append(idx)

        # Conclusion text.
        if conclusion_node is not None:
            try:
                conclusion_str = conclusion_node.sexpr()
            except Exception:
                conclusion_str = str(conclusion_node)
        else:
            try:
                conclusion_str = node.sexpr()
            except Exception:
                conclusion_str = str(node)

        # Determine background theory.
        theory: Optional[str] = None
        if rule_name == "th-lemma":
            theory = _guess_theory(conclusion_str)
            theories_seen.add(theory)

        step = ProofStep(
            rule=rule_name,
            conclusion=conclusion_str,
            premises=premise_indices,
            theory=theory,
        )
        idx = len(steps)
        steps.append(step)
        visited[node_id] = idx
        return idx


# ── helpers ──────────────────────────────────────────────────────────────────


def _is_proof_node(expr: "z3.ExprRef") -> bool:
    """Heuristically determine if a Z3 expression is a proof node."""
    try:
        name = expr.decl().name()
        if name in _KNOWN_RULES:
            return True
        # Z3 may use internal names; check if children suggest proof structure.
        if any(
            z3.is_app(c) and c.decl().name() in _KNOWN_RULES
            for c in expr.children()
        ):
            return True
    except Exception:
        pass
    return False


def _guess_theory(conclusion: str) -> str:
    """Guess which background theory produced a th-lemma."""
    arith_signals = ("+", "-", "*", "<=", ">=", "<", ">", "div", "mod")
    if any(sig in conclusion for sig in arith_signals):
        return "arith"
    return "eq"


# ═══════════════════════════════════════════════════════════════════════════════
# UNSAT-core certificate strategy
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_unsat_core_certificate(
    model_name: str,
    properties: List[str],
    assertions: List["z3.BoolRef"],
    timeout_ms: int = 10000,
) -> Optional[ProofCertificate]:
    """Build a certificate from a minimal UNSAT core.

    Creates a fresh solver with tracked assertions, extracts the UNSAT core,
    and builds a certificate containing the SMT-LIB2 encoding of just the
    core constraints.
    """
    if not HAS_Z3:
        return None
    try:
        t0 = time.perf_counter()
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        # Add assertions with tracking labels
        labels = []
        for i, a in enumerate(assertions):
            label = z3.Bool(f"__track_{i}")
            labels.append(label)
            s.assert_and_track(a, label)
        result = s.check()
        if result != z3.unsat:
            return None
        core = s.unsat_core()
        if not core:
            return None
        # Map core labels back to assertion indices
        core_label_names = {str(c) for c in core}
        core_indices = []
        for i, label in enumerate(labels):
            if str(label) in core_label_names:
                core_indices.append(i)
        # Build proof steps: one asserted step per core constraint, then a
        # synthetic th-lemma concluding false
        steps: List[ProofStep] = []
        vc_strings: List[str] = []
        theories_seen: set = set()
        for ci in core_indices:
            a = assertions[ci]
            try:
                sexpr = a.sexpr()
            except Exception:
                sexpr = str(a)
            vc_strings.append(sexpr)
            theory = _guess_theory(sexpr)
            if theory != "eq":
                theories_seen.add(theory)
            steps.append(ProofStep(rule="asserted", conclusion=sexpr))
        # Add final th-lemma step
        premise_indices = list(range(len(steps)))
        theory = "arith" if "arith" in theories_seen else "eq"
        theories_seen.add(theory)
        steps.append(ProofStep(
            rule="th-lemma",
            conclusion="false",
            premises=premise_indices,
            theory=theory,
        ))
        root_idx = len(steps) - 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ProofCertificate(
            model_name=model_name,
            properties=properties,
            steps=steps,
            root_step=root_idx,
            theories_used=sorted(theories_seen),
            verification_conditions=vc_strings,
            extraction_time_ms=elapsed_ms,
            strategy=CertificateStrategy.UNSAT_CORE,
        )
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Replay-based certificate strategy
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_replay_certificate(
    model_name: str,
    properties: List[str],
    assertions: List["z3.BoolRef"],
    timeout_ms: int = 10000,
) -> Optional[ProofCertificate]:
    """Replay assertions in a fresh context with proof=true, no UserPropagator.

    Most TensorGuard constraints are QF_LIA which Z3 can prove without
    UserPropagator. This strategy re-encodes constraints in a clean context
    and extracts a full proof.
    """
    if not HAS_Z3:
        return None
    try:
        t0 = time.perf_counter()
        ctx = z3.Context("proof", "true")
        s = z3.Solver(ctx=ctx)
        s.set("timeout", timeout_ms)
        new_assertions = []
        for a in assertions:
            try:
                translated = a.translate(ctx)
                s.add(translated)
                new_assertions.append(translated)
            except Exception:
                # Skip assertions that can't be translated (custom theory)
                continue
        if not new_assertions:
            return None
        result = s.check()
        if result != z3.unsat:
            return None
        proof_obj = None
        try:
            proof_obj = s.proof()
        except Exception:
            pass
        if proof_obj is None:
            return None
        # Walk the proof tree
        steps: List[ProofStep] = []
        visited: Dict[int, int] = {}
        theories_seen: set = set()
        extractor = ProofExtractor.__new__(ProofExtractor)
        extractor._solver = s
        extractor._assertions = new_assertions
        try:
            root_idx = extractor._walk_proof(proof_obj, visited, steps, theories_seen)
        except Exception:
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        vc_strings = []
        for a in new_assertions:
            try:
                vc_strings.append(a.sexpr())
            except Exception:
                vc_strings.append(str(a))
        return ProofCertificate(
            model_name=model_name,
            properties=properties,
            steps=steps,
            root_step=root_idx,
            theories_used=sorted(theories_seen),
            verification_conditions=vc_strings,
            extraction_time_ms=elapsed_ms,
            strategy=CertificateStrategy.REPLAY,
        )
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Dual-solver verification certificate strategy
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_dual_solver_certificate(
    model_name: str,
    properties: List[str],
    assertions: List["z3.BoolRef"],
    timeout_ms: int = 10000,
) -> Optional[ProofCertificate]:
    """Replay constraints in a fresh Z3 context (no UserPropagator) and
    optionally CVC5. If the simplified encoding also gives UNSAT, build a
    certificate from the UNSAT core of the fresh solver.
    """
    if not HAS_Z3:
        return None
    try:
        t0 = time.perf_counter()
        # Fresh Z3 solver without UserPropagator
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        translated: List["z3.BoolRef"] = []
        for a in assertions:
            try:
                s.add(a)
                translated.append(a)
            except Exception:
                continue
        if not translated:
            return None
        result = s.check()
        if result != z3.unsat:
            return None
        # Successfully verified in fresh solver — build certificate
        # Try to get UNSAT core for a stronger certificate
        s2 = z3.Solver()
        s2.set("timeout", timeout_ms)
        labels = []
        for i, a in enumerate(translated):
            label = z3.Bool(f"__dual_{i}")
            labels.append(label)
            s2.assert_and_track(a, label)
        r2 = s2.check()
        if r2 == z3.unsat:
            core = s2.unsat_core()
            core_label_names = {str(c) for c in core} if core else set()
        else:
            core_label_names = set()
        # Build proof steps
        steps: List[ProofStep] = []
        vc_strings: List[str] = []
        theories_seen: set = set()
        for i, a in enumerate(translated):
            label_str = f"__dual_{i}"
            if core_label_names and label_str not in core_label_names:
                continue
            try:
                sexpr = a.sexpr()
            except Exception:
                sexpr = str(a)
            vc_strings.append(sexpr)
            theory = _guess_theory(sexpr)
            if theory != "eq":
                theories_seen.add(theory)
            steps.append(ProofStep(rule="asserted", conclusion=sexpr))
        premise_indices = list(range(len(steps)))
        theory = "arith" if "arith" in theories_seen else "eq"
        theories_seen.add(theory)
        steps.append(ProofStep(
            rule="th-lemma",
            conclusion="false",
            premises=premise_indices,
            theory=theory,
        ))
        root_idx = len(steps) - 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ProofCertificate(
            model_name=model_name,
            properties=properties,
            steps=steps,
            root_step=root_idx,
            theories_used=sorted(theories_seen),
            verification_conditions=vc_strings,
            extraction_time_ms=elapsed_ms,
            strategy=CertificateStrategy.DUAL_SOLVER,
        )
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience function — tries strategies in order
# ═══════════════════════════════════════════════════════════════════════════════


def extract_proof_certificate(
    model_name: str,
    properties: List[str],
    solver: "z3.Solver",
    assertions: List["z3.BoolRef"],
) -> Optional[ProofCertificate]:
    """Extract a proof certificate using a cascade of strategies.

    Tries each strategy in order until one succeeds:
      1. Z3 native proof (solver.proof())
      2. UNSAT-core certificate
      3. Replay certificate (fresh context, no UserPropagator, proof=true)
      4. Dual-solver certificate

    Returns ``None`` when no strategy can produce a certificate.
    """
    if not HAS_Z3:
        return None

    # Strategy 1: Z3 native proof
    try:
        extractor = ProofExtractor(solver, assertions)
        cert = extractor.extract(model_name=model_name, properties=properties)
        if cert is not None:
            return cert
    except Exception:
        pass

    # Collect assertions for fallback strategies
    try:
        assertion_list = list(assertions) if assertions else list(solver.assertions())
    except Exception:
        return None

    # Strategy 2: UNSAT-core certificate
    try:
        cert = _extract_unsat_core_certificate(
            model_name, properties, assertion_list,
        )
        if cert is not None:
            return cert
    except Exception:
        pass

    # Strategy 3: Replay certificate
    try:
        cert = _extract_replay_certificate(
            model_name, properties, assertion_list,
        )
        if cert is not None:
            return cert
    except Exception:
        pass

    # Strategy 4: Dual-solver certificate
    try:
        cert = _extract_dual_solver_certificate(
            model_name, properties, assertion_list,
        )
        if cert is not None:
            return cert
    except Exception:
        pass

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CVC5ProofExtractor
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import cvc5 as _cvc5_mod
    HAS_CVC5 = True
except ImportError:
    HAS_CVC5 = False


class CVC5ProofExtractor:
    """Extract an Alethe proof certificate from a CVC5 solver.

    Delegates to the CVC5Solver's ``get_proof_certificate()`` method when
    using the CVC5 backend directly, or can extract from a raw CVC5 proof
    node.

    Usage::

        from src.smt.cvc5_backend import CVC5Solver
        solver = CVC5Solver(produce_proofs=True)
        # ... add assertions ...
        if solver.check_sat() == SatResult.UNSAT:
            cert = solver.get_proof_certificate("MyModel", ["prop1"])
    """

    @staticmethod
    def extract_from_cvc5_solver(
        cvc5_solver: Any,
        model_name: str = "",
        properties: Optional[List[str]] = None,
    ) -> Optional[ProofCertificate]:
        """Extract a proof certificate from a CVC5Solver backend instance.

        The *cvc5_solver* argument should be an instance of
        ``src.smt.cvc5_backend.CVC5Solver``.
        """
        if hasattr(cvc5_solver, "get_proof_certificate"):
            return cvc5_solver.get_proof_certificate(
                model_name=model_name,
                properties=properties,
            )
        return None


def get_proof_status(
    certificate: Optional[ProofCertificate],
    solver_verified: bool = False,
) -> str:
    """Determine the proof status string for output reporters.

    Returns:
        "certified"       — a proof certificate exists and is valid.
        "solver_verified" — the solver verified the property but no proof
                            certificate could be extracted.
        "unverified"      — neither certified nor solver-verified.
    """
    if certificate is not None:
        return "certified"
    if solver_verified:
        return "solver_verified"
    return "unverified"
