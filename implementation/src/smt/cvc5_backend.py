"""
CVC5 SMT Backend for Refinement Type Inference.

Provides CVC5 integration with Alethe proof production, mirroring the Z3
backend functionality. Supports shape/device/phase constraints and extracts
Alethe proof objects when CVC5 returns UNSAT.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.smt.solver import (
    SmtSolver,
    SatResult,
    SmtModel,
    SmtUnsatCore,
    Predicate,
    Comparison,
    IsInstance,
    IsNone,
    IsTruthy,
    HasAttr,
    And,
    Or,
    Not,
    Implies,
    Iff,
    BoolLit,
    Var,
    Const,
    Len,
    BinOp,
    UnaryOp,
    ArithOp,
    UnaryArithOp,
    ComparisonOp,
    Sort,
    Expr,
)
from src.proof_certificate import (
    ProofStep,
    ProofCertificate,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CVC5 import
# ---------------------------------------------------------------------------

try:
    import cvc5
    from cvc5 import Kind, ProofRule

    CVC5_AVAILABLE = True
except ImportError:
    CVC5_AVAILABLE = False

    class _StubKind:
        """Minimal stub so module loads without cvc5."""
        pass

    Kind = _StubKind  # type: ignore[misc,assignment]


class CVC5SolverError(Exception):
    """Error raised by CVC5 solver operations."""
    pass


class CVC5Solver(SmtSolver):
    """Full CVC5 integration for QF_UFLIA + type tags with Alethe proofs.

    Mirrors the Z3Solver API surface so it can serve as a drop-in backend.
    When CVC5 returns UNSAT, Alethe proofs are available via
    ``get_proof_certificate()``.
    """

    DEFAULT_TAGS: List[str] = [
        "int", "float", "str", "bool", "list", "tuple", "dict", "set",
        "bytes", "NoneType", "complex", "frozenset", "bytearray",
        "memoryview", "range", "type", "function", "object",
    ]

    def __init__(
        self,
        *,
        timeout_ms: int = 30000,
        logic: str = "QF_UFLIA",
        tag_domain: Optional[List[str]] = None,
        produce_proofs: bool = True,
    ) -> None:
        if not CVC5_AVAILABLE:
            raise ImportError(
                "cvc5 package is required for CVC5Solver. "
                "Install with: pip install cvc5"
            )

        self._tm = cvc5.TermManager()
        self._solver = cvc5.Solver(self._tm)

        # Core options
        self._solver.setOption("produce-models", "true")
        self._solver.setOption("produce-unsat-cores", "true")
        if produce_proofs:
            self._solver.setOption("produce-proofs", "true")
        self._produce_proofs = produce_proofs
        self._timeout_ms = timeout_ms
        self._solver.setOption("tlimit", str(timeout_ms))
        self._logic = logic
        self._solver.setLogic(logic)

        # Sort declarations
        self._int_sort = self._tm.getIntegerSort()
        self._bool_sort = self._tm.getBooleanSort()
        self._str_sort = self._tm.getStringSort()

        # Tag sort — finite enum via uninterpreted sort + distinct constants
        self._tag_names = tag_domain or list(self.DEFAULT_TAGS)
        self._tag_sort = self._tm.mkUninterpretedSort(
            f"Tag_{uuid.uuid4().hex[:8]}"
        )
        self._tag_constants: Dict[str, Any] = {}
        tag_consts_list = []
        for name in self._tag_names:
            c = self._tm.mkConst(self._tag_sort, f"tag_{name}")
            self._tag_constants[name] = c
            tag_consts_list.append(c)
        # Assert all tag constants are distinct
        if len(tag_consts_list) >= 2:
            self._solver.assertFormula(
                self._tm.mkTerm(Kind.DISTINCT, *tag_consts_list)
            )

        # Declared variables
        self._int_vars: Dict[str, Any] = {}
        self._bool_vars: Dict[str, Any] = {}
        self._tag_vars: Dict[str, Any] = {}
        self._str_vars: Dict[str, Any] = {}

        # Function declarations
        self._len_fn = self._tm.mkConst(
            self._tm.mkFunctionSort([self._int_sort], self._int_sort), "len"
        )
        self._isinstance_fn = self._tm.mkConst(
            self._tm.mkFunctionSort(
                [self._int_sort, self._tag_sort], self._bool_sort
            ),
            "isinstance_fn",
        )
        self._is_none_fn = self._tm.mkConst(
            self._tm.mkFunctionSort([self._int_sort], self._bool_sort),
            "is_none_fn",
        )
        self._is_truthy_fn = self._tm.mkConst(
            self._tm.mkFunctionSort([self._int_sort], self._bool_sort),
            "is_truthy_fn",
        )
        self._hasattr_fn = self._tm.mkConst(
            self._tm.mkFunctionSort(
                [self._int_sort, self._str_sort], self._bool_sort
            ),
            "hasattr_fn",
        )
        self._typeof_fn = self._tm.mkConst(
            self._tm.mkFunctionSort([self._int_sort], self._tag_sort),
            "typeof",
        )

        # Track labels for unsat core
        self._labels: Dict[str, Any] = {}
        self._label_counter = 0

        # Last result
        self._last_result: Optional[SatResult] = None

        # Assertion tracking for proof extraction
        self._assertions: List[Any] = []

    # -- scope management --------------------------------------------------

    def push(self) -> None:
        self._solver.push()

    def pop(self, n: int = 1) -> None:
        for _ in range(n):
            self._solver.pop()

    def reset(self) -> None:
        # CVC5 doesn't have a direct reset; recreate
        self.__init__(
            timeout_ms=self._timeout_ms,
            logic=self._logic,
            produce_proofs=self._produce_proofs,
        )

    def set_timeout(self, milliseconds: int) -> None:
        self._timeout_ms = milliseconds
        self._solver.setOption("tlimit", str(milliseconds))

    def set_logic(self, logic: str) -> None:
        self._logic = logic

    # -- variable declarations ---------------------------------------------

    def declare_int(self, name: str) -> None:
        if name not in self._int_vars:
            self._int_vars[name] = self._tm.mkConst(self._int_sort, name)

    def declare_bool(self, name: str) -> None:
        if name not in self._bool_vars:
            self._bool_vars[name] = self._tm.mkConst(self._bool_sort, name)

    def declare_tag(self, name: str, domain: Optional[List[str]] = None) -> None:
        if name not in self._tag_vars:
            self._tag_vars[name] = self._tm.mkConst(self._tag_sort, name)

    def declare_str(self, name: str) -> None:
        if name not in self._str_vars:
            self._str_vars[name] = self._tm.mkConst(self._str_sort, name)

    # -- formula assertion -------------------------------------------------

    def assert_formula(
        self, formula: Predicate, label: Optional[str] = None
    ) -> None:
        cvc5_formula = self._encode_predicate(formula)
        if label is not None:
            lbl = self._tm.mkConst(self._bool_sort, label)
            self._labels[label] = lbl
            # Use named assertions for unsat core tracking
            named = self._tm.mkTerm(Kind.IMPLIES, lbl, cvc5_formula)
            self._solver.assertFormula(named)
            self._solver.assertFormula(lbl)
            self._assertions.append(cvc5_formula)
        else:
            self._solver.assertFormula(cvc5_formula)
            self._assertions.append(cvc5_formula)

    # -- check sat ---------------------------------------------------------

    def check_sat(self) -> SatResult:
        start = time.monotonic()
        try:
            result = self._solver.checkSat()
        except Exception as e:
            logger.warning("CVC5 check_sat error: %s", e)
            self._last_result = SatResult.UNKNOWN
            return self._last_result
        elapsed = time.monotonic() - start
        self._last_result = self._translate_result(result)
        logger.debug(
            "CVC5 check_sat: %s (%.3fs)", self._last_result.value, elapsed
        )
        return self._last_result

    def check_sat_assuming(self, assumptions: List[Predicate]) -> SatResult:
        cvc5_assumptions = [self._encode_predicate(a) for a in assumptions]
        start = time.monotonic()
        try:
            result = self._solver.checkSatAssuming(*cvc5_assumptions)
        except Exception as e:
            logger.warning("CVC5 check_sat_assuming error: %s", e)
            self._last_result = SatResult.UNKNOWN
            return self._last_result
        elapsed = time.monotonic() - start
        self._last_result = self._translate_result(result)
        logger.debug(
            "CVC5 check_sat_assuming: %s (%.3fs)",
            self._last_result.value,
            elapsed,
        )
        return self._last_result

    # -- model extraction --------------------------------------------------

    def get_model(self) -> Optional[SmtModel]:
        if self._last_result != SatResult.SAT:
            return None
        return self._extract_model()

    def _extract_model(self) -> SmtModel:
        values: Dict[str, Any] = {}
        for name, var in self._int_vars.items():
            try:
                val = self._solver.getValue(var)
                values[name] = val.getIntegerValue()
            except Exception:
                values[name] = str(var)
        for name, var in self._bool_vars.items():
            try:
                val = self._solver.getValue(var)
                values[name] = val.getBooleanValue()
            except Exception:
                values[name] = False
        for name, var in self._tag_vars.items():
            try:
                val = self._solver.getValue(var)
                values[name] = str(val)
            except Exception:
                values[name] = str(var)

        return SmtModel(variable_values=values)

    # -- unsat core --------------------------------------------------------

    def get_unsat_core(self) -> Optional[SmtUnsatCore]:
        if self._last_result != SatResult.UNSAT:
            return None
        try:
            core = self._solver.getUnsatCore()
            labels = [str(c) for c in core]
            return SmtUnsatCore(core=[], labels=labels)
        except Exception:
            return SmtUnsatCore(core=[], labels=[])

    # -- Alethe proof extraction -------------------------------------------

    def get_proof_certificate(
        self,
        model_name: str = "",
        properties: Optional[List[str]] = None,
    ) -> Optional[ProofCertificate]:
        """Extract an Alethe proof certificate after UNSAT result.

        Returns None if proofs are not available or solver didn't return UNSAT.
        """
        if not self._produce_proofs:
            return None
        if self._last_result != SatResult.UNSAT:
            return None

        if properties is None:
            properties = []

        t0 = time.perf_counter()

        try:
            proofs = self._solver.getProof()
            if not proofs:
                return None
            proof = proofs[0] if isinstance(proofs, (list, tuple)) else proofs
        except Exception as e:
            logger.debug("CVC5 proof extraction failed: %s", e)
            return None

        steps: List[ProofStep] = []
        visited: Dict[int, int] = {}
        theories_seen: set = set()

        try:
            root_idx = self._walk_cvc5_proof(
                proof, visited, steps, theories_seen
            )
        except Exception as e:
            logger.debug("CVC5 proof walk failed: %s", e)
            return None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        vc_strings = [str(a) for a in self._assertions]

        cert = ProofCertificate(
            model_name=model_name,
            properties=properties,
            steps=steps,
            root_step=root_idx,
            theories_used=sorted(theories_seen),
            verification_conditions=vc_strings,
            extraction_time_ms=elapsed_ms,
            proof_source="cvc5",
        )
        return cert

    def _walk_cvc5_proof(
        self,
        node: Any,
        visited: Dict[int, int],
        steps: List[ProofStep],
        theories_seen: set,
    ) -> int:
        """Recursively walk the CVC5 proof DAG, returning the step index."""
        node_id = id(node)
        if node_id in visited:
            return visited[node_id]

        # Extract rule name from CVC5 proof node
        try:
            rule = node.getRule()
            rule_name = _cvc5_rule_to_name(rule)
        except Exception:
            rule_name = "unknown"

        # Get children (premises)
        try:
            children = node.getChildren()
        except Exception:
            children = []

        premise_indices: List[int] = []
        for child in children:
            idx = self._walk_cvc5_proof(child, visited, steps, theories_seen)
            premise_indices.append(idx)

        # Get conclusion
        try:
            result = node.getResult()
            conclusion_str = str(result)
        except Exception:
            conclusion_str = str(node)

        # Determine theory
        theory: Optional[str] = None
        if rule_name in ("th-lemma", "theory-lemma", "theory_rewrite"):
            theory = _guess_theory_cvc5(conclusion_str)
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

    # -- predicate encoding ------------------------------------------------

    def _encode_predicate(self, pred: Predicate) -> Any:
        if isinstance(pred, BoolLit):
            return self._tm.mkBoolean(pred.value)

        if isinstance(pred, Comparison):
            left = self._encode_expr(pred.left)
            right = self._encode_expr(pred.right)
            return self._encode_comparison_op(pred.op, left, right)

        if isinstance(pred, IsInstance):
            var = self._get_or_declare_int(pred.var)
            tag_val = self._tag_constants.get(pred.tag)
            if tag_val is None:
                logger.warning("Unknown tag %r, treating as false", pred.tag)
                return self._tm.mkBoolean(False)
            return self._tm.mkTerm(Kind.APPLY_UF, self._isinstance_fn, var, tag_val)

        if isinstance(pred, IsNone):
            var = self._get_or_declare_int(pred.var)
            return self._tm.mkTerm(Kind.APPLY_UF, self._is_none_fn, var)

        if isinstance(pred, IsTruthy):
            var = self._get_or_declare_int(pred.var)
            return self._tm.mkTerm(Kind.APPLY_UF, self._is_truthy_fn, var)

        if isinstance(pred, HasAttr):
            var = self._get_or_declare_int(pred.var)
            key = self._tm.mkString(pred.key)
            return self._tm.mkTerm(Kind.APPLY_UF, self._hasattr_fn, var, key)

        if isinstance(pred, And):
            if not pred.conjuncts:
                return self._tm.mkBoolean(True)
            encoded = [self._encode_predicate(c) for c in pred.conjuncts]
            if len(encoded) == 1:
                return encoded[0]
            return self._tm.mkTerm(Kind.AND, *encoded)

        if isinstance(pred, Or):
            if not pred.disjuncts:
                return self._tm.mkBoolean(False)
            encoded = [self._encode_predicate(d) for d in pred.disjuncts]
            if len(encoded) == 1:
                return encoded[0]
            return self._tm.mkTerm(Kind.OR, *encoded)

        if isinstance(pred, Not):
            inner = self._encode_predicate(pred.operand)
            return self._tm.mkTerm(Kind.NOT, inner)

        if isinstance(pred, Implies):
            ante = self._encode_predicate(pred.antecedent)
            cons = self._encode_predicate(pred.consequent)
            return self._tm.mkTerm(Kind.IMPLIES, ante, cons)

        if isinstance(pred, Iff):
            left = self._encode_predicate(pred.left)
            right = self._encode_predicate(pred.right)
            return self._tm.mkTerm(Kind.EQUAL, left, right)

        raise CVC5SolverError(f"Cannot encode predicate: {type(pred)}")

    def _encode_expr(self, expr: Expr) -> Any:
        if isinstance(expr, Var):
            if expr.sort == Sort.BOOL:
                return self._get_or_declare_bool(expr.name)
            if expr.sort == Sort.TAG:
                return self._get_or_declare_tag(expr.name)
            if expr.sort == Sort.STR:
                return self._get_or_declare_str(expr.name)
            return self._get_or_declare_int(expr.name)

        if isinstance(expr, Const):
            if isinstance(expr.value, bool):
                return self._tm.mkBoolean(expr.value)
            if isinstance(expr.value, int):
                return self._tm.mkInteger(expr.value)
            if isinstance(expr.value, str):
                return self._tm.mkString(expr.value)
            return self._tm.mkInteger(int(expr.value))

        if isinstance(expr, Len):
            inner = self._encode_expr(expr.arg)
            return self._tm.mkTerm(Kind.APPLY_UF, self._len_fn, inner)

        if isinstance(expr, BinOp):
            left = self._encode_expr(expr.left)
            right = self._encode_expr(expr.right)
            return self._encode_arith_op(expr.op, left, right)

        if isinstance(expr, UnaryOp):
            operand = self._encode_expr(expr.operand)
            if expr.op == UnaryArithOp.NEG:
                return self._tm.mkTerm(Kind.NEG, operand)
            if expr.op == UnaryArithOp.ABS:
                zero = self._tm.mkInteger(0)
                cond = self._tm.mkTerm(Kind.GEQ, operand, zero)
                neg = self._tm.mkTerm(Kind.NEG, operand)
                return self._tm.mkTerm(Kind.ITE, cond, operand, neg)

        raise CVC5SolverError(f"Cannot encode expression: {type(expr)}")

    def _encode_comparison_op(self, op: ComparisonOp, left: Any, right: Any) -> Any:
        ops = {
            ComparisonOp.EQ: Kind.EQUAL,
            ComparisonOp.LT: Kind.LT,
            ComparisonOp.LE: Kind.LEQ,
            ComparisonOp.GT: Kind.GT,
            ComparisonOp.GE: Kind.GEQ,
        }
        if op == ComparisonOp.NE:
            eq = self._tm.mkTerm(Kind.EQUAL, left, right)
            return self._tm.mkTerm(Kind.NOT, eq)
        kind = ops.get(op)
        if kind is None:
            raise CVC5SolverError(f"Unknown comparison op: {op}")
        return self._tm.mkTerm(kind, left, right)

    def _encode_arith_op(self, op: ArithOp, left: Any, right: Any) -> Any:
        ops = {
            ArithOp.ADD: Kind.ADD,
            ArithOp.SUB: Kind.SUB,
            ArithOp.MUL: Kind.MULT,
            ArithOp.DIV: Kind.INTS_DIVISION,
            ArithOp.MOD: Kind.INTS_MODULUS,
        }
        kind = ops.get(op)
        if kind is None:
            raise CVC5SolverError(f"Unknown arith op: {op}")
        return self._tm.mkTerm(kind, left, right)

    # -- variable helpers --------------------------------------------------

    def _get_or_declare_int(self, name: str) -> Any:
        if name not in self._int_vars:
            self._int_vars[name] = self._tm.mkConst(self._int_sort, name)
        return self._int_vars[name]

    def _get_or_declare_bool(self, name: str) -> Any:
        if name not in self._bool_vars:
            self._bool_vars[name] = self._tm.mkConst(self._bool_sort, name)
        return self._bool_vars[name]

    def _get_or_declare_tag(self, name: str) -> Any:
        if name not in self._tag_vars:
            self._tag_vars[name] = self._tm.mkConst(self._tag_sort, name)
        return self._tag_vars[name]

    def _get_or_declare_str(self, name: str) -> Any:
        if name not in self._str_vars:
            self._str_vars[name] = self._tm.mkConst(self._str_sort, name)
        return self._str_vars[name]

    # -- result translation ------------------------------------------------

    @staticmethod
    def _translate_result(result: Any) -> SatResult:
        if result.isSat():
            return SatResult.SAT
        if result.isUnsat():
            return SatResult.UNSAT
        return SatResult.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════


# Map CVC5 ProofRule enum values to Alethe-compatible rule names
_CVC5_RULE_MAP: Dict[str, str] = {
    "ASSUME": "assume",
    "SCOPE": "scope",
    "RESOLUTION": "resolution",
    "CHAIN_RESOLUTION": "chain-resolution",
    "REFL": "refl",
    "SYMM": "symm",
    "TRANS": "trans",
    "CONG": "cong",
    "MODUS_PONENS": "mp",
    "NOT_NOT_ELIM": "not-not-elim",
    "CONTRA": "contra",
    "AND_ELIM": "and-elim",
    "AND_INTRO": "and-intro",
    "NOT_OR_ELIM": "not-or-elim",
    "IMPLIES_ELIM": "implies-elim",
    "NOT_IMPLIES1": "not-implies1",
    "NOT_IMPLIES2": "not-implies2",
    "EQUIV_ELIM1": "equiv-elim1",
    "EQUIV_ELIM2": "equiv-elim2",
    "NOT_EQUIV_ELIM1": "not-equiv-elim1",
    "NOT_EQUIV_ELIM2": "not-equiv-elim2",
    "ITE_ELIM1": "ite-elim1",
    "ITE_ELIM2": "ite-elim2",
    "NOT_ITE_ELIM1": "not-ite-elim1",
    "NOT_ITE_ELIM2": "not-ite-elim2",
    "ARITH_TRICHOTOMY": "arith-trichotomy",
    "ARITH_SUM_UB": "arith-sum-ub",
    "ARITH_MULT_POS": "arith-mult-pos",
    "ARITH_MULT_NEG": "arith-mult-neg",
    "THEORY_REWRITE": "theory_rewrite",
    "THEORY_LEMMA": "theory-lemma",
    "REWRITE": "rewrite",
    "ACI_NORM": "aci-norm",
    "INSTANTIATE": "instantiate",
    "SKOLEMIZE": "skolemize",
}


def _cvc5_rule_to_name(rule: Any) -> str:
    """Convert a CVC5 ProofRule enum to a string name."""
    try:
        rule_str = rule.name if hasattr(rule, "name") else str(rule)
        # Strip prefix like "ProofRule." if present
        if "." in rule_str:
            rule_str = rule_str.rsplit(".", 1)[-1]
        return _CVC5_RULE_MAP.get(rule_str, rule_str.lower().replace("_", "-"))
    except Exception:
        return "unknown"


def _guess_theory_cvc5(conclusion: str) -> str:
    """Guess which background theory produced a lemma."""
    arith_signals = ("+", "-", "*", "<=", ">=", "<", ">", "div", "mod")
    if any(sig in conclusion for sig in arith_signals):
        return "arith"
    return "eq"
