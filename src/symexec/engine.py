"""Top-level entrypoints for the symbolic-execution engine.

``analyze_source`` parses a Python source string, runs the interpreter over the
shipped ``if __name__ == '__main__':`` demo (the author's own harness), and over
each free function, and returns the bugs found.  Running the demo is what lets
the engine reproduce real-world failures such as titans-pytorch #60 and
OpenStrawberry #113, which only manifest through the example call sequence.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .bugs import SymBug
from .interpreter import Interpreter
from .config import SymConfig
from .state import State
from .abstain import AbstainCategory, AbstainLedger, AbstainReason
from .coverage import CoverageMeter

__all__ = ["analyze_source", "analyze_file", "SymResult", "ITERATION_CAPS"]


def _new_coverage() -> "CoverageMeter":
    return CoverageMeter()


def iteration_caps() -> dict:
    """The engine's deterministic worst-case iteration caps (Step 78).

    These constants bound the cost of analysing any single construct regardless
    of wall-clock time, which is what makes the per-file latency budget an
    enforceable contract rather than a hope.  Read live from the interpreter so
    this never drifts from the values actually in force."""
    from . import interpreter as _i

    return {
        "max_call_depth": _i._MAX_DEPTH,
        "loop_unroll": _i._LOOP_UNROLL,
        "loop_fixpoint_max": _i._LOOP_FIX_MAX,
        "loop_narrow_max": _i._LOOP_NARROW_MAX,
        "disjunction_width": Interpreter(ast.parse(""))._disj_bound,
    }


ITERATION_CAPS = iteration_caps()


@dataclass
class SymResult:
    bugs: List[SymBug]
    functions_analyzed: int
    ran_main: bool
    abstentions: AbstainLedger = field(default_factory=AbstainLedger)
    coverage: "CoverageMeter" = field(default_factory=lambda: _new_coverage())

    def fingerprint(self) -> str:
        """Deterministic SHA-256 digest of this result (Step 60).

        A function of the *set* of findings plus the abstain-coverage profile —
        identical inputs yield an identical digest on any machine, so it can be
        recorded as a golden reproducibility receipt."""
        from .footprint import footprint

        return footprint(self.bugs, self.abstentions).digest

    def footprint(self):
        """The full :class:`~src.symexec.footprint.ProofFootprint` for this
        result (digest + bug/abstain counts + coverage profile)."""
        from .footprint import footprint as _fp

        return _fp(self.bugs, self.abstentions)

    def explain(self, filename: str = "<unknown>") -> str:
        """Render the full ``--explain`` provenance view for every report
        (Step 65): the source→…→sink derivation chain plus the counterexample /
        certificate / minimal-conditions sections, location, calibrated
        confidence and fix suggestion."""
        from .explain import explain_bugs

        return explain_bugs(self.bugs, filename=filename)

    def safety(self, filename: str = "<unknown>") -> str:
        """Render the positive *"why is this safe?"* report (even_more #14):
        the verdict (was any sound forced-failure bug provable), the covered
        fragment (coverage profile), the relative-completeness guarantee (the
        kinds whose absence-of-report is a positive guarantee on that fragment),
        and the abstain ledger marking exactly where that guarantee stops."""
        from .safety import explain_safety

        return explain_safety(self, filename=filename)

    def safety_certificate(self, source: str, *, filename: str = "<unknown>"):
        """Build a proof-carrying, replayable **safety certificate** — a
        self-contained, independently re-verifiable attestation that no
        forced-failure bug of any relative-completeness kind is provable on the
        covered fragment (even_more.md "quantum leap": find → certify).

        ``source`` must be the exact program this result was produced from; the
        certificate binds to its SHA-256 and the deterministic analysis
        fingerprint so :func:`verify_safety_certificate` can re-derive the
        verdict without trusting the issuer."""
        from .safety_certificate import certify_safety

        return certify_safety(self, source, filename=filename)

    def to_dict(self, filename: str = "<unknown>") -> dict:
        """A stable JSON object for this result (Step 68): every symexec-specific
        field — calibrated confidence, structured provenance, fingerprint and
        abstain coverage — surfaced for machine consumption."""
        from .export import result_to_dict

        return result_to_dict(self, filename=filename)

    def to_sarif(self, filename: str = "<unknown>") -> dict:
        """A complete SARIF 2.1.0 log for this single result (Step 68).

        The fingerprint and abstain profile ride in the ``run`` ``properties``;
        each finding's symexec fields ride in its ``result`` ``properties``."""
        from .export import to_sarif as _to_sarif

        return _to_sarif([(filename, self)])

    def to_lsp_diagnostics(self, uri: str = "") -> list:
        """LSP ``Diagnostic[]`` for this result (Step 69) — inline editor
        squiggles via ``textDocument/publishDiagnostics``."""
        from .integrations import to_lsp_diagnostics

        return to_lsp_diagnostics(self, uri=uri)

    def to_github_annotations(self, filename: str = "") -> list:
        """GitHub Actions ``::error file=…::`` annotation commands for this
        result (Step 69) — inline PR annotations in CI."""
        from .integrations import to_github_annotations

        return to_github_annotations(self, filename=filename)

    def certificates(self, filename: str = "<unknown>") -> list:
        """Proof-carrying bug certificates for this result (Step 94) — one
        replayable certificate per report, each naming the violated runtime
        precondition and (where recoverable) the concrete witness operands."""
        from .certificate import certify_result

        return certify_result(self, filename=filename)

    def replay(self, filename: str = "<unknown>") -> list:
        """Independently re-derive every report's verdict from its certificate
        (Step 95), without re-running the analysis."""
        from .replay import replay_all

        return replay_all(self.certificates(filename=filename))

    def repros(self) -> list:
        """Runnable reproducers for every report that has a generator — minimal
        self-contained scripts that raise the predicted exception when run with
        torch (even_more.md Tier 1)."""
        from .repro import generate_repros

        return generate_repros(self)


def _find_main_block(module: ast.Module) -> Optional[List[ast.stmt]]:
    for node in module.body:
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):
                return node.body
    return None


def analyze_source(
    source: str,
    filename: str = "<unknown>",
    budget_ms: Optional[float] = None,
    config: "SymConfig | None" = None,
) -> SymResult:
    """Analyze ``source`` and return the bugs found.

    ``budget_ms`` is an optional *coarse* wall-clock guard (default ``None`` =
    unbounded, byte-identical to the historic behaviour).  The engine's
    per-construct iteration caps (recursion depth, loop unroll/fixpoint/narrow
    passes, disjunctive width — see :data:`ITERATION_CAPS`) already bound the
    cost of any single unit *deterministically*; the budget is defence-in-depth
    for pathologically large *files*: it is checked only at top-level unit
    boundaries (between free functions / demo-less methods), and when exceeded
    the remaining units are left un-analysed and a :class:`RESOURCE_BUDGET`
    abstain is recorded.  This is sound — stopping early only loses coverage,
    never invents a report — and the already-found bugs are kept.

    ``config`` (Step 86) selects the soundness mode (``sound | balanced |
    heuristic``); the default is ``balanced``, byte-identical to the historic
    behaviour.  When ``config.budget_ms`` is set and ``budget_ms`` is not given
    explicitly, the config's budget is used."""
    try:
        module = ast.parse(source, filename=filename)
    except SyntaxError as e:
        # A SyntaxError is itself a real defect (vector-quantize-pytorch #248
        # class); surface it as a bug.
        from .bugs import SymBugKind

        return SymResult(
            bugs=[
                SymBug(
                    kind=SymBugKind.UNPACK_ARITY_MISMATCH,  # generic "won't run"
                    message=f"file does not parse: {e.msg}",
                    line=e.lineno or 0,
                    col=(e.offset or 1) - 1,
                    function="<module>",
                    severity="error",
                    confidence=1.0,
                )
            ],
            functions_analyzed=0,
            ran_main=False,
        )

    interp = Interpreter(module, filename=filename, config=config)
    if budget_ms is None and config is not None:
        budget_ms = config.budget_ms
    return _analyze_module(module, interp, filename=filename, budget_ms=budget_ms)


def _analyze_module(
    module: ast.Module,
    interp: "Interpreter",
    *,
    filename: str = "<unknown>",
    budget_ms: Optional[float] = None,
    skip_ids: frozenset = frozenset(),
) -> SymResult:
    """Run the analysis passes over an already-parsed ``module``.

    Factored out of :func:`analyze_source` so the whole-package driver (Step 82)
    can reuse the exact same passes on an import-augmented module.  ``skip_ids``
    holds ``id(node)`` of top-level defs that were *injected* for cross-file
    resolution: they participate in name resolution (so calls into them are
    followed) but are not themselves re-analysed as if defined in this file,
    which avoids attributing an imported symbol's own bugs to the importer."""
    funcs_analyzed = 0
    ran_main = False

    deadline = None if budget_ms is None else time.perf_counter() + budget_ms / 1000.0

    def _over_budget() -> bool:
        return deadline is not None and time.perf_counter() > deadline

    def _record_budget_stop(node) -> None:
        interp._abstentions.record(
            AbstainReason(
                category=AbstainCategory.RESOURCE_BUDGET,
                detector="engine",
                detail=f"analysis budget of {budget_ms:.0f}ms exceeded",
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                function="<module>",
            )
        )

    # 1) Run the shipped __main__ harness if present.
    main_body = _find_main_block(module)
    if main_body is not None:
        ran_main = True
        state = State()
        # bind module-level names that the demo references (classes/functions are
        # resolved structurally; we just need a clean env)
        try:
            interp.exec_block(main_body, state)
        except Exception:
            pass  # never let an interpreter bug crash analysis (Step 79 hardening)

    # 2) Analyze each top-level free function in isolation (so library code with
    #    no demo still gets checked).  Methods are reached interprocedurally.
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            if id(node) in skip_ids:
                continue
            if _over_budget():
                _record_budget_stop(node)
                break
            funcs_analyzed += 1
            try:
                interp.run_function(node, args={}, self_val=None)
            except Exception:
                pass  # never let an interpreter bug crash analysis

    # 2.5) Step (even_more #5b) — module-structure intent checks over every
    #      indexed class (heuristic-mode only; suppressed in sound/balanced).
    #      Module-level ``class`` statements are never executed through the
    #      interpreter, so this structural scan is driven here.
    if interp.config.enable_heuristics:
        interp._cur_dim_facts = ()
        for cls in list(interp.classes.values()):
            if id(cls) in skip_ids:
                continue
            try:
                interp._check_missing_super_init(cls)
            except Exception:
                pass

    # 3) Step 48 — when there is no shipped demo, the canonical model entry
    #    points (``forward``/``__call__``) are never reached interprocedurally.
    #    Analyze them directly: instantiate the class (running ``__init__`` to
    #    populate ``self`` layers) and run the method with its parameters seeded
    #    from their type annotations (a sound contract).  This lets annotated,
    #    demo-less ``nn.Module`` code surface layer-to-layer and rank bugs.
    if main_body is None:
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if id(node) in skip_ids:
                continue
            if _over_budget():
                _record_budget_stop(node)
                break
            for mname in ("forward", "__call__"):
                method = next(
                    (
                        m
                        for m in node.body
                        if isinstance(m, ast.FunctionDef) and m.name == mname
                    ),
                    None,
                )
                if method is None:
                    continue
                try:
                    self_val = interp._instantiate(node, [], {}, None)
                    seed: dict = {}
                    # nn.Module.forward's first positional argument is, by
                    # universal convention, the input tensor.  When it is not
                    # annotated, seed it as a rank-unknown tensor so layer
                    # output shapes propagate (catching input-independent
                    # layer-to-layer mismatches); annotated params are inferred
                    # precisely by ``_bind_params``.
                    posargs = [a for a in method.args.args if a.arg != "self"]
                    if mname == "forward" and posargs and posargs[0].annotation is None:
                        from .values import TensorVal as _TV

                        seed[posargs[0].arg] = _TV(rank=None)
                    interp.run_function(method, args=seed, self_val=self_val)
                except Exception:
                    pass
                break  # forward takes precedence over __call__

    # De-duplicate identical bug reports (same kind+line+message).
    seen = set()
    unique: List[SymBug] = []
    for b in interp.bugs:
        key = (b.kind, b.line, b.col, b.message)
        if key not in seen:
            seen.add(key)
            unique.append(b)

    # Canonical ordering (Step 20): sort by source position then kind/message so
    # the report sequence is deterministic and independent of internal traversal
    # / fixpoint-pass order (two runs of the same input are byte-identical).
    unique.sort(key=lambda b: (b.line, b.col, _kind_name(b.kind), b.message))

    return SymResult(
        bugs=unique,
        functions_analyzed=funcs_analyzed,
        ran_main=ran_main,
        abstentions=interp._abstentions,
        coverage=interp._coverage,
    )


def _kind_name(kind) -> str:
    return getattr(kind, "name", str(kind))


def analyze_file(path: str, config: "SymConfig | None" = None) -> SymResult:
    with open(path, "r", encoding="utf-8") as f:
        return analyze_source(f.read(), filename=path, config=config)
