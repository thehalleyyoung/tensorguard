"""Proof-carrying **safety certificates** — certified *absence* of forced
failures (even_more.md "quantum leap": find → certify).

Every other proof artifact in this engine answers *"why is this a bug?"*.
:class:`~src.symexec.certificate.BugCertificate` distils a single report into a
replayable witness that a runtime precondition is *violated*.  This module builds
the dual, far stronger object that only a **sound + relatively-complete +
machine-checked** analyser can credibly issue:

    A self-contained, independently re-verifiable certificate that, on the
    covered fragment of a program, **no** modeled forced-failure bug of any
    relative-completeness ("COMPLETE_FOR") kind is provable — together with the
    exact boundary (the abstain ledger) where that guarantee stops.

The certificate is proof-carrying in the same sense as the bug certificates:

* its verdict rests on the **soundness ⇐ direction**, machine-checked in Lean —
  every COMPLETE_FOR kind names the ``…​.refute`` theorem whose axiom-clean proof
  guarantees the engine never *fails to fire* on a genuine forced failure with
  known operands (the contrapositive: no report ⇒ no such failure); and
* it is **replayable** — :func:`verify_safety_certificate` re-derives the verdict
  from the source alone (the engine is deterministic: a matching reproducibility
  fingerprint plus an empty sound-bug set *is* the proof of absence), without
  trusting the issuer.

A safety certificate is therefore the apex of the soundness/completeness
contract: soundness (proved in Lean) says *reports are real*; relative
completeness (``completeness_contract``) says *real failures on known operands
are reported*; together, on the covered fragment, **absence of a report is a
guarantee of absence of the failure**, and this artifact makes that guarantee
transferable and checkable offline.

Torch-free; standard library only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .completeness_contract import COMPLETE_FOR

__all__ = [
    "SAFETY_CERTIFICATE_VERSION",
    "LEAN_REFUTATION_FOR",
    "SafetyObligation",
    "SafetyCertificate",
    "SafetyVerification",
    "certify_safety",
    "verify_safety_certificate",
    "safety_certificate_to_dict",
    "safety_certificate_from_dict",
    "dumps_safety_certificate",
    "loads_safety_certificate",
    "render_safety_certificate",
    "certify_file",
    "verify_certificate_file",
]

SAFETY_CERTIFICATE_VERSION = 1

# Sound (forced-failure) reports are emitted at error severity; heuristic /
# intent suspicions are warnings and never bear on the safety verdict.
_WARNING_SEVERITY = "warning"


# --------------------------------------------------------------------------- #
# Each COMPLETE_FOR kind is backed by a machine-checked Lean refutation theorem #
# (the ⇐ "every report is real" direction); the absence-of-report guarantee is  #
# the contrapositive of its companion relative-completeness clause.  This map is #
# pinned in-sync with ``tests/test_lean_soundness.py`` (_AUDITED_THEOREMS) and   #
# ``completeness_contract.COMPLETE_FOR`` by the certificate test-suite.          #
# --------------------------------------------------------------------------- #
LEAN_REFUTATION_FOR: Dict[str, str] = {
    "matmul_dim_mismatch": "TensorGuard.Symexec.Matmul.matmul_refute",
    "broadcast_mismatch": "TensorGuard.Symexec.Broadcast.broadcast_refute",
    "layer_dim_mismatch": "TensorGuard.Symexec.Linear.refute",
    "reshape_size_mismatch": "TensorGuard.Symexec.Reshape.refute",
    "cat_shape_mismatch": "TensorGuard.Symexec.CatStack.refute",
    "einsum_dim_mismatch": "TensorGuard.Symexec.Einsum.refute",
    "axis_out_of_range": "TensorGuard.Symexec.AxisOOB.refute",
    "tensor_index_oob": "TensorGuard.Symexec.IndexOOB.refute",
    "rank_index_error": "TensorGuard.Symexec.IndexOOB.refute",
    "negative_dimension": "TensorGuard.Symexec.NegativeDim.refute",
    "division_by_zero": "TensorGuard.Symexec.DivZero.refute",
    "unpack_arity_mismatch": "TensorGuard.Symexec.UnpackArity.refute",
    "return_arity_contract": "TensorGuard.Symexec.UnpackArity.refute",
    "einops_pattern_mismatch": "TensorGuard.Symexec.EinopsRankMismatch.refute",
    "none_propagation": "TensorGuard.Symexec.NoneDeref.refute",
}

# The trusted Lean kernel axioms every cited refutation proof is audited against
# (see tests/test_lean_soundness.py); recorded in the certificate so a verifier
# knows the exact trust base.
_TRUSTED_AXIOMS = ("propext", "Classical.choice", "Quot.sound")


@dataclass(frozen=True)
class SafetyObligation:
    """One discharged proof obligation: a COMPLETE_FOR bug kind that was *not*
    reported on the covered fragment.

    ``reported_count`` is ``0`` for a discharged obligation.  ``predicate`` is the
    runtime precondition the kind's completeness clause cites; ``lean_refutation``
    is the machine-checked ``refute`` theorem backing the soundness direction;
    ``witness_condition`` is the completeness clause's witness (which operands
    must be known for the absence guarantee to bite)."""

    kind: str
    predicate: Optional[str]
    lean_refutation: Optional[str]
    witness_condition: str
    reported_count: int

    @property
    def discharged(self) -> bool:
        return self.reported_count == 0


@dataclass(frozen=True)
class SafetyCertificate:
    """A self-contained, replayable certificate of *absence* of forced failures.

    ``proven_safe`` is ``True`` exactly when no sound forced-failure bug was
    derived; the guarantee is scoped to ``obligations`` (the COMPLETE_FOR kinds)
    on the covered fragment, with ``abstain_total`` / ``abstain_by_category``
    marking where the scope stops.  ``source_sha256`` and ``fingerprint`` bind the
    certificate to an exact program and an exact deterministic analysis, so
    :func:`verify_safety_certificate` can re-derive the verdict from source."""

    version: int
    filename: str
    source_sha256: str
    fingerprint: str
    proven_safe: bool
    sound_bug_count: int
    heuristic_bug_count: int
    functions_analyzed: int
    total_statements: int
    covered_statements: int
    coverage: float
    value_coverage: float
    obligations: Tuple[SafetyObligation, ...]
    abstain_total: int
    abstain_by_category: Tuple[Tuple[str, int], ...]
    trusted_axioms: Tuple[str, ...]

    @property
    def all_obligations_discharged(self) -> bool:
        return all(o.discharged for o in self.obligations)


@dataclass(frozen=True)
class SafetyVerification:
    """The result of independently re-checking a :class:`SafetyCertificate`."""

    verified: bool
    checks: Tuple[Tuple[str, bool, str], ...]

    def reasons(self) -> List[str]:
        return [detail for name, ok, detail in self.checks if not ok]


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _clause_by_kind() -> Dict[str, object]:
    out: Dict[str, object] = {}
    for clause in COMPLETE_FOR:
        out.setdefault(clause.kind, clause)
    return out


def _build_obligations(
    sound_counts: Dict[str, int]
) -> Tuple[SafetyObligation, ...]:
    """One obligation per COMPLETE_FOR kind, in stable contract order."""
    obligations: List[SafetyObligation] = []
    seen: set = set()
    for clause in COMPLETE_FOR:
        if clause.kind in seen:
            continue
        seen.add(clause.kind)
        obligations.append(
            SafetyObligation(
                kind=clause.kind,
                predicate=clause.predicate,
                lean_refutation=LEAN_REFUTATION_FOR.get(clause.kind),
                witness_condition=clause.condition,
                reported_count=int(sound_counts.get(clause.kind, 0)),
            )
        )
    return tuple(obligations)


def certify_safety(
    result, source: str, *, filename: str = "<unknown>"
) -> SafetyCertificate:
    """Build a :class:`SafetyCertificate` from an analysed ``result`` and the
    exact ``source`` it was produced from.

    A *sound* bug is any report not at ``warning`` severity (heuristic / intent
    suspicions never bear on the safety verdict)."""
    sound_bugs = [b for b in result.bugs if b.severity != _WARNING_SEVERITY]
    heuristic_bugs = [b for b in result.bugs if b.severity == _WARNING_SEVERITY]

    sound_counts: Dict[str, int] = {}
    for b in sound_bugs:
        k = getattr(b.kind, "value", str(b.kind))
        sound_counts[k] = sound_counts.get(k, 0) + 1

    cov = result.coverage
    ledger = result.abstentions
    by_cat = sorted(
        ((cat.value, cnt) for cat, cnt in ledger.coverage().items()),
        key=lambda kv: (-kv[1], kv[0]),
    )

    return SafetyCertificate(
        version=SAFETY_CERTIFICATE_VERSION,
        filename=filename,
        source_sha256=_sha256(source),
        fingerprint=result.fingerprint(),
        proven_safe=not sound_bugs,
        sound_bug_count=len(sound_bugs),
        heuristic_bug_count=len(heuristic_bugs),
        functions_analyzed=result.functions_analyzed,
        total_statements=cov.total,
        covered_statements=cov.non_top,
        coverage=cov.coverage,
        value_coverage=cov.value_coverage,
        obligations=_build_obligations(sound_counts),
        abstain_total=ledger.total,
        abstain_by_category=tuple(by_cat),
        trusted_axioms=_TRUSTED_AXIOMS,
    )


def verify_safety_certificate(
    cert: SafetyCertificate, source: str
) -> SafetyVerification:
    """Independently re-derive the certificate's verdict from ``source`` alone.

    Trust nothing in the certificate but its claims: re-hash the source, re-run
    the (deterministic) engine, and confirm (a) the source matches, (b) the
    reproducibility fingerprint matches, (c) no sound forced-failure bug exists,
    (d) every obligation is discharged with the precondition + Lean refutation the
    *current* contract assigns its kind.  All four must hold for ``verified``."""
    from .engine import analyze_source

    checks: List[Tuple[str, bool, str]] = []

    src_ok = _sha256(source) == cert.source_sha256
    checks.append((
        "source_sha256",
        src_ok,
        "source does not match the certified program" if not src_ok else "ok",
    ))

    result = analyze_source(source)
    fp_ok = result.fingerprint() == cert.fingerprint
    checks.append((
        "fingerprint",
        fp_ok,
        "deterministic analysis fingerprint differs from the certificate"
        if not fp_ok else "ok",
    ))

    sound_bugs = [b for b in result.bugs if b.severity != _WARNING_SEVERITY]
    sound_counts: Dict[str, int] = {}
    for b in sound_bugs:
        k = getattr(b.kind, "value", str(b.kind))
        sound_counts[k] = sound_counts.get(k, 0) + 1

    no_bug = not sound_bugs
    verdict_ok = no_bug == cert.proven_safe
    checks.append((
        "verdict",
        verdict_ok and (no_bug if cert.proven_safe else True),
        "re-analysis disagrees with the certified proven_safe verdict"
        if not verdict_ok else "ok",
    ))

    # Obligations must match the live completeness contract and actually be
    # discharged (zero reports) when the certificate claims safety.
    clauses = _clause_by_kind()
    obl_ok = True
    detail = "ok"
    for o in cert.obligations:
        clause = clauses.get(o.kind)
        if clause is None:
            obl_ok = False
            detail = f"kind {o.kind!r} is no longer a COMPLETE_FOR kind"
            break
        if o.predicate != getattr(clause, "predicate", None):
            obl_ok = False
            detail = f"precondition for {o.kind!r} drifted from the contract"
            break
        if o.lean_refutation != LEAN_REFUTATION_FOR.get(o.kind):
            obl_ok = False
            detail = f"Lean refutation for {o.kind!r} drifted from the registry"
            break
        if cert.proven_safe and sound_counts.get(o.kind, 0) != 0:
            obl_ok = False
            detail = f"obligation {o.kind!r} is not discharged on re-analysis"
            break
    checks.append(("obligations", obl_ok, detail))

    verified = all(ok for _, ok, _ in checks)
    return SafetyVerification(verified=verified, checks=tuple(checks))


# --------------------------------------------------------------------------- #
# Serialization — the certificate is the on-the-wire proof-of-absence artifact. #
# --------------------------------------------------------------------------- #
def safety_certificate_to_dict(cert: SafetyCertificate) -> dict:
    return {
        "version": cert.version,
        "filename": cert.filename,
        "source_sha256": cert.source_sha256,
        "fingerprint": cert.fingerprint,
        "proven_safe": cert.proven_safe,
        "sound_bug_count": cert.sound_bug_count,
        "heuristic_bug_count": cert.heuristic_bug_count,
        "functions_analyzed": cert.functions_analyzed,
        "total_statements": cert.total_statements,
        "covered_statements": cert.covered_statements,
        "coverage": round(cert.coverage, 6),
        "value_coverage": round(cert.value_coverage, 6),
        "obligations": [
            {
                "kind": o.kind,
                "predicate": o.predicate,
                "lean_refutation": o.lean_refutation,
                "witness_condition": o.witness_condition,
                "reported_count": o.reported_count,
            }
            for o in cert.obligations
        ],
        "abstain_total": cert.abstain_total,
        "abstain_by_category": [
            {"category": cat, "count": cnt}
            for cat, cnt in cert.abstain_by_category
        ],
        "trusted_axioms": list(cert.trusted_axioms),
    }


def safety_certificate_from_dict(d: dict) -> SafetyCertificate:
    return SafetyCertificate(
        version=int(d.get("version", SAFETY_CERTIFICATE_VERSION)),
        filename=d.get("filename", "<unknown>"),
        source_sha256=d["source_sha256"],
        fingerprint=d["fingerprint"],
        proven_safe=bool(d["proven_safe"]),
        sound_bug_count=int(d.get("sound_bug_count", 0)),
        heuristic_bug_count=int(d.get("heuristic_bug_count", 0)),
        functions_analyzed=int(d.get("functions_analyzed", 0)),
        total_statements=int(d.get("total_statements", 0)),
        covered_statements=int(d.get("covered_statements", 0)),
        coverage=float(d.get("coverage", 0.0)),
        value_coverage=float(d.get("value_coverage", 0.0)),
        obligations=tuple(
            SafetyObligation(
                kind=o["kind"],
                predicate=o.get("predicate"),
                lean_refutation=o.get("lean_refutation"),
                witness_condition=o.get("witness_condition", ""),
                reported_count=int(o.get("reported_count", 0)),
            )
            for o in d.get("obligations", [])
        ),
        abstain_total=int(d.get("abstain_total", 0)),
        abstain_by_category=tuple(
            (e["category"], int(e["count"]))
            for e in d.get("abstain_by_category", [])
        ),
        trusted_axioms=tuple(d.get("trusted_axioms", _TRUSTED_AXIOMS)),
    )


def dumps_safety_certificate(cert: SafetyCertificate, *, indent: int = 2) -> str:
    return json.dumps(safety_certificate_to_dict(cert), indent=indent, sort_keys=True)


def loads_safety_certificate(text: str) -> SafetyCertificate:
    return safety_certificate_from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# Rendering.                                                                    #
# --------------------------------------------------------------------------- #
def render_safety_certificate(cert: SafetyCertificate) -> str:
    """Render a :class:`SafetyCertificate` as a deterministic Markdown document."""
    lines: List[str] = []
    lines.append(f"# Safety certificate for `{cert.filename}`")
    lines.append("")
    if cert.proven_safe:
        lines.append(
            "✅ **Certified: no forced-failure bug is provable** on the covered "
            "fragment for any relative-completeness kind below."
        )
    else:
        lines.append(
            f"❌ **Not certified** — {cert.sound_bug_count} sound forced-failure "
            "bug(s) were proven; this is a bug certificate's job, not a safety "
            "certificate's."
        )
    lines.append("")
    lines.append(
        f"- Source SHA-256: `{cert.source_sha256}`"
    )
    lines.append(f"- Analysis fingerprint: `{cert.fingerprint}`")
    lines.append(
        f"- Functions analysed: **{cert.functions_analyzed}**; statement coverage "
        f"**{cert.covered_statements}/{cert.total_statements}** "
        f"(**{cert.coverage:.0%}**), value coverage **{cert.value_coverage:.0%}**."
    )
    lines.append("")

    lines.append("## Discharged obligations (absence ⇐ machine-checked soundness)")
    lines.append("")
    lines.append(
        "Each kind below is *relative-complete*: on the covered fragment a genuine "
        "forced failure on known operands **would have been reported** (the ⇒ "
        "completeness clause), and every such report is real (the ⇐ Lean "
        "`refute` theorem, axiom-clean). No report ⇒ no such failure."
    )
    lines.append("")
    lines.append("| Bug kind | Precondition | Lean refutation | Reports |")
    lines.append("| --- | --- | --- | --- |")
    for o in cert.obligations:
        lines.append(
            f"| `{o.kind}` | `{o.predicate}` | `{o.lean_refutation}` | "
            f"{o.reported_count} |"
        )
    lines.append("")

    lines.append("## Where the guarantee stops (abstentions)")
    lines.append("")
    if cert.abstain_total == 0:
        lines.append(
            "The engine did not abstain anywhere on the covered fragment."
        )
    else:
        lines.append(
            f"The engine abstained **{cert.abstain_total}** time(s); the safety "
            "claim does **not** extend to those operations (⊤ operands or "
            "outside the modeled fragment):"
        )
        lines.append("")
        lines.append("| Abstain category | Count |")
        lines.append("| --- | --- |")
        for cat, cnt in cert.abstain_by_category:
            lines.append(f"| `{cat}` | {cnt} |")
    lines.append("")

    lines.append("## Trust base")
    lines.append("")
    lines.append(
        "The ⇐ soundness direction of every obligation is machine-checked in Lean, "
        "audited to depend only on the trusted kernel axioms "
        f"{{{', '.join('`' + a + '`' for a in cert.trusted_axioms)}}} (no "
        "`sorryAx`). The certificate is replayable: re-running the deterministic "
        "engine on the SHA-256-pinned source must reproduce the fingerprint and an "
        "empty sound-bug set."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# File-level entry points (the certifier *product* surface).                    #
# --------------------------------------------------------------------------- #
def certify_file(path: str, *, config=None) -> SafetyCertificate:
    """Read ``path``, analyse it, and certify it in one step.

    The certificate's ``filename`` is ``path`` and its source binding is the
    exact bytes read, so :func:`verify_certificate_file` can re-verify it offline
    against the same file."""
    from .engine import analyze_source

    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    result = analyze_source(source, filename=path, config=config)
    return certify_safety(result, source, filename=path)


def verify_certificate_file(cert: SafetyCertificate, path: str) -> SafetyVerification:
    """Independently re-verify ``cert`` against the current contents of ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    return verify_safety_certificate(cert, source)
