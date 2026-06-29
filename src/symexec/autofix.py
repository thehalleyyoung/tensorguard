"""Verified auto-repair (even_more.md Tier 1, idea #2).

Most "auto-fix" tools emit a textual suggestion and hope.  TensorGuard can do
something stronger: because the analyzer is sound and fast, it can **re-verify
its own repair**.  For each report this module proposes a minimal, canonical
source edit that should make the violated runtime precondition hold, then
*re-runs the engine on the patched source* and only surfaces the fix when

  * the original bug is gone, **and**
  * no new bug kind was introduced.

So every fix this module returns is **machine-verified**, not guessed.  Fixes
are returned as candidates a maintainer can apply (with a unified diff); they
are deliberately conservative — a strategy that cannot apply unambiguously
yields nothing rather than a risky rewrite.

The repair changes *source the user would edit*; it does not touch the analyzer,
so it never affects which bugs report on the original program or the proof
fingerprint.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

__all__ = [
    "FixCandidate",
    "VerifiedFix",
    "propose_fix",
    "verify_fix",
    "repair",
]


@dataclass(frozen=True)
class FixCandidate:
    """A proposed (not-yet-verified) source edit for one report."""

    kind: str
    line: int
    strategy: str
    description: str
    patched_source: str


@dataclass(frozen=True)
class VerifiedFix:
    """A repair that the engine re-checked: original bug gone, no new bug."""

    kind: str
    line: int
    strategy: str
    description: str
    patched_source: str
    diff: str
    verified: bool
    detail: str


# --------------------------------------------------------------------------- #
# Per-kind rewrite strategies. Each takes the source lines (1-based access via  #
# index line-1) and the bug, and returns (patched_source, strategy, desc) or    #
# None when it cannot apply unambiguously. Edits are line-local so line numbers  #
# are preserved for re-verification matching.                                   #
# --------------------------------------------------------------------------- #
_RESHAPE_CALL = re.compile(r"\.(reshape|view)\(([^()]*)\)")
_NEG_INT = re.compile(r"-(\d+)")
_FORWARD_CALL = re.compile(r"\.forward\(")
_DOT_DATA = re.compile(r"\.data\b")
_DEF_INIT = re.compile(r"^(\s*)def\s+__init__\s*\(")
_LAYER_DIM_MSG = re.compile(
    r"expects? last input dim\s+(\d+)\s+but received\s+(\d+)", re.IGNORECASE
)
_LINEAR_DEF = re.compile(r"nn\.Linear\(\s*(\d+)\b")


def _fix_reshape(lines: List[str], bug) -> Optional[Tuple[str, str, str]]:
    i = bug.line - 1
    if not (0 <= i < len(lines)):
        return None
    line = lines[i]
    m = _RESHAPE_CALL.search(line)
    if not m:
        return None
    # Flatten: a single -1 target is always numel-valid for a contiguous tensor.
    new_line = line[: m.start()] + f".{m.group(1)}(-1)" + line[m.end():]
    if new_line == line:
        return None
    patched = lines[:]
    patched[i] = new_line
    return (
        "\n".join(patched),
        "reshape-flatten",
        f"replace the mismatched `.{m.group(1)}(...)` target with `-1` "
        "(flatten), which always matches the element count",
    )


def _fix_negdim(lines: List[str], bug) -> Optional[Tuple[str, str, str]]:
    i = bug.line - 1
    if not (0 <= i < len(lines)):
        return None
    line = lines[i]
    if not _NEG_INT.search(line):
        return None
    # Make the constructed dimension non-negative by dropping the sign on the
    # first negative integer literal on the offending line.
    new_line = _NEG_INT.sub(lambda mo: mo.group(1), line, count=1)
    if new_line == line:
        return None
    patched = lines[:]
    patched[i] = new_line
    return (
        "\n".join(patched),
        "negdim-abs",
        "make the negative dimension argument non-negative",
    )


def _fix_missing_super_init(lines: List[str], bug) -> Optional[Tuple[str, str, str]]:
    """Insert ``super().__init__()`` as the first statement of an ``__init__``
    that forgot to call it. The bug is reported on the ``def __init__`` line."""
    i = bug.line - 1
    if not (0 <= i < len(lines)):
        return None
    m = _DEF_INIT.match(lines[i])
    if not m:
        return None
    def_indent = m.group(1)
    body_indent = def_indent + "    "
    # Don't duplicate an existing call already present in the body.
    for probe in lines[i + 1 : i + 6]:
        if "super().__init__(" in probe.replace(" ", ""):
            return None
    patched = lines[:]
    patched.insert(i + 1, f"{body_indent}super().__init__()")
    return (
        "\n".join(patched),
        "insert-super-init",
        "insert `super().__init__()` as the first statement of `__init__` so "
        "submodules/params register correctly",
    )


def _fix_direct_forward_call(lines: List[str], bug) -> Optional[Tuple[str, str, str]]:
    """Rewrite ``module.forward(x)`` to the canonical ``module(x)`` so hooks and
    ``__call__`` machinery run."""
    i = bug.line - 1
    if not (0 <= i < len(lines)):
        return None
    line = lines[i]
    if not _FORWARD_CALL.search(line):
        return None
    new_line = _FORWARD_CALL.sub("(", line, count=1)
    if new_line == line:
        return None
    patched = lines[:]
    patched[i] = new_line
    return (
        "\n".join(patched),
        "forward-to-call",
        "call the module directly (`module(x)`) instead of `module.forward(x)` "
        "so registered hooks run",
    )


def _fix_tensor_data_access(lines: List[str], bug) -> Optional[Tuple[str, str, str]]:
    """Rewrite the unsafe ``.data`` attribute access to ``.detach()``."""
    i = bug.line - 1
    if not (0 <= i < len(lines)):
        return None
    line = lines[i]
    if not _DOT_DATA.search(line):
        return None
    new_line = _DOT_DATA.sub(".detach()", line, count=1)
    if new_line == line:
        return None
    patched = lines[:]
    patched[i] = new_line
    return (
        "\n".join(patched),
        "data-to-detach",
        "replace the autograd-unsafe `.data` access with `.detach()`",
    )


def _fix_layer_dim_mismatch(lines: List[str], bug) -> Optional[Tuple[str, str, str]]:
    """Repair an ``nn.Linear`` whose ``in_features`` does not match the feature
    dim it is fed. The message records the expected (declared) and received
    (actual) sizes; we rewrite the *unique* matching layer definition's first
    constructor argument to the received size."""
    msg = getattr(bug, "message", "") or ""
    m = _LAYER_DIM_MSG.search(msg)
    if not m:
        return None
    declared, received = int(m.group(1)), int(m.group(2))
    if declared == received:
        return None
    # Find the unique `nn.Linear(<declared>, ...)` definition line.
    hits = [
        j
        for j, ln in enumerate(lines)
        if (mm := _LINEAR_DEF.search(ln)) and int(mm.group(1)) == declared
    ]
    if len(hits) != 1:
        return None
    j = hits[0]
    line = lines[j]
    new_line = _LINEAR_DEF.sub(f"nn.Linear({received}", line, count=1)
    if new_line == line:
        return None
    patched = lines[:]
    patched[j] = new_line
    return (
        "\n".join(patched),
        "linear-in-features",
        f"set `nn.Linear` in_features {declared} -> {received} to match the "
        "feature dimension actually flowing in",
    )


_STRATEGIES: Dict[str, Callable[[List[str], object], Optional[Tuple[str, str, str]]]] = {
    "reshape_size_mismatch": _fix_reshape,
    "negative_dimension": _fix_negdim,
    "missing_super_init": _fix_missing_super_init,
    "direct_forward_call": _fix_direct_forward_call,
    "tensor_data_access": _fix_tensor_data_access,
    "layer_dim_mismatch": _fix_layer_dim_mismatch,
}


def propose_fix(bug, source: str) -> Optional[FixCandidate]:
    """Propose a minimal source edit for one bug, or ``None`` if no strategy
    applies unambiguously."""
    kind = getattr(bug.kind, "value", str(bug.kind))
    strat = _STRATEGIES.get(kind)
    if strat is None:
        return None
    lines = source.splitlines()
    try:
        produced = strat(lines, bug)
    except Exception:
        produced = None
    if produced is None:
        return None
    patched, name, desc = produced
    # Preserve the source's trailing newline so unified diffs stay minimal
    # (strategies build the body with "\n".join, which drops a final newline).
    if source.endswith("\n") and not patched.endswith("\n"):
        patched += "\n"
    return FixCandidate(
        kind=kind,
        line=int(getattr(bug, "line", 0)),
        strategy=name,
        description=desc,
        patched_source=patched,
    )


def _kind_line_set(bugs) -> set:
    return {(b.kind.value, b.line) for b in bugs}


def _kind_set(bugs) -> set:
    return {b.kind.value for b in bugs}


def verify_fix(
    candidate: FixCandidate,
    original_bugs,
    *,
    filename: str = "<unknown>",
    config=None,
) -> VerifiedFix:
    """Re-run the analyzer on the patched source and decide whether the fix is
    sound: the targeted bug must be gone and no new bug kind may appear."""
    from .engine import analyze_source

    result = analyze_source(candidate.patched_source, filename=filename,
                            config=config)
    after_kind_line = _kind_line_set(result.bugs)
    after_kinds = _kind_set(result.bugs)
    before_kinds = _kind_set(original_bugs)

    target_gone = (candidate.kind, candidate.line) not in after_kind_line
    new_kinds = after_kinds - before_kinds
    verified = target_gone and not new_kinds

    if not target_gone:
        detail = "the targeted bug still fires after the edit"
    elif new_kinds:
        detail = f"the edit introduced new bug kind(s): {sorted(new_kinds)}"
    else:
        detail = "re-verified: targeted bug gone, no new bug introduced"

    return VerifiedFix(
        kind=candidate.kind,
        line=candidate.line,
        strategy=candidate.strategy,
        description=candidate.description,
        patched_source=candidate.patched_source,
        diff="",  # filled in by repair() which holds the original source
        verified=verified,
        detail=detail,
    )


def _unified_diff(original: str, patched: str, filename: str) -> str:
    a = original.splitlines(keepends=True)
    b = patched.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(a, b, fromfile=f"a/{filename}", tofile=f"b/{filename}")
    )


def repair(source: str, *, filename: str = "<unknown>", config=None,
           verified_only: bool = True) -> List[VerifiedFix]:
    """Analyze ``source``, propose a minimal fix per report, re-verify each, and
    return the verified repairs (with unified diffs).

    With ``verified_only=False`` the unverified candidates are returned too (with
    ``verified=False`` and a reason), which is useful for diagnostics."""
    from .engine import analyze_source

    result = analyze_source(source, filename=filename, config=config)
    original_bugs = result.bugs
    fixes: List[VerifiedFix] = []
    for bug in original_bugs:
        candidate = propose_fix(bug, source)
        if candidate is None:
            continue
        vf = verify_fix(candidate, original_bugs, filename=filename, config=config)
        vf = VerifiedFix(
            kind=vf.kind,
            line=vf.line,
            strategy=vf.strategy,
            description=vf.description,
            patched_source=vf.patched_source,
            diff=_unified_diff(source, candidate.patched_source, filename),
            verified=vf.verified,
            detail=vf.detail,
        )
        if vf.verified or not verified_only:
            fixes.append(vf)
    return fixes
