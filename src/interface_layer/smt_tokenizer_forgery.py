"""A bounded model checker for the tokenizer boundary, via SMT (Z3).

Every other PromptABI forgery analyzer is *enumerative*: it injects a concrete
witness and replays the real operator. That is sound but never *complete* -- it
cannot prove the absence of a forgery, and it can be fooled by the single most
LLM-specific fact about tokenization:

    a control marker's *substring being present* is NOT the same as the
    tokenizer *emitting that marker's token id*.

Under greedy longest-match sub-word segmentation a **longer** added token can
straddle and consume the marker, so its id is never produced (e.g. the vocab has
both ``<|im_start|>`` and ``X<|im_start|>``; the text ``X<|im_start|>`` segments
to the *longer* token, and ``<|im_start|>``'s id is absent). A substring/regex
filter cannot see this; the tokenizer's segmentation *precedence* decides it.

This module encodes that precedence -- the real :class:`ByteLevelTokenizer`'s
greedy longest-match-with-byte-fallback segmenter -- as SMT constraints over a
bounded attacker window, and decides:

* **forge**: synthesize an attacker string that makes the tokenizer *emit* a
  target control-token id inside a rendered prompt (respecting greedy
  precedence, and optionally avoiding a set of blocked substrings a prompt-level
  filter would reject); or
* **prove**: return UNSAT -- a *certificate* that **no** attacker input up to the
  bound (over the modelled alphabet) can make the tokenizer emit that id.

Every SAT witness is re-tokenized with the real :class:`ByteLevelTokenizer`; if
the model and the real segmenter ever disagree the result is downgraded to
``unknown`` rather than trusted. Z3 is an optional dependency (mirrors the
optional jinja2 backend); without it the prover reports ``unavailable``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .chat_templates import ChatTemplateParseError, parse_hf_chat_template_config
from .control_token_injection import (
    _build_render_fn,
    _build_tokenizer,
    _extract_special_token_decls,
)
from .tokenizers import ByteLevelTokenizer

try:  # optional, like the jinja2 backend
    import z3  # type: ignore

    _HAS_Z3 = True
except Exception:  # pragma: no cover - exercised only when z3 absent
    _HAS_Z3 = False

SMT_TOKENIZER_FORGERY_VERSION = "1.0.0"

_SENTINEL = "\u241ePROMPTABISLOT\u241e"  # SYMBOL FOR RECORD SEPARATOR + tag


class ForgeryStatus(StrEnum):
    BYPASS_FOUND = "bypass-found"
    PROVEN_UNFORGEABLE = "proven-unforgeable"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SegmentationForgeryResult:
    """Outcome of a bounded forge-or-prove query for one target control token."""

    status: ForgeryStatus
    target_token: str
    target_id: int | None = None
    bound: int | None = None
    alphabet_size: int | None = None
    witness_attacker: str | None = None
    witness_text: str | None = None
    witness_token_ids: tuple[int, ...] | None = None
    blocked_substrings: tuple[str, ...] = ()
    note: str = ""
    solver_ms: float | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "status": str(self.status),
            "target_token": self.target_token,
            "target_id": self.target_id,
            "bound": self.bound,
            "alphabet_size": self.alphabet_size,
            "note": self.note,
        }
        if self.blocked_substrings:
            data["blocked_substrings"] = list(self.blocked_substrings)
        if self.witness_attacker is not None:
            data["witness_attacker"] = self.witness_attacker
            data["witness_text"] = self.witness_text
            data["witness_token_ids"] = list(self.witness_token_ids or ())
        if self.solver_ms is not None:
            data["solver_ms"] = round(self.solver_ms, 2)
        return data


@dataclass(frozen=True, slots=True)
class TokenForgeryReport:
    """A forge-or-prove sweep over every control token of a config."""

    results: tuple[SegmentationForgeryResult, ...]
    bound: int
    abstained: bool = False
    abstain_reason: str | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def forgeable(self) -> tuple[SegmentationForgeryResult, ...]:
        return tuple(r for r in self.results if r.status == ForgeryStatus.BYPASS_FOUND)

    @property
    def proven(self) -> tuple[SegmentationForgeryResult, ...]:
        return tuple(r for r in self.results if r.status == ForgeryStatus.PROVEN_UNFORGEABLE)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": SMT_TOKENIZER_FORGERY_VERSION,
            "bound": self.bound,
            "z3_available": _HAS_Z3,
            "results": [r.to_dict() for r in self.results],
            "forgeable_count": len(self.forgeable),
            "proven_count": len(self.proven),
        }
        if self.abstained:
            data["abstained"] = True
            if self.abstain_reason:
                data["abstain_reason"] = self.abstain_reason
        if self.extras:
            data["extras"] = self.extras
        return data


def z3_available() -> bool:
    return _HAS_Z3


# --------------------------------------------------------------------------- #
# Core: forge-or-prove for one target token under the greedy segmenter.
# --------------------------------------------------------------------------- #

def prove_or_forge_token(
    *,
    target_token: str,
    added_tokens: dict[str, int],
    prefix: str = "",
    suffix: str = "",
    max_attacker_len: int,
    alphabet: set[str] | None = None,
    blocked_substrings: tuple[str, ...] = (),
    timeout_ms: int = 5000,
) -> SegmentationForgeryResult:
    """Decide whether some attacker window can make the tokenizer emit ``target_token``.

    ``added_tokens`` maps every added/special token surface to its id (this is the
    real segmentation vocabulary -- modelling it is what captures greedy
    precedence). The rendered text is ``prefix + <attacker window> + suffix`` with
    the window free over ``alphabet``. Returns a synthesized witness (validated by
    replaying the real tokenizer) or an UNSAT non-forgeability certificate.
    """

    target_id = added_tokens.get(target_token)
    if not _HAS_Z3:
        return SegmentationForgeryResult(
            status=ForgeryStatus.UNAVAILABLE,
            target_token=target_token,
            target_id=target_id,
            bound=max_attacker_len,
            note="z3 is not installed; install the 'smt' extra to enable proofs",
        )
    if target_id is None:
        return SegmentationForgeryResult(
            status=ForgeryStatus.UNKNOWN,
            target_token=target_token,
            bound=max_attacker_len,
            note="target token is not in the added-token vocabulary",
        )

    if alphabet is None:
        alphabet = set()
        for surface in added_tokens:
            alphabet.update(surface)
        alphabet.update(target_token)
        alphabet.update("a")  # a benign filler char
    alpha = sorted(alphabet)

    pre = [ord(c) for c in prefix]
    suf = [ord(c) for c in suffix]
    n = len(pre) + max_attacker_len + len(suf)

    solver = z3.Solver()
    solver.set("timeout", int(timeout_ms))
    char = [z3.Int(f"c{i}") for i in range(n)]
    for i, cp in enumerate(pre):
        solver.add(char[i] == cp)
    for k in range(max_attacker_len):
        i = len(pre) + k
        solver.add(z3.Or([char[i] == ord(a) for a in alpha]))
    for j, cp in enumerate(suf):
        solver.add(char[len(pre) + max_attacker_len + j] == cp)

    # Greedy longest-match precedence, mirroring ByteLevelTokenizer._match_added_token
    # (sorted by descending length, then surface) with single-char/byte fallback.
    toks = sorted(set(added_tokens), key=lambda t: (-len(t), t))

    def match_at(t: str, i: int):
        if i + len(t) > n:
            return z3.BoolVal(False)
        return z3.And([char[i + j] == ord(t[j]) for j in range(len(t))])

    tlen = []
    for i in range(n):
        expr = z3.IntVal(1)  # byte / single-char fallback
        for t in reversed(toks):  # longest checked first => add it last
            expr = z3.If(match_at(t, i), z3.IntVal(len(t)), expr)
        tlen.append(expr)

    # reach[k] : the greedy walk lands a token start at position k.
    reach = [z3.Bool(f"r{i}") for i in range(n + 1)]
    solver.add(reach[0])
    for k in range(1, n + 1):
        solver.add(reach[k] == z3.Or([z3.And(reach[i], (i + tlen[i]) == k) for i in range(k)]))

    # target emitted *because of the attacker*: a reached start whose greedy
    # longest match is exactly target AND whose span overlaps the attacker window
    # (a match wholly inside the fixed framing is not a forgery).
    win_lo = len(pre)
    win_hi = len(pre) + max_attacker_len
    emit = []
    for i in range(n - len(target_token) + 1):
        span_lo, span_hi = i, i + len(target_token)
        if span_lo < win_hi and span_hi > win_lo:  # overlaps attacker window
            emit.append(z3.And(reach[i], match_at(target_token, i), tlen[i] == len(target_token)))
    if not emit:
        return SegmentationForgeryResult(
            status=ForgeryStatus.PROVEN_UNFORGEABLE,
            target_token=target_token,
            target_id=target_id,
            bound=max_attacker_len,
            alphabet_size=len(alpha),
            blocked_substrings=blocked_substrings,
            note=(
                "target cannot overlap the attacker window at any position given the "
                "surrounding template framing"
            ),
        )
    solver.add(z3.Or(emit))

    for b in blocked_substrings:
        if not b:
            continue
        for i in range(n - len(b) + 1):
            solver.add(z3.Not(z3.And([char[i + j] == ord(b[j]) for j in range(len(b))])))

    started = time.perf_counter()
    verdict = solver.check()
    solver_ms = (time.perf_counter() - started) * 1000.0

    if verdict == z3.unsat:
        return SegmentationForgeryResult(
            status=ForgeryStatus.PROVEN_UNFORGEABLE,
            target_token=target_token,
            target_id=target_id,
            bound=max_attacker_len,
            alphabet_size=len(alpha),
            blocked_substrings=blocked_substrings,
            note=(
                f"UNSAT: no attacker window of length <= {max_attacker_len} over a "
                f"{len(alpha)}-symbol alphabet can make the tokenizer emit token id "
                f"{target_id} (greedy segmentation modelled exactly)."
            ),
            solver_ms=solver_ms,
        )
    if verdict != z3.sat:
        return SegmentationForgeryResult(
            status=ForgeryStatus.UNKNOWN,
            target_token=target_token,
            target_id=target_id,
            bound=max_attacker_len,
            alphabet_size=len(alpha),
            blocked_substrings=blocked_substrings,
            note=f"solver returned {verdict} (likely timeout at {timeout_ms} ms)",
            solver_ms=solver_ms,
        )

    model = solver.model()
    text = "".join(chr(model[char[i]].as_long()) for i in range(n))
    attacker = text[len(pre): len(pre) + max_attacker_len]

    # Defense in depth: replay the REAL tokenizer with a benign baseline and
    # require the witness to emit *strictly more* of the target than the framing
    # alone -- this excludes target ids contributed by the fixed prefix/suffix.
    benign_window = "a" * max_attacker_len
    benign_text = prefix + benign_window + suffix
    real_ids = _real_token_ids(added_tokens, text)
    benign_ids = _real_token_ids(added_tokens, benign_text)
    if real_ids.count(target_id) <= benign_ids.count(target_id):
        return SegmentationForgeryResult(
            status=ForgeryStatus.UNKNOWN,
            target_token=target_token,
            target_id=target_id,
            bound=max_attacker_len,
            alphabet_size=len(alpha),
            blocked_substrings=blocked_substrings,
            note="SMT model and real tokenizer disagreed on attacker-caused emission; not trusting the witness",
            solver_ms=solver_ms,
        )
    return SegmentationForgeryResult(
        status=ForgeryStatus.BYPASS_FOUND,
        target_token=target_token,
        target_id=target_id,
        bound=max_attacker_len,
        alphabet_size=len(alpha),
        witness_attacker=attacker,
        witness_text=text,
        witness_token_ids=tuple(real_ids),
        blocked_substrings=blocked_substrings,
        note=(
            "SAT: synthesized an attacker window that makes the real tokenizer emit "
            f"token id {target_id} (replay-validated), respecting greedy segmentation"
            + (" and the blocked-substring filter" if blocked_substrings else "")
            + "."
        ),
        solver_ms=solver_ms,
    )


def _real_token_ids(added_tokens: dict[str, int], text: str) -> list[int]:
    decl_specials = {surface: tid for surface, tid in added_tokens.items()}
    tok = ByteLevelTokenizer(added_tokens=list(added_tokens), special_tokens=decl_specials)
    return [t.token_id for t in tok.encode(text).tokens]


# --------------------------------------------------------------------------- #
# Config-level sweep: derive prefix/suffix from the real template.
# --------------------------------------------------------------------------- #

def _user_slot_frame(render_fn) -> tuple[str, str] | None:
    """Render a single user turn with a sentinel; return the (prefix, suffix) text
    surrounding attacker-controlled user content."""

    try:
        text = render_fn([{"role": "user", "content": _SENTINEL}], add_generation_prompt=True)
    except Exception:
        return None
    idx = text.find(_SENTINEL)
    if idx < 0:
        return None
    return text[:idx], text[idx + len(_SENTINEL):]


def scan_token_forgery(
    config: dict[str, object],
    *,
    max_attacker_len: int = 16,
    blocked_substrings: tuple[str, ...] = (),
    timeout_ms: int = 5000,
) -> TokenForgeryReport:
    """Forge-or-prove every special control token for the user-content slot."""

    if not isinstance(config, dict):
        return TokenForgeryReport(
            results=(), bound=max_attacker_len, abstained=True,
            abstain_reason="config root is not a JSON object",
        )
    try:
        parsed = parse_hf_chat_template_config(config)
    except ChatTemplateParseError as exc:
        return TokenForgeryReport(
            results=(), bound=max_attacker_len, abstained=True,
            abstain_reason=f"chat template could not be parsed: {exc}",
        )
    decls = _extract_special_token_decls(config)
    added: dict[str, int] = {
        d.text: (d.token_id if d.token_id is not None else 1000 + i)
        for i, d in enumerate(decls)
    }
    if not added:
        return TokenForgeryReport(
            results=(), bound=max_attacker_len, abstained=True,
            abstain_reason="config declares no special/added tokens",
        )
    render_fn, backend = _build_render_fn(parsed, config)
    frame = _user_slot_frame(render_fn)
    if frame is None:
        return TokenForgeryReport(
            results=(), bound=max_attacker_len, abstained=True,
            abstain_reason="could not locate the user-content slot in the template",
        )
    prefix, suffix = frame
    targets = [d.text for d in decls if d.special]
    results = []
    for target in targets:
        results.append(
            prove_or_forge_token(
                target_token=target,
                added_tokens=added,
                prefix=prefix,
                suffix=suffix,
                max_attacker_len=max_attacker_len,
                blocked_substrings=blocked_substrings,
                timeout_ms=timeout_ms,
            )
        )
    return TokenForgeryReport(
        results=tuple(results),
        bound=max_attacker_len,
        extras={"render_backend": backend, "z3_available": _HAS_Z3},
    )


def scan_token_forgery_file(path: str | Path, **kwargs) -> TokenForgeryReport:
    import json

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return scan_token_forgery(config, **kwargs)


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

def render_token_forgery_report_json(report: TokenForgeryReport) -> str:
    import json

    return json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)


def render_token_forgery_report_text(report: TokenForgeryReport) -> str:
    lines = [f"SMT token-forgery sweep (v{SMT_TOKENIZER_FORGERY_VERSION})"]
    lines.append(f"z3 available: {_HAS_Z3} | bound: {report.bound}")
    if report.abstained:
        lines.append(f"ABSTAIN: {report.abstain_reason}")
        return "\n".join(lines)
    lines.append(
        f"results: {len(report.results)} "
        f"({len(report.forgeable)} forgeable, {len(report.proven)} proven-unforgeable)"
    )
    for r in report.results:
        lines.append(f"  [{r.status}] {r.target_token!r} (id={r.target_id})")
        if r.witness_attacker is not None:
            lines.append(f"      witness attacker text: {r.witness_attacker!r}")
        if r.note:
            lines.append(f"      {r.note}")
    return "\n".join(lines)
