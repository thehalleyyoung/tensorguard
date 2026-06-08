"""Surface-ban soundness: does an id-level ``bad_words``/``suppress_tokens`` ban
actually prevent a forbidden *surface*?

A very common safety / brand-safety control is to ban specific **token ids** from
ever being generated:

* ``suppress_tokens`` / ``SuppressTokensLogitsProcessor`` -- a *set* of token ids
  whose logits are forced to ``-inf`` at every step;
* ``bad_words_ids`` / ``NoBadWordsLogitsProcessor`` -- a list of token-id
  *sequences*; the final id of a sequence is masked only when the immediately
  preceding generated ids match the sequence prefix (so it bans exact id-tuples
  as contiguous runs of the output).

The intent is almost always to prevent a forbidden *string* (a slur, a competitor
name, a leaked secret marker, a refusal-bypass phrase). But the vocabulary is
**redundant**: the same characters can be produced by a *different* tokenization.
Banning the ids of the word's natural tokenization -- even banning *every* token
whose surface literally contains the word -- does not prevent the model from
spelling it with other tokens (in a byte-level vocab, character by character).
So an id-level ban is, in general, an **unsound** way to ban a surface.

This module decides the question exactly. The set of outputs that *contain* the
forbidden surface, restricted to allowed tokens and avoiding every banned id-tuple,
is recognized by a **finite product automaton**:

    KMP(target)            -- matches the forbidden surface in the character stream
  x AhoCorasick(bad-words) -- tracks how much of a banned id-tuple has been built

with one edge per *allowed* vocabulary token (an edge is blocked exactly when the
``NoBadWordsLogitsProcessor`` would mask that token's id in the current id-context).
Reachability of a "target matched" state in this finite graph is decidable
**with no length bound**: a reachable witness is a concrete token sequence the
model can emit despite the ban (a *bypass*), and unreachability is a genuine
**soundness certificate** -- the id-ban provably prevents the surface forever.

The analysis is solver-free (pure automata reachability), complementing the
bounded SMT provers elsewhere in PromptABI with an *unbounded* decision.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .constrained_decoding_feasibility import _byte_level_decoder

SURFACE_BAN_SOUNDNESS_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Vocabulary loading (id <-> surface).
# --------------------------------------------------------------------------- #


def load_id_surfaces_from_tokenizer_json(path: str | Path) -> tuple[tuple[int, str], ...]:
    """Return ``(token_id, decoded_surface)`` pairs from a HF ``tokenizer.json``.

    Supports byte-level BPE (GPT-2 byte table) and SentencePiece unigram
    (``U+2581`` -> space). Special ``<...>`` tokens and surfaces that are not valid
    UTF-8 are dropped (a decoder never emits them as standalone text).
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    model = raw.get("model") if isinstance(raw, dict) else None
    vocab = model.get("vocab") if isinstance(model, dict) else None
    decoder = _byte_level_decoder()
    pairs: list[tuple[int, str]] = []
    if isinstance(vocab, dict):
        for key, idx in vocab.items():
            try:
                surface = bytes(decoder[ch] for ch in key).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue
            if surface:
                pairs.append((int(idx), surface))
    elif isinstance(vocab, list):
        for idx, entry in enumerate(vocab):
            tok = entry[0] if isinstance(entry, (list, tuple)) and entry else None
            if not isinstance(tok, str) or not tok:
                continue
            if tok.startswith("<") and tok.endswith(">"):
                continue
            pairs.append((idx, tok.replace("\u2581", " ")))
    else:
        raise ValueError("tokenizer.json model.vocab is neither a byte-level map nor a unigram list")
    return tuple(pairs)


# --------------------------------------------------------------------------- #
# KMP automaton for the forbidden surface.
# --------------------------------------------------------------------------- #


def _kmp_failure(pattern: str) -> list[int]:
    fail = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k > 0 and pattern[i] != pattern[k]:
            k = fail[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        fail[i] = k
    return fail


def _kmp_advance_char(pattern: str, fail: list[int], state: int, ch: str) -> int:
    k = state
    while k > 0 and ch != pattern[k]:
        k = fail[k - 1]
    if ch == pattern[k]:
        k += 1
    return k


def _kmp_run_token(pattern: str, fail: list[int], state: int, surface: str) -> tuple[int, bool]:
    """Advance the KMP state over a whole token surface; report if it ever matched."""

    k = state
    matched = False
    L = len(pattern)
    for ch in surface:
        k = _kmp_advance_char(pattern, fail, k, ch)
        if k == L:
            matched = True
            k = fail[k - 1] if L > 0 else 0
    return k, matched


# --------------------------------------------------------------------------- #
# Aho-Corasick over banned token-id tuples.
# --------------------------------------------------------------------------- #


class _BadWordAutomaton:
    """Recognizes banned token-id tuples as contiguous runs; a token id is *masked*
    from a state exactly when appending it would *complete* a banned tuple."""

    def __init__(self, sequences: Sequence[Sequence[int]]) -> None:
        # goto: state -> {id: state}; output: states that complete a banned tuple
        self._goto: list[dict[int, int]] = [dict()]
        self._fail: list[int] = [0]
        self._terminal: set[int] = set()
        self._sequences = [tuple(s) for s in sequences if s]
        for seq in self._sequences:
            self._add(seq)
        self._build_fail()

    @property
    def trivial(self) -> bool:
        return not self._sequences

    def _add(self, seq: Sequence[int]) -> None:
        node = 0
        for tid in seq:
            nxt = self._goto[node].get(tid)
            if nxt is None:
                nxt = len(self._goto)
                self._goto.append(dict())
                self._fail.append(0)
                self._goto[node][tid] = nxt
            node = nxt
        self._terminal.add(node)

    def _build_fail(self) -> None:
        queue: deque[int] = deque()
        for nxt in self._goto[0].values():
            self._fail[nxt] = 0
            queue.append(nxt)
        while queue:
            node = queue.popleft()
            for tid, nxt in self._goto[node].items():
                queue.append(nxt)
                f = self._fail[node]
                while f and tid not in self._goto[f]:
                    f = self._fail[f]
                self._fail[nxt] = self._goto[f].get(tid, 0)
                if self._fail[nxt] == nxt:  # only possible degenerate case
                    self._fail[nxt] = 0
                if self._fail[nxt] in self._terminal:
                    self._terminal.add(nxt)

    def step(self, state: int, tid: int) -> int | None:
        """Next state after consuming token id ``tid``; None if that id is *masked*
        (would complete a banned tuple and therefore be suppressed by the processor)."""

        node = state
        while node and tid not in self._goto[node]:
            node = self._fail[node]
        nxt = self._goto[node].get(tid, 0)
        if nxt in self._terminal:
            return None  # NoBadWordsLogitsProcessor masks this id here
        return nxt


# --------------------------------------------------------------------------- #
# Report types.
# --------------------------------------------------------------------------- #


class BanSoundnessStatus(StrEnum):
    BYPASS_FOUND = "bypass-found"
    PROVEN_SOUND = "proven-sound"
    ABSTAINED = "abstained"


@dataclass(frozen=True, slots=True)
class BanBypassWitness:
    """A concrete allowed-token sequence whose decoded surface contains the target."""

    token_ids: tuple[int, ...]
    token_surfaces: tuple[str, ...]
    decoded_text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "token_ids": list(self.token_ids),
            "token_surfaces": list(self.token_surfaces),
            "decoded_text": self.decoded_text,
        }


@dataclass(frozen=True, slots=True)
class SurfaceBanReport:
    target: str
    status: BanSoundnessStatus
    vocab_size: int
    suppressed_id_count: int
    bad_word_tuple_count: int
    allowed_token_count: int
    product_states_explored: int
    witness: BanBypassWitness | None = None
    reason: str | None = None
    assumptions: tuple[str, ...] = ()

    @property
    def sound(self) -> bool:
        return self.status is BanSoundnessStatus.PROVEN_SOUND

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": SURFACE_BAN_SOUNDNESS_VERSION,
            "target": self.target,
            "status": self.status.value,
            "vocab_size": self.vocab_size,
            "suppressed_id_count": self.suppressed_id_count,
            "bad_word_tuple_count": self.bad_word_tuple_count,
            "allowed_token_count": self.allowed_token_count,
            "product_states_explored": self.product_states_explored,
            "assumptions": list(self.assumptions),
        }
        if self.witness is not None:
            data["witness"] = self.witness.to_dict()
        if self.reason is not None:
            data["reason"] = self.reason
        return data


# --------------------------------------------------------------------------- #
# Core decision.
# --------------------------------------------------------------------------- #


def prove_surface_ban(
    id_surfaces: Sequence[tuple[int, str]],
    target: str,
    *,
    suppressed_ids: Iterable[int] = (),
    bad_word_id_seqs: Sequence[Sequence[int]] = (),
    max_product_states: int = 200_000,
) -> SurfaceBanReport:
    """Decide whether an id-level ban soundly prevents the forbidden surface ``target``.

    Returns ``bypass-found`` with a concrete allowed-token witness whose decoded
    surface contains ``target``, or ``proven-sound`` -- an *unbounded* certificate
    that no allowed token sequence (respecting the bad-word masking) can ever spell
    ``target``. The product automaton ``KMP(target) x AhoCorasick(bad_word_ids)`` is
    finite, so reachability is exact.
    """

    if not target:
        raise ValueError("target surface must be non-empty")

    assumptions = (
        "ban-is-over-token-ids (suppress_tokens / bad_words_ids semantics)",
        "decoder-emits-token-surfaces-concatenated",
        "product-automaton-reachability-is-complete-and-unbounded",
    )
    suppressed = frozenset(suppressed_ids)
    allowed = [(tid, surf) for tid, surf in id_surfaces if tid not in suppressed]
    bad = _BadWordAutomaton(bad_word_id_seqs)

    fail = _kmp_failure(target)
    L = len(target)

    # Precompute, for each KMP state, the token outcome (next kmp, matched) per token.
    # Stored KMP states are always < L (we reset on match), so L states suffice.
    token_effect: list[list[tuple[int, bool]]] = []
    for q in range(L):
        row = [_kmp_run_token(target, fail, q, surf) for _tid, surf in allowed]
        token_effect.append(row)

    start = (0, 0)  # (kmp_state, ac_state)
    parent: dict[tuple[int, int], tuple[tuple[int, int] | None, int]] = {start: (None, -1)}
    queue: deque[tuple[int, int]] = deque([start])
    explored = 0
    hit_state: tuple[int, int] | None = None
    bypass_edge: int = -1
    bypass_from: tuple[int, int] | None = None

    while queue:
        kq, aq = queue.popleft()
        explored += 1
        if explored > max_product_states:
            return SurfaceBanReport(
                target=target,
                status=BanSoundnessStatus.ABSTAINED,
                vocab_size=len(id_surfaces),
                suppressed_id_count=len(suppressed),
                bad_word_tuple_count=len(bad_word_id_seqs),
                allowed_token_count=len(allowed),
                product_states_explored=explored,
                reason=f"product exploration exceeded max_product_states={max_product_states}",
                assumptions=assumptions,
            )
        row = token_effect[kq]
        for edge, (tid, _surf) in enumerate(allowed):
            new_ac = bad.step(aq, tid)
            if new_ac is None:
                continue  # this id is masked in the current context
            new_kmp, matched = row[edge]
            if matched:
                hit_state = (kq, aq)
                bypass_edge = edge
                bypass_from = (kq, aq)
                queue.clear()
                break
            nxt = (new_kmp, new_ac)
            if nxt not in parent:
                parent[nxt] = ((kq, aq), edge)
                queue.append(nxt)

    if hit_state is not None and bypass_from is not None:
        # Reconstruct the token path: predecessors up to bypass_from, then the matching edge.
        path_edges: list[int] = []
        node: tuple[int, int] | None = bypass_from
        while node is not None and parent[node][0] is not None:
            prev, edge = parent[node]
            path_edges.append(edge)
            node = prev
        path_edges.reverse()
        path_edges.append(bypass_edge)
        ids = tuple(allowed[e][0] for e in path_edges)
        surfaces = tuple(allowed[e][1] for e in path_edges)
        decoded = "".join(surfaces)
        witness = BanBypassWitness(token_ids=ids, token_surfaces=surfaces, decoded_text=decoded)
        # Honest validation: the decoded surface must actually contain the target.
        if target not in decoded:
            return SurfaceBanReport(
                target=target,
                status=BanSoundnessStatus.ABSTAINED,
                vocab_size=len(id_surfaces),
                suppressed_id_count=len(suppressed),
                bad_word_tuple_count=len(bad_word_id_seqs),
                allowed_token_count=len(allowed),
                product_states_explored=explored,
                reason="internal: reconstructed witness did not contain target",
                assumptions=assumptions,
            )
        return SurfaceBanReport(
            target=target,
            status=BanSoundnessStatus.BYPASS_FOUND,
            vocab_size=len(id_surfaces),
            suppressed_id_count=len(suppressed),
            bad_word_tuple_count=len(bad_word_id_seqs),
            allowed_token_count=len(allowed),
            product_states_explored=explored,
            witness=witness,
            assumptions=assumptions,
        )

    return SurfaceBanReport(
        target=target,
        status=BanSoundnessStatus.PROVEN_SOUND,
        vocab_size=len(id_surfaces),
        suppressed_id_count=len(suppressed),
        bad_word_tuple_count=len(bad_word_id_seqs),
        allowed_token_count=len(allowed),
        product_states_explored=explored,
        assumptions=assumptions,
    )


def naive_substring_suppression(
    id_surfaces: Sequence[tuple[int, str]], target: str
) -> tuple[int, ...]:
    """The ids a thorough-but-naive defender would suppress: every token whose
    surface literally contains the forbidden ``target`` substring."""

    return tuple(tid for tid, surf in id_surfaces if target in surf)


# --------------------------------------------------------------------------- #
# Reporters.
# --------------------------------------------------------------------------- #


def render_surface_ban_report_json(report: SurfaceBanReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def render_surface_ban_report_text(report: SurfaceBanReport) -> str:
    lines = [
        f"surface-ban soundness: {report.target!r}",
        f"  status: {report.status.value}",
        f"  vocab: {report.vocab_size} ids; suppressed {report.suppressed_id_count}; "
        f"bad-word tuples {report.bad_word_tuple_count}; allowed {report.allowed_token_count}",
        f"  product states explored: {report.product_states_explored}",
    ]
    if report.reason:
        lines.append(f"  note: {report.reason}")
    if report.witness is not None:
        w = report.witness
        lines.append(
            f"  BYPASS via {len(w.token_ids)} allowed token(s) "
            f"ids={list(w.token_ids)} surfaces={list(w.token_surfaces)}"
        )
        lines.append(f"    decoded output contains target: {w.decoded_text!r}")
    if report.status is BanSoundnessStatus.PROVEN_SOUND:
        lines.append("  -> no allowed token sequence can ever spell the target (sound, unbounded)")
    return "\n".join(lines)
