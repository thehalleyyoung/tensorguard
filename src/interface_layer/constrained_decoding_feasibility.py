"""Constrained-decoding feasibility: does the *token-restricted* decoder get stuck?

Guided / constrained decoding (``guided_regex`` / ``guided_json`` in vLLM,
outlines, xgrammar, llguidance, lm-format-enforcer) compiles a user grammar into
a character-level automaton ``G`` and then, at every step, masks the model's
logits to the vocabulary tokens that keep the output inside ``L(G)``. The grammar
is over **characters**, but the decoder may only ever emit whole **vocabulary
tokens** whose surfaces are multi-character. That mismatch is the source of a
whole class of subtle, LLM-specific bugs that a regex/grammar checker alone --
and even a tokenizer-only checker -- cannot see:

* **Decoder stall (dead-end / livelock).** A grammar state is reachable by legal
  tokens, but *no* vocabulary token can leave it while staying in the grammar
  (every token that starts correctly *overshoots* into an illegal character), and
  the state is not accepting. The constrained decoder is now stuck: it must emit
  an off-grammar token, force EOS in the middle of a value, or loop until the
  length cap. Classic real example: the grammar needs a closing ``"`` but every
  vocab token containing ``"`` is a *merged* token like ``",`` or ``"}`` that
  carries extra characters the grammar rejects at that point.

* **Expressivity gap.** A string is in ``L(G)`` -- the user considers it a valid
  output -- but it has **no** segmentation into vocabulary tokens, so guided
  decoding can *never* produce it. The constraint silently shrinks the model's
  output language; some legal answers become unreachable.

This module decides both. The **stall** analysis is an exact greatest-fixpoint
over the *token-lifted* grammar automaton (the grammar's states, with an edge
``q -> q'`` whenever some vocabulary token's surface drives ``G`` from ``q`` to
``q'`` without ever leaving the language): complete, no bound. The **gap**
analysis is a bounded SMT model check -- and it uses Z3 *cleverly*: the existence
of a tokenization is itself a positive reachability predicate ``seg[k]`` (``seg``
is fully determined by the chosen characters), so the otherwise quantifier-
alternating question "``exists s in L(G)`` such that ``not exists`` a
tokenization" becomes a single quantifier-free assertion ``accept(s) and
not seg[len(s)]``. Z3 then synthesizes the shortest valid-but-unemittable string
or returns UNSAT -- a bounded certificate that the grammar is fully decodable
under this tokenizer.

The regex front-end compiles a practical regex subset to a
:class:`~promptabi.formal.DeterministicFiniteAutomaton` (alphabet-relative, like
the rest of PromptABI's bounded provers). Z3 is optional; without it the stall
analysis still runs (it needs no solver) and the gap check reports
``unavailable``.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .formal import DeterministicFiniteAutomaton

try:  # optional, like the jinja2 / z3 backends elsewhere
    import z3  # type: ignore

    _HAS_Z3 = True
except Exception:  # pragma: no cover - exercised only when z3 absent
    _HAS_Z3 = False

CONSTRAINED_DECODING_FEASIBILITY_VERSION = "1.0.0"


def z3_available() -> bool:
    """Whether the optional Z3 solver is importable (needed only for gap checks)."""

    return _HAS_Z3


# --------------------------------------------------------------------------- #
# Regex -> DFA front-end (a practical, alphabet-relative subset).
# --------------------------------------------------------------------------- #


class RegexCompileError(ValueError):
    """The regex used a construct outside PromptABI's supported decoding subset."""


_ANY = object()  # marker for '.'


@dataclass(frozen=True, slots=True)
class _CharSet:
    """A single-position matcher: an explicit char set, or the negation of one."""

    chars: frozenset[str]
    negated: bool = False
    any_char: bool = False

    def resolve(self, alphabet: frozenset[str]) -> frozenset[str]:
        if self.any_char:
            return alphabet
        if self.negated:
            return frozenset(alphabet - self.chars)
        return frozenset(self.chars & alphabet)


# AST: ("cs", _CharSet) | ("concat", [..]) | ("alt", [..]) | ("star"|"plus"|"opt", node)
#      | ("rep", node, m, n|None) | ("empty",)


class _RegexParser:
    _CLASS_SHORTHAND = {
        "d": frozenset("0123456789"),
        "w": frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"),
        "s": frozenset(" \t\n\r\f\v"),
    }

    def __init__(self, pattern: str) -> None:
        self._s = pattern
        self._i = 0
        self.literals: set[str] = set()

    def parse(self):
        node = self._alt()
        if self._i != len(self._s):
            raise RegexCompileError(f"unexpected character at position {self._i}: {self._s[self._i]!r}")
        return node

    def _peek(self) -> str | None:
        return self._s[self._i] if self._i < len(self._s) else None

    def _next(self) -> str:
        ch = self._s[self._i]
        self._i += 1
        return ch

    def _alt(self):
        branches = [self._concat()]
        while self._peek() == "|":
            self._next()
            branches.append(self._concat())
        if len(branches) == 1:
            return branches[0]
        return ("alt", branches)

    def _concat(self):
        items = []
        while True:
            ch = self._peek()
            if ch is None or ch in "|)":
                break
            items.append(self._repeat())
        if not items:
            return ("empty",)
        if len(items) == 1:
            return items[0]
        return ("concat", items)

    def _repeat(self):
        atom = self._atom()
        ch = self._peek()
        if ch == "*":
            self._next()
            return ("star", atom)
        if ch == "+":
            self._next()
            return ("plus", atom)
        if ch == "?":
            self._next()
            return ("opt", atom)
        if ch == "{":
            return self._counted(atom)
        return atom

    def _counted(self, atom):
        assert self._next() == "{"
        digits_lo = ""
        while self._peek() is not None and self._peek().isdigit():
            digits_lo += self._next()
        if digits_lo == "":
            raise RegexCompileError("'{' quantifier requires a lower bound")
        lo = int(digits_lo)
        hi: int | None = lo
        if self._peek() == ",":
            self._next()
            digits_hi = ""
            while self._peek() is not None and self._peek().isdigit():
                digits_hi += self._next()
            hi = int(digits_hi) if digits_hi else None
        if self._peek() != "}":
            raise RegexCompileError("unterminated '{' quantifier")
        self._next()
        if hi is not None and hi < lo:
            raise RegexCompileError(f"quantifier upper bound {hi} below lower bound {lo}")
        return ("rep", atom, lo, hi)

    def _atom(self):
        ch = self._peek()
        if ch is None:
            raise RegexCompileError("unexpected end of pattern (expected an atom)")
        if ch == "(":
            self._next()
            # non-capturing or capturing groups both behave the same here.
            if self._s[self._i : self._i + 2] == "?:":
                self._i += 2
            node = self._alt()
            if self._peek() != ")":
                raise RegexCompileError("unbalanced '('")
            self._next()
            return node
        if ch == "[":
            return ("cs", self._char_class())
        if ch == ".":
            self._next()
            return ("cs", _CharSet(frozenset(), any_char=True))
        if ch == "\\":
            return ("cs", self._escape())
        if ch in ")*+?{}|":
            raise RegexCompileError(f"unexpected metacharacter {ch!r} at position {self._i}")
        self._next()
        self.literals.add(ch)
        return ("cs", _CharSet(frozenset({ch})))

    def _escape(self) -> _CharSet:
        assert self._next() == "\\"
        if self._peek() is None:
            raise RegexCompileError("trailing backslash")
        esc = self._next()
        if esc in self._CLASS_SHORTHAND:
            chars = self._CLASS_SHORTHAND[esc]
            self.literals |= chars
            return _CharSet(chars)
        if esc in {"D", "W", "S"}:
            chars = self._CLASS_SHORTHAND[esc.lower()]
            return _CharSet(chars, negated=True)
        mapped = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v"}.get(esc, esc)
        self.literals.add(mapped)
        return _CharSet(frozenset({mapped}))

    def _char_class(self) -> _CharSet:
        assert self._next() == "["
        negated = False
        if self._peek() == "^":
            self._next()
            negated = True
        chars: set[str] = set()
        if self._peek() == "]":  # literal ] as first member
            chars.add(self._next())
        while True:
            ch = self._peek()
            if ch is None:
                raise RegexCompileError("unterminated character class")
            if ch == "]":
                self._next()
                break
            if ch == "\\":
                sub = self._escape()
                if sub.negated:
                    raise RegexCompileError("negated shorthand inside a character class is unsupported")
                chars |= set(sub.chars)
                continue
            self._next()
            if self._peek() == "-" and self._i + 1 < len(self._s) and self._s[self._i + 1] != "]":
                self._next()  # consume '-'
                hi = self._next()
                if ord(hi) < ord(ch):
                    raise RegexCompileError(f"inverted character range {ch!r}-{hi!r}")
                for code in range(ord(ch), ord(hi) + 1):
                    chars.add(chr(code))
            else:
                chars.add(ch)
        self.literals |= chars
        return _CharSet(frozenset(chars), negated=negated)


class _NFA:
    """Thompson NFA with explicit epsilon edges over integer states."""

    def __init__(self) -> None:
        self.trans: dict[int, list[tuple[str | None, int]]] = {}
        self._n = 0

    def new_state(self) -> int:
        s = self._n
        self._n += 1
        self.trans[s] = []
        return s

    def add(self, src: int, sym: str | None, dst: int) -> None:
        self.trans[src].append((sym, dst))


def _ast_to_nfa(node, nfa: _NFA, alphabet: frozenset[str]) -> tuple[int, int]:
    kind = node[0]
    if kind == "empty":
        s = nfa.new_state()
        return s, s
    if kind == "cs":
        start = nfa.new_state()
        accept = nfa.new_state()
        for ch in sorted(node[1].resolve(alphabet)):
            nfa.add(start, ch, accept)
        return start, accept
    if kind == "concat":
        start = None
        prev_accept = None
        for child in node[1]:
            cs, ca = _ast_to_nfa(child, nfa, alphabet)
            if start is None:
                start = cs
            else:
                nfa.add(prev_accept, None, cs)
            prev_accept = ca
        if start is None:  # empty concat
            s = nfa.new_state()
            return s, s
        return start, prev_accept
    if kind == "alt":
        start = nfa.new_state()
        accept = nfa.new_state()
        for child in node[1]:
            cs, ca = _ast_to_nfa(child, nfa, alphabet)
            nfa.add(start, None, cs)
            nfa.add(ca, None, accept)
        return start, accept
    if kind in {"star", "plus", "opt"}:
        cs, ca = _ast_to_nfa(node[1], nfa, alphabet)
        start = nfa.new_state()
        accept = nfa.new_state()
        nfa.add(start, None, cs)
        nfa.add(ca, None, accept)
        if kind != "opt":  # star/plus loop back
            nfa.add(ca, None, cs)
        if kind != "plus":  # star/opt may skip
            nfa.add(start, None, accept)
        return start, accept
    if kind == "rep":
        _, child, lo, hi = node
        pieces: list[tuple[int, int]] = []
        if hi is None:
            for _ in range(lo):
                pieces.append(_ast_to_nfa(child, nfa, alphabet))
            star_s, star_a = _ast_to_nfa(("star", child), nfa, alphabet)
            pieces.append((star_s, star_a))
        else:
            for idx in range(hi):
                if idx < lo:
                    pieces.append(_ast_to_nfa(child, nfa, alphabet))
                else:
                    pieces.append(_ast_to_nfa(("opt", child), nfa, alphabet))
        if not pieces:
            s = nfa.new_state()
            return s, s
        for (_, a), (s2, _) in zip(pieces, pieces[1:]):
            nfa.add(a, None, s2)
        return pieces[0][0], pieces[-1][1]
    raise RegexCompileError(f"unsupported AST node {kind!r}")


def _epsilon_closure(nfa: _NFA, states: Iterable[int]) -> frozenset[int]:
    stack = list(states)
    seen = set(stack)
    while stack:
        s = stack.pop()
        for sym, dst in nfa.trans[s]:
            if sym is None and dst not in seen:
                seen.add(dst)
                stack.append(dst)
    return frozenset(seen)


def regex_to_dfa(
    pattern: str,
    *,
    extra_alphabet: Iterable[str] = (),
    name: str = "regex",
) -> DeterministicFiniteAutomaton:
    """Compile a regex (full-match) to a partial DFA over a closed working alphabet.

    The alphabet is ``regex literals + extra_alphabet`` (e.g. the characters that
    appear in the tokenizer's vocabulary). ``.`` and negated classes resolve
    against this closed alphabet, so results are *alphabet-relative* -- the same
    honest bound-relativity the rest of PromptABI's provers use.
    """

    parser = _RegexParser(pattern)
    ast = parser.parse()
    alphabet = frozenset(parser.literals) | frozenset(extra_alphabet)
    alphabet = frozenset(ch for ch in alphabet if ch != "")
    if not alphabet:
        raise RegexCompileError("empty working alphabet (regex matched no characters)")

    nfa = _NFA()
    nfa_start, nfa_accept = _ast_to_nfa(ast, nfa, alphabet)

    start_set = _epsilon_closure(nfa, [nfa_start])
    state_names: dict[frozenset[int], str] = {start_set: "d0"}
    transitions: dict[tuple[str, str], str] = {}
    accepts: set[str] = set()
    order = [start_set]
    queue: deque[frozenset[int]] = deque([start_set])
    while queue:
        cur = queue.popleft()
        cur_name = state_names[cur]
        if nfa_accept in cur:
            accepts.add(cur_name)
        for sym in sorted(alphabet):
            move: set[int] = set()
            for s in cur:
                for tsym, dst in nfa.trans[s]:
                    if tsym == sym:
                        move.add(dst)
            if not move:
                continue
            target = _epsilon_closure(nfa, move)
            if target not in state_names:
                state_names[target] = f"d{len(state_names)}"
                order.append(target)
                queue.append(target)
            transitions[(cur_name, sym)] = state_names[target]

    states = frozenset(state_names.values())
    return DeterministicFiniteAutomaton(
        states=states,
        alphabet=tuple(sorted(alphabet)),
        start="d0",
        accepts=frozenset(accepts),
        transitions={(src, sym): dst for (src, sym), dst in transitions.items()},
        name=name,
    )


# --------------------------------------------------------------------------- #
# Tokenizer vocabulary surfaces.
# --------------------------------------------------------------------------- #


def _byte_level_decoder() -> dict[str, int]:
    """Map each GPT-2 byte-level *surface character* back to its byte value."""

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(
        range(ord("\xae"), ord("\xff") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load_vocab_surfaces_from_tokenizer_json(path: str | Path) -> tuple[str, ...]:
    """Decode the real subword surfaces from a HF ``tokenizer.json`` ``model.vocab``.

    Byte-level BPE keys (with ``Ġ`` etc.) are mapped back to bytes via the GPT-2
    table and decoded as UTF-8. Surfaces that are not valid UTF-8 are dropped (a
    decoder never emits them as standalone text at a grammar boundary).
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    model = raw.get("model") if isinstance(raw, dict) else None
    vocab = model.get("vocab") if isinstance(model, dict) else None
    decoder = _byte_level_decoder()
    surfaces: list[str] = []
    if isinstance(vocab, dict):
        # byte-level BPE: keys are GPT-2 byte-surface strings.
        for key in vocab:
            try:
                data = bytes(decoder[ch] for ch in key)
                surfaces.append(data.decode("utf-8"))
            except (KeyError, UnicodeDecodeError):
                continue
    elif isinstance(vocab, list):
        # SentencePiece unigram: entries are [token, score]; U+2581 is the space marker.
        for entry in vocab:
            tok = entry[0] if isinstance(entry, (list, tuple)) and entry else None
            if not isinstance(tok, str) or not tok:
                continue
            if tok.startswith("<") and tok.endswith(">"):
                continue  # special/control token, not emittable text at a boundary
            surfaces.append(tok.replace("\u2581", " "))
    else:
        raise ValueError("tokenizer.json model.vocab is neither a byte-level map nor a unigram list")
    # also include any explicit added_tokens content surfaces
    for entry in raw.get("added_tokens", []) if isinstance(raw, dict) else []:
        content = entry.get("content") if isinstance(entry, dict) else None
        if isinstance(content, str) and content:
            surfaces.append(content)
    return tuple(dict.fromkeys(s for s in surfaces if s))


# --------------------------------------------------------------------------- #
# Feasibility report types.
# --------------------------------------------------------------------------- #


class FeasibilityStatus(StrEnum):
    STALL_FOUND = "stall-found"
    GAP_FOUND = "gap-found"
    FEASIBLE = "feasible"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class StallWitness:
    """A grammar state the token-restricted decoder can reach but never legally leave."""

    state: str
    kind: str  # "dead-end" (no legal token at all) | "livelock" (tokens loop, never accept)
    reach_text: str  # a concatenation of vocab surfaces that drives G into this state
    reach_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "kind": self.kind,
            "reach_text": self.reach_text,
            "reach_tokens": list(self.reach_tokens),
        }


@dataclass(frozen=True, slots=True)
class GapWitness:
    """A grammar-accepted string that has no tokenization (the decoder can't emit it)."""

    text: str
    length: int

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "length": self.length}


@dataclass(frozen=True, slots=True)
class DecodingFeasibilityReport:
    grammar_name: str
    status: FeasibilityStatus
    grammar_state_count: int
    grammar_accept_count: int
    vocab_size: int
    pruned_vocab_size: int
    gap_bound: int
    gap_certified: bool
    stalls: tuple[StallWitness, ...] = ()
    gaps: tuple[GapWitness, ...] = ()
    reason: str | None = None
    assumptions: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.status is FeasibilityStatus.FEASIBLE

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": CONSTRAINED_DECODING_FEASIBILITY_VERSION,
            "grammar_name": self.grammar_name,
            "status": self.status.value,
            "grammar_state_count": self.grammar_state_count,
            "grammar_accept_count": self.grammar_accept_count,
            "vocab_size": self.vocab_size,
            "pruned_vocab_size": self.pruned_vocab_size,
            "gap_bound": self.gap_bound,
            "gap_certified": self.gap_certified,
            "stall_count": len(self.stalls),
            "gap_count": len(self.gaps),
            "stalls": [s.to_dict() for s in self.stalls],
            "gaps": [g.to_dict() for g in self.gaps],
            "assumptions": list(self.assumptions),
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


# --------------------------------------------------------------------------- #
# Core feasibility decision.
# --------------------------------------------------------------------------- #


def _token_step(dfa: DeterministicFiniteAutomaton, state: str, surface: str) -> str | None:
    """Walk a whole token surface through the char DFA; None if it ever leaves L(G)."""

    cur: str | None = state
    for ch in surface:
        cur = dfa.step(cur, ch)
        if cur is None:
            return None
    return cur


def _prune_vocab(dfa: DeterministicFiniteAutomaton, vocab: Sequence[str]) -> tuple[str, ...]:
    # Only characters that actually label a grammar transition can ever be legal;
    # a surface containing any other character overshoots from every state.
    effective = {sym for (_src, sym) in dfa.transitions}
    pruned = [s for s in dict.fromkeys(vocab) if s and all(ch in effective for ch in s)]
    return tuple(pruned)


def _index_by_first_char(vocab: Sequence[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for surface in vocab:
        index.setdefault(surface[0], []).append(surface)
    return index


def _find_stalls(
    dfa: DeterministicFiniteAutomaton,
    vocab: Sequence[str],
    *,
    max_stalls: int,
) -> tuple[StallWitness, ...]:
    index = _index_by_first_char(vocab)
    # The only tokens worth trying from a state are those whose first char has a
    # defined transition out of that state (every other token overshoots at char 0).
    defined_first: dict[str, set[str]] = {}
    for (src, sym) in dfa.transitions:
        defined_first.setdefault(src, set()).add(sym)

    # Forward reachability over token edges, recording a shortest token path.
    reach_path: dict[str, tuple[str, ...]] = {dfa.start: ()}
    edges: dict[str, set[str]] = {dfa.start: set()}
    queue: deque[str] = deque([dfa.start])
    while queue:
        q = queue.popleft()
        for ch in defined_first.get(q, ()):  # type: ignore[arg-type]
            for surface in index.get(ch, ()):
                nxt = _token_step(dfa, q, surface)
                if nxt is None:
                    continue
                edges.setdefault(q, set()).add(nxt)
                if nxt not in reach_path:
                    reach_path[nxt] = reach_path[q] + (surface,)
                    edges.setdefault(nxt, set())
                    queue.append(nxt)

    reachable = set(reach_path)
    # Greatest set of states that can still legally finish: accepting, or a token
    # edge into another finishable state (co-reachability fixpoint).
    good = {q for q in reachable if q in dfa.accepts}
    changed = True
    while changed:
        changed = False
        for q in reachable:
            if q in good:
                continue
            if any(t in good for t in edges.get(q, ())):
                good.add(q)
                changed = True

    stalls: list[StallWitness] = []
    for q in sorted(reachable - good):
        kind = "dead-end" if not edges.get(q) else "livelock"
        tokens = reach_path[q]
        stalls.append(
            StallWitness(state=q, kind=kind, reach_text="".join(tokens), reach_tokens=tokens)
        )
        if len(stalls) >= max_stalls:
            break
    return tuple(stalls)


def _segmentable(vocab: Sequence[str], text: str) -> bool:
    """Reference DP: can `text` be split into a concatenation of vocab surfaces?"""

    n = len(text)
    seg = [False] * (n + 1)
    seg[0] = True
    bylen: dict[int, list[str]] = {}
    for w in vocab:
        bylen.setdefault(len(w), []).append(w)
    for k in range(1, n + 1):
        for length, words in bylen.items():
            if length > k or not seg[k - length]:
                continue
            chunk = text[k - length : k]
            if chunk in words:
                seg[k] = True
                break
    return seg[n]


def _find_gap_smt(
    dfa: DeterministicFiniteAutomaton,
    vocab: Sequence[str],
    *,
    max_len: int,
    timeout_ms: int,
) -> GapWitness | None:
    state_ids = {name: i for i, name in enumerate(sorted(dfa.states))}
    dead = len(state_ids)
    accept_ids = frozenset(state_ids[s] for s in dfa.accepts)
    alpha = sorted({ord(ch) for ch in dfa.alphabet})
    # transition table keyed by (state_id, codepoint)
    table: dict[tuple[int, int], int] = {}
    for (src, sym), dst in dfa.transitions.items():
        table[(state_ids[src], ord(sym))] = state_ids[dst]

    bylen: dict[int, list[str]] = {}
    for w in vocab:
        bylen.setdefault(len(w), []).append(w)

    for n in range(1, max_len + 1):
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        chars = [z3.Int(f"c{i}") for i in range(n)]
        for c in chars:
            solver.add(z3.Or([c == cp for cp in alpha]))
        st = [z3.Int(f"st{i}") for i in range(n + 1)]
        solver.add(st[0] == state_ids[dfa.start])
        for k in range(n):
            expr = z3.IntVal(dead)
            for (sid, cp), dst in table.items():
                expr = z3.If(z3.And(st[k] == sid, chars[k] == cp), z3.IntVal(dst), expr)
            solver.add(st[k + 1] == expr)
        solver.add(z3.Or([st[n] == a for a in accept_ids]))

        seg = [z3.Bool(f"seg{i}") for i in range(n + 1)]
        solver.add(seg[0])
        for k in range(1, n + 1):
            options = []
            for length, words in bylen.items():
                if length > k:
                    continue
                for w in words:
                    match = [chars[k - length + j] == ord(w[j]) for j in range(length)]
                    options.append(z3.And(seg[k - length], *match))
            solver.add(seg[k] == (z3.Or(options) if options else z3.BoolVal(False)))
        solver.add(z3.Not(seg[n]))

        verdict = solver.check()
        if verdict == z3.sat:
            model = solver.model()
            text = "".join(chr(model[c].as_long()) for c in chars)
            # Validate honestly against the real DFA + reference segmentation DP.
            if dfa.accepts_text(text) and not _segmentable(vocab, text):
                return GapWitness(text=text, length=n)
        # unsat at length n: keep widening; only the final n decides certification.
    return None


def prove_decoding_feasibility(
    dfa: DeterministicFiniteAutomaton,
    vocab: Sequence[str],
    *,
    grammar_name: str = "grammar",
    max_gap_len: int = 24,
    max_stalls: int = 16,
    max_gap_vocab: int = 1500,
    timeout_ms: int = 5000,
) -> DecodingFeasibilityReport:
    """Decide whether a guided-decoding grammar is feasible under a tokenizer vocab.

    Returns a report containing any **stalls** (states the token-restricted decoder
    can reach but never legally leave -- complete, no bound) and any **expressivity
    gaps** (grammar-accepted strings with no tokenization -- bounded SMT, shortest
    witness). ``FEASIBLE`` means no stalls *and* no gaps up to ``max_gap_len``.
    """

    assumptions = (
        "guided-decoding-masks-to-whole-vocabulary-tokens",
        "grammar-is-character-level-dfa",
        f"gap-search-bounded-len<={max_gap_len}",
        "alphabet-relative-to-vocab-and-grammar",
    )
    pruned = _prune_vocab(dfa, vocab)
    stalls = _find_stalls(dfa, pruned, max_stalls=max_stalls)

    gaps: tuple[GapWitness, ...] = ()
    gap_certified = False
    reason: str | None = None
    if not _HAS_Z3:
        reason = "z3 not installed: stalls analyzed (complete); gap check skipped"
    elif len(pruned) > max_gap_vocab:
        reason = (
            f"gap check skipped: {len(pruned)} grammar-relevant surfaces exceeds "
            f"max_gap_vocab={max_gap_vocab} (stalls analyzed completely)"
        )
    else:
        witness = _find_gap_smt(dfa, pruned, max_len=max_gap_len, timeout_ms=timeout_ms)
        if witness is not None:
            gaps = (witness,)
        else:
            gap_certified = True

    if stalls:
        status = FeasibilityStatus.STALL_FOUND
    elif gaps:
        status = FeasibilityStatus.GAP_FOUND
    elif gap_certified:
        status = FeasibilityStatus.FEASIBLE
    else:
        status = FeasibilityStatus.UNAVAILABLE

    return DecodingFeasibilityReport(
        grammar_name=grammar_name,
        status=status,
        grammar_state_count=len(dfa.states),
        grammar_accept_count=len(dfa.accepts),
        vocab_size=len(tuple(dict.fromkeys(vocab))),
        pruned_vocab_size=len(pruned),
        gap_bound=max_gap_len,
        gap_certified=gap_certified,
        stalls=stalls,
        gaps=gaps,
        reason=reason,
        assumptions=assumptions,
    )


def prove_regex_decoding_feasibility(
    pattern: str,
    vocab: Sequence[str],
    *,
    grammar_name: str | None = None,
    max_gap_len: int = 24,
    max_stalls: int = 16,
    max_gap_vocab: int = 1500,
    timeout_ms: int = 5000,
) -> DecodingFeasibilityReport:
    """Compile a ``guided_regex`` pattern and decide its decoding feasibility."""

    vocab_alpha = {ch for surface in vocab for ch in surface}
    dfa = regex_to_dfa(pattern, extra_alphabet=vocab_alpha, name=grammar_name or pattern)
    return prove_decoding_feasibility(
        dfa,
        vocab,
        grammar_name=grammar_name or pattern,
        max_gap_len=max_gap_len,
        max_stalls=max_stalls,
        max_gap_vocab=max_gap_vocab,
        timeout_ms=timeout_ms,
    )


# --------------------------------------------------------------------------- #
# Reporters.
# --------------------------------------------------------------------------- #


def render_decoding_feasibility_report_json(report: DecodingFeasibilityReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def render_decoding_feasibility_report_text(report: DecodingFeasibilityReport) -> str:
    lines = [
        f"constrained-decoding feasibility: {report.grammar_name}",
        f"  status: {report.status.value}",
        f"  grammar states: {report.grammar_state_count} ({report.grammar_accept_count} accepting)",
        f"  vocab: {report.vocab_size} surfaces ({report.pruned_vocab_size} grammar-relevant)",
        f"  gap search bound: {report.gap_bound} (certified={report.gap_certified})",
    ]
    if report.reason:
        lines.append(f"  note: {report.reason}")
    for stall in report.stalls:
        lines.append(
            f"  STALL [{stall.kind}] state={stall.state} "
            f"after {len(stall.reach_tokens)} token(s) -> {stall.reach_text!r}"
        )
    for gap in report.gaps:
        lines.append(f"  GAP unemittable valid output (len {gap.length}): {gap.text!r}")
    if report.status is FeasibilityStatus.FEASIBLE:
        lines.append("  -> no stalls; no expressivity gaps within bound (decoding feasible)")
    return "\n".join(lines)
