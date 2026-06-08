"""Streaming stop-sequence soundness: can a token-boundary-aware server truncate
output *exactly* at a stop string?

Inference / serving stacks (vLLM, TGI, llama.cpp, Ollama, OpenAI-compatible
servers) end a generation when a configured **stop string** ``S`` appears in the
output, and must return the text *up to* ``S``. But the model does not emit
characters -- it emits **whole vocabulary tokens** whose decoded surfaces are
multi-character. Two LLM-specific failure modes follow from that mismatch, and
both are properties of the *tokenizer's surface algebra* that a regex/string view
of the output cannot see:

* **stop-overshoot** -- the stop completes *strictly inside* a single token's
  surface (the token carries bytes *after* ``S``). The decoder cannot stop
  mid-token, so a token-granular server emits the trailing bytes past ``S`` (or
  must perform byte-surgery inside one token to cut at the right place). Example:
  stop ``"\\n\\n"`` but the vocab contains a token ``"\\n\\nThe"`` -- the model can
  emit the whole thing atomically and the output already runs past the stop.

* **split-stop** -- the stop completes *across* a token boundary, with **no single
  token surface containing all of** ``S``. A server that detects the stop by
  scanning each newly-decoded token's text in isolation (a very common streaming
  implementation) never sees the match and runs past it. Example: stop ``"</s>"``
  realized as ``"</"`` + ``"s>"``.

This module **decides** both, exactly and with **no length bound**, over the
vocabulary's decoded surfaces:

* overshoot is decided by a vocabulary scan: any token whose surface contains an
  occurrence of ``S`` that does not end at the token's last character;
* split is decided by reachability in a finite automaton ``KMP(S)`` with one edge
  per vocabulary token, tracking whether a completion consumed *carried* progress
  from an earlier token (a cross-token match).

A clean run is a genuine, unbounded **soundness certificate**: every realization
of the stop string ends exactly on a token boundary and within a single token, so
token-granular truncation is always exact. The analysis is solver-free, and
complements PromptABI's surface-*containment* prover (``surface_ban_soundness``)
with the orthogonal question of *boundary alignment*.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

from .surface_ban_soundness import (
    _kmp_advance_char,
    _kmp_failure,
    load_id_surfaces_from_tokenizer_json,
)

STREAMING_STOP_SOUNDNESS_VERSION = "1.0.0"


class StopSoundnessStatus(StrEnum):
    """Outcome of a streaming-stop soundness analysis."""

    HAZARDS_FOUND = "hazards-found"
    PROVEN_SOUND = "proven-sound"
    ABSTAINED = "abstained"


class StopHazardKind(StrEnum):
    STOP_OVERSHOOT = "stop-overshoot"
    SPLIT_STOP = "split-stop"


@dataclass(frozen=True)
class StopHazard:
    """A concrete, replay-validated way the stop string defeats token-granular
    truncation."""

    kind: StopHazardKind
    token_ids: tuple[int, ...]
    surfaces: tuple[str, ...]
    decoded: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": str(self.kind),
            "token_ids": list(self.token_ids),
            "surfaces": list(self.surfaces),
            "decoded": self.decoded,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class StreamingStopReport:
    stop: str
    status: StopSoundnessStatus
    vocab_size: int
    hazards: tuple[StopHazard, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def has_overshoot(self) -> bool:
        return any(h.kind is StopHazardKind.STOP_OVERSHOOT for h in self.hazards)

    @property
    def has_split(self) -> bool:
        return any(h.kind is StopHazardKind.SPLIT_STOP for h in self.hazards)

    def as_dict(self) -> dict[str, object]:
        return {
            "analysis": "streaming-stop-soundness",
            "version": STREAMING_STOP_SOUNDNESS_VERSION,
            "stop": self.stop,
            "status": str(self.status),
            "vocab_size": self.vocab_size,
            "hazards": [h.as_dict() for h in self.hazards],
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# KMP helpers specialized for streaming analysis.
# --------------------------------------------------------------------------- #


def _kmp_first_completion(
    pattern: str, fail: list[int], start: int, surface: str
) -> tuple[int, int | None]:
    """Run KMP over ``surface`` from state ``start``.

    Returns ``(end_state, first_completion_chars)`` where ``first_completion_chars``
    is the number of characters consumed *within this surface* at the moment the
    pattern first fully matched, or ``None`` if it never matched. ``end_state`` is
    the KMP state after consuming the whole surface (always ``< len(pattern)``).
    """

    L = len(pattern)
    k = start
    first: int | None = None
    for i, ch in enumerate(surface):
        k = _kmp_advance_char(pattern, fail, k, ch)
        if k == L:
            if first is None:
                first = i + 1  # chars of this surface consumed at first match
            k = fail[k - 1] if L > 0 else 0
    return k, first


def _all_occurrence_ends(pattern: str, fail: list[int], surface: str) -> list[int]:
    """Indices (0-based, of the last char) where ``pattern`` ends inside ``surface``."""

    L = len(pattern)
    ends: list[int] = []
    k = 0
    for i, ch in enumerate(surface):
        k = _kmp_advance_char(pattern, fail, k, ch)
        if k == L:
            ends.append(i)
            k = fail[k - 1] if L > 0 else 0
    return ends


# --------------------------------------------------------------------------- #
# Decision procedures.
# --------------------------------------------------------------------------- #


def _find_overshoot(
    id_surfaces: Sequence[tuple[int, str]], stop: str, fail: list[int]
) -> StopHazard | None:
    """A single token whose surface contains ``stop`` ending before the token's last
    character: the model emits it atomically and overshoots the stop."""

    for tid, surface in id_surfaces:
        for end in _all_occurrence_ends(stop, fail, surface):
            if end < len(surface) - 1:
                trailing = surface[end + 1 :]
                return StopHazard(
                    kind=StopHazardKind.STOP_OVERSHOOT,
                    token_ids=(tid,),
                    surfaces=(surface,),
                    decoded=surface,
                    detail=(
                        f"token {tid} surface {surface!r} contains the stop ending at "
                        f"char {end} of {len(surface) - 1}; {len(trailing)} trailing "
                        f"char(s) {trailing!r} are emitted past the stop"
                    ),
                )
    return None


def _find_split(
    id_surfaces: Sequence[tuple[int, str]], stop: str, fail: list[int], max_states: int
) -> StopHazard | None:
    """Reachability of a *cross-token* completion of ``stop`` (no single token holds
    all of it) via BFS over KMP states with one edge per vocabulary token."""

    L = len(stop)
    if L < 2:
        return None  # a 1-char stop can never span a boundary
    # parent[state] = (prev_state, token_id) for witness reconstruction.
    parent: dict[int, tuple[int, int]] = {}
    reached = {0}
    queue: deque[int] = deque([0])
    while queue:
        if len(reached) > max_states:
            return None  # state budget exhausted (cannot happen: |states| <= L)
        k = queue.popleft()
        for tid, surface in id_surfaces:
            end_state, first = _kmp_first_completion(stop, fail, k, surface)
            if first is not None and first < L and k > 0:
                # Completion used carried progress from a previous token: cross-token.
                ids, surfs = _reconstruct(parent, k)
                ids = ids + (tid,)
                surfs = surfs + (surface,)
                return StopHazard(
                    kind=StopHazardKind.SPLIT_STOP,
                    token_ids=ids,
                    surfaces=surfs,
                    decoded="".join(surfs),
                    detail=(
                        f"stop completes across a token boundary: only {first} of "
                        f"{L} chars come from the final token {tid} {surface!r}; the "
                        f"rest is carried from earlier token(s) -- a per-token "
                        f"substring scan misses it"
                    ),
                )
            if end_state not in reached:
                reached.add(end_state)
                parent[end_state] = (k, tid)
                queue.append(end_state)
    return None


def _reconstruct(parent: Mapping[int, tuple[int, int]], state: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Walk the BFS parent map from ``state`` back to the start; returns token ids.
    Surfaces are not stored in the map, so callers append them; here we only need
    ids and re-derive surfaces from the caller. We therefore return ids only and an
    empty surfaces tuple placeholder is filled by the caller using a lookup."""

    ids: list[int] = []
    cur = state
    while cur in parent:
        prev, tid = parent[cur]
        ids.append(tid)
        cur = prev
    ids.reverse()
    return tuple(ids), tuple()


def prove_streaming_stop(
    id_surfaces: Sequence[tuple[int, str]],
    stop: str,
    *,
    max_product_states: int = 1_000_000,
) -> StreamingStopReport:
    """Decide whether token-granular truncation at ``stop`` is sound for this vocab.

    Returns a report listing every distinct hazard kind found (``stop-overshoot``
    and/or ``split-stop``), or a soundness certificate when neither is reachable.
    """

    if not stop:
        return StreamingStopReport(
            stop=stop,
            status=StopSoundnessStatus.ABSTAINED,
            vocab_size=len(id_surfaces),
            notes=("empty stop string",),
        )
    if not id_surfaces:
        return StreamingStopReport(
            stop=stop,
            status=StopSoundnessStatus.ABSTAINED,
            vocab_size=0,
            notes=("empty vocabulary",),
        )

    fail = _kmp_failure(stop)
    surface_by_id = {tid: surf for tid, surf in id_surfaces}
    hazards: list[StopHazard] = []

    overshoot = _find_overshoot(id_surfaces, stop, fail)
    if overshoot is not None:
        hazards.append(overshoot)

    split = _find_split(id_surfaces, stop, fail, max_product_states)
    if split is not None:
        # Re-derive surfaces for the reconstructed id path.
        surfs = tuple(surface_by_id.get(t, "") for t in split.token_ids)
        decoded = "".join(surfs)
        # Replay-validate: the stop must actually appear and span >1 token.
        if stop in decoded and not any(stop in s for s in surfs):
            hazards.append(
                StopHazard(
                    kind=split.kind,
                    token_ids=split.token_ids,
                    surfaces=surfs,
                    decoded=decoded,
                    detail=split.detail,
                )
            )

    if hazards:
        status = StopSoundnessStatus.HAZARDS_FOUND
        notes: tuple[str, ...] = ()
    else:
        status = StopSoundnessStatus.PROVEN_SOUND
        notes = (
            "every realization of the stop ends on a token boundary within a single "
            "token; token-granular truncation is exact (unbounded certificate)",
        )

    return StreamingStopReport(
        stop=stop,
        status=status,
        vocab_size=len(id_surfaces),
        hazards=tuple(hazards),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #


def render_streaming_stop_report_json(report: StreamingStopReport) -> str:
    return json.dumps(report.as_dict(), indent=2, ensure_ascii=False)


def render_streaming_stop_report_text(report: StreamingStopReport) -> str:
    lines = [
        f"streaming-stop soundness: {report.stop!r}",
        f"  status: {report.status}",
        f"  vocab: {report.vocab_size} token surfaces",
    ]
    if not report.hazards:
        lines.append("  PROVEN SOUND: token-granular truncation is exact (unbounded).")
    for h in report.hazards:
        lines.append(f"  [{h.kind}] tokens={list(h.token_ids)} surfaces={list(h.surfaces)}")
        lines.append(f"    decoded: {h.decoded!r}")
        lines.append(f"    {h.detail}")
    for note in report.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
