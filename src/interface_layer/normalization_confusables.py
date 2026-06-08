"""Normalization-confusable control-marker forgery (the pre-segmentation layer).

This is the dual-use generalization of role-separator forgery (which injects the
*literal* separator) into the *normalization* layer. A tokenizer's normalizer
``N`` runs before segmentation; if a control marker ``m`` (a role separator, or
a normalized added token) has a **non-literal preimage** ``x`` with
``N(x) == N(m)`` and ``m`` not literally present in ``x``, then an attacker can
forge ``m``'s token sequence with input that **defeats any filter scanning for
the literal marker**. Unicode confusables (fullwidth under NFKC, U+2581 under
SentencePiece metaspace, case folds under lowercasing, zero-width insertion
under ``clean_text``) are exactly such preimages.

Three confirmed-by-replay bug classes:

* ``confusable-marker-forgery`` -- a verified non-literal preimage of a control
  marker, confirmed by rendering it into an attacker field and observing the
  marker's normalized token subsequence appear (benign -> attack).
* ``marker-collision`` -- two distinct control markers that ``N`` maps to the
  same surface, collapsing their recognizer semantics.
* ``normalized-marker-destruction`` -- a registered marker flagged
  ``normalized: true`` whose own surface ``N`` rewrites, so the writer emits a
  marker the post-normalization recognizer no longer sees atomically.

No model is loaded and no Python is modeled: only the real chat template, the
real ``tokenizer.json`` normalizer, and the byte-level segmenter are replayed.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .chat_templates import ChatTemplateParseError, parse_hf_chat_template_config
from .control_token_injection import (
    _build_render_fn,
    _build_tokenizer,
    _extract_special_token_decls,
)
from .normalizers import Normalizer, parse_normalizer
from .role_separator_forgery import (
    _BENIGN_SENTINEL,
    _attacker_messages,
    _derive_separator,
    _subsequence_count,
)
from .tokenizers import TokenizerError

NORMALIZATION_CONFUSABLES_VERSION = "1.0.0"

_METASPACE = "\u2581"


class ConfusableBugClass(StrEnum):
    CONFUSABLE_MARKER_FORGERY = "confusable-marker-forgery"
    MARKER_COLLISION = "marker-collision"
    NORMALIZED_MARKER_DESTRUCTION = "normalized-marker-destruction"


@dataclass(frozen=True, slots=True)
class ConfusableWitness:
    marker: str
    preimage: str
    strategy: str
    normalized_form: str
    attacker_field: str | None = None
    benign_occurrences: int | None = None
    attack_occurrences: int | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "marker": self.marker,
            "preimage": self.preimage,
            "strategy": self.strategy,
            "normalized_form": self.normalized_form,
        }
        if self.attacker_field is not None:
            data["attacker_field"] = self.attacker_field
            data["benign_occurrences"] = self.benign_occurrences
            data["attack_occurrences"] = self.attack_occurrences
        return data


@dataclass(frozen=True, slots=True)
class ConfusableFinding:
    bug_class: ConfusableBugClass
    severity: str
    verdict: str
    explanation: str
    witness: ConfusableWitness

    def to_dict(self) -> dict[str, object]:
        return {
            "bug_class": str(self.bug_class),
            "severity": self.severity,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "witness": self.witness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class NormalizationConfusablesReport:
    findings: tuple[ConfusableFinding, ...]
    normalizer_name: str = "identity"
    abstained: bool = False
    abstain_reason: str | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def confirmed(self) -> tuple[ConfusableFinding, ...]:
        return tuple(f for f in self.findings if f.verdict == "confirmed")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": NORMALIZATION_CONFUSABLES_VERSION,
            "normalizer": self.normalizer_name,
            "findings": [f.to_dict() for f in self.findings],
            "confirmed_count": len(self.confirmed),
        }
        if self.abstained:
            data["abstained"] = True
            if self.abstain_reason:
                data["abstain_reason"] = self.abstain_reason
        if self.extras:
            data["extras"] = self.extras
        return data


# --------------------------------------------------------------------------- #
# Confusable preimage proposers (proposals are always verified against N).
# --------------------------------------------------------------------------- #

def _fullwidth(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        out.append(chr(code - 0x20 + 0xFF00) if 0x21 <= code <= 0x7E else ch)
    return "".join(out)


def _metaspace_spaces(s: str) -> str:
    return s.replace(" ", _METASPACE)


def _accent_then_compose(s: str) -> str:
    # Decompose composed accents: an é (U+00E9) preimage is e + U+0301.
    return unicodedata.normalize("NFD", s)


def _inject_zero_width(s: str) -> str:
    # Insert a zero-width space after the first character.
    if len(s) < 2:
        return s
    return s[0] + "\u200b" + s[1:]


def confusable_preimages(marker: str, normalizer: Normalizer) -> list[tuple[str, str]]:
    """Return verified ``(preimage, strategy)`` pairs for ``marker`` under ``N``.

    Every returned preimage ``x`` satisfies ``N(x) == N(marker)``, ``x != marker``
    and ``marker not in x`` (so a literal-substring filter for ``marker`` misses
    it). Proposals are generated from the normalizer's capabilities; correctness
    is established by replaying the real ``N``.
    """

    if normalizer.is_identity or normalizer.capabilities.unsupported:
        return []
    caps = normalizer.capabilities
    target = normalizer.apply(marker)
    proposals: list[tuple[str, str]] = []
    if caps.compatibility_fold:
        proposals.append((_fullwidth(marker), "fullwidth-compatibility"))
    if caps.space_to_metaspace and " " in marker:
        proposals.append((_metaspace_spaces(marker), "metaspace-space"))
    if caps.lowercases and marker.lower() != marker:
        proposals.append((marker.upper(), "case-fold"))
        proposals.append((marker.capitalize(), "case-fold"))
    if caps.strips_accents:
        proposals.append((_accent_then_compose(marker), "accent-strip"))
    if caps.cleans_text:
        proposals.append((_inject_zero_width(marker), "zero-width-insert"))

    verified: list[tuple[str, str]] = []
    seen: set[str] = set()
    for preimage, strategy in proposals:
        if preimage in seen or preimage == marker or marker in preimage:
            continue
        if normalizer.apply(preimage) == target:
            seen.add(preimage)
            verified.append((preimage, strategy))
    return verified


# --------------------------------------------------------------------------- #
# Marker collection
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class _Marker:
    text: str
    source: str           # "special-token" | "role-separator"
    role: str | None = None
    normalized_flag: bool = False
    attacker_field: str | None = None  # for role separators: where to inject


def _normalized_flags(config: dict[str, object]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    decoder = config.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        for value in decoder.values():
            if isinstance(value, dict) and isinstance(value.get("content"), str):
                flags[value["content"]] = bool(value.get("normalized", False))
    return flags


def _collect_markers(config: dict[str, object], render_fn) -> list[_Marker]:
    markers: list[_Marker] = []
    flags = _normalized_flags(config)
    for decl in _extract_special_token_decls(config):
        markers.append(
            _Marker(
                text=decl.text,
                source="special-token",
                normalized_flag=flags.get(decl.text, False),
            )
        )
    for role, field_name in (("assistant", "user-content"), ("system", "user-content"), ("tool", "user-content")):
        sep = _derive_separator(render_fn, role)
        if sep:
            markers.append(_Marker(text=sep, source="role-separator", role=role, attacker_field=field_name))
    return markers


_ROLE_SEVERITY = {"system": "critical", "assistant": "critical", "tool": "high"}


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #

def scan_normalization_confusables(
    config: dict[str, object],
    *,
    normalizer_spec: object | None = None,
) -> NormalizationConfusablesReport:
    """Scan one config for normalization-confusable control-marker forgery."""

    if not isinstance(config, dict):
        return NormalizationConfusablesReport(
            findings=(), abstained=True, abstain_reason="config root is not a JSON object",
        )
    normalizer = parse_normalizer(normalizer_spec)
    if normalizer.is_identity:
        return NormalizationConfusablesReport(
            findings=(), normalizer_name=normalizer.name, abstained=True,
            abstain_reason="tokenizer declares no (modelled) normalizer; nothing to fold",
        )
    if normalizer.capabilities.unsupported:
        return NormalizationConfusablesReport(
            findings=(), normalizer_name=normalizer.name, abstained=True,
            abstain_reason=f"normalizer '{normalizer.name}' uses an unmodelled primitive",
        )
    try:
        parsed = parse_hf_chat_template_config(config)
    except ChatTemplateParseError as exc:
        return NormalizationConfusablesReport(
            findings=(), normalizer_name=normalizer.name, abstained=True,
            abstain_reason=f"chat template could not be parsed: {exc}",
        )
    decls = _extract_special_token_decls(config)
    try:
        tokenizer = _build_tokenizer(decls)
    except TokenizerError as exc:
        return NormalizationConfusablesReport(
            findings=(), normalizer_name=normalizer.name, abstained=True,
            abstain_reason=f"could not build a replay tokenizer: {exc}",
        )
    render_fn, backend = _build_render_fn(parsed, config)
    markers = _collect_markers(config, render_fn)

    findings: list[ConfusableFinding] = []

    # Class 1: confusable forgery (role separators + normalized:true tokens).
    for marker in markers:
        if marker.source == "special-token" and not marker.normalized_flag:
            # normalized:false markers are matched pre-normalization; a confusable
            # cannot reach them. (Conservative: avoids false positives.)
            continue
        for preimage, strategy in confusable_preimages(marker.text, normalizer):
            finding = _confirm_forgery(
                marker, preimage, strategy, normalizer, tokenizer, render_fn,
            )
            if finding is not None:
                findings.append(finding)
                break  # one witness per marker is enough

    # Class 2: marker collision under N.
    findings.extend(_marker_collisions(markers, normalizer))

    # Class 3: normalized:true marker whose own surface N rewrites.
    findings.extend(_marker_destruction(markers, normalizer))

    return NormalizationConfusablesReport(
        findings=tuple(findings),
        normalizer_name=normalizer.name,
        extras={"render_backend": backend},
    )


def _confirm_forgery(
    marker: _Marker, preimage: str, strategy: str, normalizer: Normalizer, tokenizer, render_fn,
) -> ConfusableFinding | None:
    field_name = marker.attacker_field or "user-content"
    norm_marker_ids = [t.token_id for t in tokenizer.encode(normalizer.apply(marker.text)).tokens]
    if not norm_marker_ids:
        return None
    try:
        benign_text = render_fn(_attacker_messages(field_name, _BENIGN_SENTINEL), add_generation_prompt=False)
        attack_text = render_fn(_attacker_messages(field_name, preimage), add_generation_prompt=False)
    except Exception:
        return None
    benign_ids = [t.token_id for t in tokenizer.encode(normalizer.apply(benign_text)).tokens]
    attack_ids = [t.token_id for t in tokenizer.encode(normalizer.apply(attack_text)).tokens]
    benign_occ = _subsequence_count(benign_ids, norm_marker_ids)
    attack_occ = _subsequence_count(attack_ids, norm_marker_ids)
    if attack_occ <= benign_occ:
        return None
    role = marker.role
    severity = _ROLE_SEVERITY.get(role, "high") if role else "high"
    where = f"the {role}-turn separator" if role else "the control marker"
    explanation = (
        f"{where} {marker.text!r} can be forged with the non-literal preimage "
        f"{preimage!r} ({strategy}): the tokenizer normalizer {normalizer.name} folds it to "
        f"the same surface {normalizer.apply(marker.text)!r}, so injecting it into the "
        f"{field_name} field reproduces the marker's normalized token subsequence "
        f"({benign_occ} -> {attack_occ}). Crucially the preimage does NOT contain the literal "
        f"marker, so an input filter that rejects/strips the literal marker string is bypassed "
        f"while the model still reads a forged boundary."
    )
    return ConfusableFinding(
        bug_class=ConfusableBugClass.CONFUSABLE_MARKER_FORGERY,
        severity=severity,
        verdict="confirmed",
        explanation=explanation,
        witness=ConfusableWitness(
            marker=marker.text,
            preimage=preimage,
            strategy=strategy,
            normalized_form=normalizer.apply(marker.text),
            attacker_field=field_name,
            benign_occurrences=benign_occ,
            attack_occurrences=attack_occ,
        ),
    )


def _marker_collisions(markers: list[_Marker], normalizer: Normalizer) -> list[ConfusableFinding]:
    findings: list[ConfusableFinding] = []
    by_norm: dict[str, list[_Marker]] = {}
    for m in markers:
        by_norm.setdefault(normalizer.apply(m.text), []).append(m)
    for norm_form, group in by_norm.items():
        surfaces = sorted({m.text for m in group})
        if len(surfaces) > 1:
            findings.append(
                ConfusableFinding(
                    bug_class=ConfusableBugClass.MARKER_COLLISION,
                    severity="high",
                    verdict="confirmed",
                    explanation=(
                        f"distinct control markers {surfaces!r} all normalize to {norm_form!r} under "
                        f"{normalizer.name}; after normalization the recognizer cannot tell them apart, "
                        f"collapsing their role/turn semantics."
                    ),
                    witness=ConfusableWitness(
                        marker=" | ".join(surfaces),
                        preimage=surfaces[0],
                        strategy="marker-collision",
                        normalized_form=norm_form,
                    ),
                )
            )
    return findings


def _marker_destruction(markers: list[_Marker], normalizer: Normalizer) -> list[ConfusableFinding]:
    findings: list[ConfusableFinding] = []
    for m in markers:
        if m.source != "special-token" or not m.normalized_flag:
            continue
        norm = normalizer.apply(m.text)
        if norm != m.text:
            findings.append(
                ConfusableFinding(
                    bug_class=ConfusableBugClass.NORMALIZED_MARKER_DESTRUCTION,
                    severity="medium",
                    verdict="confirmed",
                    explanation=(
                        f"the registered marker {m.text!r} is flagged normalized:true, but {normalizer.name} "
                        f"rewrites its surface to {norm!r}; the writer emits {m.text!r} while a "
                        f"post-normalization recognizer reads {norm!r}, so the marker is not preserved atomically."
                    ),
                    witness=ConfusableWitness(
                        marker=m.text,
                        preimage=m.text,
                        strategy="normalized-true",
                        normalized_form=norm,
                    ),
                )
            )
    return findings


# --------------------------------------------------------------------------- #
# File entry point + renderers
# --------------------------------------------------------------------------- #

def scan_normalization_confusables_file(path: str | Path) -> NormalizationConfusablesReport:
    """Scan a ``tokenizer_config.json``; read the sibling ``tokenizer.json`` normalizer."""

    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return NormalizationConfusablesReport(
            findings=(), abstained=True, abstain_reason="config root is not a JSON object",
        )
    normalizer_spec: object | None = None
    sibling = config_path.with_name("tokenizer.json")
    if sibling.exists():
        try:
            tj = json.loads(sibling.read_text(encoding="utf-8"))
            if isinstance(tj, dict):
                normalizer_spec = tj.get("normalizer")
        except (OSError, json.JSONDecodeError):
            normalizer_spec = None
    return scan_normalization_confusables(config, normalizer_spec=normalizer_spec)


def render_normalization_confusables_report_json(report: NormalizationConfusablesReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)


def render_normalization_confusables_report_text(report: NormalizationConfusablesReport) -> str:
    lines = [f"normalization-confusable forgery scan (v{NORMALIZATION_CONFUSABLES_VERSION})"]
    lines.append(f"normalizer: {report.normalizer_name}")
    if report.abstained:
        lines.append(f"ABSTAIN: {report.abstain_reason}")
        return "\n".join(lines)
    lines.append(f"findings: {len(report.findings)} ({len(report.confirmed)} confirmed)")
    for f in report.findings:
        lines.append(f"  {f.verdict.upper()} [{f.severity}/{f.bug_class}]")
        lines.append(f"      {f.explanation}")
    return "\n".join(lines)
