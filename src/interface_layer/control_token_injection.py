"""Tokenizer-layer control-token injection (special-token smuggling) analysis.

This lifts the tool-call **boundary-soundness** property to the *prompt/role wire
format*. The tool-call detectors ask whether attacker-influenced *argument* text
can forge a tool-call delimiter; this asks the dual, deeper question one layer
down: can attacker-influenced *message* text forge one of the tokenizer's own
**special tokens** -- the reserved alphabet the chat template uses to mark roles,
turn ends, and tool calls?

A chat template ``T`` together with a tokenizer special-token vocabulary ``S`` is
**prompt-sound** iff no user-controlled field, once rendered by ``T`` and
segmented by the tokenizer, can introduce a control token in ``S`` that the
conversation author did not place there. Hugging Face tokenizers match
added/special tokens with a longest-match trie *before* the model tokenizer, so a
content string equal to (or merely *containing*) a special token's surface form
is promoted to that control token id -- regardless of the surrounding quoting.

Two properties make this distinct from, and stronger than, template-string-level
role-boundary analysis (``role_boundaries``):

1. **It confirms at the token-id layer**, by replaying the tokenizer's *own*
   added-token segmenter (:class:`promptabi.tokenizers.ByteLevelTokenizer`) on a
   differential render pair. The model sees token ids, not template text, so this
   is the layer where the forgery actually happens.
2. **JSON-encoding ("tojson") is not a defense here.** A control token such as
   ``<|im_start|>`` contains no JSON metacharacter, so ``content | tojson``
   emits ``"<|im_start|>"`` and the trie still matches the marker *inside the
   quotes*. Template-level analysis that treats ``tojson`` as a sanitizer reports
   such a template safe; at the tokenizer layer it is still forgeable.

The injected token's role gives the severity class: a turn/role opener is a role
hijack (system-prompt override / prompt injection); an end-of-turn/EOS token is a
generation-truncation / denial-of-service; a tool marker is prompt-layer tool
forgery; a BOS marker corrupts the context prefix.

Everything here operates on the **discrete interface**: the declarative template,
the declared special-token vocabulary, and the tokenizer's added-token matcher.
No model is loaded and no LLM semantics are modeled.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .chat_templates import (
    ChatTemplateParseError,
    ChatTemplateRenderError,
    parse_hf_chat_template_config,
    render_chat_template_supported_fragment,
)
from .tokenizers import ByteLevelTokenizer, TokenizerError

try:  # optional high-fidelity backend; the library never hard-depends on Jinja
    from jinja2.sandbox import ImmutableSandboxedEnvironment as _JinjaSandbox
    from jinja2.exceptions import TemplateError as _JinjaTemplateError

    _HAS_JINJA = True
except Exception:  # pragma: no cover - jinja2 not installed
    _HAS_JINJA = False

CONTROL_TOKEN_INJECTION_VERSION = "1.0.0"

_BENIGN_PLACEHOLDER = "PROMPTABISAFEPLACEHOLDER"


def _raise_exception(message: str) -> None:  # pragma: no cover - exercised via templates
    raise _JinjaTemplateError(message)


def _strftime_now(fmt: str) -> str:  # pragma: no cover - exercised via templates
    import datetime as _dt

    return _dt.datetime.now().strftime(fmt)


def _tojson(value: object, *, indent: int | None = None, **_: object) -> str:  # pragma: no cover
    return json.dumps(value, ensure_ascii=False, indent=indent)


_JINJA_ENV = None


def _jinja_env():
    """Build (once) a Jinja environment matching HF ``apply_chat_template``."""

    global _JINJA_ENV
    if _JINJA_ENV is None:
        env = _JinjaSandbox(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = _strftime_now
        _JINJA_ENV = env
    return _JINJA_ENV


class ControlTokenClass(StrEnum):
    """Severity class of a forgeable control token, by its structural role."""

    ROLE_HIJACK = "role-hijack"
    GENERATION_TRUNCATION = "generation-truncation"
    TOOL_FORGERY = "tool-forgery"
    CONTEXT_PREFIX_CORRUPTION = "context-prefix-corruption"
    GENERIC_SMUGGLING = "control-token-smuggling"


_SEVERITY = {
    ControlTokenClass.ROLE_HIJACK: "critical",
    ControlTokenClass.TOOL_FORGERY: "critical",
    ControlTokenClass.GENERATION_TRUNCATION: "high",
    ControlTokenClass.CONTEXT_PREFIX_CORRUPTION: "high",
    ControlTokenClass.GENERIC_SMUGGLING: "medium",
}

# Surface-form signatures. Matched case-insensitively against the token text.
_ROLE_OPENER_RE = re.compile(
    r"(im_start|start_header_id|start_of_turn|<\|user\|>|<\|assistant\|>|<\|system\|>"
    r"|<\|developer\|>|<\|role\|>|\[INST\]|<<SYS>>|<\|begin_of_role\|>|<\|channel\|>"
    r"|<\|start\|>)",
    re.IGNORECASE,
)
_TURN_END_RE = re.compile(
    r"(im_end|eot_id|eom_id|end_of_turn|<\|end\|>|</s>|<\|endoftext\|>|\[/INST\]"
    r"|<<\/SYS>>|<\|end_of_text\|>|<end_of_turn>|<\|return\|>)",
    re.IGNORECASE,
)
_TOOL_RE = re.compile(
    r"(tool_call|tool_calls|tool_response|tool_results?|</?tools>|<\|tool\|>|function_call|\[tool)",
    re.IGNORECASE,
)
_BOS_RE = re.compile(r"(begin_of_text|<\|startoftext\|>|^<s>$|<bos>|<\|bos\|>)", re.IGNORECASE)

# Token-config keys whose value is, by definition, a special control token.
_NAMED_SPECIAL_KEYS = {
    "bos_token": ControlTokenClass.CONTEXT_PREFIX_CORRUPTION,
    "eos_token": ControlTokenClass.GENERATION_TRUNCATION,
    "pad_token": ControlTokenClass.GENERIC_SMUGGLING,
    "unk_token": ControlTokenClass.GENERIC_SMUGGLING,
    "sep_token": ControlTokenClass.GENERIC_SMUGGLING,
    "cls_token": ControlTokenClass.GENERIC_SMUGGLING,
    "mask_token": ControlTokenClass.GENERIC_SMUGGLING,
}


def classify_control_token(text: str, *, config_key: str | None = None) -> ControlTokenClass:
    """Classify a special token's surface form into a severity class.

    Surface-form role/tool/turn signatures take priority over the declaring
    config key, because the structural meaning of the marker (what an injected
    copy *does* to the prompt) dominates its declared name.
    """

    if _ROLE_OPENER_RE.search(text):
        return ControlTokenClass.ROLE_HIJACK
    if _TOOL_RE.search(text):
        return ControlTokenClass.TOOL_FORGERY
    if _TURN_END_RE.search(text):
        return ControlTokenClass.GENERATION_TRUNCATION
    if _BOS_RE.search(text):
        return ControlTokenClass.CONTEXT_PREFIX_CORRUPTION
    if config_key in _NAMED_SPECIAL_KEYS:
        return _NAMED_SPECIAL_KEYS[config_key]
    return ControlTokenClass.GENERIC_SMUGGLING


@dataclass(frozen=True, slots=True)
class SpecialTokenDecl:
    """A special/added token surface form declared by a tokenizer config."""

    text: str
    source: str
    special: bool
    token_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"text": self.text, "source": self.source, "special": self.special}
        if self.token_id is not None:
            data["token_id"] = self.token_id
        return data


@dataclass(frozen=True, slots=True)
class InjectionProbe:
    """A single attacker-controllable field position to drive a content value into."""

    label: str
    field_path: str
    description: str


# The conversation fields an adversary routinely controls. Each builder returns a
# messages list (and optional tools) given a value to place in the target field.
_PROBES: tuple[InjectionProbe, ...] = (
    InjectionProbe("user-content", "messages[user].content", "text of a user turn"),
    InjectionProbe("system-content", "messages[system].content", "text of a system turn (e.g. RAG-populated)"),
    InjectionProbe("tool-content", "messages[tool].content", "a tool/function result fed back into the prompt"),
    InjectionProbe("assistant-content", "messages[assistant].content", "prior assistant text replayed in history"),
    InjectionProbe("message-name", "messages[user].name", "the speaker name field of a turn"),
)


def _probe_messages(probe: InjectionProbe, value: str) -> list[dict[str, object]]:
    if probe.label == "user-content":
        return [{"role": "user", "content": value}]
    if probe.label == "system-content":
        return [{"role": "system", "content": value}, {"role": "user", "content": "hi"}]
    if probe.label == "tool-content":
        return [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": value},
        ]
    if probe.label == "assistant-content":
        return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": value}]
    if probe.label == "message-name":
        return [{"role": "user", "content": "hi", "name": value}]
    raise AssertionError(probe.label)


@dataclass(frozen=True, slots=True)
class ControlTokenInjectionWitness:
    """A differential render+tokenize witness proving a control token was forged."""

    probe_label: str
    token_text: str
    benign_value: str
    attack_value: str
    benign_control_count: int
    attack_control_count: int
    rendered_attack_excerpt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_label": self.probe_label,
            "token_text": self.token_text,
            "benign_value": self.benign_value,
            "attack_value": self.attack_value,
            "benign_control_count": self.benign_control_count,
            "attack_control_count": self.attack_control_count,
            "rendered_attack_excerpt": self.rendered_attack_excerpt,
        }


@dataclass(frozen=True, slots=True)
class ControlTokenInjectionFinding:
    """A confirmed (or candidate) prompt-soundness violation."""

    token_text: str
    token_class: ControlTokenClass
    severity: str
    probe_label: str
    verdict: str  # confirmed | rejected | abstained
    explanation: str
    witness: ControlTokenInjectionWitness | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "token_text": self.token_text,
            "token_class": str(self.token_class),
            "severity": self.severity,
            "probe_label": self.probe_label,
            "verdict": self.verdict,
            "explanation": self.explanation,
        }
        if self.witness is not None:
            data["witness"] = self.witness.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class ControlTokenInjectionReport:
    """All prompt-soundness findings for one tokenizer config."""

    special_tokens: tuple[SpecialTokenDecl, ...]
    findings: tuple[ControlTokenInjectionFinding, ...]
    abstained: bool = False
    abstain_reason: str = ""
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def confirmed(self) -> tuple[ControlTokenInjectionFinding, ...]:
        return tuple(f for f in self.findings if f.verdict == "confirmed")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": CONTROL_TOKEN_INJECTION_VERSION,
            "special_tokens": [t.to_dict() for t in self.special_tokens],
            "findings": [f.to_dict() for f in self.findings],
            "confirmed_count": len(self.confirmed),
        }
        if self.abstained:
            data["abstained"] = True
            data["abstain_reason"] = self.abstain_reason
        if self.extras:
            data["extras"] = dict(self.extras)
        return data


def _extract_special_token_decls(config: dict[str, object]) -> tuple[SpecialTokenDecl, ...]:
    decls: dict[str, SpecialTokenDecl] = {}

    def add(text: str | None, source: str, special: bool, token_id: int | None = None) -> None:
        if not text:
            return
        existing = decls.get(text)
        if existing is None or (special and not existing.special):
            decls[text] = SpecialTokenDecl(text=text, source=source, special=special, token_id=token_id)

    for key, cls in _NAMED_SPECIAL_KEYS.items():
        add(_token_text(config.get(key)), key, True)
    additional = config.get("additional_special_tokens")
    if isinstance(additional, list):
        for index, value in enumerate(additional):
            add(_token_text(value), f"additional_special_tokens[{index}]", True)
    decoder = config.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        for raw_id, value in decoder.items():
            if not isinstance(value, dict):
                continue
            text = _token_text(value)
            special = bool(value.get("special", False))
            try:
                token_id: int | None = int(raw_id)
            except (TypeError, ValueError):
                token_id = None
            add(text, f"added_tokens_decoder[{raw_id}]", special, token_id)
    return tuple(decls[text] for text in sorted(decls))


def _token_text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content:
            return content
    return None


def _build_tokenizer(decls: tuple[SpecialTokenDecl, ...]) -> ByteLevelTokenizer:
    added = [d.text for d in decls]
    special = {d.text: (d.token_id if d.token_id is not None else 1000 + i) for i, d in enumerate(decls) if d.special}
    return ByteLevelTokenizer(added_tokens=added, special_tokens=special or None)


def _count_control_text(tokenizer: ByteLevelTokenizer, text: str, token_text: str) -> int:
    count = 0
    for tok in tokenizer.encode(text).tokens:
        if (getattr(tok, "special", False) or getattr(tok, "added", False)) and tok.text == token_text:
            count += 1
    return count


def scan_control_token_injection(config: dict[str, object]) -> ControlTokenInjectionReport:
    """Scan one tokenizer config for forgeable control tokens through rendered content.

    For every declared special token and every attacker-controllable field, render
    a benign vs. attack message pair through PromptABI's supported chat-template
    fragment and replay the tokenizer's added-token segmenter on both. A token is
    *confirmed* forgeable when the attack rendering yields a strictly greater count
    of that control token at the injected position than the benign rendering.
    """

    try:
        parsed = parse_hf_chat_template_config(config)
    except ChatTemplateParseError as exc:
        return ControlTokenInjectionReport(
            special_tokens=(), findings=(), abstained=True,
            abstain_reason=f"chat template could not be parsed: {exc}",
        )

    decls = _extract_special_token_decls(config)
    if not decls:
        return ControlTokenInjectionReport(
            special_tokens=(), findings=(), abstained=True,
            abstain_reason="no special tokens declared in tokenizer config",
        )

    try:
        tokenizer = _build_tokenizer(decls)
    except TokenizerError as exc:
        return ControlTokenInjectionReport(
            special_tokens=decls, findings=(), abstained=True,
            abstain_reason=f"could not build a replay tokenizer: {exc}",
        )

    # Prefer the real Jinja engine (the chat template *is* a Jinja program; running
    # it on a differential input pair is operator-replay, like running the parser's
    # regex). Fall back to PromptABI's strict supported fragment when Jinja is absent.
    render_fn, backend = _build_render_fn(parsed, config)

    render_variants = (
        {"add_generation_prompt": False},
        {"add_generation_prompt": True},
    )

    findings: list[ControlTokenInjectionFinding] = []
    any_rendered = False
    for probe in _PROBES:
        benign_render = _render_first(render_fn, probe, _BENIGN_PLACEHOLDER, render_variants)
        if benign_render is None:
            continue  # this field is not rendered on any supported path
        rendered_variant, benign_text = benign_render
        any_rendered = True
        # The placeholder must be inert (contribute no control tokens).
        for decl in decls:
            token = decl.text
            try:
                attack_text = render_fn(_probe_messages(probe, token), **rendered_variant)
            except Exception:
                continue
            benign_count = _count_control_text(tokenizer, benign_text, token)
            attack_count = _count_control_text(tokenizer, attack_text, token)
            token_class = classify_control_token(token, config_key=_decl_key(decl))
            severity = _SEVERITY[token_class]
            if attack_count > benign_count:
                findings.append(
                    ControlTokenInjectionFinding(
                        token_text=token,
                        token_class=token_class,
                        severity=severity,
                        probe_label=probe.label,
                        verdict="confirmed",
                        explanation=(
                            f"{probe.description} containing the special token {token!r} renders "
                            f"unescaped and the tokenizer's added-token matcher promotes it to a "
                            f"control token ({benign_count} -> {attack_count} occurrences). "
                            f"An attacker who controls this field forges a {token_class} via the prompt."
                        ),
                        witness=ControlTokenInjectionWitness(
                            probe_label=probe.label,
                            token_text=token,
                            benign_value=_BENIGN_PLACEHOLDER,
                            attack_value=token,
                            benign_control_count=benign_count,
                            attack_control_count=attack_count,
                            rendered_attack_excerpt=_excerpt(attack_text, token),
                        ),
                    )
                )
    if not any_rendered:
        return ControlTokenInjectionReport(
            special_tokens=decls, findings=(), abstained=True,
            abstain_reason=(
                "chat template could not be rendered for any attacker-controlled field"
                if backend == "jinja"
                else "chat template uses constructs outside the supported render fragment"
            ),
            extras={"render_backend": backend},
        )
    return ControlTokenInjectionReport(
        special_tokens=decls, findings=tuple(findings), extras={"render_backend": backend}
    )


def _build_render_fn(parsed, config: dict[str, object]):
    """Return ``(render(messages, *, add_generation_prompt) -> str, backend_name)``."""

    token_context = {
        key: text
        for key in _NAMED_SPECIAL_KEYS
        if (text := _token_text(config.get(key))) is not None
    }
    if _HAS_JINJA:
        template = _jinja_env().from_string(parsed.template_source)

        def render_jinja(messages, *, add_generation_prompt: bool = False) -> str:
            return template.render(
                messages=messages,
                tools=None,
                add_generation_prompt=add_generation_prompt,
                **token_context,
            )

        return render_jinja, "jinja"

    def render_fragment(messages, *, add_generation_prompt: bool = False) -> str:
        return render_chat_template_supported_fragment(
            parsed, messages, add_generation_prompt=add_generation_prompt
        )

    return render_fragment, "supported-fragment"


def _decl_key(decl: SpecialTokenDecl) -> str | None:
    if decl.source in _NAMED_SPECIAL_KEYS:
        return decl.source
    return None


def _render_first(render_fn, probe: InjectionProbe, value: str, variants):
    for variant in variants:
        try:
            text = render_fn(_probe_messages(probe, value), **variant)
        except Exception:
            continue
        if value in text:
            return variant, text
    return None


def _excerpt(text: str, token: str, radius: int = 24) -> str:
    idx = text.find(token)
    if idx < 0:
        return text[:64]
    start = max(0, idx - radius)
    end = min(len(text), idx + len(token) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def scan_control_token_injection_file(path: str | Path) -> ControlTokenInjectionReport:
    """Scan a Hugging Face ``tokenizer_config.json`` on disk."""

    raw = Path(path).read_text(encoding="utf-8")
    config = json.loads(raw)
    if not isinstance(config, dict):
        return ControlTokenInjectionReport(
            special_tokens=(), findings=(), abstained=True,
            abstain_reason="tokenizer_config.json must contain an object",
        )
    return scan_control_token_injection(config)


def render_control_token_injection_report_json(report: ControlTokenInjectionReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def render_control_token_injection_report_text(report: ControlTokenInjectionReport) -> str:
    lines: list[str] = []
    lines.append(f"control-token-injection scan (v{CONTROL_TOKEN_INJECTION_VERSION})")
    lines.append(f"declared special tokens: {len(report.special_tokens)}")
    if report.abstained:
        lines.append(f"ABSTAINED: {report.abstain_reason}")
        return "\n".join(lines)
    confirmed = report.confirmed
    lines.append(f"confirmed forgeable control tokens: {len(confirmed)}")
    for finding in report.findings:
        marker = "CONFIRMED" if finding.verdict == "confirmed" else finding.verdict.upper()
        lines.append(
            f"  {marker} [{finding.severity}/{finding.token_class}] {finding.token_text!r} "
            f"via {finding.probe_label}"
        )
        lines.append(f"      {finding.explanation}")
        if finding.witness is not None:
            lines.append(f"      render: {finding.witness.rendered_attack_excerpt!r}")
    return "\n".join(lines)
