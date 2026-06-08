"""Composable model of Hugging Face *tokenizer normalizers*.

A `tokenizer.json` carries a ``normalizer`` block that runs over text *before*
sub-word segmentation (and, for added tokens whose ``normalized`` flag is true,
over the marker surface itself). PromptABI's other analyzers reason about the
segmenter; this module reasons about the layer *above* it -- the normalization
homomorphism ``N : raw -> normalized`` -- which is exactly the layer a Unicode
confusable / homoglyph attack lives in.

We model only the deterministic, well-documented normalizer primitives and
compose them faithfully. Anything we do not recognize degrades to an explicit
``unsupported`` capability so callers can abstain rather than guess.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field, replace
from typing import Callable

_METASPACE = "\u2581"  # 'LOWER ONE EIGHTH BLOCK' -- SentencePiece space marker

# Characters BertNormalizer.clean_text removes (control / zero-width / BOM) and
# the whitespace it collapses to a single ASCII space.
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"


@dataclass(frozen=True, slots=True)
class NormalizerCapabilities:
    """What folding behaviours a parsed normalizer exhibits.

    These drive which confusable preimage families are worth proposing. They are
    *hints*: every proposed confusable is still verified by replaying the real
    normalizer, so an over-broad capability never produces a false positive.
    """

    compatibility_fold: bool = False  # NFKC / NFKD: fullwidth, ligatures, ...
    canonical_fold: bool = False      # NFC / NFD: combining-mark composition
    lowercases: bool = False
    strips_accents: bool = False
    space_to_metaspace: bool = False  # Replace(" " -> U+2581) / Metaspace
    cleans_text: bool = False         # strips control/zero-width, collapses ws
    unsupported: bool = False         # contained a primitive we do not model


@dataclass(frozen=True, slots=True)
class Normalizer:
    """A composed, deterministic text normalizer."""

    name: str
    fns: tuple[Callable[[str], str], ...] = ()
    capabilities: NormalizerCapabilities = field(default_factory=NormalizerCapabilities)

    def apply(self, text: str) -> str:
        for fn in self.fns:
            text = fn(text)
        return text

    @property
    def is_identity(self) -> bool:
        return not self.fns


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _nfkd(s: str) -> str:
    return unicodedata.normalize("NFKD", s)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def _bert_clean_text(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0 or code == 0xFFFD or _is_control(ch):
            continue
        if ch in _ZERO_WIDTH:
            continue
        out.append(" " if _is_whitespace(ch) else ch)
    return "".join(out)


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def parse_normalizer(spec: object) -> Normalizer:
    """Build a :class:`Normalizer` from a ``tokenizer.json`` normalizer block."""

    if spec is None:
        return Normalizer(name="identity")
    if not isinstance(spec, dict):
        return Normalizer(
            name="unsupported",
            capabilities=NormalizerCapabilities(unsupported=True),
        )

    kind = spec.get("type")
    caps = NormalizerCapabilities()

    if kind == "Sequence":
        children = spec.get("normalizers")
        fns: list[Callable[[str], str]] = []
        names: list[str] = []
        if isinstance(children, list):
            for child in children:
                sub = parse_normalizer(child)
                fns.extend(sub.fns)
                names.append(sub.name)
                caps = _merge_caps(caps, sub.capabilities)
        return Normalizer(name="Sequence[" + ",".join(names) + "]", fns=tuple(fns), capabilities=caps)

    if kind in ("NFC", "NFD", "NFKC", "NFKD"):
        fn = {"NFC": _nfc, "NFD": _nfd, "NFKC": _nfkc, "NFKD": _nfkd}[kind]
        caps = replace(
            caps,
            compatibility_fold=kind in ("NFKC", "NFKD"),
            canonical_fold=kind in ("NFC", "NFD"),
        )
        return Normalizer(name=kind, fns=(fn,), capabilities=caps)

    if kind == "Lowercase":
        return Normalizer(name="Lowercase", fns=(str.lower,), capabilities=replace(caps, lowercases=True))

    if kind == "StripAccents":
        return Normalizer(name="StripAccents", fns=(_strip_accents,), capabilities=replace(caps, strips_accents=True))

    if kind == "Strip":
        left = bool(spec.get("strip_left", True))
        right = bool(spec.get("strip_right", True))

        def _strip(s: str, left=left, right=right) -> str:
            if left and right:
                return s.strip()
            if left:
                return s.lstrip()
            if right:
                return s.rstrip()
            return s

        return Normalizer(name="Strip", fns=(_strip,))

    if kind == "Prepend":
        prepend = spec.get("prepend", "")
        if not isinstance(prepend, str):
            prepend = ""

        def _prepend(s: str, prepend=prepend) -> str:
            return prepend + s if s else s

        caps = replace(caps, space_to_metaspace=prepend == _METASPACE)
        return Normalizer(name="Prepend", fns=(_prepend,), capabilities=caps)

    if kind == "Replace":
        pattern = spec.get("pattern")
        content = spec.get("content", "")
        if isinstance(pattern, dict) and isinstance(pattern.get("String"), str) and isinstance(content, str):
            needle = pattern["String"]

            def _replace(s: str, needle=needle, content=content) -> str:
                return s.replace(needle, content)

            caps = replace(caps, space_to_metaspace=needle == " " and content == _METASPACE)
            return Normalizer(name="Replace", fns=(_replace,), capabilities=caps)
        # Regex-pattern Replace: we do not model regex rewriting -> unsupported.
        return Normalizer(name="Replace(regex)", capabilities=NormalizerCapabilities(unsupported=True))

    if kind == "BertNormalizer":
        lowercase = bool(spec.get("lowercase", True))
        strip_accents = spec.get("strip_accents")
        clean_text = bool(spec.get("clean_text", True))
        fns = []
        if clean_text:
            fns.append(_bert_clean_text)
        # strip_accents defaults to the value of lowercase when null in HF.
        do_strip = lowercase if strip_accents is None else bool(strip_accents)
        if do_strip:
            fns.append(_strip_accents)
        if lowercase:
            fns.append(str.lower)
        caps = NormalizerCapabilities(lowercases=lowercase, strips_accents=do_strip, cleans_text=clean_text)
        return Normalizer(name="BertNormalizer", fns=tuple(fns), capabilities=caps)

    if kind in ("Nmt",):
        # Nmt cleans a fixed set of control chars; approximate with clean_text.
        return Normalizer(name="Nmt", fns=(_bert_clean_text,), capabilities=NormalizerCapabilities(cleans_text=True))

    if kind in ("Precompiled",):
        # Opaque precompiled charsmap (SentencePiece) -- we cannot model it.
        return Normalizer(name="Precompiled", capabilities=NormalizerCapabilities(unsupported=True))

    return Normalizer(name=f"unsupported:{kind}", capabilities=NormalizerCapabilities(unsupported=True))


def _merge_caps(a: NormalizerCapabilities, b: NormalizerCapabilities) -> NormalizerCapabilities:
    return NormalizerCapabilities(
        compatibility_fold=a.compatibility_fold or b.compatibility_fold,
        canonical_fold=a.canonical_fold or b.canonical_fold,
        lowercases=a.lowercases or b.lowercases,
        strips_accents=a.strips_accents or b.strips_accents,
        space_to_metaspace=a.space_to_metaspace or b.space_to_metaspace,
        cleans_text=a.cleans_text or b.cleans_text,
        unsupported=a.unsupported or b.unsupported,
    )
